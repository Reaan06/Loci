"""Authenticated local Evolution webhook and durable inbox/outbox worker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from typing import Any, Callable, Protocol

from .agent import Agent
from .core import BrainStore, Clock, ReminderDispatcher, SystemClock, encode_datetime, require_utc_datetime
from .evolution import EvolutionConfig, parse_evolution_message

MAX_WEBHOOK_REQUEST_BYTES = 100_000
MAX_WEBHOOK_RESPONSE_BYTES = 1_000
BUSY_REPLY = "Error: message is already being processed"


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes = b""


class OutboundTextTransport(Protocol):
    def send_text(self, text: str) -> None: ...


class WebhookApplication:
    """Framework-free handler. The HTTP server binds loopback unless explicitly opted in."""

    max_request_bytes = MAX_WEBHOOK_REQUEST_BYTES

    def __init__(self, store: BrainStore, config: EvolutionConfig):
        self.store = store
        self.config = config

    def handle(self, method: str, path: str, headers: dict[str, str], body: bytes) -> HttpResponse:
        if method != "POST" or path != "/evolution/webhook":
            return HttpResponse(404)
        secret = next((value for key, value in headers.items() if key.lower() == "x-webhook-secret"), None)
        if not isinstance(secret, str) or not hmac.compare_digest(secret.encode(), self.config.webhook_secret.encode()):
            return HttpResponse(401)
        if len(body) > self.max_request_bytes:
            return HttpResponse(413)
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return HttpResponse(400)
        message = parse_evolution_message(payload, self.config)
        if message is None:
            return HttpResponse(202)
        try:
            self.store.accept_inbound_message(
                message.message_id,
                message.text,
                channel="evolution",
                binding=self.config.binding,
            )
        except ValueError:
            return HttpResponse(409)
        return HttpResponse(202)


def create_webhook_server(host: str, port: int, database_path: Path | str, config: EvolutionConfig, *, clock: Clock | None = None) -> ThreadingHTTPServer:
    """Create a deliberately small server; only loopback is allowed by default."""
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("non-loopback webhook binding requires an explicit private-network deployment wrapper")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = self.headers.get("Content-Length")
            try:
                request_size = int(length) if length is not None else -1
            except ValueError:
                request_size = -1
            if request_size < 0 or request_size > MAX_WEBHOOK_REQUEST_BYTES:
                self.send_response(413)
                self.end_headers()
                return
            body = self.rfile.read(request_size)
            store = BrainStore(database_path, clock=clock)
            try:
                response = WebhookApplication(store, config).handle("POST", self.path, dict(self.headers.items()), body)
            finally:
                store.close()
            self.send_response(response.status)
            self.send_header("Content-Length", str(len(response.body)))
            self.end_headers()
            if response.body:
                self.wfile.write(response.body)

        def log_message(self, format: str, *args: object) -> None:
            # Never log webhook bodies, API keys, phone identities, or headers.
            return

    return ThreadingHTTPServer((host, port), Handler)


class EvolutionWorker:
    """One short polling pass with distinct SQLite connections per work item."""

    def __init__(
        self,
        database_path: Path | str,
        agent_factory: Callable[[BrainStore], Agent],
        transport: OutboundTextTransport,
        *,
        config: EvolutionConfig | None = None,
        clock: Clock | None = None,
    ):
        self.database_path = Path(database_path)
        self.agent_factory = agent_factory
        self.transport = transport
        self.clock = clock or SystemClock()
        resolved_config = config or getattr(transport, "config", None)
        if not isinstance(resolved_config, EvolutionConfig):
            raise ValueError("EvolutionWorker requires a valid EvolutionConfig from transport or config argument")
        self.config = resolved_config
        self.binding = self.config.binding

    def process_once(self) -> int:
        processed = self._process_inbox()
        return processed + self._process_outbox()

    def _process_inbox(self) -> int:
        store = BrainStore(self.database_path, clock=self.clock)
        try:
            claim = store.claim_inbound_message(
                self.clock.now(), channel="evolution", binding=self.binding
            )
            if claim is None:
                return 0
            message, token = claim
            try:
                reply = self.agent_factory(store).handle(message.text, message_id=message.id)
                if reply.text == BUSY_REPLY:
                    store.retry_inbound_message(message.id, token, self.clock.now())
                elif not store.complete_inbound_message(message.id, token, reply.text):
                    return 0
            except Exception:
                store.retry_inbound_message(message.id, token, self.clock.now())
            return 1
        finally:
            store.close()

    def _process_outbox(self) -> int:
        store = BrainStore(self.database_path, clock=self.clock)
        try:
            claim = store.claim_outbox_message(
                self.clock.now(), channel="evolution", binding=self.binding
            )
            if claim is None:
                return 0
            message, token = claim
            try:
                self.transport.send_text(message.text)
            except Exception:
                store.retry_outbox_message(message.id, token, self.clock.now())
            else:
                store.mark_outbox_sent(message.id, token)
            return 1
        finally:
            store.close()


class TelegramWorkerProcess:
    """Worker process for Telegram inbox, outbox, and automatic reminder routing."""

    def __init__(
        self,
        database_path: Path | str,
        agent_factory: Callable[[BrainStore], Agent],
        client: Any,
        config: Any,
        *,
        clock: Clock | None = None,
    ):
        self.database_path = Path(database_path)
        self.agent_factory = agent_factory
        self.client = client
        self.config = config
        self.clock = clock or SystemClock()
        self.binding = self.config.binding

    def process_once(self, stop: threading.Event | None = None) -> int:
        if stop is not None and stop.is_set():
            return 0
        processed = self._process_inbox()
        if stop is not None and stop.is_set():
            return processed
        processed += self._process_outbox()
        if stop is not None and stop.is_set():
            return processed
        processed += self.process_reminders(stop=stop)
        return processed

    def drain_due(self, max_iterations: int = 100, stop: threading.Event | None = None) -> tuple[int, bool]:
        """
        Drain currently due inbox, outbox, and reminder work for this configured channel and binding.
        Stops when no work is due, max_iterations is reached, or stop event is set.
        Failed rows are backed off in the database and not immediately re-claimed in the same pass.
        Returns (processed_count, cap_reached).
        """
        total_processed = 0
        claimed_inbound_ids: set[str] = set()
        claimed_outbox_ids: set[str] = set()
        dispatched_reminder_ids: set[str] = set()

        for _ in range(max_iterations):
            if stop is not None and stop.is_set():
                break

            work_done_this_iter = 0

            # 1. Inbox
            if stop is not None and stop.is_set():
                break
            inbox_done = self._process_inbox_drain(claimed_inbound_ids)
            work_done_this_iter += inbox_done

            # 2. Outbox
            if stop is not None and stop.is_set():
                break
            outbox_done = self._process_outbox_drain(claimed_outbox_ids)
            work_done_this_iter += outbox_done

            # 3. Reminders
            if stop is not None and stop.is_set():
                break
            reminders_done = self._process_reminders_drain(dispatched_reminder_ids)
            work_done_this_iter += reminders_done

            total_processed += work_done_this_iter
            if work_done_this_iter == 0:
                return total_processed, False

        return total_processed, True

    def _process_inbox_drain(self, already_claimed: set[str]) -> int:
        store = BrainStore(self.database_path, clock=self.clock)
        try:
            claim = store.claim_inbound_message(
                self.clock.now(), channel="telegram", binding=self.binding
            )
            if claim is None:
                return 0
            message, token = claim
            if message.id in already_claimed:
                # We already claimed/handled this item during this drain pass, avoid infinite loop on failed items
                store.retry_inbound_message(message.id, token, self.clock.now())
                return 0
            already_claimed.add(message.id)
            try:
                reply = self.agent_factory(store).handle(message.text, message_id=message.id)
                if reply.text == BUSY_REPLY:
                    store.retry_inbound_message(message.id, token, self.clock.now())
                    return 0
                elif not store.complete_inbound_message(message.id, token, reply.text):
                    return 0
            except Exception as e:
                store.retry_inbound_message(message.id, token, self.clock.now())
                sys.stderr.write(f"telegram_worker_error: inbox_failure: {type(e).__name__}\n")
                return 0
            return 1
        finally:
            store.close()

    def _process_outbox_drain(self, already_claimed: set[str]) -> int:
        store = BrainStore(self.database_path, clock=self.clock)
        try:
            claim = store.claim_outbox_message(
                self.clock.now(), channel="telegram", binding=self.binding
            )
            if claim is None:
                return 0
            message, token = claim
            if message.id in already_claimed:
                store.retry_outbox_message(message.id, token, self.clock.now())
                return 0
            already_claimed.add(message.id)
            try:
                self.client.send_message(self.config.owner_chat_id, message.text)
            except Exception as e:
                store.retry_outbox_message(message.id, token, self.clock.now())
                sys.stderr.write(f"telegram_worker_error: outbox_failure: {type(e).__name__}\n")
                return 0
            else:
                store.mark_outbox_sent(message.id, token)
            return 1
        finally:
            store.close()

    def _process_reminders_drain(self, already_dispatched: set[str]) -> int:
        from .telegram import TelegramTransport

        store = BrainStore(self.database_path, clock=self.clock)
        try:
            transport = TelegramTransport(self.client, self.config)
            claim = store.claim_due_reminder(
                self.clock.now(),
                channel="telegram",
                binding=self.binding,
            )
            if claim is None:
                return 0
            reminder, token = claim
            if reminder.id in already_dispatched:
                store.mark_delivery_failed(reminder.id, token, self.clock.now())
                return 0
            already_dispatched.add(reminder.id)
            try:
                transport.send(reminder)
            except Exception as e:
                store.mark_delivery_failed(reminder.id, token, self.clock.now())
                sys.stderr.write(f"telegram_worker_error: reminder_failure: {type(e).__name__}\n")
            else:
                store.mark_sent(reminder.id, token)
            return 1
        finally:
            store.close()

    def _process_inbox(self) -> int:
        store = BrainStore(self.database_path, clock=self.clock)
        try:
            claim = store.claim_inbound_message(
                self.clock.now(), channel="telegram", binding=self.binding
            )
            if claim is None:
                return 0
            message, token = claim
            try:
                reply = self.agent_factory(store).handle(message.text, message_id=message.id)
                if reply.text == BUSY_REPLY:
                    store.retry_inbound_message(message.id, token, self.clock.now())
                    return 0
                elif not store.complete_inbound_message(message.id, token, reply.text):
                    return 0
            except Exception as e:
                store.retry_inbound_message(message.id, token, self.clock.now())
                sys.stderr.write(f"telegram_worker_error: inbox_failure: {type(e).__name__}\n")
                return 0
            return 1
        finally:
            store.close()

    def _process_outbox(self) -> int:
        store = BrainStore(self.database_path, clock=self.clock)
        try:
            claim = store.claim_outbox_message(
                self.clock.now(), channel="telegram", binding=self.binding
            )
            if claim is None:
                return 0
            message, token = claim
            try:
                self.client.send_message(self.config.owner_chat_id, message.text)
            except Exception as e:
                store.retry_outbox_message(message.id, token, self.clock.now())
                sys.stderr.write(f"telegram_worker_error: outbox_failure: {type(e).__name__}\n")
                return 0
            else:
                store.mark_outbox_sent(message.id, token)
            return 1
        finally:
            store.close()

    def process_reminders(self, limit: int = 100, stop: threading.Event | None = None) -> int:
        from .telegram import TelegramTransport

        if limit <= 0:
            raise ValueError("limit must be positive")
        if stop is not None and stop.is_set():
            return 0

        store = BrainStore(self.database_path, clock=self.clock)
        try:
            transport = TelegramTransport(self.client, self.config)
            attempts = 0
            for _ in range(limit):
                if stop is not None and stop.is_set():
                    break
                claim = store.claim_due_reminder(
                    self.clock.now(),
                    channel="telegram",
                    binding=self.binding,
                )
                if claim is None:
                    break
                reminder, token = claim
                attempts += 1
                try:
                    transport.send(reminder)
                except Exception as e:
                    store.mark_delivery_failed(reminder.id, token, self.clock.now())
                    sys.stderr.write(f"telegram_worker_error: reminder_failure: {type(e).__name__}\n")
                else:
                    store.mark_sent(reminder.id, token)
            return attempts
        finally:
            store.close()
