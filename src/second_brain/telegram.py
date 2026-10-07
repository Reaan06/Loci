from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from os import environ
import re
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler

from .core import (
    BrainStore,
    Clock,
    Reminder,
    SystemClock,
    Transport,
    encode_datetime,
    require_utc_datetime,
)

logger = logging.getLogger(__name__)

TELEGRAM_MAX_TEXT_LENGTH = 4096
TELEGRAM_TRUNCATION_MARKER = "... [truncated]"
TELEGRAM_MAX_PAYLOAD_BYTES = 512 * 1024


class TelegramError(Exception):
    """Base error for Telegram operations."""


class TelegramConfigError(TelegramError):
    """Raised when Telegram configuration is invalid."""


class TelegramRequestError(TelegramError):
    """Raised when an HTTP error occurs talking to Telegram."""


class TelegramConflictError(TelegramRequestError):
    """Raised when Telegram returns HTTP 409 Conflict (e.g. concurrent poller)."""


class TelegramEnvelopeError(TelegramError):
    """Raised when an update envelope is malformed."""


@dataclass(frozen=True)
class InboundTelegramMessage:
    message_id: str
    text: str


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    owner_user_id: int
    owner_chat_id: int
    bot_namespace: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.bot_token, str) or not self.bot_token.strip():
            raise ValueError("bot_token must be a non-empty string")
        if not re.match(r"^[A-Za-z0-9_:-]+$", self.bot_token):
            raise ValueError("bot_token contains invalid characters")
        if not isinstance(self.owner_user_id, int) or isinstance(self.owner_user_id, bool) or self.owner_user_id <= 0:
            raise ValueError("owner_user_id must be a positive integer")
        if not isinstance(self.owner_chat_id, int) or isinstance(self.owner_chat_id, bool) or self.owner_chat_id <= 0:
            raise ValueError("owner_chat_id must be a positive integer")
        if not isinstance(self.bot_namespace, str) or not self.bot_namespace.strip():
            raise ValueError("bot_namespace must be a non-empty string")

    def __repr__(self) -> str:
        return (
            f"TelegramConfig(bot_token='***', owner_user_id={self.owner_user_id}, "
            f"owner_chat_id={self.owner_chat_id}, bot_namespace={self.bot_namespace!r})"
        )

    @property
    def binding(self) -> str:
        digest = hashlib.sha256(
            f"{self.bot_namespace}:{self.owner_user_id}:{self.owner_chat_id}".encode()
        ).hexdigest()[:16]
        return f"telegram:{digest}"

    @classmethod
    def from_environment(cls, env: dict[str, str] | None = None) -> TelegramConfig:
        lookup = environ if env is None else env
        raw_token = lookup.get("SECOND_BRAIN_TELEGRAM_BOT_TOKEN")
        raw_user_id = lookup.get("SECOND_BRAIN_TELEGRAM_OWNER_USER_ID")
        raw_chat_id = lookup.get("SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID")
        raw_ns = lookup.get("SECOND_BRAIN_TELEGRAM_BOT_NAMESPACE")

        if not raw_token or not raw_token.strip():
            raise ValueError("SECOND_BRAIN_TELEGRAM_BOT_TOKEN is required")
        if not raw_user_id or not raw_user_id.strip():
            raise ValueError("SECOND_BRAIN_TELEGRAM_OWNER_USER_ID is required")
        if not raw_chat_id or not raw_chat_id.strip():
            raise ValueError("SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID is required")

        try:
            owner_user_id = int(raw_user_id.strip())
        except (ValueError, TypeError):
            raise ValueError("SECOND_BRAIN_TELEGRAM_OWNER_USER_ID must be a positive integer")

        try:
            owner_chat_id = int(raw_chat_id.strip())
        except (ValueError, TypeError):
            raise ValueError("SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID must be a positive integer")

        bot_token = raw_token.strip()
        if raw_ns and raw_ns.strip():
            bot_namespace = raw_ns.strip()
        else:
            prefix_match = re.match(r"^(\d+):", bot_token)
            if prefix_match:
                bot_namespace = prefix_match.group(1)
            else:
                bot_namespace = "default"

        return cls(
            bot_token=bot_token,
            owner_user_id=owner_user_id,
            owner_chat_id=owner_chat_id,
            bot_namespace=bot_namespace,
        )


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise TelegramRequestError(f"Unexpected redirect from Telegram API (HTTP {code})")


def _default_transport(request: Request, timeout: float) -> bytes:
    opener = build_opener(_NoRedirectHandler)
    with opener.open(request, timeout=timeout) as response:
        return response.read()


TELEGRAM_API_BASE_URL = "https://api.telegram.org"


class TelegramClient:
    def __init__(
        self,
        config: TelegramConfig,
        *,
        transport: Callable[[Request, float], bytes] | None = None,
        base_url: str = TELEGRAM_API_BASE_URL,
    ):
        self.config = config
        self._transport = transport or _default_transport
        if base_url != TELEGRAM_API_BASE_URL:
            raise ValueError(f"Telegram base_url must be {TELEGRAM_API_BASE_URL}")
        self._base_url = TELEGRAM_API_BASE_URL

    def _request(self, method: str, payload: dict[str, Any], timeout: float = 35.0) -> Any:
        url = f"{self._base_url}/bot{self.config.bot_token}/{method}"
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        err_to_raise: Exception | None = None
        response_bytes: bytes = b""
        try:
            response_bytes = self._transport(req, timeout)
        except HTTPError as e:
            code = e.code
            e.close()
            if code == 409:
                err_to_raise = TelegramConflictError("Telegram returned HTTP 409 Conflict (active concurrent poll)")
            else:
                err_to_raise = TelegramRequestError(f"Telegram API HTTP error {code}")
        except TelegramRequestError:
            err_to_raise = TelegramRequestError("Telegram API request failed: transport error")
        except URLError:
            err_to_raise = TelegramRequestError("Telegram API request failed: network or connection error")
        except Exception as e:
            err_to_raise = TelegramRequestError(f"Telegram API request failed: {type(e).__name__}")

        if err_to_raise is not None:
            err_to_raise.__cause__ = None
            err_to_raise.__context__ = None
            err_to_raise.__suppress_context__ = True
            raise err_to_raise

        result: Any = None
        json_failed = False
        try:
            result = json.loads(response_bytes.decode("utf-8"))
        except Exception:
            json_failed = True

        if json_failed:
            err = TelegramEnvelopeError("Malformed JSON response from Telegram API")
            err.__cause__ = None
            err.__context__ = None
            err.__suppress_context__ = True
            raise err

        if not isinstance(result, dict) or result.get("ok") is not True or "result" not in result:
            error_code = result.get("error_code") if isinstance(result, dict) else None
            if error_code == 409:
                raise TelegramConflictError("Telegram returned HTTP 409 Conflict (active concurrent poll)")
            if error_code is not None and isinstance(error_code, int) and not isinstance(error_code, bool):
                raise TelegramRequestError(f"Telegram API error {error_code}")
            raise TelegramRequestError("Telegram API error")

        return result.get("result")

    def get_updates(
        self,
        *,
        offset: int | None = None,
        limit: int = 100,
        timeout: int = 25,
    ) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not (1 <= limit <= 100):
            raise ValueError("limit must be an integer between 1 and 100")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 0:
            raise ValueError("timeout must be a non-negative integer")
        payload: dict[str, Any] = {
            "timeout": timeout,
            "limit": limit,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
                raise ValueError("offset must be a non-negative integer")
            payload["offset"] = offset
        res = self._request("getUpdates", payload, timeout=float(timeout + 10))
        if not isinstance(res, list):
            raise TelegramEnvelopeError("getUpdates result must be a list")
        return res

    def send_message(self, chat_id: int, text: str) -> dict[str, Any]:
        if not isinstance(chat_id, int) or isinstance(chat_id, bool) or chat_id <= 0:
            raise ValueError("chat_id must be a positive integer")
        if len(text) > TELEGRAM_MAX_TEXT_LENGTH:
            cut = TELEGRAM_MAX_TEXT_LENGTH - len(TELEGRAM_TRUNCATION_MARKER)
            text = text[:cut] + TELEGRAM_TRUNCATION_MARKER
        payload = {
            "chat_id": chat_id,
            "text": text,
        }
        res = self._request("sendMessage", payload, timeout=20.0)
        if not isinstance(res, dict):
            raise TelegramRequestError("Malformed Telegram API response: result is not an object")

        msg_id = res.get("message_id")
        if not isinstance(msg_id, int) or isinstance(msg_id, bool) or msg_id <= 0:
            raise TelegramRequestError("Malformed Telegram API response: invalid message_id")

        chat = res.get("chat")
        if not isinstance(chat, dict):
            raise TelegramRequestError("Malformed Telegram API response: missing or invalid chat object")
        cid = chat.get("id")
        if not isinstance(cid, int) or isinstance(cid, bool) or cid <= 0:
            raise TelegramRequestError("Malformed Telegram API response: invalid chat id")
        if cid != chat_id:
            raise TelegramRequestError("Mismatched Telegram API response: destination chat mismatch")
        ctype = chat.get("type")
        if ctype != "private":
            raise TelegramRequestError("Mismatched Telegram API response: chat type is not private")

        return res


def parse_telegram_update(update: dict[str, Any], config: TelegramConfig) -> InboundTelegramMessage | None:
    if not isinstance(update, dict):
        return None
    message = update.get("message")
    if not isinstance(message, dict):
        return None

    sender = message.get("from")
    if not isinstance(sender, dict):
        return None
    if sender.get("is_bot") is not False:
        return None
    sender_id = sender.get("id")
    if not isinstance(sender_id, int) or isinstance(sender_id, bool) or sender_id <= 0:
        return None
    if sender_id != config.owner_user_id:
        return None

    chat = message.get("chat")
    if not isinstance(chat, dict):
        return None
    if chat.get("type") != "private":
        return None
    chat_id = chat.get("id")
    if not isinstance(chat_id, int) or isinstance(chat_id, bool) or chat_id <= 0:
        return None
    if chat_id != config.owner_chat_id:
        return None

    text = message.get("text")
    if not isinstance(text, str) or not text.strip():
        return None

    raw_msg_id = message.get("message_id")
    if not isinstance(raw_msg_id, int) or isinstance(raw_msg_id, bool) or raw_msg_id <= 0:
        return None

    stable_id = f"telegram:{config.bot_namespace}:{config.owner_chat_id}:{raw_msg_id}"
    return InboundTelegramMessage(message_id=stable_id, text=text.strip())


def validate_and_parse_batch(
    updates: list[dict[str, Any]],
    config: TelegramConfig,
) -> tuple[int | None, list[InboundTelegramMessage]]:
    highest_update_id: int | None = None
    accepted: list[InboundTelegramMessage] = []

    for item in updates:
        if not isinstance(item, dict):
            raise TelegramEnvelopeError("Update item is not a dictionary")
        uid = item.get("update_id")
        if not isinstance(uid, int) or isinstance(uid, bool) or uid < 0:
            raise TelegramEnvelopeError("Update item has invalid update_id")

        if "message" not in item:
            # Edit, channel post, or non-message update -> safely skip
            if highest_update_id is None or uid > highest_update_id:
                highest_update_id = uid
            continue

        message = item.get("message")
        if not isinstance(message, dict):
            raise TelegramEnvelopeError("Update message is not a dictionary")

        chat = message.get("chat")
        if not isinstance(chat, dict):
            raise TelegramEnvelopeError("Message chat is not a dictionary")

        chat_type = chat.get("type")
        if chat_type in ("group", "supergroup", "channel"):
            # Legitimate group or channel update -> safely skip
            if highest_update_id is None or uid > highest_update_id:
                highest_update_id = uid
            continue

        if chat_type != "private":
            raise TelegramEnvelopeError("Message chat type is not private")

        sender = message.get("from")
        if not isinstance(sender, dict):
            raise TelegramEnvelopeError("Message from is not a dictionary")

        if sender.get("is_bot") is True:
            # Legitimate bot message -> safely skip
            if highest_update_id is None or uid > highest_update_id:
                highest_update_id = uid
            continue

        sender_id = sender.get("id")
        chat_id = chat.get("id")

        if isinstance(sender_id, bool) or not isinstance(sender_id, int) or sender_id <= 0:
            raise TelegramEnvelopeError("Candidate direct message has malformed sender ID")

        if isinstance(chat_id, bool) or not isinstance(chat_id, int) or chat_id <= 0:
            raise TelegramEnvelopeError("Candidate direct message has malformed chat ID")

        if sender_id != config.owner_user_id:
            # Legitimate other owner/user -> safely skip
            if highest_update_id is None or uid > highest_update_id:
                highest_update_id = uid
            continue

        if chat_id != config.owner_chat_id:
            # Legitimate other chat -> safely skip
            if highest_update_id is None or uid > highest_update_id:
                highest_update_id = uid
            continue

        # Candidate direct message for owner: check message_id
        raw_msg_id = message.get("message_id")
        if isinstance(raw_msg_id, bool) or not isinstance(raw_msg_id, int) or raw_msg_id <= 0:
            raise TelegramEnvelopeError("Candidate direct message has invalid message_id")

        text = message.get("text")
        if text is None:
            media_keys = ("photo", "voice", "audio", "video", "document", "sticker", "animation", "contact", "location")
            if any(k in message for k in media_keys):
                # Legitimate media message from owner -> safely skip
                if highest_update_id is None or uid > highest_update_id:
                    highest_update_id = uid
                continue
            if highest_update_id is None or uid > highest_update_id:
                highest_update_id = uid
            continue

        if not isinstance(text, str) or not text.strip():
            if highest_update_id is None or uid > highest_update_id:
                highest_update_id = uid
            continue

        stable_id = f"telegram:{config.bot_namespace}:{config.owner_chat_id}:{raw_msg_id}"
        accepted.append(InboundTelegramMessage(message_id=stable_id, text=text.strip()))

        if highest_update_id is None or uid > highest_update_id:
            highest_update_id = uid

    next_offset = (highest_update_id + 1) if highest_update_id is not None else None
    return next_offset, accepted


class TelegramTransport(Transport):
    def __init__(self, client: TelegramClient, config: TelegramConfig):
        self.client = client
        self.config = config

    def send(self, reminder: Reminder) -> None:
        self.client.send_message(self.config.owner_chat_id, reminder.text)


class TelegramWorker:
    def __init__(
        self,
        store: BrainStore,
        client: TelegramClient,
        config: TelegramConfig,
        *,
        clock: Clock | None = None,
        lease_seconds: int = 60,
    ):
        self.store = store
        self.client = client
        self.config = config
        self.clock = clock or SystemClock()
        self.lease_seconds = lease_seconds

    def _now(self) -> datetime:
        return self.clock.now()

    def process_outbox(self, limit: int = 20) -> int:
        if limit <= 0:
            raise ValueError("limit must be positive")
        attempts = 0
        for _ in range(limit):
            claim = self.store.claim_outbox_message(
                self._now(),
                channel="telegram",
                binding=self.config.binding,
                claim_seconds=self.lease_seconds,
            )
            if claim is None:
                break
            message, token = claim
            attempts += 1
            try:
                self.client.send_message(self.config.owner_chat_id, message.text)
            except Exception:
                self.store.retry_outbox_message(message.id, token, self._now())
            else:
                self.store.mark_outbox_sent(message.id, token)
        return attempts


class TelegramPoller:
    def __init__(
        self,
        store: BrainStore,
        client: TelegramClient,
        config: TelegramConfig,
        *,
        clock: Clock | None = None,
        lease_seconds: int = 60,
    ):
        self.store = store
        self.client = client
        self.config = config
        self.clock = clock or SystemClock()
        self.lease_seconds = lease_seconds

    def _now(self) -> datetime:
        return self.clock.now()

    def poll_once(self) -> int:
        now = self._now()
        lease = self.store.acquire_poll_lease(
            self.config.bot_namespace,
            self.config.binding,
            now,
            lease_seconds=self.lease_seconds,
        )
        if lease is None:
            return 0
        lease_token, current_offset = lease

        try:
            updates = self.client.get_updates(offset=current_offset, timeout=10)
            if not updates:
                return 0
            next_offset, accepted = validate_and_parse_batch(updates, self.config)
            if next_offset is None:
                return 0

            batch_pairs = [(m.message_id, m.text) for m in accepted]
            committed = self.store.commit_poll_batch(
                self.config.bot_namespace,
                self.config.binding,
                lease_token,
                next_offset,
                batch_pairs,
                self._now(),
            )
            return len(accepted) if committed else 0
        finally:
            self.store.release_poll_lease(
                self.config.bot_namespace,
                self.config.binding,
                lease_token,
            )
