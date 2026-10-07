import hashlib
import hmac
import json
import sys
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_brain.agent import Agent
from second_brain.calendar import CalendarRetryableError
from second_brain.core import BrainStore, FakeClock
from second_brain.evolution import EvolutionConfig
from second_brain.model import ModelRequestError, ModelResponse, ModelToolCall
from second_brain.server import (
    EvolutionWorker,
    TelegramWorkerProcess,
    WebhookApplication,
    create_webhook_server,
)
from second_brain.telegram import (
    TelegramClient,
    TelegramConfig,
    TelegramTransport,
)


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingTransport:
    def __init__(self, failures=0, config=None):
        self.failures = failures
        self.sent = []
        self.config = config

    def send_text(self, text):
        self.sent.append(text)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("temporary Evolution failure")


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "brain.sqlite3"
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.clock = FakeClock(self.now)
        self.config = EvolutionConfig(
            base_url="https://evolution.example", api_key="key", instance_name="assistant",
            owner_jid="15551234567@s.whatsapp.net", webhook_secret="webhook-secret",
        )
        self.store = BrainStore(self.db, clock=self.clock)
        self.application = WebhookApplication(self.store, self.config)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def payload(self, text="save this", message_id="provider-id"):
        return json.dumps({
            "event": "messages.upsert", "instance": "assistant",
            "data": {"key": {"id": message_id, "fromMe": False, "remoteJid": "15551234567@s.whatsapp.net"}, "message": {"conversation": text}},
        }).encode()

    def request(self, body, secret="webhook-secret"):
        return self.application.handle("POST", "/evolution/webhook", {"X-Webhook-Secret": secret}, body)

    def test_auth_and_malformed_input_are_rejected_without_persistence(self):
        self.assertEqual(401, self.request(self.payload(), "wrong").status)
        self.assertEqual(404, self.application.handle("POST", "/evolution/webhook?secret=webhook-secret", {}, self.payload()).status)
        self.assertEqual(400, self.request(b"not-json").status)
        self.assertEqual(404, self.application.handle("POST", "/wrong", {"X-Webhook-Secret": "webhook-secret"}, self.payload()).status)
        self.assertEqual([], self.store.list_inbound_messages())
        self.assertEqual(413, self.request(b"x" * (self.application.max_request_bytes + 1)).status)

    def test_accepted_input_is_durable_before_ack_and_duplicate_id_is_safe(self):
        first = self.request(self.payload())
        self.assertEqual(202, first.status)
        rows = self.store.list_inbound_messages()
        self.assertEqual(1, len(rows))
        self.store.close()
        self.store = BrainStore(self.db, clock=self.clock)
        self.application = WebhookApplication(self.store, self.config)
        self.assertEqual(202, self.request(self.payload()).status)
        self.assertEqual(409, self.request(self.payload("other text")).status)
        self.assertEqual(1, len(self.store.list_inbound_messages()))

    def test_worker_completes_once_then_outbox_failure_retries_without_replaying_model(self):
        self.request(self.payload("save a note"))
        model = FakeModel([ModelResponse(None, (ModelToolCall("one", "save_note", {"text": "durable"}),))])
        transport = RecordingTransport(failures=1, config=self.config)
        worker = EvolutionWorker(self.db, lambda store: Agent(store, model), transport, clock=self.clock)

        self.assertEqual(2, worker.process_once())
        self.assertEqual(["Saved note: durable"], transport.sent)
        self.assertEqual(["durable"], [note.text for note in self.store.search_notes("durable")])
        self.clock.advance(seconds=2)
        self.assertEqual(1, worker.process_once())
        self.assertEqual(["Saved note: durable", "Saved note: durable"], transport.sent)
        self.assertEqual(1, model.calls)
        self.assertEqual("sent", self.store.list_outbox_messages()[0].status)
        self.assertEqual(202, self.request(self.payload("save a note")).status)
        self.assertEqual(1, model.calls)

    def test_model_failure_and_busy_reply_are_retried_instead_of_completed(self):
        self.request(self.payload("save", "retry-model"))
        failed = EvolutionWorker(
            self.db, lambda store: Agent(store, FakeModel([ModelRequestError("temporary")])), RecordingTransport(config=self.config), clock=self.clock,
        )
        self.assertEqual(1, failed.process_once())
        row = self.store.list_inbound_messages()[0]
        self.assertEqual("pending", row.status)
        self.clock.advance(seconds=2)
        recovered_model = FakeModel([ModelResponse(None, (ModelToolCall("ok", "save_note", {"text": "recovered"}),))])
        recovered = EvolutionWorker(self.db, lambda store: Agent(store, recovered_model), RecordingTransport(config=self.config), clock=self.clock)
        self.assertEqual(2, recovered.process_once())
        self.assertEqual("completed", self.store.list_inbound_messages()[0].status)

    def test_transient_calendar_failure_keeps_inbox_retryable(self):
        self.request(self.payload("calendar", "calendar-retry"))

        class Calendar:
            config = object()
            def normalize_create(self, arguments, config):
                return {"calendar_id": "primary"}
            def create_binding(self, fields):
                return "test-config-binding"
            def create_event(self, fields, message_id):
                raise CalendarRetryableError("temporary")

        model = FakeModel([ModelResponse(None, (ModelToolCall("calendar", "calendar_create", {
            "summary": "Plan", "start": "2026-06-01T09:00:00Z", "end": "2026-06-01T10:00:00Z"
        }),))])
        worker = EvolutionWorker(self.db, lambda store: Agent(store, model, calendar=Calendar()), RecordingTransport(config=self.config), clock=self.clock)
        self.assertEqual(1, worker.process_once())
        self.assertEqual("pending", self.store.list_inbound_messages()[0].status)

    def test_actual_loopback_server_accepts_authenticated_delivery(self):
        server = create_webhook_server("127.0.0.1", 0, self.db, self.config, clock=self.clock)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            request = Request(
                f"http://127.0.0.1:{server.server_port}/evolution/webhook",
                data=self.payload(), headers={"X-Webhook-Secret": "webhook-secret", "Content-Type": "application/json"}, method="POST",
            )
            with urlopen(request, timeout=2) as response:
                self.assertEqual(202, response.status)
            self.assertEqual(1, len(self.store.list_inbound_messages()))
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_leases_are_fenced_and_recovered_by_another_connection(self):
        self.request(self.payload("save", "lease-id"))
        claimed = self.store.claim_inbound_message(self.now, claim_seconds=30)
        other = BrainStore(self.db, clock=FakeClock(self.now + timedelta(seconds=31)))
        try:
            takeover = other.claim_inbound_message(self.now + timedelta(seconds=31), claim_seconds=30)
            self.assertEqual(claimed[0].id, takeover[0].id)
            self.assertFalse(self.store.retry_inbound_message(claimed[0].id, claimed[1], self.now))
            self.assertTrue(other.retry_inbound_message(takeover[0].id, takeover[1], self.now + timedelta(seconds=31)))
        finally:
            other.close()

    def test_evolution_worker_does_not_send_telegram_unbound_or_different_owner(self):
        with self.store.connection:
            # 1. Unbound legacy outbox message
            self.store.connection.execute(
                "INSERT INTO outbox_messages (id, text, status, attempts, next_attempt_at, lease_token, lease_until, channel, binding) "
                "VALUES ('out-unbound', 'unbound text', 'pending', 0, ?, NULL, NULL, 'evolution', NULL)",
                (self.now.isoformat().replace("+00:00", "Z"),),
            )
            # 2. Telegram outbox message
            self.store.connection.execute(
                "INSERT INTO outbox_messages (id, text, status, attempts, next_attempt_at, lease_token, lease_until, channel, binding) "
                "VALUES ('out-tg', 'telegram text', 'pending', 0, ?, NULL, NULL, 'telegram', 'some-tg-binding')",
                (self.now.isoformat().replace("+00:00", "Z"),),
            )
            # 3. Different owner outbox message
            other_config = EvolutionConfig(
                base_url="https://evolution.example", api_key="key", instance_name="assistant",
                owner_jid="15559999999@s.whatsapp.net", webhook_secret="webhook-secret",
            )
            self.store.connection.execute(
                "INSERT INTO outbox_messages (id, text, status, attempts, next_attempt_at, lease_token, lease_until, channel, binding) "
                "VALUES ('out-other', 'other text', 'pending', 0, ?, NULL, NULL, 'evolution', ?)",
                (self.now.isoformat().replace("+00:00", "Z"), other_config.binding),
            )

        transport = RecordingTransport(config=self.config)
        worker = EvolutionWorker(self.db, lambda store: Agent(store, FakeModel([])), transport, clock=self.clock)

        # Worker with self.config must NOT claim or send unbound, telegram, or other owner rows
        self.assertEqual(0, worker.process_once())
        self.assertEqual([], transport.sent)

        # Verify all 3 messages remain pending in SQLite
        outbox = {row.id: row.status for row in self.store.list_outbox_messages()}
        self.assertEqual("pending", outbox["out-unbound"])
        self.assertEqual("pending", outbox["out-tg"])
        self.assertEqual("pending", outbox["out-other"])

        # Insert a matching outbox message for this worker
        with self.store.connection:
            self.store.connection.execute(
                "INSERT INTO outbox_messages (id, text, status, attempts, next_attempt_at, lease_token, lease_until, channel, binding) "
                "VALUES ('out-matching', 'matching text', 'pending', 0, ?, NULL, NULL, 'evolution', ?)",
                (self.now.isoformat().replace("+00:00", "Z"), self.config.binding),
            )

        # Worker now claims and sends ONLY the matching message
        self.assertEqual(1, worker.process_once())
        self.assertEqual(["matching text"], transport.sent)
        self.assertEqual("sent", [m.status for m in self.store.list_outbox_messages() if m.id == "out-matching"][0])

        # Worker without transport config is rejected
        with self.assertRaisesRegex(ValueError, "EvolutionConfig"):
            EvolutionWorker(self.db, lambda store: Agent(store, FakeModel([])), RecordingTransport(config=None), clock=self.clock)

    def test_telegram_worker_process_processes_inbox_outbox_and_reminders(self):
        tg_config = TelegramConfig(
            bot_token="123:test-token",
            owner_user_id=100,
            owner_chat_id=200,
            bot_namespace="test-bot",
        )
        sent_messages = []

        def recording_transport(request: Request, timeout: float) -> bytes:
            sent_messages.append(json.loads(request.data))
            return json.dumps({"ok": True, "result": {"message_id": 999, "chat": {"id": 200, "type": "private"}}}).encode()

        tg_client = TelegramClient(tg_config, transport=recording_transport)

        # 1. Enqueue inbound telegram message
        self.store.accept_inbound_message(
            "tg-msg-1", "remind me to call mom", channel="telegram", binding=tg_config.binding
        )

        model = FakeModel([ModelResponse(None, (ModelToolCall("r1", "save_note", {"text": "call mom"}),))])
        worker_proc = TelegramWorkerProcess(
            self.db,
            lambda store: Agent(store, model),
            tg_client,
            tg_config,
            clock=self.clock,
        )

        # Process inbox and outbox
        self.assertEqual(2, worker_proc.process_once())
        self.assertEqual(1, len(sent_messages))
        self.assertEqual(200, sent_messages[0]["chat_id"])
        self.assertEqual("Saved note: call mom", sent_messages[0]["text"])

        # 2. Test reminder dispatch routes to Telegram
        # Create a pending reminder bound to nothing or bound to telegram
        self.store.create_reminder("Buy groceries", self.now, incoming_message_id="rem-init")
        sent_messages.clear()

        # Reminder process dispatch
        dispatched = worker_proc.process_reminders()
        self.assertEqual(1, dispatched)
        self.assertEqual(1, len(sent_messages))
        self.assertEqual(200, sent_messages[0]["chat_id"])
        self.assertEqual("Buy groceries", sent_messages[0]["text"])

        # Check in DB that reminder is now sent and bound to telegram and tg_config.binding
        rem = self.store.list_reminders(include_finished=True)[0]
        self.assertEqual("sent", rem.status)
        self.assertEqual("telegram", rem.channel)
        self.assertEqual(tg_config.binding, rem.binding)


    def test_main_cli_telegram_once(self):
        from second_brain.__main__ import build_parser, run
        from unittest.mock import patch

        parser = build_parser()
        args = parser.parse_args(["--db", str(self.db), "telegram", "--once"])
        self.assertEqual("telegram", args.command)
        self.assertTrue(args.once)

        with patch("second_brain.__main__.run_telegram") as mock_run_tg:
            mock_run_tg.return_value = 0
            code = run(args)
            self.assertEqual(0, code)
            mock_run_tg.assert_called_once_with(args)

    def test_regression_telegram_409_conflict_stops_with_explanation_and_nonzero(self):
        import io
        from unittest.mock import patch, MagicMock
        from second_brain.__main__ import build_parser, run_telegram
        from second_brain.telegram import TelegramConflictError

        parser = build_parser()
        # Test 1a: --once mode with 409
        args_once = parser.parse_args(["--db", str(self.db), "telegram", "--once"])
        mock_poller_once = MagicMock()
        mock_poller_once.poll_once.side_effect = TelegramConflictError("HTTP 409 Conflict: webhook active")

        stderr_capture = io.StringIO()
        with patch.dict("os.environ", {
            "SECOND_BRAIN_TELEGRAM_BOT_TOKEN": "123:test-secret-token",
            "SECOND_BRAIN_TELEGRAM_OWNER_USER_ID": "100",
            "SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID": "200",
            "SECOND_BRAIN_PROVIDER_URL": "https://api.example.com/v1",
            "SECOND_BRAIN_PROVIDER_KEY": "test-key",
            "SECOND_BRAIN_PROVIDER_MODEL": "test-model",
        }), patch("second_brain.telegram.TelegramPoller", return_value=mock_poller_once), \
           patch("sys.stderr", stderr_capture):
            code = run_telegram(args_once)

        self.assertNotEqual(0, code)
        err = stderr_capture.getvalue()
        self.assertIn("409", err)
        self.assertIn("webhook", err.lower())
        self.assertNotIn("test-secret-token", err)

        # Test 1b: Continuous mode with 409
        args_cont = parser.parse_args(["--db", str(self.db), "telegram"])
        mock_poller_cont = MagicMock()
        mock_poller_cont.poll_once.side_effect = TelegramConflictError("HTTP 409 Conflict: another poller active")

        stderr_capture_cont = io.StringIO()
        with patch.dict("os.environ", {
            "SECOND_BRAIN_TELEGRAM_BOT_TOKEN": "123:test-secret-token",
            "SECOND_BRAIN_TELEGRAM_OWNER_USER_ID": "100",
            "SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID": "200",
            "SECOND_BRAIN_PROVIDER_URL": "https://api.example.com/v1",
            "SECOND_BRAIN_PROVIDER_KEY": "test-key",
            "SECOND_BRAIN_PROVIDER_MODEL": "test-model",
        }), patch("second_brain.telegram.TelegramPoller", return_value=mock_poller_cont), \
           patch("sys.stderr", stderr_capture_cont):
            code_cont = run_telegram(args_cont)

        self.assertNotEqual(0, code_cont)
        err_cont = stderr_capture_cont.getvalue()
        self.assertIn("409", err_cont)
        self.assertIn("conflict", err_cont.lower())
        self.assertNotIn("test-secret-token", err_cont)

    def test_regression_telegram_network_backoff_and_reset(self):
        import io
        from unittest.mock import patch, MagicMock
        from second_brain.__main__ import build_parser, run_telegram
        from second_brain.telegram import TelegramRequestError

        parser = build_parser()
        args = parser.parse_args(["--db", str(self.db), "telegram"])

        mock_poller = MagicMock()
        # Fail 3 times, then succeed, then stop
        poll_count = 0
        def fake_poll():
            nonlocal poll_count
            poll_count += 1
            if poll_count <= 3:
                raise TelegramRequestError("Connection reset")
            return 1

        mock_poller.poll_once.side_effect = fake_poll

        wait_calls = []
        stop = threading.Event()
        real_wait = stop.wait

        def tracked_wait(timeout=None):
            if timeout is not None:
                wait_calls.append(timeout)
            if poll_count >= 4:
                stop.set()
            return real_wait(0.001)

        stop.wait = tracked_wait

        stderr_capture = io.StringIO()
        with patch.dict("os.environ", {
            "SECOND_BRAIN_TELEGRAM_BOT_TOKEN": "123:test-token",
            "SECOND_BRAIN_TELEGRAM_OWNER_USER_ID": "100",
            "SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID": "200",
            "SECOND_BRAIN_PROVIDER_URL": "https://api.example.com/v1",
            "SECOND_BRAIN_PROVIDER_KEY": "test-key",
            "SECOND_BRAIN_PROVIDER_MODEL": "test-model",
        }), patch("second_brain.telegram.TelegramPoller", return_value=mock_poller), \
           patch("sys.stderr", stderr_capture):
            code = run_telegram(args, stop=stop)

        self.assertEqual(0, code)
        # Should have backed off 1.0, 2.0, 4.0
        self.assertIn(1.0, wait_calls)
        self.assertIn(2.0, wait_calls)
        self.assertIn(4.0, wait_calls)
        self.assertIn("telegram_request_error", stderr_capture.getvalue().lower())

    def test_regression_telegram_worker_exceptions_visible(self):
        import io
        from unittest.mock import patch
        from second_brain.telegram import TelegramConfig, TelegramClient

        tg_config = TelegramConfig(
            bot_token="123:test-token",
            owner_user_id=100,
            owner_chat_id=200,
            bot_namespace="test-bot",
        )

        def failing_transport(request: Request, timeout: float) -> bytes:
            raise RuntimeError("Outbox transport network failure")

        tg_client = TelegramClient(tg_config, transport=failing_transport)

        # Enqueue an inbound message
        self.store.accept_inbound_message(
            "msg-fail-test", "hello world", channel="telegram", binding=tg_config.binding
        )
        # Process inbound successfully to generate outbox message
        model = FakeModel([ModelResponse("Hello back", ())])
        worker_proc = TelegramWorkerProcess(
            self.db,
            lambda store: Agent(store, model),
            tg_client,
            tg_config,
            clock=self.clock,
        )

        # Complete inbound so outbox has a pending message
        worker_proc._process_inbox()

        # Now processing outbox should fail and write sanitized error to stderr
        stderr_capture = io.StringIO()
        with patch("sys.stderr", stderr_capture):
            worker_proc._process_outbox()

        err = stderr_capture.getvalue()
        self.assertTrue(len(err) > 0)
        self.assertIn("telegram_worker_error: outbox_failure", err)
        self.assertIn("TelegramRequestError", err)
        self.assertNotIn("test-token", err)

    def test_regression_telegram_drain_due_processes_all_due_items_exact_binding(self):
        from second_brain.telegram import TelegramConfig, TelegramClient

        tg_config = TelegramConfig(
            bot_token="123:test-token",
            owner_user_id=100,
            owner_chat_id=200,
            bot_namespace="test-bot",
        )
        sent_messages = []

        def recording_transport(request: Request, timeout: float) -> bytes:
            sent_messages.append(json.loads(request.data))
            return json.dumps({"ok": True, "result": {"message_id": 999, "chat": {"id": 200, "type": "private"}}}).encode()

        tg_client = TelegramClient(tg_config, transport=recording_transport)

        # Enqueue 3 inbound telegram messages
        for i in range(1, 4):
            self.store.accept_inbound_message(
                f"tg-in-{i}", f"message {i}", channel="telegram", binding=tg_config.binding
            )

        # Create 1 due reminder bound to telegram
        self.store.create_reminder("Test reminder", self.now, incoming_message_id="rem-due-1")

        model = FakeModel([
            ModelResponse("Reply 1", ()),
            ModelResponse("Reply 2", ()),
            ModelResponse("Reply 3", ()),
        ])
        worker_proc = TelegramWorkerProcess(
            self.db,
            lambda store: Agent(store, model),
            tg_client,
            tg_config,
            clock=self.clock,
        )

        # Use drain_due method (or run via --once)
        processed, cap_reached = worker_proc.drain_due(max_iterations=100)
        self.assertFalse(cap_reached)
        self.assertGreaterEqual(processed, 7)  # 3 inbounds + 3 outbox replies + 1 reminder

        # Check all 3 inbounds completed
        inbounds = self.store.list_inbound_messages(channel="telegram", binding=tg_config.binding)
        self.assertEqual(3, len(inbounds))
        self.assertTrue(all(m.status == "completed" for m in inbounds))

        # Check reminder sent
        reminders = self.store.list_reminders(include_finished=True)
        self.assertEqual("sent", reminders[0].status)

    def test_regression_telegram_cli_once_drains_multiple_messages(self):
        from unittest.mock import patch, MagicMock
        from second_brain.__main__ import build_parser, run_telegram
        from second_brain.telegram import TelegramConfig, TelegramClient

        tg_config = TelegramConfig(
            bot_token="123:test-token",
            owner_user_id=100,
            owner_chat_id=200,
            bot_namespace="123",
        )
        sent = []
        def recording_transport(request: Request, timeout: float) -> bytes:
            sent.append(json.loads(request.data))
            return json.dumps({"ok": True, "result": {"message_id": 999, "chat": {"id": 200, "type": "private"}}}).encode()

        # Enqueue 3 inbound messages
        for i in range(1, 4):
            self.store.accept_inbound_message(
                f"cli-once-{i}", f"cli message {i}", channel="telegram", binding=tg_config.binding
            )

        parser = build_parser()
        args_once = parser.parse_args(["--db", str(self.db), "telegram", "--once"])

        mock_poller = MagicMock()
        mock_poller.poll_once.return_value = 0

        model = FakeModel([
            ModelResponse("Reply 1", ()),
            ModelResponse("Reply 2", ()),
            ModelResponse("Reply 3", ()),
        ])

        with patch.dict("os.environ", {
            "SECOND_BRAIN_TELEGRAM_BOT_TOKEN": "123:test-token",
            "SECOND_BRAIN_TELEGRAM_OWNER_USER_ID": "100",
            "SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID": "200",
            "SECOND_BRAIN_PROVIDER_URL": "https://api.example.com/v1",
            "SECOND_BRAIN_PROVIDER_KEY": "test-key",
            "SECOND_BRAIN_PROVIDER_MODEL": "test-model",
        }), patch("second_brain.telegram.TelegramPoller", return_value=mock_poller), \
           patch("second_brain.telegram.TelegramClient", return_value=TelegramClient(tg_config, transport=recording_transport)), \
           patch("second_brain.model.OpenAICompatibleModel.from_environment", return_value=model):
            code = run_telegram(args_once)

        self.assertEqual(0, code)
        inbounds = self.store.list_inbound_messages(channel="telegram", binding=tg_config.binding)
        # ALL 3 should be completed, not just 1!
        self.assertTrue(all(m.status == "completed" for m in inbounds))
        self.assertEqual(3, len(sent))

    def test_regression_telegram_drain_cap_leaves_durable_work_and_warns(self):
        import io
        from unittest.mock import patch
        from second_brain.telegram import TelegramConfig, TelegramClient

        tg_config = TelegramConfig(
            bot_token="123:test-token",
            owner_user_id=100,
            owner_chat_id=200,
            bot_namespace="test-bot",
        )
        def recording_transport(request: Request, timeout: float) -> bytes:
            return json.dumps({"ok": True, "result": {"message_id": 999, "chat": {"id": 200, "type": "private"}}}).encode()

        tg_client = TelegramClient(tg_config, transport=recording_transport)

        # Enqueue 5 messages
        for i in range(1, 6):
            self.store.accept_inbound_message(
                f"cap-msg-{i}", f"text {i}", channel="telegram", binding=tg_config.binding
            )

        model = FakeModel([
            ModelResponse(f"Reply {i}", ()) for i in range(1, 10)
        ])
        worker_proc = TelegramWorkerProcess(
            self.db,
            lambda store: Agent(store, model),
            tg_client,
            tg_config,
            clock=self.clock,
        )

        # Drain with max_iterations=2 (each iteration handles 1 inbox + 1 outbox)
        processed, cap_reached = worker_proc.drain_due(max_iterations=2)
        self.assertTrue(cap_reached)

        # Verify durable queue still has remaining pending items
        pending_inbounds = [
            m for m in self.store.list_inbound_messages(channel="telegram", binding=tg_config.binding)
            if m.status == "pending"
        ]
        self.assertGreater(len(pending_inbounds), 0)

    def test_regression_telegram_drain_channel_isolation_no_steal(self):
        from second_brain.telegram import TelegramConfig, TelegramClient

        tg_config = TelegramConfig(
            bot_token="123:test-token",
            owner_user_id=100,
            owner_chat_id=200,
            bot_namespace="test-bot",
        )
        sent_messages = []
        def recording_transport(request: Request, timeout: float) -> bytes:
            sent_messages.append(json.loads(request.data))
            return json.dumps({"ok": True, "result": {"message_id": 999, "chat": {"id": 200, "type": "private"}}}).encode()

        tg_client = TelegramClient(tg_config, transport=recording_transport)

        # Enqueue message for Evolution channel
        self.store.accept_inbound_message(
            "evo-msg-1", "evolution message", channel="evolution", binding="evo:instance"
        )
        # Enqueue message for a different telegram binding
        self.store.accept_inbound_message(
            "tg-diff-msg", "diff binding message", channel="telegram", binding="telegram:other_binding"
        )

        model = FakeModel([ModelResponse("Reply", ())])
        worker_proc = TelegramWorkerProcess(
            self.db,
            lambda store: Agent(store, model),
            tg_client,
            tg_config,
            clock=self.clock,
        )

        processed, cap_reached = worker_proc.drain_due(max_iterations=10)
        self.assertEqual(0, processed)
        self.assertFalse(cap_reached)
        self.assertEqual(0, len(sent_messages))

        # Check neither was touched
        evo_msg = self.store.list_inbound_messages(channel="evolution", binding="evo:instance")[0]
        self.assertEqual("pending", evo_msg.status)
        diff_msg = self.store.list_inbound_messages(channel="telegram", binding="telegram:other_binding")[0]
        self.assertEqual("pending", diff_msg.status)

    def test_regression_telegram_sigint_clean_shutdown(self):
        from unittest.mock import patch, MagicMock
        from second_brain.__main__ import build_parser, run_telegram

        parser = build_parser()
        args = parser.parse_args(["--db", str(self.db), "telegram"])

        mock_poller = MagicMock()
        def fake_poll():
            raise KeyboardInterrupt()

        mock_poller.poll_once.side_effect = fake_poll

        with patch.dict("os.environ", {
            "SECOND_BRAIN_TELEGRAM_BOT_TOKEN": "123:test-token",
            "SECOND_BRAIN_TELEGRAM_OWNER_USER_ID": "100",
            "SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID": "200",
            "SECOND_BRAIN_PROVIDER_URL": "https://api.example.com/v1",
            "SECOND_BRAIN_PROVIDER_KEY": "test-key",
            "SECOND_BRAIN_PROVIDER_MODEL": "test-model",
        }), patch("second_brain.telegram.TelegramPoller", return_value=mock_poller):
            code = run_telegram(args)

        self.assertEqual(0, code)

    def test_regression_telegram_stop_event_checked_before_outbox_and_reminders(self):
        """
        Verify that when stop is set between stages:
        - Setting stop before process_once does not claim inbox.
        - Setting stop between inbox and outbox prevents claiming outbox.
        """
        from second_brain.core import encode_datetime
        from second_brain.telegram import TelegramConfig, TelegramClient

        tg_config = TelegramConfig(
            bot_token="123:test-token",
            owner_user_id=100,
            owner_chat_id=200,
            bot_namespace="test-bot",
        )
        sent = []
        def recording_transport(request: Request, timeout: float) -> bytes:
            sent.append(json.loads(request.data))
            return json.dumps({"ok": True, "result": {"message_id": 999, "chat": {"id": 200, "type": "private"}}}).encode()

        tg_client = TelegramClient(tg_config, transport=recording_transport)

        # Enqueue an outbox message directly
        self.store.accept_inbound_message("inbox-1", "msg 1", channel="telegram", binding=tg_config.binding)
        # Put another item directly in outbox
        with self.store._transaction():
            self.store.connection.execute(
                """INSERT INTO outbox_messages
                   (id, text, status, attempts, next_attempt_at, lease_token, lease_until, channel, binding)
                   VALUES ('outbox-1', 'reply 1', 'pending', 0, ?, NULL, NULL, 'telegram', ?)""",
                (encode_datetime(self.clock.now()), tg_config.binding),
            )

        stop = threading.Event()
        stop.set()  # Already stopped!

        model = FakeModel([ModelResponse("Reply", ())])
        worker_proc = TelegramWorkerProcess(
            self.db,
            lambda store: Agent(store, model),
            tg_client,
            tg_config,
            clock=self.clock,
        )

        # process_once with stop already set should process 0 items and make 0 claims
        processed = worker_proc.process_once(stop=stop)
        self.assertEqual(0, processed)
        self.assertEqual(0, len(sent))

        # Test stop set after inbox (hook inside agent_factory)
        stop_after_inbox = threading.Event()
        def agent_factory_with_stop(store):
            stop_after_inbox.set()
            return Agent(store, model)

        worker_proc2 = TelegramWorkerProcess(
            self.db,
            agent_factory_with_stop,
            tg_client,
            tg_config,
            clock=self.clock,
        )
        processed2 = worker_proc2.process_once(stop=stop_after_inbox)
        self.assertEqual(1, processed2)  # only inbox processed, outbox was blocked by stop
        self.assertEqual(0, len(sent))  # outbox was not sent

    def test_regression_telegram_stop_during_reminders_leaves_subsequent_reminders_pending(self):
        """
        Verify that setting stop during reminder dispatch stops further claims
        and leaves subsequent reminders in pending status without claiming them.
        """
        from second_brain.telegram import TelegramConfig, TelegramClient

        tg_config = TelegramConfig(
            bot_token="123:test-token",
            owner_user_id=100,
            owner_chat_id=200,
            bot_namespace="test-bot",
        )
        sent = []
        stop = threading.Event()

        def recording_transport(request: Request, timeout: float) -> bytes:
            sent.append(json.loads(request.data))
            # Stop right after first send!
            stop.set()
            return json.dumps({"ok": True, "result": {"message_id": 999, "chat": {"id": 200, "type": "private"}}}).encode()

        tg_client = TelegramClient(tg_config, transport=recording_transport)

        # Schedule 3 reminders with distinct due times to guarantee claim order
        rem1 = self.store.create_reminder("Reminder 1", self.clock.now(), incoming_message_id="rem-1")
        rem2 = self.store.create_reminder("Reminder 2", self.clock.now() + timedelta(seconds=1), incoming_message_id="rem-2")
        rem3 = self.store.create_reminder("Reminder 3", self.clock.now() + timedelta(seconds=2), incoming_message_id="rem-3")

        model = FakeModel([])
        worker_proc = TelegramWorkerProcess(
            self.db,
            lambda store: Agent(store, model),
            tg_client,
            tg_config,
            clock=self.clock,
        )

        dispatched = worker_proc.process_reminders(limit=10, stop=stop)
        self.assertEqual(1, dispatched)
        self.assertEqual(1, len(sent))

        # Verify reminders r2 and r3 remain pending and unleased
        reminders = {r.id: r for r in self.store.list_reminders(include_finished=True)}
        self.assertEqual("sent", reminders[rem1.id].status)
        self.assertEqual("pending", reminders[rem2.id].status)
        self.assertIsNone(self.store.connection.execute("SELECT claim_token FROM reminders WHERE id = ?", (rem2.id,)).fetchone()["claim_token"])
        self.assertEqual("pending", reminders[rem3.id].status)
        self.assertIsNone(self.store.connection.execute("SELECT claim_token FROM reminders WHERE id = ?", (rem3.id,)).fetchone()["claim_token"])

    def test_regression_telegram_slow_inflight_worker_finishes_before_db_close_on_sigint(self):
        """
        Simulate an in-flight model call that takes > join_warn_seconds.
        On SIGINT:
        - The worker is non-daemon and allowed to finish cleanly.
        - The main thread does not abruptly close SQLite or exit while worker is alive.
        - A diagnostic warning is emitted when join exceeds threshold, but wait continues.
        - The result is completed and persisted; exit code is 0.
        """
        import io
        import time
        from unittest.mock import patch, MagicMock
        from second_brain.__main__ import build_parser, run_telegram
        from second_brain.telegram import TelegramConfig, TelegramClient

        tg_config = TelegramConfig(
            bot_token="123:test-token",
            owner_user_id=100,
            owner_chat_id=200,
            bot_namespace="123",
        )
        sent = []
        def recording_transport(request: Request, timeout: float) -> bytes:
            sent.append(json.loads(request.data))
            return json.dumps({"ok": True, "result": {"message_id": 999, "chat": {"id": 200, "type": "private"}}}).encode()

        # Enqueue 1 message
        self.store.accept_inbound_message(
            "slow-msg-1", "slow message", channel="telegram", binding=tg_config.binding
        )

        parser = build_parser()
        args = parser.parse_args(["--db", str(self.db), "telegram"])

        mock_poller = MagicMock()
        mock_poller.poll_once.return_value = 0

        # Slow model response that signals it has started and waits for SIGINT simulation
        in_flight_event = threading.Event()
        release_model_event = threading.Event()

        class SlowModel:
            def complete(self, prompt, system_instruction=None, json_schema=None):
                in_flight_event.set()
                # Wait until simulated join starts
                release_model_event.wait(timeout=2.0)
                return ModelResponse("Slow model finished", ())

        # When main poller loop starts, wait for model to be in flight, then simulate SIGINT
        def fake_poll_once():
            if in_flight_event.wait(timeout=2.0):
                # Trigger shutdown while worker is still in-flight
                raise KeyboardInterrupt()
            raise TimeoutError("Model never entered in-flight state")

        mock_poller.poll_once.side_effect = fake_poll_once

        stderr_capture = io.StringIO()

        def delayed_release():
            time.sleep(0.08)
            release_model_event.set()

        releaser = threading.Thread(target=delayed_release)
        releaser.start()

        try:
            with patch.dict("os.environ", {
                "SECOND_BRAIN_TELEGRAM_BOT_TOKEN": "123:test-token",
                "SECOND_BRAIN_TELEGRAM_OWNER_USER_ID": "100",
                "SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID": "200",
                "SECOND_BRAIN_PROVIDER_URL": "https://api.example.com/v1",
                "SECOND_BRAIN_PROVIDER_KEY": "test-key",
                "SECOND_BRAIN_PROVIDER_MODEL": "test-model",
            }), patch("second_brain.telegram.TelegramPoller", return_value=mock_poller), \
               patch("second_brain.telegram.TelegramClient", return_value=TelegramClient(tg_config, transport=recording_transport)), \
               patch("second_brain.model.OpenAICompatibleModel.from_environment", return_value=SlowModel()), \
               patch("sys.stderr", stderr_capture):
                code = run_telegram(args, join_warn_seconds=0.02)
        finally:
            release_model_event.set()
            releaser.join(timeout=5.0)
        self.assertEqual(0, code)

        # Verify warning was emitted because join took > 0.02s
        err = stderr_capture.getvalue()
        self.assertIn("telegram_shutdown_warning", err)
        self.assertIn("waiting for in-flight work", err.lower())

        # Verify inbound completed and outbox row exists in durable database
        inbound = self.store.list_inbound_messages(channel="telegram", binding=tg_config.binding)[0]
        self.assertEqual("completed", inbound.status)
        outbox = self.store.list_outbox_messages(channel="telegram", binding=tg_config.binding)[0]
        # Inbound completed and generated an outbox row. Because stop was set while inbox was in-flight,
        # outbox was not claimed/sent after stop, safely remaining pending for the next run.
        self.assertEqual("pending", outbox.status)
        self.assertEqual("No action was performed. Please clarify what you need.", outbox.text)

    def test_regression_telegram_once_sigint_during_drain_exits_cleanly_without_traceback(self):
        """
        Verify that in --once mode, an actual SIGINT during drain_due:
        - Traps cleanly without raising KeyboardInterrupt or dumping a raw traceback to stderr.
        - Sets the stop event so no subsequent items are claimed after the interrupt.
        - Exits cleanly (code 0) without leaking secrets or crashing.
        - Restores original signal handler.
        """
        import io
        import signal
        from unittest.mock import patch, MagicMock
        from second_brain.__main__ import build_parser, run_telegram
        from second_brain.telegram import TelegramConfig, TelegramClient

        tg_config = TelegramConfig(
            bot_token="123:test-token",
            owner_user_id=100,
            owner_chat_id=200,
            bot_namespace="123",
        )
        sent = []
        def recording_transport(request: Request, timeout: float) -> bytes:
            sent.append(json.loads(request.data))
            return json.dumps({"ok": True, "result": {"message_id": 999, "chat": {"id": 200, "type": "private"}}}).encode()

        # Enqueue 2 inbound messages
        self.store.accept_inbound_message("once-sigint-1", "first text", channel="telegram", binding=tg_config.binding)
        self.store.accept_inbound_message("once-sigint-2", "second text", channel="telegram", binding=tg_config.binding)

        parser = build_parser()
        args = parser.parse_args(["--db", str(self.db), "telegram", "--once"])

        mock_poller = MagicMock()
        mock_poller.poll_once.return_value = 0

        # Model that triggers SIGINT on the first completion
        call_count = 0
        def fake_complete(prompt, system_instruction=None, json_schema=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Trigger real OS signal SIGINT to self
                import os
                os.kill(os.getpid(), signal.SIGINT)
            return ModelResponse("Done", ())

        mock_model = MagicMock()
        mock_model.complete.side_effect = fake_complete

        stderr_capture = io.StringIO()
        orig_handler = signal.getsignal(signal.SIGINT)
        observed_handler = None

        try:
            with patch.dict("os.environ", {
                "SECOND_BRAIN_TELEGRAM_BOT_TOKEN": "123:test-token",
                "SECOND_BRAIN_TELEGRAM_OWNER_USER_ID": "100",
                "SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID": "200",
                "SECOND_BRAIN_PROVIDER_URL": "https://api.example.com/v1",
                "SECOND_BRAIN_PROVIDER_KEY": "test-key",
                "SECOND_BRAIN_PROVIDER_MODEL": "test-model",
            }), patch("second_brain.telegram.TelegramPoller", return_value=mock_poller), \
               patch("second_brain.telegram.TelegramClient", return_value=TelegramClient(tg_config, transport=recording_transport)), \
               patch("second_brain.model.OpenAICompatibleModel.from_environment", return_value=mock_model), \
               patch("sys.stderr", stderr_capture):
                code = run_telegram(args)
            observed_handler = signal.getsignal(signal.SIGINT)
        finally:
            signal.signal(signal.SIGINT, orig_handler)

        self.assertEqual(0, code)
        # Original signal handler must be restored
        self.assertEqual(orig_handler, observed_handler)

        # Check stderr: no raw traceback or secrets
        err = stderr_capture.getvalue()
        self.assertNotIn("Traceback", err)
        self.assertNotIn("test-key", err)
        self.assertNotIn("test-token", err)

        # Message 1 completed
        m1 = self.store.list_inbound_messages(channel="telegram", binding=tg_config.binding)
        m1_map = {m.id: m for m in m1}
        self.assertEqual("completed", m1_map["once-sigint-1"].status)
        # Message 2 was NOT claimed or processed because stop was set upon SIGINT
        self.assertEqual("pending", m1_map["once-sigint-2"].status)

    def test_regression_telegram_once_subprocess_sigint_during_drain_exits_cleanly(self):
        """
        Verify with an actual external subprocess that --once receiving SIGINT during drain_due:
        - Exits cleanly without raw traceback or secret leakage.
        - Leaves subsequent queue items intact in pending state.
        """
        import os
        import selectors
        import signal
        import subprocess
        import sys
        import time

        from second_brain.telegram import TelegramConfig

        db_path = str(self.db)
        child_env = {
            "SECOND_BRAIN_TELEGRAM_BOT_TOKEN": "123:test-token",
            "SECOND_BRAIN_TELEGRAM_OWNER_USER_ID": "100",
            "SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID": "200",
            "SECOND_BRAIN_TELEGRAM_BOT_NAMESPACE": "123",
            "SECOND_BRAIN_PROVIDER_URL": "https://api.example.com/v1",
            "SECOND_BRAIN_PROVIDER_KEY": "secret-provider-key",
            "SECOND_BRAIN_PROVIDER_MODEL": "test-model",
        }
        tg_config = TelegramConfig.from_environment(child_env)
        tg_binding = tg_config.binding

        # Enqueue 2 inbound messages
        self.store.accept_inbound_message("sub-once-1", "first sub text", channel="telegram", binding=tg_binding)
        self.store.accept_inbound_message("sub-once-2", "second sub text", channel="telegram", binding=tg_binding)

        # Run python script via subprocess that mocks TelegramPoller to avoid network, but runs real __main__
        code = """
import os, sys, time, signal
from unittest.mock import patch, MagicMock
from second_brain.model import ModelResponse
from second_brain.telegram import TelegramConfig, TelegramClient

def fake_complete(*args, **kwargs):
    sys.stderr.write("IN_FLIGHT_CALL\\n")
    sys.stderr.flush()
    time.sleep(0.5)
    return ModelResponse("Subprocess done", ())

mock_model = MagicMock()
mock_model.complete.side_effect = fake_complete

mock_poller = MagicMock()
mock_poller.poll_once.return_value = 0

tg_config = TelegramConfig(
    bot_token="123:test-token",
    owner_user_id=100,
    owner_chat_id=200,
    bot_namespace="123",
)

def recording_transport(req, timeout):
    return b'{"ok": true, "result": {"message_id": 999, "chat": {"id": 200, "type": "private"}}}'

with patch.dict("os.environ", {
    "SECOND_BRAIN_TELEGRAM_BOT_TOKEN": "123:test-token",
    "SECOND_BRAIN_TELEGRAM_OWNER_USER_ID": "100",
    "SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID": "200",
    "SECOND_BRAIN_TELEGRAM_BOT_NAMESPACE": "123",
    "SECOND_BRAIN_PROVIDER_URL": "https://api.example.com/v1",
    "SECOND_BRAIN_PROVIDER_KEY": "secret-provider-key",
    "SECOND_BRAIN_PROVIDER_MODEL": "test-model",
}), patch("second_brain.telegram.TelegramPoller", return_value=mock_poller), \
   patch("second_brain.telegram.TelegramClient", return_value=TelegramClient(tg_config, transport=recording_transport)), \
   patch("second_brain.model.OpenAICompatibleModel.from_environment", return_value=mock_model):
    from second_brain.__main__ import main
    sys.exit(main())
"""
        proc = None
        sel = None
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", code, "--db", db_path, "telegram", "--once"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, **child_env, "PYTHONPATH": "src"},
            )

            sel = selectors.DefaultSelector()
            sel.register(proc.stderr, selectors.EVENT_READ)
            stderr_chunks: list[bytes] = []
            marker_found = False
            deadline = time.monotonic() + 5.0

            while time.monotonic() < deadline:
                remaining = max(0.01, deadline - time.monotonic())
                events = sel.select(timeout=min(remaining, 0.2))
                if events:
                    try:
                        chunk = os.read(proc.stderr.fileno(), 1024)
                    except OSError:
                        chunk = b""
                    if chunk:
                        stderr_chunks.append(chunk)
                        if b"IN_FLIGHT_CALL" in b"".join(stderr_chunks):
                            marker_found = True
                            break
                    else:
                        # EOF received on pipe
                        break
                if proc.poll() is not None:
                    break

            if not marker_found:
                poll_status = proc.poll()
                try:
                    leftover = os.read(proc.stderr.fileno(), 4096)
                    if leftover:
                        stderr_chunks.append(leftover)
                except OSError:
                    pass
                captured = b"".join(stderr_chunks).decode("utf-8", errors="replace")
                self.fail(
                    f"Child process did not emit marker IN_FLIGHT_CALL within timeout. "
                    f"Child exit status={poll_status}, captured stderr: {captured!r}"
                )

            # Send SIGINT while in flight
            proc.send_signal(signal.SIGINT)

            stdout_bytes, stderr_rest = proc.communicate(timeout=10.0)
            stderr_chunks.append(stderr_rest)
            full_stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")

            # Exit code must be 0
            self.assertEqual(0, proc.returncode)

            # Check stderr for cleanliness
            self.assertNotIn("Traceback", full_stderr)
            self.assertNotIn("secret-provider-key", full_stderr)
            self.assertNotIn("123:test-token", full_stderr)

            # First message completed, second remained pending
            inbounds = {m.id: m for m in self.store.list_inbound_messages(channel="telegram", binding=tg_binding)}
            self.assertEqual("completed", inbounds["sub-once-1"].status)
            self.assertEqual("pending", inbounds["sub-once-2"].status)

            # Durable response persisted and new outbox remains unclaimed (pending)
            outboxes = self.store.list_outbox_messages(channel="telegram", binding=tg_binding)
            self.assertEqual(1, len(outboxes))
            self.assertEqual("pending", outboxes[0].status)
            self.assertIsNone(self.store.connection.execute("SELECT lease_token FROM outbox_messages WHERE id = ?", (outboxes[0].id,)).fetchone()["lease_token"])
            self.assertEqual("No action was performed. Please clarify what you need.", outboxes[0].text)
        finally:
            if sel is not None:
                try:
                    sel.close()
                except Exception:
                    pass
            if proc is not None:
                if proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.communicate(timeout=3.0)
                    except (subprocess.TimeoutExpired, OSError):
                        try:
                            proc.kill()
                            proc.communicate(timeout=3.0)
                        except (subprocess.TimeoutExpired, OSError):
                            pass
                if proc.stdout and not proc.stdout.closed:
                    try:
                        proc.stdout.close()
                    except OSError:
                        pass
                if proc.stderr and not proc.stderr.closed:
                    try:
                        proc.stderr.close()
                    except OSError:
                        pass
                try:
                    proc.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, OSError):
                    pass



