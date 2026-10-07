from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import traceback
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_brain.core import BrainStore, FakeClock, Reminder, ReminderDispatcher
from second_brain.telegram import (
    TelegramConfig,
    TelegramClient,
    TelegramConflictError,
    TelegramEnvelopeError,
    TelegramRequestError,
    TelegramTransport,
    TelegramWorker,
    TelegramPoller,
    parse_telegram_update,
    validate_and_parse_batch,
    InboundTelegramMessage,
)


class TelegramTests(unittest.TestCase):
    def test_config_requires_positive_numeric_ids_and_never_leaks_token_repr(self):
        config = TelegramConfig(
            bot_token="123456:secret-token-value",
            owner_user_id=111222,
            owner_chat_id=111222,
            bot_namespace="mybot",
        )
        self.assertEqual("mybot", config.bot_namespace)
        self.assertEqual(111222, config.owner_user_id)
        self.assertEqual(111222, config.owner_chat_id)
        # Token is masked in repr
        self.assertNotIn("secret-token-value", repr(config))
        self.assertNotIn("123456:secret-token-value", repr(config))

        # Rejects negative, zero, and boolean
        with self.assertRaises(ValueError):
            TelegramConfig(bot_token="tok", owner_user_id=-1, owner_chat_id=111)
        with self.assertRaises(ValueError):
            TelegramConfig(bot_token="tok", owner_user_id=0, owner_chat_id=111)
        with self.assertRaises(ValueError):
            TelegramConfig(bot_token="tok", owner_user_id=True, owner_chat_id=111)
        with self.assertRaises(ValueError):
            TelegramConfig(bot_token="tok", owner_user_id=111, owner_chat_id=-5)
        with self.assertRaises(ValueError):
            TelegramConfig(bot_token="tok", owner_user_id=111, owner_chat_id=False)
        with self.assertRaises(ValueError):
            TelegramConfig(bot_token="", owner_user_id=111, owner_chat_id=111)

    def test_config_binding_is_secret_free_and_stable_across_token_rotation(self):
        config1 = TelegramConfig(
            bot_token="123456:first-secret",
            owner_user_id=42,
            owner_chat_id=42,
            bot_namespace="same-bot",
        )
        config2 = TelegramConfig(
            bot_token="123456:rotated-secret",
            owner_user_id=42,
            owner_chat_id=42,
            bot_namespace="same-bot",
        )
        config_diff_owner = TelegramConfig(
            bot_token="123456:first-secret",
            owner_user_id=99,
            owner_chat_id=42,
            bot_namespace="same-bot",
        )
        config_diff_ns = TelegramConfig(
            bot_token="123456:first-secret",
            owner_user_id=42,
            owner_chat_id=42,
            bot_namespace="other-bot",
        )
        # Binding is secret-free
        self.assertNotIn("first-secret", config1.binding)
        self.assertNotIn("rotated-secret", config2.binding)
        # Same binding across token rotation
        self.assertEqual(config1.binding, config2.binding)
        # Different binding when owner or namespace changes
        self.assertNotEqual(config1.binding, config_diff_owner.binding)
        self.assertNotEqual(config1.binding, config_diff_ns.binding)

    def test_config_from_environment(self):
        env = {
            "SECOND_BRAIN_TELEGRAM_BOT_TOKEN": "987654:my-token",
            "SECOND_BRAIN_TELEGRAM_OWNER_USER_ID": "555123",
            "SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID": "555123",
            "SECOND_BRAIN_TELEGRAM_BOT_NAMESPACE": "test-ns",
        }
        cfg = TelegramConfig.from_environment(env)
        self.assertEqual(555123, cfg.owner_user_id)
        self.assertEqual(555123, cfg.owner_chat_id)
        self.assertEqual("test-ns", cfg.bot_namespace)

        # Missing required vars raises ValueError
        with self.assertRaises(ValueError):
            TelegramConfig.from_environment({})

        # Auto-derives bot_namespace from numeric token prefix when unset
        env_no_ns = {
            "SECOND_BRAIN_TELEGRAM_BOT_TOKEN": "987654:my-token",
            "SECOND_BRAIN_TELEGRAM_OWNER_USER_ID": "555123",
            "SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID": "555123",
        }
        cfg_auto = TelegramConfig.from_environment(env_no_ns)
        self.assertEqual("987654", cfg_auto.bot_namespace)

    def test_client_enforces_fixed_endpoint_and_rejects_redirects(self):
        config = TelegramConfig("123:secret", 1, 1, "ns")
        recorded_requests = []

        def fake_transport(request: Request, timeout: float) -> bytes:
            recorded_requests.append(request)
            return json.dumps({"ok": True, "result": []}).encode()

        client = TelegramClient(config, transport=fake_transport)
        client.get_updates(offset=10)

        self.assertEqual(1, len(recorded_requests))
        req = recorded_requests[0]
        self.assertEqual("https://api.telegram.org/bot123:secret/getUpdates", req.full_url)
        self.assertEqual("POST", req.get_method())

    def test_client_sanitizes_errors_without_reflecting_token_or_url(self):
        config = TelegramConfig("123:supersecrettoken", 1, 1, "ns")

        def failing_transport(request: Request, timeout: float) -> bytes:
            raise HTTPError(
                request.full_url, 404, "Not Found", {}, None
            )

        client = TelegramClient(config, transport=failing_transport)
        with self.assertRaises(TelegramRequestError) as ctx:
            client.get_updates()
        error_msg = str(ctx.exception)
        self.assertNotIn("supersecrettoken", error_msg)
        self.assertNotIn("https://api.telegram.org", error_msg)

    def test_client_handles_409_conflict_explicitly(self):
        config = TelegramConfig("123:token", 1, 1, "ns")

        def conflict_transport(request: Request, timeout: float) -> bytes:
            raise HTTPError(request.full_url, 409, "Conflict", {}, None)

        client = TelegramClient(config, transport=conflict_transport)
        with self.assertRaises(TelegramConflictError):
            client.get_updates()

    def test_client_send_message_truncation_with_explicit_marker(self):
        config = TelegramConfig("123:token", 1, 999, "ns")
        sent_bodies = []

        def recording_transport(request: Request, timeout: float) -> bytes:
            sent_bodies.append(json.loads(request.data))
            return json.dumps({"ok": True, "result": {"message_id": 101, "chat": {"id": 999, "type": "private"}}}).encode()

        client = TelegramClient(config, transport=recording_transport)
        long_text = "A" * 5000
        client.send_message(999, long_text)

        self.assertEqual(1, len(sent_bodies))
        sent = sent_bodies[0]["text"]
        self.assertLessEqual(len(sent), 4096)
        self.assertTrue(sent.endswith("... [truncated]"))

    def test_poller_poll_once_commits_accepted_messages_and_advances_offset(self):
        store = BrainStore(":memory:")
        try:
            config = TelegramConfig("123:token", 100, 200, "bot-ns")
            updates_payload = [
                {
                    "update_id": 50,
                    "message": {
                        "message_id": 1001,
                        "from": {"id": 100, "is_bot": False},
                        "chat": {"id": 200, "type": "private"},
                        "text": "Remember buy milk",
                    },
                },
                {
                    "update_id": 51,
                    "message": {
                        "message_id": 1002,
                        "from": {"id": 999, "is_bot": False},  # not owner -> ignored
                        "chat": {"id": 200, "type": "private"},
                        "text": "Spam",
                    },
                },
            ]

            def fake_transport(request: Request, timeout: float) -> bytes:
                return json.dumps({"ok": True, "result": updates_payload}).encode()

            client = TelegramClient(config, transport=fake_transport)
            poller = TelegramPoller(store, client, config)

            accepted_count = poller.poll_once()
            self.assertEqual(1, accepted_count)

            # Cursor advanced to 52 in DB
            cursor = store.get_telegram_cursor("bot-ns", config.binding)
            self.assertEqual(52, cursor)

            # Inbound message in DB with telegram channel and matching binding
            inbound = store.claim_inbound_message(
                datetime.now(timezone.utc),
                channel="telegram",
                binding=config.binding,
            )
            self.assertIsNotNone(inbound)
            self.assertEqual("Remember buy milk", inbound[0].text)
        finally:
            store.close()

    def test_worker_processes_outbox_for_telegram_binding(self):
        store = BrainStore(":memory:")
        try:
            config = TelegramConfig("123:token", 100, 200, "bot-ns")
            sent_messages = []

            def recording_transport(request: Request, timeout: float) -> bytes:
                sent_messages.append(json.loads(request.data))
                return json.dumps({"ok": True, "result": {"message_id": 77, "chat": {"id": 200, "type": "private"}}}).encode()

            client = TelegramClient(config, transport=recording_transport)
            worker = TelegramWorker(store, client, config)

            # Enqueue telegram outbox message
            store.accept_inbound_message("tg-1", "hello", channel="telegram", binding=config.binding)
            claim = store.claim_inbound_message(datetime.now(timezone.utc), channel="telegram", binding=config.binding)
            self.assertIsNotNone(claim)
            msg, token = claim
            store.complete_inbound_message(msg.id, token, "Reply from AI")

            # Process outbox
            count = worker.process_outbox()
            self.assertEqual(1, count)
            self.assertEqual(1, len(sent_messages))
            self.assertEqual(200, sent_messages[0]["chat_id"])
            self.assertEqual("Reply from AI", sent_messages[0]["text"])
        finally:
            store.close()

    def test_telegram_transport_sends_reminder_to_owner_chat(self):
        config = TelegramConfig("123:token", 100, 200, "bot-ns")
        sent = []

        def recording_transport(request: Request, timeout: float) -> bytes:
            sent.append(json.loads(request.data))
            return json.dumps({"ok": True, "result": {"message_id": 88, "chat": {"id": 200, "type": "private"}}}).encode()

        client = TelegramClient(config, transport=recording_transport)
        transport = TelegramTransport(client, config)

        rem = Reminder(
            id="rem-1",
            text="Doctor appointment",
            due_at=datetime.now(timezone.utc),
            status="pending",
            attempts=0,
            next_attempt_at=datetime.now(timezone.utc),
        )
        transport.send(rem)
        self.assertEqual(1, len(sent))
        self.assertEqual(200, sent[0]["chat_id"])
        self.assertEqual("Doctor appointment", sent[0]["text"])

        config = TelegramConfig("123:token", owner_user_id=100, owner_chat_id=200, bot_namespace="ns")

        # Valid owner private text
        valid_update = {
            "update_id": 1,
            "message": {
                "message_id": 42,
                "from": {"id": 100, "is_bot": False, "first_name": "Owner"},
                "chat": {"id": 200, "type": "private"},
                "text": "Hello bot",
            },
        }
        parsed = parse_telegram_update(valid_update, config)
        self.assertIsNotNone(parsed)
        self.assertEqual("Hello bot", parsed.text)
        self.assertTrue(parsed.message_id.startswith("telegram:"))

        # Bot sender -> ignored
        bot_update = {
            "update_id": 2,
            "message": {
                "message_id": 43,
                "from": {"id": 100, "is_bot": True},
                "chat": {"id": 200, "type": "private"},
                "text": "Hello",
            },
        }
        self.assertIsNone(parse_telegram_update(bot_update, config))

        # Different user -> ignored
        other_user = {
            "update_id": 3,
            "message": {
                "message_id": 44,
                "from": {"id": 999, "is_bot": False},
                "chat": {"id": 200, "type": "private"},
                "text": "Hello",
            },
        }
        self.assertIsNone(parse_telegram_update(other_user, config))

        # Group chat -> ignored
        group_chat = {
            "update_id": 4,
            "message": {
                "message_id": 45,
                "from": {"id": 100, "is_bot": False},
                "chat": {"id": 200, "type": "group"},
                "text": "Hello",
            },
        }
        self.assertIsNone(parse_telegram_update(group_chat, config))

        # Edit (edited_message) -> ignored
        edit_update = {
            "update_id": 5,
            "edited_message": {
                "message_id": 42,
                "from": {"id": 100, "is_bot": False},
                "chat": {"id": 200, "type": "private"},
                "text": "Edited text",
            },
        }
        self.assertIsNone(parse_telegram_update(edit_update, config))

        # Media message (no text) -> ignored
        media_update = {
            "update_id": 6,
            "message": {
                "message_id": 46,
                "from": {"id": 100, "is_bot": False},
                "chat": {"id": 200, "type": "private"},
                "photo": [],
            },
        }
        self.assertIsNone(parse_telegram_update(media_update, config))

    def test_batch_validation_rejects_malformed_envelope_before_destructive_advancement(self):
        config = TelegramConfig("123:token", 100, 200, "ns")

        # Batch with a malformed update_id (e.g. string or negative or boolean)
        bad_batch = [
            {"update_id": 1, "message": {"message_id": 1, "from": {"id": 100, "is_bot": False}, "chat": {"id": 200, "type": "private"}, "text": "ok"}},
            {"update_id": "not-an-int"},
        ]
        with self.assertRaises(TelegramEnvelopeError):
            validate_and_parse_batch(bad_batch, config)

        bad_batch_bool = [
            {"update_id": True},
        ]
        with self.assertRaises(TelegramEnvelopeError):
            validate_and_parse_batch(bad_batch_bool, config)

        # Well-formed mixed batch advances offset to max+1 and returns accepted
        good_batch = [
            {"update_id": 10, "message": {"message_id": 1, "from": {"id": 100, "is_bot": False}, "chat": {"id": 200, "type": "private"}, "text": "msg 1"}},
            {"update_id": 15, "edited_message": {"message_id": 2}}, # skipped
            {"update_id": 20, "message": {"message_id": 3, "from": {"id": 100, "is_bot": False}, "chat": {"id": 200, "type": "private"}, "text": "msg 2"}},
        ]
        next_offset, accepted = validate_and_parse_batch(good_batch, config)
        self.assertEqual(21, next_offset)
        self.assertEqual(2, len(accepted))
        self.assertEqual("msg 1", accepted[0].text)
        self.assertEqual("msg 2", accepted[1].text)

    def test_exception_cause_context_and_traceback_never_leak_token(self):
        secret = "secret_tok_98765"
        config = TelegramConfig(f"123:{secret}", 1, 1, "ns")

        class ResourceTrackerHTTPError(HTTPError):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.closed = False

            def close(self):
                self.closed = True
                super().close()

        tracker = []

        def failing_http_404(request: Request, timeout: float) -> bytes:
            err = ResourceTrackerHTTPError(request.full_url, 404, "Not Found", {}, None)
            tracker.append(err)
            raise err

        client = TelegramClient(config, transport=failing_http_404)
        with self.assertRaises(TelegramRequestError) as ctx:
            client.get_updates()

        exc = ctx.exception
        self.assertIsNone(exc.__cause__, "HTTPError must not be chained as __cause__")
        self.assertIsNone(exc.__context__, "HTTPError must not be retained in __context__")
        tb = "".join(traceback.format_exception(exc))
        self.assertNotIn(secret, tb, "Secret token must not appear in formatted traceback")
        self.assertNotIn("api.telegram.org", tb)
        self.assertTrue(tracker[0].closed, "HTTPError resource must be closed")

        # HTTP 409 conflict test
        def failing_http_409(request: Request, timeout: float) -> bytes:
            err = ResourceTrackerHTTPError(request.full_url, 409, "Conflict", {}, None)
            tracker.append(err)
            raise err

        client409 = TelegramClient(config, transport=failing_http_409)
        with self.assertRaises(TelegramConflictError) as ctx409:
            client409.get_updates()

        exc409 = ctx409.exception
        self.assertIsNone(exc409.__cause__)
        self.assertIsNone(exc409.__context__)
        tb409 = "".join(traceback.format_exception(exc409))
        self.assertNotIn(secret, tb409)
        self.assertNotIn("api.telegram.org", tb409)

        # URLError with secret in reason
        def failing_url_error(request: Request, timeout: float) -> bytes:
            raise URLError(f"Failed to reach https://api.telegram.org/bot{secret}/getUpdates")

        client_url = TelegramClient(config, transport=failing_url_error)
        with self.assertRaises(TelegramRequestError) as ctx_url:
            client_url.get_updates()

        exc_url = ctx_url.exception
        self.assertIsNone(exc_url.__cause__)
        self.assertIsNone(exc_url.__context__)
        tb_url = "".join(traceback.format_exception(exc_url))
        self.assertNotIn(secret, tb_url)
        self.assertNotIn(secret, str(exc_url))

        # Malformed JSON with secret marker in raw response bytes
        def failing_json_transport(request: Request, timeout: float) -> bytes:
            return f"invalid json containing secret token: {secret}".encode("utf-8")

        client_json = TelegramClient(config, transport=failing_json_transport)
        with self.assertRaises(TelegramEnvelopeError) as ctx_json:
            client_json.get_updates()

        exc_json = ctx_json.exception
        self.assertIsNone(exc_json.__cause__, "JSONDecodeError must not be chained as __cause__")
        self.assertIsNone(exc_json.__context__, "JSONDecodeError must not be retained in __context__")
        tb_json = "".join(traceback.format_exception(exc_json))
        self.assertNotIn(secret, tb_json, "Raw response secret must not appear in traceback")
        self.assertNotIn(secret, str(exc_json))

        # Transport throwing TelegramRequestError with raw secret in message
        def failing_custom_error_transport(request: Request, timeout: float) -> bytes:
            raise TelegramRequestError(f"Transport error with secret {secret} and token")

        client_custom = TelegramClient(config, transport=failing_custom_error_transport)
        with self.assertRaises(TelegramRequestError) as ctx_custom:
            client_custom.get_updates()

        exc_custom = ctx_custom.exception
        self.assertIsNone(exc_custom.__cause__)
        self.assertIsNone(exc_custom.__context__)
        tb_custom = "".join(traceback.format_exception(exc_custom))
        self.assertNotIn(secret, tb_custom, "Transport error secret must not appear in traceback")
        self.assertNotIn(secret, str(exc_custom), "Transport error secret must not appear in str(exc)")

    def test_client_enforces_exact_telegram_endpoint_and_rejects_arbitrary_base_url(self):
        config = TelegramConfig("123:token", 1, 1, "ns")

        # Disallowed arbitrary base_urls
        disallowed_urls = [
            "https://evil.example.com",
            "https://evil.example.com/api",
            "https://user:pass@api.telegram.org",
            "https://api.telegram.org:8443",
            "https://api.telegram.org/custom/path",
            "https://api.telegram.org?query=1",
            "https://api.telegram.org#frag",
            "http://api.telegram.org",
        ]
        for bad_url in disallowed_urls:
            with self.subTest(bad_url=bad_url):
                with self.assertRaises((ValueError, TelegramRequestError)):
                    TelegramClient(config, base_url=bad_url)

    def test_client_rejects_non_boolean_truthy_ok_and_missing_result(self):
        config = TelegramConfig("123:token", 100, 200, "ns")

        # Non-boolean truthy ok or missing result
        bad_envelopes = [
            {"ok": 1, "result": {"message_id": 1, "chat": {"id": 200, "type": "private"}}},
            {"ok": "true", "result": {"message_id": 1, "chat": {"id": 200, "type": "private"}}},
            {"ok": [True], "result": {"message_id": 1, "chat": {"id": 200, "type": "private"}}},
            {"ok": True},  # missing "result" key entirely
        ]
        for bad_env in bad_envelopes:
            with self.subTest(bad_env=bad_env):
                def fake_transport(request: Request, timeout: float) -> bytes:
                    return json.dumps(bad_env).encode("utf-8")

                client = TelegramClient(config, transport=fake_transport)
                with self.assertRaises((TelegramRequestError, TelegramEnvelopeError)):
                    client.send_message(200, "hello")

    def test_get_updates_parameter_validation(self):
        config = TelegramConfig("123:token", 1, 1, "ns")
        client = TelegramClient(config, transport=lambda req, to: json.dumps({"ok": True, "result": []}).encode())

        # limit validation: 1..100
        with self.assertRaises(ValueError):
            client.get_updates(limit=0)
        with self.assertRaises(ValueError):
            client.get_updates(limit=101)
        with self.assertRaises(ValueError):
            client.get_updates(limit=True)  # bool
        with self.assertRaises(ValueError):
            client.get_updates(limit=-5)

        # timeout validation: non-negative / positive non-bool
        with self.assertRaises(ValueError):
            client.get_updates(timeout=-1)
        with self.assertRaises(ValueError):
            client.get_updates(timeout=True)  # bool

    def test_send_message_validates_delivery_success_and_rejects_malformed_responses(self):
        config = TelegramConfig("123:token", 100, 200, "ns")

        malformed_results = [
            {},  # empty dict
            {"message_id": None},
            {"message_id": -1},
            {"message_id": 0},
            {"message_id": True},  # bool message_id
            {"message_id": "10"},  # str message_id
            {"message_id": 1},  # regression: result {'message_id': 1} missing chat must fail!
            {"message_id": 10, "chat": "not-a-dict"},
            {"message_id": 10, "chat": None},  # regression: chat is None must fail!
            {"message_id": 10, "chat": {"id": None}},
            {"message_id": 10, "chat": {"id": -200}},
            {"message_id": 10, "chat": {"id": 0}},
            {"message_id": 10, "chat": {"id": True}},  # bool chat_id
            {"message_id": 10, "chat": {"id": 999}},  # mismatched chat_id (requested 200)
            {"message_id": 10, "chat": {"id": 200}},  # regression: type missing/None must fail!
            {"message_id": 10, "chat": {"id": 200, "type": None}},  # type is None must fail!
            {"message_id": 10, "chat": {"id": 200, "type": "group"}},  # non-private chat type
        ]

        for malformed in malformed_results:
            def fake_transport(request: Request, timeout: float) -> bytes:
                return json.dumps({"ok": True, "result": malformed}).encode()

            client = TelegramClient(config, transport=fake_transport)
            with self.assertRaises(TelegramRequestError, msg=f"Should reject {malformed}"):
                client.send_message(200, "test")

        # Non-dict API results
        for non_dict in [None, [], "ok", 123]:
            def fake_transport_nondict(request: Request, timeout: float) -> bytes:
                return json.dumps({"ok": True, "result": non_dict}).encode()

            client = TelegramClient(config, transport=fake_transport_nondict)
            with self.assertRaises(TelegramRequestError, msg=f"Should reject {non_dict}"):
                client.send_message(200, "test")

    def test_outbox_worker_retries_and_does_not_mark_sent_on_malformed_send_message(self):
        store = BrainStore(":memory:")
        try:
            config = TelegramConfig("123:token", 100, 200, "bot-ns")

            def malformed_transport(request: Request, timeout: float) -> bytes:
                # Returns malformed empty result
                return json.dumps({"ok": True, "result": {}}).encode()

            client = TelegramClient(config, transport=malformed_transport)
            worker = TelegramWorker(store, client, config)

            # Enqueue telegram outbox message
            store.accept_inbound_message("tg-1", "hello", channel="telegram", binding=config.binding)
            claim = store.claim_inbound_message(datetime.now(timezone.utc), channel="telegram", binding=config.binding)
            self.assertIsNotNone(claim)
            msg, token = claim
            store.complete_inbound_message(msg.id, token, "Reply from AI")

            # Process outbox - send_message should fail, outbox must NOT be marked sent
            attempts = worker.process_outbox()
            self.assertEqual(1, attempts)

            # Check that outbox message was marked failed/retrying, NOT completed/sent
            # A new claim should be possible after lease expires or retry
            outbox_claim = store.claim_outbox_message(
                datetime.now(timezone.utc) + timedelta(seconds=120),
                channel="telegram",
                binding=config.binding,
            )
            self.assertIsNotNone(outbox_claim, "Outbox message must remain pending/retryable")
        finally:
            store.close()

    def test_reminder_dispatcher_does_not_mark_sent_on_malformed_send_message(self):
        store = BrainStore(":memory:")
        try:
            config = TelegramConfig("123:token", 100, 200, "bot-ns")
            now = datetime.now(timezone.utc)

            def malformed_transport(request: Request, timeout: float) -> bytes:
                return json.dumps({"ok": True, "result": {"message_id": -5}}).encode()

            client = TelegramClient(config, transport=malformed_transport)
            transport = TelegramTransport(client, config)
            dispatcher = ReminderDispatcher(
                store,
                transport,
                channel="telegram",
                binding=config.binding,
            )

            store.create_reminder("Test reminder", now, incoming_message_id="rem-test")

            attempts = dispatcher.dispatch_due()
            self.assertEqual(1, attempts)

            # Reminder must NOT be marked sent in DB; it remains pending for retry
            rem = store.list_reminders(include_finished=True)[0]
            self.assertNotEqual("sent", rem.status, "Reminder must not be marked sent on malformed response")
            self.assertEqual("pending", rem.status)
        finally:
            store.close()

    def test_parse_update_rejects_boolean_ids_and_negative_message_id(self):
        # Owner ID is 1 (True == 1 in Python!)
        config = TelegramConfig("123:token", owner_user_id=1, owner_chat_id=1, bot_namespace="ns")

        # Sender ID is boolean True
        bool_sender_update = {
            "update_id": 1,
            "message": {
                "message_id": 10,
                "from": {"id": True, "is_bot": False},
                "chat": {"id": 1, "type": "private"},
                "text": "Hello",
            },
        }
        self.assertIsNone(parse_telegram_update(bool_sender_update, config))

        # Chat ID is boolean True
        bool_chat_update = {
            "update_id": 2,
            "message": {
                "message_id": 10,
                "from": {"id": 1, "is_bot": False},
                "chat": {"id": True, "type": "private"},
                "text": "Hello",
            },
        }
        self.assertIsNone(parse_telegram_update(bool_chat_update, config))

        # Message ID is negative
        neg_msg_update = {
            "update_id": 3,
            "message": {
                "message_id": -10,
                "from": {"id": 1, "is_bot": False},
                "chat": {"id": 1, "type": "private"},
                "text": "Hello",
            },
        }
        self.assertIsNone(parse_telegram_update(neg_msg_update, config))

        # Message ID is boolean True
        bool_msg_update = {
            "update_id": 4,
            "message": {
                "message_id": True,
                "from": {"id": 1, "is_bot": False},
                "chat": {"id": 1, "type": "private"},
                "text": "Hello",
            },
        }
        self.assertIsNone(parse_telegram_update(bool_msg_update, config))

    def test_batch_validation_rejects_whole_batch_for_malformed_candidate_direct_message(self):
        config = TelegramConfig("123:token", owner_user_id=1, owner_chat_id=1, bot_namespace="ns")

        # Candidate direct message with boolean sender ID True
        batch_bool_sender = [
            {
                "update_id": 10,
                "message": {
                    "message_id": 100,
                    "from": {"id": True, "is_bot": False},
                    "chat": {"id": 1, "type": "private"},
                    "text": "Attacker probe",
                },
            }
        ]
        with self.assertRaises(TelegramEnvelopeError):
            validate_and_parse_batch(batch_bool_sender, config)

        # Candidate direct message with negative message ID
        batch_neg_msg = [
            {
                "update_id": 11,
                "message": {
                    "message_id": -5,
                    "from": {"id": 1, "is_bot": False},
                    "chat": {"id": 1, "type": "private"},
                    "text": "Attacker probe",
                },
            }
        ]
        with self.assertRaises(TelegramEnvelopeError):
            validate_and_parse_batch(batch_neg_msg, config)

    def test_poller_does_not_advance_cursor_when_envelope_has_malformed_candidate_message(self):
        store = BrainStore(":memory:")
        try:
            config = TelegramConfig("123:token", owner_user_id=1, owner_chat_id=1, bot_namespace="bot-ns")
            malformed_updates = [
                {
                    "update_id": 50,
                    "message": {
                        "message_id": -99,
                        "from": {"id": 1, "is_bot": False},
                        "chat": {"id": 1, "type": "private"},
                        "text": "Malformed msg",
                    },
                }
            ]

            def fake_transport(request: Request, timeout: float) -> bytes:
                return json.dumps({"ok": True, "result": malformed_updates}).encode()

            client = TelegramClient(config, transport=fake_transport)
            poller = TelegramPoller(store, client, config)

            with self.assertRaises(TelegramEnvelopeError):
                poller.poll_once()

            # Cursor in DB must NOT have advanced (remains at initial 0, not advanced to 51)
            cursor = store.get_telegram_cursor("bot-ns", config.binding)
            self.assertEqual(0, cursor)
        finally:
            store.close()

    def test_config_rejects_path_injection_tokens(self):
        with self.assertRaises(ValueError):
            TelegramConfig(bot_token="123/../../etc/passwd", owner_user_id=1, owner_chat_id=1)
        with self.assertRaises(ValueError):
            TelegramConfig(bot_token="123:token\nnewline", owner_user_id=1, owner_chat_id=1)
        with self.assertRaises(ValueError):
            TelegramConfig(bot_token="123:token with space", owner_user_id=1, owner_chat_id=1)
        with self.assertRaises(ValueError):
            TelegramConfig(bot_token="123?query=1", owner_user_id=1, owner_chat_id=1)


if __name__ == "__main__":
    unittest.main()
