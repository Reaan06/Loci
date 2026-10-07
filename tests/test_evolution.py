import json
import sys
import threading
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_brain.core import Reminder
from second_brain.evolution import (
    EvolutionConfig,
    EvolutionRequestError,
    EvolutionTransport,
    parse_evolution_message,
)


class EvolutionTests(unittest.TestCase):
    def config(self, **changes):
        values = {
            "base_url": "https://evolution.example",
            "api_key": "not-a-real-key",
            "instance_name": "assistant",
            "owner_jid": "15551234567@s.whatsapp.net",
            "webhook_secret": "webhook-secret",
        }
        values.update(changes)
        return EvolutionConfig(**values)

    def event(self, **changes):
        key = {"id": "provider-message-id", "fromMe": False, "remoteJid": "15551234567@s.whatsapp.net"}
        data = {"key": key, "message": {"conversation": "save this"}}
        payload = {"event": "messages.upsert", "instance": "assistant", "data": data}
        for key_name, value in changes.items():
            if key_name == "data":
                payload["data"] = value
            else:
                payload[key_name] = value
        return payload

    def test_parser_accepts_only_configured_owner_text_and_stable_hashed_identity(self):
        parsed = parse_evolution_message(self.event(), self.config())
        self.assertIsNotNone(parsed)
        self.assertEqual("save this", parsed.text)
        self.assertTrue(parsed.message_id.startswith("evolution:"))
        self.assertNotIn("provider-message-id", parsed.message_id)
        self.assertEqual(parsed.message_id, parse_evolution_message(self.event(), self.config()).message_id)

        extended = self.event(data={
            "key": {"id": "extended", "fromMe": False, "remoteJid": "15551234567@s.whatsapp.net"},
            "message": {"extendedTextMessage": {"text": "extended text"}},
        })
        self.assertEqual("extended text", parse_evolution_message(extended, self.config()).text)

    def test_parser_rejects_untrusted_malformed_and_oversized_events(self):
        cases = [
            self.event(event="MESSAGES_UPSERT"),
            self.event(instance="other-instance"),
            self.event(data={"key": {"id": "x", "fromMe": True, "remoteJid": "15551234567@s.whatsapp.net"}, "message": {"conversation": "x"}}),
            self.event(data={"key": {"id": "x", "fromMe": 0, "remoteJid": "15551234567@s.whatsapp.net"}, "message": {"conversation": "x"}}),
            self.event(data={"key": {"id": "x", "fromMe": False, "remoteJid": "123@g.us"}, "message": {"conversation": "x"}}),
            self.event(data={"key": {"id": "x", "fromMe": False, "remoteJid": "15551234567@lid"}, "message": {"conversation": "x"}}),
            self.event(data={"key": {"id": "x", "fromMe": False, "remoteJid": "15551234567@s.whatsapp.net"}, "message": {"imageMessage": {}}}),
            self.event(data=None),
            self.event(data={"key": {"id": "x", "fromMe": False, "remoteJid": "15551234567@s.whatsapp.net"}, "message": {"conversation": "x" * 6000}}),
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertIsNone(parse_evolution_message(payload, self.config()))

    def test_transport_uses_schema_text_message_and_requires_positive_response(self):
        sent = []

        def transport(request, timeout):
            sent.append((request, timeout))
            return json.dumps({"key": {"id": "outbound"}, "message": {"conversation": "hi"}, "status": "PENDING"}).encode()

        client = EvolutionTransport(self.config(), transport=transport)
        client.send_text("hello")
        request, timeout = sent[0]
        self.assertEqual("not-a-real-key", request.get_header("Apikey"))
        self.assertIn("/message/sendText/assistant", request.full_url)
        self.assertEqual({"number": "15551234567", "textMessage": {"text": "hello"}}, json.loads(request.data))
        self.assertGreater(timeout, 0)
        with self.assertRaisesRegex(EvolutionRequestError, "acceptance"):
            EvolutionTransport(self.config(), transport=lambda request, timeout: b'{"status":"ERROR"}').send_text("no")

    def test_config_rejects_missing_hosts_and_malformed_ports(self):
        for base_url in (
            "https://",
            "https:///path",
            "https://?query",
            "https://:443",
            "https://evolution.example:not-a-port",
        ):
            with self.subTest(base_url=base_url):
                with self.assertRaisesRegex(ValueError, "valid authority"):
                    self.config(base_url=base_url)

    def test_config_rejects_query_fragment_and_userinfo_base_urls(self):
        for base_url in (
            "https://evolution.example/root?tenant=alpha",
            "https://evolution.example/root#fragment",
            "https://user:placeholder@evolution.example/root",
            "https://@evolution.example/root",
            "https://:@evolution.example/root",
        ):
            with self.subTest(base_url=base_url):
                with self.assertRaisesRegex(ValueError, "must not include query, fragment, or userinfo"):
                    self.config(base_url=base_url)

    def test_transport_preserves_valid_path_prefix_and_api_key(self):
        for base_url in ("https://evolution.example/root", "https://evolution.example/root/"):
            with self.subTest(base_url=base_url):
                sent = []

                def transport(request, timeout):
                    sent.append(request)
                    return b'{"key":{"id":"outbound"},"message":{},"status":"PENDING"}'

                EvolutionTransport(self.config(base_url=base_url), transport=transport).send_text("hello")
                self.assertEqual("https://evolution.example/root/message/sendText/assistant", sent[0].full_url)
                self.assertEqual("not-a-real-key", sent[0].get_header("Apikey"))

    def test_text_dialect_is_explicit_and_http_requires_trusted_private_opt_in(self):
        sent = []
        client = EvolutionTransport(self.config(text_dialect="text"), transport=lambda request, timeout: sent.append(request) or b'{"key":{"id":"x"},"message":{},"status":"PENDING"}')
        client.send(Reminder("r", "remember", datetime(2026, 1, 1, tzinfo=timezone.utc), "pending", 0, datetime(2026, 1, 1, tzinfo=timezone.utc)))
        self.assertEqual({"number": "15551234567", "text": "Reminder: remember"}, json.loads(sent[0].data))
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            self.config(base_url="http://evolution.example")
        self.assertEqual(
            "http://127.0.0.1:8080",
            self.config(base_url="http://127.0.0.1:8080", allow_insecure_private_http=True).base_url,
        )

    def test_default_transport_rejects_redirect_before_target_is_requested(self):
        target_hits = []

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(307)
                self.send_header("Location", f"http://127.0.0.1:{target.server_port}/target")
                self.end_headers()
            def log_message(self, *_):
                pass

        class TargetHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                target_hits.append(self.headers.get("apikey"))
                self.send_response(200)
                self.end_headers()
            def log_message(self, *_):
                pass

        target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
        redirector = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        target_thread = threading.Thread(target=target.serve_forever)
        redirect_thread = threading.Thread(target=redirector.serve_forever)
        target_thread.start(); redirect_thread.start()
        try:
            client = EvolutionTransport(self.config(base_url=f"http://127.0.0.1:{redirector.server_port}", allow_insecure_private_http=True))
            with self.assertRaises(EvolutionRequestError):
                client.send_text("hello")
            self.assertEqual([], target_hits)
        finally:
            redirector.shutdown(); target.shutdown()
            redirect_thread.join(); target_thread.join()
            redirector.server_close(); target.server_close()

    def test_binding_is_derived_from_instance_and_owner_and_secret_free(self):
        cfg = self.config()
        binding = cfg.binding
        self.assertIsInstance(binding, str)
        self.assertEqual(64, len(binding))
        self.assertEqual(binding, self.config().binding)
        # Secret-free: credentials or url changes do not alter binding
        other_secrets = self.config(api_key="different-key", webhook_secret="different-secret", base_url="https://other.example")
        self.assertEqual(binding, other_secrets.binding)
        # Instance or owner change produces distinct binding
        diff_instance = self.config(instance_name="other-instance")
        self.assertNotEqual(binding, diff_instance.binding)
        diff_owner = self.config(owner_jid="15559876543@s.whatsapp.net")
        self.assertNotEqual(binding, diff_owner.binding)


if __name__ == "__main__":
    unittest.main()
