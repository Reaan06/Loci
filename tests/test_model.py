import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_brain.model import (
    MAX_PROVIDER_RESPONSE_BYTES,
    ModelConfigurationError,
    ModelRequestError,
    ModelResponse,
    OpenAICompatibleModel,
)


class RecordingTransport:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        return self.response


class ModelTests(unittest.TestCase):
    def test_openai_compatible_request_uses_explicit_configuration(self):
        response = json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode()
        transport = RecordingTransport(response)
        model = OpenAICompatibleModel(
            endpoint="https://provider.example/v1/chat/completions",
            api_key="test-key",
            model="test-model",
            timeout_seconds=4,
            transport=transport,
        )

        result = model.complete([{"role": "user", "content": "hello"}], [])

        self.assertEqual(ModelResponse(content="hello", tool_calls=()), result)
        request, timeout = transport.requests[0]
        self.assertEqual(4, timeout)
        self.assertEqual("Bearer test-key", request.get_header("Authorization"))
        payload = json.loads(request.data)
        self.assertEqual("test-model", payload["model"])
        self.assertEqual("hello", payload["messages"][0]["content"])

    def test_default_transport_rejects_local_redirect_without_reaching_target_or_forwarding_auth(self):
        reached_target = []

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{target.server_port}/target")
                self.end_headers()

            def log_message(self, format, *args):
                pass

        class TargetHandler(BaseHTTPRequestHandler):
            def _record_request(self):
                reached_target.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()

            def do_GET(self):
                self._record_request()

            def do_POST(self):
                self._record_request()

            def log_message(self, format, *args):
                pass

        target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
        redirector = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        target_thread = threading.Thread(target=target.serve_forever)
        redirect_thread = threading.Thread(target=redirector.serve_forever)
        target_thread.start()
        redirect_thread.start()
        try:
            model = OpenAICompatibleModel(
                f"http://127.0.0.1:{redirector.server_port}/chat",
                "dummy-token",
                "test-model",
                allow_insecure_localhost=True,
            )
            with self.assertRaisesRegex(ModelRequestError, "provider request failed"):
                model.complete([{"role": "user", "content": "hello"}], [])
            self.assertEqual([], reached_target)
        finally:
            redirector.shutdown()
            target.shutdown()
            redirect_thread.join()
            target_thread.join()
            redirector.server_close()
            target.server_close()

    def test_configuration_and_oversized_provider_responses_are_refused(self):
        with self.assertRaises(ModelConfigurationError):
            OpenAICompatibleModel("http://provider.example/v1", "key", "model")
        with self.assertRaises(ModelConfigurationError):
            OpenAICompatibleModel("https://provider.example/v1", "", "model")
        oversized = RecordingTransport(b"x" * (MAX_PROVIDER_RESPONSE_BYTES + 1))
        model = OpenAICompatibleModel("https://provider.example/v1", "key", "model", transport=oversized)
        with self.assertRaisesRegex(ValueError, "response exceeds"):
            model.complete([{"role": "user", "content": "hello"}], [])


if __name__ == "__main__":
    unittest.main()
