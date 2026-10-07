import json
import sys
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.request import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import second_brain.calendar as calendar_module
from second_brain.calendar import (
    CalendarClient,
    CalendarConfig,
    CalendarConfigurationError,
    CalendarResponse,
    CalendarRetryableError,
    parse_calendar_time,
)


class Recorder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def response(status, body):
    return CalendarResponse(status, json.dumps(body).encode())


class CalendarTests(unittest.TestCase):
    def config(self, **changes):
        values = {
            "client_id": "client-id",
            "client_secret": "client-secret",
            "refresh_token": "refresh-token",
            "timezone_name": "Europe/Berlin",
        }
        values.update(changes)
        return CalendarConfig(**values)

    def token(self):
        return response(200, {"access_token": "access-token", "expires_in": 3600})

    def event(self, event_id="a" * 64, summary="Planning", start="2026-06-01T09:00:00+02:00", end="2026-06-01T10:00:00+02:00", description="Discuss", fingerprint="PLACEHOLDER"):
        return {"id": event_id, "status": "confirmed", "summary": summary, "description": description,
                "start": {"dateTime": start}, "end": {"dateTime": end},
                "extendedProperties": {"private": {"second_brain_fingerprint": fingerprint}}}

    def test_environment_is_complete_or_disabled_without_leaking_secrets(self):
        self.assertIsNone(CalendarConfig.from_environment({}))
        with self.assertRaisesRegex(CalendarConfigurationError, "complete"):
            CalendarConfig.from_environment({"SECOND_BRAIN_GOOGLE_CLIENT_ID": "partial"})
        with self.assertRaisesRegex(CalendarConfigurationError, "timezone"):
            self.config(timezone_name="Not/AZone")
        self.assertNotIn("client-secret", repr(self.config()))
        self.assertNotIn("refresh-token", repr(self.config()))

    def test_strict_time_parsing_rejects_dst_fold_gap_duration_and_boolean_max(self):
        self.assertEqual("2026-06-01T07:00:00+00:00", parse_calendar_time("2026-06-01T09:00:00+02:00", "Europe/Berlin").isoformat())
        self.assertEqual("2026-06-01T07:00:00+00:00", parse_calendar_time("2026-06-01T09:00:00", "Europe/Berlin").isoformat())
        for value in ("2026-03-29T02:30:00", "2026-10-25T02:30:00"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_calendar_time(value, "Europe/Berlin")
        with self.assertRaises(ValueError):
            CalendarClient.normalize_create({"summary": "x", "start": "2026-06-01T00:00:00Z", "end": "2026-06-16T00:00:01Z"}, self.config())
        with self.assertRaises(ValueError):
            CalendarClient.normalize_list({"time_min": "2026-06-01T00:00:00Z", "time_max": "2026-07-03T00:00:00Z"}, self.config())
        with self.assertRaises(ValueError):
            CalendarClient.normalize_list({"time_min": "2026-06-01T00:00:00Z", "time_max": "2026-06-02T00:00:00Z", "max_results": True}, self.config())

    def test_refresh_form_is_encoded_cached_and_401_is_retried_once(self):
        client = CalendarClient(self.config(), monotonic=lambda: 100)
        intent = client.normalize_create({"summary": "Planning", "description": "Discuss", "start": "2026-06-01T09:00:00+02:00", "end": "2026-06-01T10:00:00+02:00"}, self.config())
        recorder = Recorder([self.token(), response(401, {}), self.token(), response(200, self.event(event_id=client.provider_event_id("message"), fingerprint=calendar_module._fingerprint(intent)) )])
        client.transport = recorder
        created = client.create_event(intent, "message")
        self.assertEqual("Planning", created.summary)
        self.assertEqual(4, len(recorder.requests))
        self.assertIn(b"grant_type=refresh_token", recorder.requests[0].data)
        self.assertIn(b"client_secret=client-secret", recorder.requests[0].data)
        self.assertEqual("Bearer access-token", recorder.requests[-1].get_header("Authorization"))

    def test_token_cache_expires_on_injected_clock_and_repeated_401_is_bounded(self):
        now = [100.0]
        client = CalendarClient(self.config(), monotonic=lambda: now[0])
        intent = client.normalize_list({"time_min": "2026-06-01T00:00:00Z", "time_max": "2026-06-02T00:00:00Z"}, self.config())
        recorder = Recorder([
            response(200, {"access_token": "first", "expires_in": 31}), response(200, {"items": []}),
            response(200, {"access_token": "second", "expires_in": 31}), response(200, {"items": []}),
        ])
        client.transport = recorder
        client.list_events(intent)
        now[0] += 2
        client.list_events(intent)
        self.assertEqual(4, len(recorder.requests))
        self.assertEqual("Bearer second", recorder.requests[-1].get_header("Authorization"))

        repeated_401 = CalendarClient(self.config(), transport=Recorder([
            self.token(), response(401, {}), self.token(), response(401, {}),
        ]))
        with self.assertRaises(CalendarRetryableError):
            repeated_401.list_events(intent)
        self.assertEqual(4, len(repeated_401.transport.requests))

    def test_malformed_token_response_is_safe_and_does_not_echo_credentials(self):
        client = CalendarClient(self.config(), transport=Recorder([response(200, {"expires_in": 10})]))
        intent = client.normalize_list({"time_min": "2026-06-01T00:00:00Z", "time_max": "2026-06-02T00:00:00Z"}, self.config())
        with self.assertRaises(CalendarRetryableError) as caught:
            client.list_events(intent)
        self.assertNotIn("refresh-token", str(caught.exception))
        self.assertNotIn("client-secret", str(caught.exception))

    def test_configuration_binding_is_opaque_domain_separated_and_changes_with_credentials(self):
        intent = CalendarClient.normalize_create({"summary": "Planning", "start": "2026-06-01T09:00:00+02:00", "end": "2026-06-01T10:00:00+02:00"}, self.config())
        base = CalendarClient(self.config()).create_binding(intent)

        self.assertEqual(64, len(base))
        self.assertNotEqual(base, calendar_module._fingerprint(intent))
        self.assertNotEqual(base, CalendarClient(self.config(client_id="other-client")).create_binding(intent))
        self.assertNotEqual(base, CalendarClient(self.config(client_secret="other-secret")).create_binding(intent))
        self.assertNotEqual(base, CalendarClient(self.config(refresh_token="other-refresh")).create_binding(intent))
        self.assertNotIn("client-secret", base)
        self.assertNotIn("refresh-token", base)

    def test_conflict_uses_same_id_and_exact_content_or_fails_closed(self):
        client = CalendarClient(self.config())
        intent = client.normalize_create({"summary": "Planning", "description": "Discuss", "start": "2026-06-01T09:00:00+02:00", "end": "2026-06-01T10:00:00+02:00"}, self.config())
        fingerprint = calendar_module._fingerprint(intent)
        recorder = Recorder([self.token(), response(409, {}), response(200, self.event(event_id=client.provider_event_id("stable-message"), fingerprint=fingerprint))])
        client.transport = recorder
        event = client.create_event(intent, "stable-message")
        self.assertEqual(client.provider_event_id("stable-message"), event.id)
        self.assertTrue(recorder.requests[2].full_url.endswith("/events/" + client.provider_event_id("stable-message")))

        mismatch = Recorder([self.token(), response(409, {}), response(200, self.event(event_id=client.provider_event_id("stable-message"), summary="Other", fingerprint=fingerprint))])
        with self.assertRaisesRegex(ValueError, "does not match"):
            CalendarClient(self.config(), transport=mismatch).create_event(intent, "stable-message")

    def test_acceptance_then_timeout_recovers_with_one_provider_event_and_stable_id(self):
        class AcceptanceThenTimeout:
            def __init__(self, event):
                self.event = event
                self.requests = []
                self.accepted_event_ids = []
                self.create_attempts = []

            def __call__(self, request, timeout):
                self.requests.append(request)
                if request.full_url == calendar_module.GOOGLE_TOKEN_URL:
                    return response(200, {"access_token": "access-token", "expires_in": 3600})
                if request.method == "POST":
                    event_id = json.loads(request.data)["id"]
                    self.create_attempts.append(event_id)
                    if not self.accepted_event_ids:
                        self.accepted_event_ids.append(event_id)
                        raise CalendarRetryableError("accepted then connection lost")
                    return response(409, {})
                return response(200, self.event)

        config = self.config()
        intent = CalendarClient.normalize_create({"summary": "Planning", "description": "Discuss", "start": "2026-06-01T09:00:00+02:00", "end": "2026-06-01T10:00:00+02:00"}, config)
        event_id = CalendarClient(config).provider_event_id("durable-message")
        transport = AcceptanceThenTimeout(self.event(event_id=event_id, fingerprint=calendar_module._fingerprint(intent)))

        with self.assertRaises(CalendarRetryableError):
            CalendarClient(config, transport=transport).create_event(intent, "durable-message")
        recovered = CalendarClient(config, transport=transport).create_event(intent, "durable-message")

        self.assertEqual(event_id, recovered.id)
        self.assertEqual([event_id], transport.accepted_event_ids)
        self.assertEqual([event_id, event_id], transport.create_attempts)
        self.assertTrue(transport.requests[-1].full_url.endswith("/events/" + event_id))

    def test_default_transport_refuses_token_redirect_before_target_receives_form(self):
        target_requests = []

        class TargetHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                target_requests.append(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                self.send_response(200)
                self.end_headers()
            def log_message(self, *_):
                pass

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(307)
                self.send_header("Location", f"http://127.0.0.1:{target.server_port}/token")
                self.end_headers()
            def log_message(self, *_):
                pass

        target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
        redirector = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        target_thread = threading.Thread(target=target.serve_forever)
        redirect_thread = threading.Thread(target=redirector.serve_forever)
        target_thread.start(); redirect_thread.start()
        original_token = calendar_module.GOOGLE_TOKEN_URL
        original_api = calendar_module.GOOGLE_CALENDAR_URL
        try:
            calendar_module.GOOGLE_TOKEN_URL = f"http://127.0.0.1:{redirector.server_port}/token"
            client = CalendarClient(self.config())
            with self.assertRaises(ValueError):
                client.list_events({"time_min": "2026-06-01T00:00:00+00:00", "time_max": "2026-06-02T00:00:00+00:00", "max_results": 1})
            calendar_module.GOOGLE_CALENDAR_URL = f"http://127.0.0.1:{redirector.server_port}/events"
            client._access_token = "access-token"
            client._token_expires_at = float("inf")
            with self.assertRaises(CalendarRetryableError):
                client.list_events({"time_min": "2026-06-01T00:00:00+00:00", "time_max": "2026-06-02T00:00:00+00:00", "max_results": 1})
            self.assertEqual([], target_requests)
        finally:
            calendar_module.GOOGLE_TOKEN_URL = original_token
            calendar_module.GOOGLE_CALENDAR_URL = original_api
            redirector.shutdown(); target.shutdown()
            redirect_thread.join(); target_thread.join()
            redirector.server_close(); target.server_close()

    def test_list_honors_requested_limit_and_marks_extra_accepted_items_truncated(self):
        events = [
            {"id": "one", "status": "confirmed", "summary": "First", "start": {"date": "2026-06-01"}, "end": {"date": "2026-06-02"}},
            {"id": "two", "status": "confirmed", "summary": "Second", "start": {"date": "2026-06-02"}, "end": {"date": "2026-06-03"}},
        ]
        recorder = Recorder([self.token(), response(200, {"items": events})])
        client = CalendarClient(self.config(), transport=recorder)
        intent = client.normalize_list({"time_min": "2026-06-01T00:00:00Z", "time_max": "2026-06-03T00:00:00Z", "max_results": 1}, self.config())

        result = client.list_events(intent)

        self.assertEqual(("one",), tuple(event.id for event in result.events))
        self.assertTrue(result.truncated)

    def test_list_skips_semantically_invalid_all_day_ranges(self):
        valid = {"id": "valid", "status": "confirmed", "summary": "Leap day", "start": {"date": "2028-02-29"}, "end": {"date": "2028-03-01"}}
        invalid = [
            {"id": "bad-text", "summary": "x", "start": {"date": "not-a-date"}, "end": {"date": "also-not-a-date"}},
            {"id": "bad-day", "summary": "x", "start": {"date": "2026-02-29"}, "end": {"date": "2026-03-01"}},
            {"id": "bad-month", "summary": "x", "start": {"date": "2026-13-01"}, "end": {"date": "2026-14-01"}},
            {"id": "same-day", "summary": "x", "start": {"date": "2026-06-01"}, "end": {"date": "2026-06-01"}},
            {"id": "reversed", "summary": "x", "start": {"date": "2026-06-02"}, "end": {"date": "2026-06-01"}},
            {"id": "mixed", "summary": "x", "start": {"date": "2026-06-01"}, "end": {"dateTime": "2026-06-02T00:00:00Z"}},
            {"id": "typed", "summary": "x", "start": {"date": 20260601}, "end": {"date": "2026-06-02"}},
        ]
        recorder = Recorder([self.token(), response(200, {"items": [valid, *invalid]})])
        client = CalendarClient(self.config(), transport=recorder)
        intent = client.normalize_list({"time_min": "2026-02-01T00:00:00Z", "time_max": "2026-03-01T00:00:00Z", "max_results": 20}, self.config())

        result = client.list_events(intent)

        self.assertEqual(("valid",), tuple(event.id for event in result.events))
        self.assertTrue(result.events[0].all_day)

    def test_list_omits_cancelled_marks_truncation_and_preserves_all_day_honestly(self):
        events = {"items": [
            {"id": "one", "status": "confirmed", "summary": "All day", "start": {"date": "2026-06-01"}, "end": {"date": "2026-06-02"}},
            {"id": "gone", "status": "cancelled", "summary": "Deleted", "start": {"dateTime": "2026-06-01T10:00:00Z"}, "end": {"dateTime": "2026-06-01T11:00:00Z"}},
        ], "nextPageToken": "more"}
        recorder = Recorder([self.token(), response(200, events)])
        client = CalendarClient(self.config(), transport=recorder)
        intent = client.normalize_list({"time_min": "2026-06-01T00:00:00Z", "time_max": "2026-06-02T00:00:00Z", "max_results": 2}, self.config())
        result = client.list_events(intent)
        self.assertTrue(result.truncated)
        self.assertEqual(1, len(result.events))
        self.assertTrue(result.events[0].all_day)
        self.assertIn("singleEvents=true", recorder.requests[-1].full_url)
        self.assertIn("orderBy=startTime", recorder.requests[-1].full_url)


if __name__ == "__main__":
    unittest.main()
