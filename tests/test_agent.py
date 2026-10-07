import json
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_brain.agent import Agent, MAX_TOOL_CALLS, ToolCall
from second_brain.calendar import CalendarClient, CalendarConfig, CalendarResponse, CalendarRetryableError
from second_brain.core import BrainStore, FakeClock
from second_brain.model import ModelResponse, ModelRequestError


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


class ClockAdvancingModel(FakeModel):
    def __init__(self, clock, seconds, responses):
        super().__init__(responses)
        self.clock = clock
        self.seconds = seconds

    def complete(self, messages, tools):
        self.clock.advance(seconds=self.seconds)
        return super().complete(messages, tools)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        self.clock = FakeClock(self.now)
        self.store = BrainStore(Path(self.temporary_directory.name) / "brain.sqlite3", clock=self.clock)

    def tearDown(self):
        self.store.close()
        self.temporary_directory.cleanup()

    def agent(self, *responses):
        return Agent(self.store, FakeModel(responses))

    def test_tool_result_not_hallucinated_text_is_the_action_confirmation(self):
        reply = self.agent(
            ModelResponse(None, (ToolCall("untrusted-call", "save_note", {"text": "Call the dentist"}),)),
            ModelResponse("I did something entirely different", ()),
        ).handle("save a note", message_id="message-1")

        self.assertEqual("Saved note: Call the dentist", reply.text)
        self.assertEqual(["Call the dentist"], [note.text for note in self.store.search_notes("dentist")])

    def test_malformed_arguments_unknown_tools_and_naive_dates_do_not_mutate(self):
        cases = [
            ToolCall("one", "save_note", {"text": "x", "extra": True}),
            ToolCall("two", "save_note", {"text": 1}),
            ToolCall("three", "save_note", {"text": "x", "pinned": 1}),
            ToolCall("four", "calendar_create", {}),
            ToolCall("five", "create_reminder", {"text": "x", "due_at": "2026-06-01T09:00:00"}),
        ]
        for index, call in enumerate(cases):
            with self.subTest(call=call.name):
                reply = self.agent(ModelResponse(None, (call,))).handle("request", message_id=f"bad-{index}")
                self.assertTrue(reply.text.startswith(("Error:", "Clarification required:")))
        self.assertEqual([], self.store.search_notes("x"))
        self.assertEqual([], self.store.list_reminders())

    def test_replayed_delivery_uses_durable_message_identity_not_model_call_identity(self):
        first = self.agent(ModelResponse(None, (ToolCall("a", "save_note", {"text": "repeat me"}),)))
        second = self.agent(ModelResponse(None, (ToolCall("different", "save_note", {"text": "repeat me"}),)))

        self.assertEqual("Saved note: repeat me", first.handle("save note", message_id="replay").text)
        self.assertEqual("Saved note: repeat me", second.handle("save note", message_id="replay").text)
        self.assertEqual(1, len(self.store.search_notes("repeat me")))
        with self.assertRaisesRegex(ValueError, "different input"):
            second.handle("different request", message_id="replay")

    def test_failed_model_attempt_is_safe_to_retry_and_tool_loop_is_bounded(self):
        failed = self.agent(ModelRequestError("temporary provider failure"))
        with self.assertRaises(ModelRequestError):
            failed.handle("save note", message_id="retry")
        successful = self.agent(ModelResponse(None, (ToolCall("recovered", "save_note", {"text": "safe retry"}),)))
        self.assertEqual("Saved note: safe retry", successful.handle("save note", message_id="retry").text)

        calls = tuple(ToolCall(str(index), "search_notes", {"query": "safe"}) for index in range(MAX_TOOL_CALLS + 1))
        exhausted = self.agent(ModelResponse(None, calls))
        self.assertEqual("Error: tool call limit exhausted", exhausted.handle("search", message_id="limit").text)

    def test_recovery_refuses_a_changed_plan_after_a_durable_tool_reservation(self):
        reservation = self.store.reserve_conversation("crash-replay", "save this")
        self.store.activate_conversation_lease("crash-replay", reservation.lease_token)
        self.store.reserve_tool_action("crash-replay", "save_note", {"text": "first plan"})
        self.store.deactivate_conversation_lease("crash-replay")
        self.clock.advance(seconds=61)
        changed = self.agent(ModelResponse(None, (ToolCall("new-id", "save_note", {"text": "changed plan"}),)))

        self.assertEqual(
            "Error: incoming message ID was already used for a different tool plan",
            changed.handle("save this", message_id="crash-replay").text,
        )
        self.assertEqual([], self.store.search_notes("plan"))

    def test_crash_before_model_recovers_after_the_conversation_lease_expires(self):
        self.store.reserve_conversation("crash-before-model", "recover me")
        self.store.close()
        recovered_clock = FakeClock(self.now + timedelta(seconds=61))
        self.store = BrainStore(Path(self.temporary_directory.name) / "brain.sqlite3", clock=recovered_clock)
        model = FakeModel([ModelResponse(None, (ToolCall("recovered", "save_note", {"text": "recover me"}),))])

        reply = Agent(self.store, model).handle("recover me", message_id="crash-before-model")

        self.assertEqual("Saved note: recover me", reply.text)
        self.assertEqual(1, model.calls)
        self.assertEqual(["recover me"], [note.text for note in self.store.search_notes("recover")])

    def test_active_lease_stays_protected_but_an_expired_owner_is_fenced(self):
        original = self.store.reserve_conversation("lease", "save")
        self.store.activate_conversation_lease("lease", original.lease_token)
        active_store = BrainStore(Path(self.temporary_directory.name) / "brain.sqlite3", clock=FakeClock(self.now))
        try:
            active_model = FakeModel([ModelResponse(None, (ToolCall("active", "save_note", {"text": "wrong"}),))])
            active_reply = Agent(active_store, active_model).handle("save", message_id="lease")
            self.assertEqual("Error: message is already being processed", active_reply.text)
            self.assertEqual(0, active_model.calls)
        finally:
            active_store.close()

        self.clock.advance(seconds=61)
        replacement_store = BrainStore(Path(self.temporary_directory.name) / "brain.sqlite3", clock=self.clock)
        try:
            replacement = replacement_store.reserve_conversation("lease", "save")
            self.assertEqual("owned", replacement.state)
            with self.assertRaisesRegex(ValueError, "reservation is unavailable"):
                self.store.reserve_tool_action("lease", "save_note", {"text": "stale"})
            with self.assertRaisesRegex(ValueError, "reservation is unavailable"):
                self.store.complete_conversation("lease", "save", "stale response", lease_token=original.lease_token)
            self.assertEqual([], self.store.search_notes("stale"))
        finally:
            self.store.deactivate_conversation_lease("lease")
            replacement_store.close()

    def test_no_tool_provider_prose_never_confirms_an_unperformed_action(self):
        reply = self.agent(ModelResponse("Saved a note I did not save", ())).handle(
            "save a note", message_id="false-confirmation"
        )

        self.assertEqual("No action was performed. Please clarify what you need.", reply.text)
        self.assertFalse(reply.action_performed)
        self.assertEqual([], self.store.search_notes("save"))

    def test_no_tool_finalization_requires_an_unexpired_lease(self):
        for seconds, expires in ((59, False), (60, True), (61, True)):
            with self.subTest(seconds=seconds):
                message_id = f"delayed-no-tool-{seconds}"
                model = ClockAdvancingModel(self.clock, seconds, [ModelResponse("done", ())])
                agent = Agent(self.store, model)

                if expires:
                    with self.assertRaisesRegex(ValueError, "reservation is unavailable"):
                        agent.handle("status", message_id=message_id)
                    retry = self.agent(ModelResponse("done", ()))
                    self.assertEqual(
                        "No action was performed. Please clarify what you need.",
                        retry.handle("status", message_id=message_id).text,
                    )
                else:
                    self.assertEqual(
                        "No action was performed. Please clarify what you need.",
                        agent.handle("status", message_id=message_id).text,
                    )
                    self.assertEqual(1, model.calls)

    def test_delayed_after_tool_finalization_is_fenced_and_retries_idempotently(self):
        class ToolDelayingStore(BrainStore):
            def save_note(self, text, *, incoming_message_id):
                note = super().save_note(text, incoming_message_id=incoming_message_id)
                self.clock.advance(seconds=60)
                return note

        self.store.close()
        self.store = ToolDelayingStore(
            Path(self.temporary_directory.name) / "brain.sqlite3", clock=self.clock
        )
        expired = Agent(
            self.store,
            FakeModel([ModelResponse(None, (ToolCall("first", "save_note", {"text": "once"}),))]),
        )
        with self.assertRaisesRegex(ValueError, "reservation is unavailable"):
            expired.handle("save", message_id="after-tool-expiry")
        self.assertEqual(["once"], [note.text for note in self.store.search_notes("once")])
        self.store.close()
        self.store = BrainStore(Path(self.temporary_directory.name) / "brain.sqlite3", clock=self.clock)

        retry = self.agent(ModelResponse(None, (ToolCall("second", "save_note", {"text": "once"}),)))
        self.assertEqual("Saved note: once", retry.handle("save", message_id="after-tool-expiry").text)
        self.assertEqual(["once"], [note.text for note in self.store.search_notes("once")])

    def test_calendar_is_hidden_when_disabled_and_transient_calendar_errors_retry_with_frozen_intent(self):
        hidden_model = FakeModel([ModelResponse(None, (ToolCall("calendar", "calendar_list", {
            "time_min": "2026-06-01T00:00:00Z", "time_max": "2026-06-02T00:00:00Z"
        }),))])
        reply = Agent(self.store, hidden_model).handle("calendar", message_id="calendar-disabled")
        self.assertEqual("Error: Google Calendar is not configured", reply.text)

        class TransientCalendar:
            config = object()
            def normalize_create(self, arguments, config):
                return {"calendar_id": "primary", "timezone": "Europe/Berlin", "summary": "Plan", "description": "", "start": "2026-06-01T07:00:00+00:00", "end": "2026-06-01T08:00:00+00:00"}
            def create_binding(self, fields):
                return "same-config-binding"
            def create_event(self, fields, message_id):
                raise CalendarRetryableError("temporary")

        call = ToolCall("calendar", "calendar_create", {"summary": "Plan", "start": "2026-06-01T09:00:00", "end": "2026-06-01T10:00:00"})
        with self.assertRaises(CalendarRetryableError):
            Agent(self.store, FakeModel([ModelResponse(None, (call,))]), calendar=TransientCalendar()).handle("calendar", message_id="calendar-retry")
        self.assertEqual("owned", self.store.reserve_conversation("calendar-retry", "calendar").state)

        self.clock.advance(seconds=61)
        retry = Agent(self.store, FakeModel([ModelResponse(None, (call,))]), calendar=TransientCalendar())
        with self.assertRaises(CalendarRetryableError):
            retry.handle("calendar", message_id="calendar-retry")

    def test_calendar_create_lease_expiry_recovers_with_stable_event_id(self):
        class Calendar:
            config = object()
            def normalize_create(self, arguments, config):
                return {"calendar_id": "primary", "timezone": "Europe/Berlin", "summary": "Plan", "description": "", "start": "2026-06-01T07:00:00+00:00", "end": "2026-06-01T08:00:00+00:00"}
            def create_binding(self, fields):
                return "unchanged-config"
            def create_event(self, fields, message_id):
                return type("Event", (), {"id": f"event-{message_id}", "summary": "Plan", "start": fields["start"], "end": fields["end"]})()

        class ExpiringCalendar(Calendar):
            def create_event(self, fields, message_id):
                event = super().create_event(fields, message_id)
                self.clock.advance(seconds=60)
                return event

        call = ToolCall("calendar", "calendar_create", {"summary": "Plan", "start": "2026-06-01T09:00:00", "end": "2026-06-01T10:00:00"})
        first = ExpiringCalendar()
        first.clock = self.clock
        with self.assertRaisesRegex(ValueError, "reservation is unavailable"):
            Agent(self.store, FakeModel([ModelResponse(None, (call,))]), calendar=first).handle("calendar", message_id="calendar-lease")

        retry = Calendar()
        reply = Agent(self.store, FakeModel([ModelResponse(None, (call,))]), calendar=retry).handle("calendar", message_id="calendar-lease")
        self.assertEqual("Created calendar event: Plan from 2026-06-01T07:00:00+00:00 to 2026-06-01T08:00:00+00:00", reply.text)

    def test_calendar_retry_rejects_changed_configuration_before_provider_transport(self):
        class Recorder:
            def __init__(self, responses):
                self.responses = list(responses)
                self.requests = []
            def __call__(self, request, timeout):
                self.requests.append(request)
                return self.responses.pop(0)

        def response(status, body):
            return CalendarResponse(status, json.dumps(body).encode())

        call = ToolCall("calendar", "calendar_create", {"summary": "Plan", "start": "2026-06-01T09:00:00", "end": "2026-06-01T10:00:00"})
        first_recorder = Recorder([response(200, {"access_token": "access", "expires_in": 3600}), response(500, {})])
        first = CalendarClient(
            CalendarConfig("client-sentinel", "secret-sentinel", "refresh-sentinel", "Europe/Berlin"),
            transport=first_recorder,
        )
        with self.assertRaises(CalendarRetryableError):
            Agent(self.store, FakeModel([ModelResponse(None, (call,))]), calendar=first).handle("calendar", message_id="calendar-binding")
        self.assertEqual(2, len(first_recorder.requests))

        self.clock.advance(seconds=61)
        changed_recorder = Recorder([])
        changed = CalendarClient(
            CalendarConfig("client-sentinel", "secret-sentinel", "rotated-refresh-sentinel", "Europe/Berlin"),
            transport=changed_recorder,
        )
        reply = Agent(self.store, FakeModel([ModelResponse(None, (call,))]), calendar=changed).handle("calendar", message_id="calendar-binding")
        persisted_hash = self.store.connection.execute(
            "SELECT arguments_hash FROM conversation_tools WHERE incoming_message_id = ?", ("calendar-binding",)
        ).fetchone()[0]

        self.assertEqual("Error: incoming message ID was already used for a different tool plan", reply.text)
        self.assertEqual(0, len(changed_recorder.requests))
        for sentinel in ("client-sentinel", "secret-sentinel", "refresh-sentinel", "rotated-refresh-sentinel"):
            self.assertNotIn(sentinel, persisted_hash)
            self.assertNotIn(sentinel, reply.text)
            self.assertNotIn(sentinel, repr(changed.config))

    def test_concurrent_same_message_does_not_double_mutate(self):
        replies = []
        errors = []

        def handle(call_id):
            try:
                store = BrainStore(Path(self.temporary_directory.name) / "brain.sqlite3", clock=FakeClock(self.now))
                try:
                    agent = Agent(store, FakeModel([ModelResponse(None, (ToolCall(call_id, "save_note", {"text": "once"}),))]))
                    replies.append(agent.handle("save", message_id="concurrent").text)
                finally:
                    store.close()
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=handle, args=(str(index),)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual([], errors)
        self.assertEqual(1, len(self.store.search_notes("once")))
        self.assertEqual(2, len(replies))
        self.assertTrue(all(reply in {"Saved note: once", "Error: message is already being processed"} for reply in replies))


if __name__ == "__main__":
    unittest.main()
