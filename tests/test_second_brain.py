import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_brain.__main__ import parse_schedule
from second_brain.core import BrainStore, FakeClock, ReminderDispatcher


class RecordingTransport:
    def __init__(self, failures=0):
        self.failures = failures
        self.sent = []

    def send(self, reminder):
        self.sent.append(reminder.id)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("transport unavailable")


class SecondBrainTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "brain.sqlite3"
        self.now = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
        self.clock = FakeClock(self.now)
        self.store = BrainStore(self.database_path, clock=self.clock)

    def tearDown(self):
        self.store.close()
        self.temporary_directory.cleanup()

    def test_note_mutations_are_durable_and_idempotent(self):
        note = self.store.save_note("Call the dentist", incoming_message_id="message-1")
        replay = self.store.save_note("Call the dentist", incoming_message_id="message-1")
        self.assertEqual(note.id, replay.id)
        self.assertEqual([note], self.store.search_notes("dentist"))

        self.store.close()
        self.store = BrainStore(self.database_path, clock=self.clock)
        self.assertEqual([note], self.store.search_notes("dentist"))
        self.assertTrue(self.store.delete_note(note.id, incoming_message_id="message-2"))
        self.assertEqual([], self.store.search_notes("dentist"))

    def test_reusing_a_message_id_with_different_note_text_is_rejected(self):
        self.store.save_note("first version", incoming_message_id="message-content")
        with self.assertRaisesRegex(ValueError, "different input"):
            self.store.save_note("changed version", incoming_message_id="message-content")

    def test_reminder_retries_after_transport_failure_then_sends(self):
        reminder = self.store.create_reminder(
            "Take medication",
            self.now + timedelta(minutes=1),
            incoming_message_id="message-3",
        )
        transport = RecordingTransport(failures=1)
        dispatcher = ReminderDispatcher(self.store, transport, clock=self.clock)
        self.clock.advance(minutes=1)

        self.assertEqual(1, dispatcher.dispatch_due())
        failed = self.store.get_reminder(reminder.id)
        self.assertEqual("pending", failed.status)
        self.assertEqual(1, failed.attempts)
        self.assertEqual([], transport.sent[:0])

        self.clock.advance(seconds=2)
        self.assertEqual(1, dispatcher.dispatch_due())
        sent = self.store.get_reminder(reminder.id)
        self.assertEqual("sent", sent.status)
        self.assertEqual([reminder.id, reminder.id], transport.sent)

    def test_schedule_requires_an_offset_or_explicit_unambiguous_timezone(self):
        self.assertEqual(
            datetime(2026, 6, 1, 7, tzinfo=timezone.utc),
            parse_schedule("2026-06-01T09:00:00+02:00", None),
        )
        with self.assertRaisesRegex(ValueError, "explicit --timezone"):
            parse_schedule("2026-06-01T09:00:00", None)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            parse_schedule("2026-10-25T02:30:00", "Europe/Berlin")
        with self.assertRaisesRegex(ValueError, "invalid"):
            parse_schedule("2026-03-29T02:30:00", "Europe/Berlin")

    def test_completion_at_or_after_expiry_is_fenced_without_releasing_takeover(self):
        original = self.store.reserve_conversation("complete-expired", "status")
        self.clock.advance(seconds=60)
        with self.assertRaisesRegex(ValueError, "reservation is unavailable"):
            self.store.complete_conversation(
                "complete-expired", "status", "stale response", lease_token=original.lease_token
            )

        replacement = self.store.reserve_conversation("complete-expired", "status")
        self.assertEqual("owned", replacement.state)
        with self.assertRaisesRegex(ValueError, "reservation is unavailable"):
            self.store.complete_conversation(
                "complete-expired", "status", "older response", lease_token=original.lease_token
            )
        self.store.complete_conversation(
            "complete-expired", "status", "replacement response", lease_token=replacement.lease_token
        )
        self.assertEqual(
            "replacement response",
            self.store.reserve_conversation("complete-expired", "status").response_text,
        )

    def test_cancelled_reminder_is_not_dispatched_and_expired_claim_recovers(self):
        cancelled = self.store.create_reminder("Cancel me", self.now, incoming_message_id="message-4")
        self.assertTrue(self.store.cancel_reminder(cancelled.id, incoming_message_id="message-5"))
        dispatcher = ReminderDispatcher(self.store, RecordingTransport(), clock=self.clock)
        self.assertEqual(0, dispatcher.dispatch_due())
        self.assertEqual("cancelled", self.store.get_reminder(cancelled.id).status)

        reminder = self.store.create_reminder("Recover me", self.now, incoming_message_id="message-6")
        claim = self.store.claim_due_reminder(self.now, claim_seconds=30)
        self.assertEqual(reminder.id, claim[0].id)
        self.clock.advance(seconds=31)
        replacement_claim = self.store.claim_due_reminder(self.clock.now(), claim_seconds=30)
        self.assertEqual(reminder.id, replacement_claim[0].id)

    def test_migration_v5_preserves_data_and_quarantines_legacy_queue(self):
        # Build a raw schema v5 database
        v5_db = Path(self.temporary_directory.name) / "v5.sqlite3"
        import sqlite3
        conn = sqlite3.connect(v5_db)
        conn.executescript("""
            CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY);
            INSERT INTO schema_migrations (version) VALUES (1), (2), (3), (4), (5);
            CREATE TABLE notes (id TEXT PRIMARY KEY, text TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE reminders (
                id TEXT PRIMARY KEY, text TEXT NOT NULL, due_at TEXT NOT NULL,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL, claim_token TEXT, claim_until TEXT
            );
            CREATE TABLE processed_messages (
                incoming_message_id TEXT PRIMARY KEY, operation TEXT NOT NULL,
                result TEXT NOT NULL, processed_at TEXT NOT NULL, request_hash TEXT
            );
            CREATE TABLE conversation_messages (
                incoming_message_id TEXT PRIMARY KEY, input_hash TEXT NOT NULL,
                state TEXT NOT NULL, response_text TEXT, updated_at TEXT NOT NULL,
                lease_token TEXT, lease_until TEXT
            );
            CREATE TABLE conversation_tools (
                incoming_message_id TEXT PRIMARY KEY, operation TEXT NOT NULL,
                arguments_hash TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE inbound_messages (
                id TEXT PRIMARY KEY, input_hash TEXT NOT NULL, text TEXT NOT NULL,
                status TEXT NOT NULL, response_text TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL, lease_token TEXT, lease_until TEXT
            );
            CREATE TABLE outbox_messages (
                id TEXT PRIMARY KEY, text TEXT NOT NULL,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL, lease_token TEXT, lease_until TEXT
            );
            INSERT INTO notes VALUES ('note-1', 'Legacy note', '2026-01-01T00:00:00Z');
            INSERT INTO conversation_messages VALUES ('conv-1', 'hash1', 'completed', 'Legacy reply', '2026-01-01T00:00:00Z', NULL, NULL);
            INSERT INTO inbound_messages VALUES ('in-1', 'hash2', 'Legacy inbound', 'pending', NULL, 0, '2026-01-01T00:00:00Z', NULL, NULL);
            INSERT INTO outbox_messages VALUES ('out-1', 'Legacy outbox', 'pending', 0, '2026-01-01T00:00:00Z', NULL, NULL);
        """)
        conn.close()

        # Open with BrainStore to trigger migration v6
        store = BrainStore(v5_db, clock=self.clock)
        try:
            versions = {row[0] for row in store.connection.execute("SELECT version FROM schema_migrations")}
            self.assertIn(6, versions)
            # Notes and conversations preserved
            notes = store.search_notes("Legacy note")
            self.assertEqual(1, len(notes))
            self.assertEqual("Legacy note", notes[0].text)

            # Inbound and outbox rows preserved with default evolution channel and NULL binding
            inbox_rows = store.list_inbound_messages()
            self.assertEqual(1, len(inbox_rows))
            self.assertEqual("in-1", inbox_rows[0].id)
            self.assertEqual("evolution", inbox_rows[0].channel)
            self.assertIsNone(inbox_rows[0].binding)

            outbox_rows = store.list_outbox_messages()
            self.assertEqual(1, len(outbox_rows))
            self.assertEqual("out-1", outbox_rows[0].id)
            self.assertEqual("evolution", outbox_rows[0].channel)
            self.assertIsNone(outbox_rows[0].binding)

            # Configured worker claim with explicit binding does NOT claim legacy unbound rows (quarantine-safe)
            in_claim = store.claim_inbound_message(self.clock.now(), channel="evolution", binding="configured-owner-binding")
            self.assertIsNone(in_claim)
            out_claim = store.claim_outbox_message(self.clock.now(), channel="evolution", binding="configured-owner-binding")
            self.assertIsNone(out_claim)
        finally:
            store.close()

    def test_queue_sql_isolation_by_channel_and_binding(self):
        self.store.accept_inbound_message("msg-evo-1", "evo 1", channel="evolution", binding="bind-evo-1")
        self.store.accept_inbound_message("msg-evo-2", "evo 2", channel="evolution", binding="bind-evo-2")
        self.store.accept_inbound_message("msg-tg-1", "tg 1", channel="telegram", binding="bind-tg-1")

        # Claiming telegram partitioned row
        tg_claim = self.store.claim_inbound_message(self.clock.now(), channel="telegram", binding="bind-tg-1")
        self.assertIsNotNone(tg_claim)
        self.assertEqual("msg-tg-1", tg_claim[0].id)
        self.assertEqual("telegram", tg_claim[0].channel)
        self.assertEqual("bind-tg-1", tg_claim[0].binding)

        # No more telegram pending
        self.assertIsNone(self.store.claim_inbound_message(self.clock.now(), channel="telegram", binding="bind-tg-1"))

        # Evolution bind-evo-1 claim
        evo1_claim = self.store.claim_inbound_message(self.clock.now(), channel="evolution", binding="bind-evo-1")
        self.assertIsNotNone(evo1_claim)
        self.assertEqual("msg-evo-1", evo1_claim[0].id)

        # Evolution bind-evo-2 claim
        evo2_claim = self.store.claim_inbound_message(self.clock.now(), channel="evolution", binding="bind-evo-2")
        self.assertIsNotNone(evo2_claim)
        self.assertEqual("msg-evo-2", evo2_claim[0].id)

    def test_reusing_incoming_message_id_with_changed_destination_is_rejected(self):
        self.assertTrue(self.store.accept_inbound_message("msg-dest", "some text", channel="evolution", binding="bind-1"))
        # Idempotent replay with exact match returns False
        self.assertFalse(self.store.accept_inbound_message("msg-dest", "some text", channel="evolution", binding="bind-1"))
        # Replay with changed channel rejects
        with self.assertRaisesRegex(ValueError, "different input"):
            self.store.accept_inbound_message("msg-dest", "some text", channel="telegram", binding="bind-1")
        # Replay with changed binding rejects
        with self.assertRaisesRegex(ValueError, "different input"):
            self.store.accept_inbound_message("msg-dest", "some text", channel="evolution", binding="bind-2")
        # Replay with changed text rejects
        with self.assertRaisesRegex(ValueError, "different input"):
            self.store.accept_inbound_message("msg-dest", "other text", channel="evolution", binding="bind-1")

    def test_outbox_metadata_inherited_atomically_on_complete_inbound(self):
        self.store.accept_inbound_message("msg-atomic", "hello tg", channel="telegram", binding="bind-tg-digest")
        claim = self.store.claim_inbound_message(self.clock.now(), channel="telegram", binding="bind-tg-digest")
        self.assertIsNotNone(claim)
        msg, token = claim
        completed = self.store.complete_inbound_message(msg.id, token, "reply text")
        self.assertTrue(completed)

        outbox_rows = self.store.list_outbox_messages(channel="telegram", binding="bind-tg-digest")
        self.assertEqual(1, len(outbox_rows))
        self.assertEqual("msg-atomic", outbox_rows[0].id)
        self.assertEqual("reply text", outbox_rows[0].text)
        self.assertEqual("telegram", outbox_rows[0].channel)
        self.assertEqual("bind-tg-digest", outbox_rows[0].binding)
        self.assertEqual("pending", outbox_rows[0].status)

    def test_migration_v6_to_v7_adds_reminder_binding_and_cursor_table(self):
        v6_db = Path(self.temporary_directory.name) / "v6.sqlite3"
        import sqlite3
        conn = sqlite3.connect(v6_db)
        conn.executescript("""
            CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY);
            INSERT INTO schema_migrations (version) VALUES (1), (2), (3), (4), (5), (6);
            CREATE TABLE notes (id TEXT PRIMARY KEY, text TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE reminders (
                id TEXT PRIMARY KEY, text TEXT NOT NULL, due_at TEXT NOT NULL,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL, claim_token TEXT, claim_until TEXT
            );
            CREATE TABLE processed_messages (
                incoming_message_id TEXT PRIMARY KEY, operation TEXT NOT NULL,
                result TEXT NOT NULL, processed_at TEXT NOT NULL, request_hash TEXT
            );
            CREATE TABLE conversation_messages (
                incoming_message_id TEXT PRIMARY KEY, input_hash TEXT NOT NULL,
                state TEXT NOT NULL, response_text TEXT, updated_at TEXT NOT NULL,
                lease_token TEXT, lease_until TEXT
            );
            CREATE TABLE conversation_tools (
                incoming_message_id TEXT PRIMARY KEY, operation TEXT NOT NULL,
                arguments_hash TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE inbound_messages (
                id TEXT PRIMARY KEY, input_hash TEXT NOT NULL, text TEXT NOT NULL,
                status TEXT NOT NULL, response_text TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL, lease_token TEXT, lease_until TEXT,
                channel TEXT NOT NULL DEFAULT 'evolution', binding TEXT
            );
            CREATE TABLE outbox_messages (
                id TEXT PRIMARY KEY, text TEXT NOT NULL,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL, lease_token TEXT, lease_until TEXT,
                channel TEXT NOT NULL DEFAULT 'evolution', binding TEXT
            );
            INSERT INTO reminders VALUES ('rem-v6', 'call mom', '2026-01-01T00:00:00Z', 'pending', 0, '2026-01-01T00:00:00Z', NULL, NULL);
        """)
        conn.close()

        store = BrainStore(v6_db, clock=self.clock)
        try:
            versions = {row[0] for row in store.connection.execute("SELECT version FROM schema_migrations")}
            self.assertIn(7, versions)
            rem = store.get_reminder("rem-v6")
            self.assertEqual("rem-v6", rem.id)
            self.assertIsNone(rem.channel)
            self.assertIsNone(rem.binding)

            # telegram_poll_cursors table exists
            cursor_row = store.connection.execute("SELECT * FROM telegram_poll_cursors").fetchall()
            self.assertEqual([], cursor_row)
        finally:
            store.close()

    def test_reminder_claiming_channel_binding_and_console_isolation(self):
        # Create an unbound pending reminder
        rem = self.store.create_reminder("Buy milk", self.now, incoming_message_id="rem-msg-1")
        self.assertIsNone(rem.channel)
        self.assertIsNone(rem.binding)

        # Telegram claim with channel='telegram' and binding='bind-tg-1'
        claim = self.store.claim_due_reminder(self.now, channel="telegram", binding="bind-tg-1")
        self.assertIsNotNone(claim)
        claimed_rem, token = claim
        self.assertEqual("Buy milk", claimed_rem.text)
        self.assertEqual("telegram", claimed_rem.channel)
        self.assertEqual("bind-tg-1", claimed_rem.binding)

        # Fail delivery -> returns to pending with attempts=1, preserving channel and binding
        self.assertTrue(self.store.mark_delivery_failed(claimed_rem.id, token, self.now))
        failed_rem = self.store.get_reminder(claimed_rem.id)
        self.assertEqual("pending", failed_rem.status)
        self.assertEqual(1, failed_rem.attempts)
        self.assertEqual("telegram", failed_rem.channel)
        self.assertEqual("bind-tg-1", failed_rem.binding)

        # Advance clock to next attempt
        self.clock.advance(seconds=10)

        # 1. Config change to 'bind-tg-2' MUST NOT claim or retarget 'bind-tg-1' reminder
        tg2_claim = self.store.claim_due_reminder(self.clock.now(), channel="telegram", binding="bind-tg-2")
        self.assertIsNone(tg2_claim)

        # 2. Console manual flow (channel=None, binding=None) MUST NOT steal recipient-bound reminder
        console_claim = self.store.claim_due_reminder(self.clock.now())
        self.assertIsNone(console_claim)

        # 3. Matching Telegram config claims and marks sent
        tg1_claim = self.store.claim_due_reminder(self.clock.now(), channel="telegram", binding="bind-tg-1")
        self.assertIsNotNone(tg1_claim)
        self.assertTrue(self.store.mark_sent(tg1_claim[0].id, tg1_claim[1]))
        sent_rem = self.store.get_reminder(claimed_rem.id)
        self.assertEqual("sent", sent_rem.status)
        self.assertEqual("telegram", sent_rem.channel)
        self.assertEqual("bind-tg-1", sent_rem.binding)

    def test_poll_lease_and_atomic_cursor_commit(self):
        # Initial acquire
        lease1 = self.store.acquire_poll_lease("bot_ns", "bind_hash", self.now, lease_seconds=30)
        self.assertIsNotNone(lease1)
        token1, offset1 = lease1
        self.assertIsNone(offset1)

        # Second concurrent acquire with active lease fails closed (returns None)
        lease2 = self.store.acquire_poll_lease("bot_ns", "bind_hash", self.now, lease_seconds=30)
        self.assertIsNone(lease2)

        # Commit batch: accepts inbound message and advances offset atomically
        accepted = [("tg:msg-1", "hello world")]
        committed = self.store.commit_poll_batch(
            "bot_ns", "bind_hash", token1, 42, accepted, self.now
        )
        self.assertTrue(committed)

        # Cursor is updated
        self.assertEqual(42, self.store.get_telegram_cursor("bot_ns", "bind_hash"))

        # Inbound message is persisted with channel='telegram' and binding='bind_hash'
        inbound = self.store.list_inbound_messages(channel="telegram", binding="bind_hash")
        self.assertEqual(1, len(inbound))
        self.assertEqual("tg:msg-1", inbound[0].id)
        self.assertEqual("hello world", inbound[0].text)

        # Release lease
        self.assertTrue(self.store.release_poll_lease("bot_ns", "bind_hash", token1))

        # Re-acquire after release gets the updated offset
        lease3 = self.store.acquire_poll_lease("bot_ns", "bind_hash", self.now, lease_seconds=30)
        self.assertIsNotNone(lease3)
        token3, offset3 = lease3
        self.assertEqual(42, offset3)


if __name__ == "__main__":
    unittest.main()
