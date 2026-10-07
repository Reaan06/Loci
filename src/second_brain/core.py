"""SQLite persistence and durable reminder delivery primitives."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import sqlite3
from typing import Protocol
from uuid import uuid4

MAX_TEXT_LENGTH = 5_000
MAX_QUERY_LENGTH = 500
RETRY_MAX_SECONDS = 3_600
CONVERSATION_LEASE_SECONDS = 60


@dataclass(frozen=True)
class Note:
    id: str
    text: str
    created_at: datetime


@dataclass(frozen=True)
class Reminder:
    id: str
    text: str
    due_at: datetime
    status: str
    attempts: int
    next_attempt_at: datetime
    channel: str | None = None
    binding: str | None = None


@dataclass(frozen=True)
class ConversationReservation:
    state: str
    response_text: str | None = None
    lease_token: str | None = None


@dataclass(frozen=True)
class InboundMessage:
    id: str
    text: str
    status: str
    attempts: int
    next_attempt_at: datetime
    channel: str = "evolution"
    binding: str | None = None


@dataclass(frozen=True)
class OutboxMessage:
    id: str
    text: str
    status: str
    attempts: int
    next_attempt_at: datetime
    channel: str = "evolution"
    binding: str | None = None


class Clock(Protocol):
    def now(self) -> datetime: ...


class Transport(Protocol):
    def send(self, reminder: Reminder) -> None: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FakeClock:
    """A controllable UTC clock for functional tests."""

    def __init__(self, current: datetime):
        self.current = require_utc_datetime(current, "current")

    def now(self) -> datetime:
        return self.current

    def advance(self, **delta: int) -> None:
        self.current += timedelta(**delta)


class BrainStore:
    def __init__(self, database_path: Path | str, clock: Clock | None = None):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or SystemClock()
        self._active_conversation_leases: dict[str, str] = {}
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def save_note(self, text: str, *, incoming_message_id: str) -> Note:
        text = require_text(text, "note text")
        message_id = require_message_id(incoming_message_id)
        with self._transaction():
            self._require_active_conversation_lease_for_mutation()
            existing = self._mutation_result(message_id, "save_note", {"text": text})
            if existing:
                return self._note_by_id(existing)
            note = Note(str(uuid4()), text, self._now())
            self.connection.execute(
                "INSERT INTO notes (id, text, created_at) VALUES (?, ?, ?)",
                (note.id, note.text, encode_datetime(note.created_at)),
            )
            self._record_mutation(message_id, "save_note", {"text": text}, note.id)
            return note

    def search_notes(self, query: str) -> list[Note]:
        query = require_query(query)
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self.connection.execute(
            "SELECT id, text, created_at FROM notes WHERE text LIKE ? ESCAPE '\\' "
            "ORDER BY created_at DESC, id DESC",
            (f"%{escaped}%",),
        ).fetchall()
        return [note_from_row(row) for row in rows]

    def delete_note(self, note_id: str, *, incoming_message_id: str) -> bool:
        message_id = require_message_id(incoming_message_id)
        with self._transaction():
            self._require_active_conversation_lease_for_mutation()
            existing = self._mutation_result(message_id, "delete_note", {"note_id": note_id})
            if existing is not None:
                return existing == "deleted"
            cursor = self.connection.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            result = "deleted" if cursor.rowcount else "not_found"
            self._record_mutation(message_id, "delete_note", {"note_id": note_id}, result)
            return result == "deleted"

    def create_reminder(self, text: str, due_at: datetime, *, incoming_message_id: str) -> Reminder:
        text = require_text(text, "reminder text")
        due_at = require_utc_datetime(due_at, "due_at")
        message_id = require_message_id(incoming_message_id)
        with self._transaction():
            self._require_active_conversation_lease_for_mutation()
            existing = self._mutation_result(
                message_id, "create_reminder", {"text": text, "due_at": encode_datetime(due_at)}
            )
            if existing:
                return self.get_reminder(existing)
            reminder = Reminder(str(uuid4()), text, due_at, "pending", 0, due_at)
            self.connection.execute(
                """INSERT INTO reminders
                   (id, text, due_at, status, attempts, next_attempt_at, claim_token, claim_until)
                   VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)""",
                (
                    reminder.id,
                    reminder.text,
                    encode_datetime(reminder.due_at),
                    reminder.status,
                    reminder.attempts,
                    encode_datetime(reminder.next_attempt_at),
                ),
            )
            self._record_mutation(
                message_id, "create_reminder", {"text": text, "due_at": encode_datetime(due_at)}, reminder.id
            )
            return reminder

    def list_reminders(self, include_finished: bool = False) -> list[Reminder]:
        statement = "SELECT * FROM reminders"
        parameters: tuple[str, ...] = ()
        if not include_finished:
            statement += " WHERE status IN ('pending', 'claimed')"
        rows = self.connection.execute(statement + " ORDER BY due_at, id", parameters).fetchall()
        return [reminder_from_row(row) for row in rows]

    def get_reminder(self, reminder_id: str) -> Reminder:
        row = self.connection.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,)).fetchone()
        if row is None:
            raise KeyError(f"reminder not found: {reminder_id}")
        return reminder_from_row(row)

    def cancel_reminder(self, reminder_id: str, *, incoming_message_id: str) -> bool:
        message_id = require_message_id(incoming_message_id)
        with self._transaction():
            self._require_active_conversation_lease_for_mutation()
            existing = self._mutation_result(message_id, "cancel_reminder", {"reminder_id": reminder_id})
            if existing is not None:
                return existing == "cancelled"
            cursor = self.connection.execute(
                "UPDATE reminders SET status = 'cancelled', claim_token = NULL, claim_until = NULL "
                "WHERE id = ? AND status IN ('pending', 'claimed')",
                (reminder_id,),
            )
            result = "cancelled" if cursor.rowcount else "not_found_or_finished"
            self._record_mutation(message_id, "cancel_reminder", {"reminder_id": reminder_id}, result)
            return result == "cancelled"

    def claim_due_reminder(
        self,
        now: datetime,
        *,
        channel: str | None = None,
        binding: str | None = None,
        claim_seconds: int = 60,
    ) -> tuple[Reminder, str] | None:
        now = require_utc_datetime(now, "now")
        if claim_seconds <= 0:
            raise ValueError("claim_seconds must be positive")
        if (channel is None) != (binding is None):
            raise ValueError("channel and binding must both be provided, or both omitted")
        claim_until = now + timedelta(seconds=claim_seconds)
        with self._transaction():
            if channel is not None and binding is not None:
                self.connection.execute(
                    """UPDATE reminders SET status = 'pending', claim_token = NULL, claim_until = NULL
                       WHERE status = 'claimed' AND claim_until <= ?
                         AND ((channel = ? AND binding = ?) OR (channel IS NULL AND binding IS NULL))""",
                    (encode_datetime(now), channel, binding),
                )
                row = self.connection.execute(
                    """SELECT * FROM reminders WHERE status = 'pending' AND next_attempt_at <= ?
                         AND ((channel = ? AND binding = ?) OR (channel IS NULL AND binding IS NULL))
                       ORDER BY next_attempt_at, id LIMIT 1""",
                    (encode_datetime(now), channel, binding),
                ).fetchone()
                if row is None:
                    return None
                token = str(uuid4())
                updated = self.connection.execute(
                    """UPDATE reminders SET status = 'claimed', claim_token = ?, claim_until = ?,
                                            channel = ?, binding = ?
                       WHERE id = ? AND status = 'pending'
                         AND ((channel = ? AND binding = ?) OR (channel IS NULL AND binding IS NULL))""",
                    (token, encode_datetime(claim_until), channel, binding, row["id"], channel, binding),
                )
            else:
                self.connection.execute(
                    """UPDATE reminders SET status = 'pending', claim_token = NULL, claim_until = NULL
                       WHERE status = 'claimed' AND claim_until <= ? AND channel IS NULL AND binding IS NULL""",
                    (encode_datetime(now),),
                )
                row = self.connection.execute(
                    """SELECT * FROM reminders WHERE status = 'pending' AND next_attempt_at <= ?
                         AND channel IS NULL AND binding IS NULL
                       ORDER BY next_attempt_at, id LIMIT 1""",
                    (encode_datetime(now),),
                ).fetchone()
                if row is None:
                    return None
                token = str(uuid4())
                updated = self.connection.execute(
                    """UPDATE reminders SET status = 'claimed', claim_token = ?, claim_until = ?
                       WHERE id = ? AND status = 'pending' AND channel IS NULL AND binding IS NULL""",
                    (token, encode_datetime(claim_until), row["id"]),
                )
            if updated.rowcount != 1:
                return None
            claimed = self.connection.execute("SELECT * FROM reminders WHERE id = ?", (row["id"],)).fetchone()
            return reminder_from_row(claimed), token

    def mark_sent(self, reminder_id: str, claim_token: str) -> bool:
        with self._transaction():
            result = self.connection.execute(
                """UPDATE reminders SET status = 'sent', claim_token = NULL, claim_until = NULL
                   WHERE id = ? AND status = 'claimed' AND claim_token = ?""",
                (reminder_id, claim_token),
            )
            return result.rowcount == 1

    def mark_delivery_failed(self, reminder_id: str, claim_token: str, now: datetime) -> bool:
        now = require_utc_datetime(now, "now")
        with self._transaction():
            row = self.connection.execute(
                "SELECT attempts FROM reminders WHERE id = ? AND status = 'claimed' AND claim_token = ?",
                (reminder_id, claim_token),
            ).fetchone()
            if row is None:
                return False
            attempts = row["attempts"] + 1
            delay_seconds = min(2**attempts, RETRY_MAX_SECONDS)
            self.connection.execute(
                """UPDATE reminders SET status = 'pending', attempts = ?, next_attempt_at = ?,
                   claim_token = NULL, claim_until = NULL WHERE id = ? AND claim_token = ?""",
                (attempts, encode_datetime(now + timedelta(seconds=delay_seconds)), reminder_id, claim_token),
            )
            return True

    def _migrate(self) -> None:
        with self.connection:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)"
            )
            applied = {row[0] for row in self.connection.execute("SELECT version FROM schema_migrations")}
            if 1 not in applied:
                self.connection.executescript(
                    """CREATE TABLE notes (
                           id TEXT PRIMARY KEY, text TEXT NOT NULL, created_at TEXT NOT NULL
                       );
                       CREATE TABLE reminders (
                           id TEXT PRIMARY KEY, text TEXT NOT NULL, due_at TEXT NOT NULL,
                           status TEXT NOT NULL CHECK(status IN ('pending', 'claimed', 'sent', 'cancelled')),
                           attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT NOT NULL,
                           claim_token TEXT, claim_until TEXT
                       );
                       CREATE TABLE processed_messages (
                           incoming_message_id TEXT PRIMARY KEY, operation TEXT NOT NULL,
                           result TEXT NOT NULL, processed_at TEXT NOT NULL
                       );
                       CREATE INDEX reminders_due_idx ON reminders (status, next_attempt_at);
                       INSERT INTO schema_migrations (version) VALUES (1);"""
                )
            if 2 not in applied:
                self.connection.execute("ALTER TABLE processed_messages ADD COLUMN request_hash TEXT")
                self.connection.executescript(
                    """CREATE TABLE conversation_messages (
                           incoming_message_id TEXT PRIMARY KEY,
                           input_hash TEXT NOT NULL,
                           state TEXT NOT NULL CHECK(state IN ('processing', 'completed')),
                           response_text TEXT,
                           updated_at TEXT NOT NULL
                       );
                       INSERT INTO schema_migrations (version) VALUES (2);"""
                )
            if 3 not in applied:
                self.connection.executescript(
                    """CREATE TABLE conversation_tools (
                           incoming_message_id TEXT PRIMARY KEY,
                           operation TEXT NOT NULL,
                           arguments_hash TEXT NOT NULL,
                           updated_at TEXT NOT NULL
                       );
                       INSERT INTO schema_migrations (version) VALUES (3);"""
                )
            if 4 not in applied:
                self.connection.executescript(
                    """ALTER TABLE conversation_messages ADD COLUMN lease_token TEXT;
                       ALTER TABLE conversation_messages ADD COLUMN lease_until TEXT;
                       CREATE INDEX conversation_leases_idx
                           ON conversation_messages (state, lease_until);
                       INSERT INTO schema_migrations (version) VALUES (4);"""
                )
            if 5 not in applied:
                self.connection.executescript(
                    """CREATE TABLE inbound_messages (
                           id TEXT PRIMARY KEY, input_hash TEXT NOT NULL, text TEXT NOT NULL,
                           status TEXT NOT NULL CHECK(status IN ('pending', 'claimed', 'completed')),
                           response_text TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                           next_attempt_at TEXT NOT NULL, lease_token TEXT, lease_until TEXT
                       );
                       CREATE TABLE outbox_messages (
                           id TEXT PRIMARY KEY, text TEXT NOT NULL,
                           status TEXT NOT NULL CHECK(status IN ('pending', 'claimed', 'sent')),
                           attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT NOT NULL,
                           lease_token TEXT, lease_until TEXT
                       );
                       CREATE INDEX inbound_due_idx ON inbound_messages (status, next_attempt_at);
                       CREATE INDEX outbox_due_idx ON outbox_messages (status, next_attempt_at);
                       INSERT INTO schema_migrations (version) VALUES (5);"""
                )
            if 6 not in applied:
                self.connection.executescript(
                    """ALTER TABLE inbound_messages ADD COLUMN channel TEXT NOT NULL DEFAULT 'evolution';
                       ALTER TABLE inbound_messages ADD COLUMN binding TEXT;
                       ALTER TABLE outbox_messages ADD COLUMN channel TEXT NOT NULL DEFAULT 'evolution';
                       ALTER TABLE outbox_messages ADD COLUMN binding TEXT;
                       CREATE INDEX IF NOT EXISTS inbound_channel_binding_due_idx
                           ON inbound_messages (channel, binding, status, next_attempt_at);
                       CREATE INDEX IF NOT EXISTS outbox_channel_binding_due_idx
                           ON outbox_messages (channel, binding, status, next_attempt_at);
                       INSERT INTO schema_migrations (version) VALUES (6);"""
                )
            if 7 not in applied:
                self.connection.executescript(
                    """ALTER TABLE reminders ADD COLUMN channel TEXT;
                       ALTER TABLE reminders ADD COLUMN binding TEXT;
                       CREATE INDEX IF NOT EXISTS reminders_channel_binding_due_idx
                           ON reminders (channel, binding, status, next_attempt_at);
                       CREATE TABLE IF NOT EXISTS telegram_poll_cursors (
                           bot_namespace TEXT NOT NULL,
                           binding TEXT NOT NULL,
                           next_offset INTEGER NOT NULL,
                           lease_token TEXT,
                           lease_until TEXT,
                           updated_at TEXT NOT NULL,
                           PRIMARY KEY (bot_namespace, binding)
                       );
                       INSERT INTO schema_migrations (version) VALUES (7);"""
                )

    def accept_inbound_message(
        self,
        incoming_message_id: str,
        text: str,
        *,
        channel: str = "evolution",
        binding: str | None = None,
    ) -> bool:
        """Durably accept a validated webhook delivery before returning an HTTP success."""
        with self._transaction():
            return self._accept_inbound_message_locked(
                incoming_message_id,
                text,
                channel=channel,
                binding=binding,
                now=self._now(),
            )

    def _accept_inbound_message_locked(
        self,
        incoming_message_id: str,
        text: str,
        *,
        channel: str,
        binding: str | None,
        now: datetime,
    ) -> bool:
        message_id = require_message_id(incoming_message_id)
        text = require_text(text, "message")
        channel = require_text(channel, "channel")
        if binding is not None:
            if not isinstance(binding, str) or not binding.strip():
                raise ValueError("binding must be a non-empty string when provided")
            binding = binding.strip()
        row = self.connection.execute(
            "SELECT input_hash FROM inbound_messages WHERE id = ?", (message_id,)
        ).fetchone()
        digest = fingerprint({"text": text, "channel": channel, "binding": binding})
        if row is not None:
            legacy_digest = fingerprint({"text": text})
            if row["input_hash"] != digest and not (
                row["input_hash"] == legacy_digest and channel == "evolution" and binding is None
            ):
                raise ValueError("incoming message ID was already used for different input")
            return False
        self.connection.execute(
            """INSERT INTO inbound_messages
               (id, input_hash, text, status, response_text, attempts, next_attempt_at, lease_token, lease_until, channel, binding)
               VALUES (?, ?, ?, 'pending', NULL, 0, ?, NULL, NULL, ?, ?)""",
            (message_id, digest, text, encode_datetime(now), channel, binding),
        )
        return True

    def list_inbound_messages(self, *, channel: str | None = None, binding: str | None = None) -> list[InboundMessage]:
        query = "SELECT * FROM inbound_messages"
        params: list[str] = []
        clauses: list[str] = []
        if channel is not None:
            clauses.append("channel = ?")
            params.append(channel)
        if binding is not None:
            clauses.append("binding = ?")
            params.append(binding)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id"
        return [inbound_from_row(row) for row in self.connection.execute(query, tuple(params))]

    def list_outbox_messages(self, *, channel: str | None = None, binding: str | None = None) -> list[OutboxMessage]:
        query = "SELECT * FROM outbox_messages"
        params: list[str] = []
        clauses: list[str] = []
        if channel is not None:
            clauses.append("channel = ?")
            params.append(channel)
        if binding is not None:
            clauses.append("binding = ?")
            params.append(binding)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id"
        return [outbox_from_row(row) for row in self.connection.execute(query, tuple(params))]

    def claim_inbound_message(
        self,
        now: datetime,
        *,
        channel: str | None = None,
        binding: str | None = None,
        claim_seconds: int = 60,
    ) -> tuple[InboundMessage, str] | None:
        return self._claim_queue("inbound_messages", now, claim_seconds, inbound_from_row, channel=channel, binding=binding)

    def claim_outbox_message(
        self,
        now: datetime,
        *,
        channel: str | None = None,
        binding: str | None = None,
        claim_seconds: int = 60,
    ) -> tuple[OutboxMessage, str] | None:
        return self._claim_queue("outbox_messages", now, claim_seconds, outbox_from_row, channel=channel, binding=binding)

    def _claim_queue(
        self,
        table: str,
        now: datetime,
        claim_seconds: int,
        factory,
        *,
        channel: str | None = None,
        binding: str | None = None,
    ):
        now = require_utc_datetime(now, "now")
        if claim_seconds <= 0:
            raise ValueError("claim_seconds must be positive")
        if (channel is None) != (binding is None):
            raise ValueError("channel and binding must both be provided, or both omitted")
        with self._transaction():
            if channel is not None and binding is not None:
                self.connection.execute(
                    f"UPDATE {table} SET status = 'pending', lease_token = NULL, lease_until = NULL "
                    f"WHERE status = 'claimed' AND lease_until <= ? AND channel = ? AND binding = ?",
                    (encode_datetime(now), channel, binding),
                )
                row = self.connection.execute(
                    f"SELECT * FROM {table} WHERE status = 'pending' AND next_attempt_at <= ? "
                    f"AND channel = ? AND binding = ? ORDER BY next_attempt_at, id LIMIT 1",
                    (encode_datetime(now), channel, binding),
                ).fetchone()
                if row is None:
                    return None
                token = str(uuid4())
                result = self.connection.execute(
                    f"UPDATE {table} SET status = 'claimed', lease_token = ?, lease_until = ? "
                    f"WHERE id = ? AND status = 'pending' AND channel = ? AND binding = ?",
                    (token, encode_datetime(now + timedelta(seconds=claim_seconds)), row["id"], channel, binding),
                )
            else:
                self.connection.execute(
                    f"UPDATE {table} SET status = 'pending', lease_token = NULL, lease_until = NULL "
                    f"WHERE status = 'claimed' AND lease_until <= ?",
                    (encode_datetime(now),),
                )
                row = self.connection.execute(
                    f"SELECT * FROM {table} WHERE status = 'pending' AND next_attempt_at <= ? "
                    f"ORDER BY next_attempt_at, id LIMIT 1",
                    (encode_datetime(now),),
                ).fetchone()
                if row is None:
                    return None
                token = str(uuid4())
                result = self.connection.execute(
                    f"UPDATE {table} SET status = 'claimed', lease_token = ?, lease_until = ? "
                    f"WHERE id = ? AND status = 'pending'",
                    (token, encode_datetime(now + timedelta(seconds=claim_seconds)), row["id"]),
                )
            if result.rowcount != 1:
                return None
            claimed = self.connection.execute(f"SELECT * FROM {table} WHERE id = ?", (row["id"],)).fetchone()
            return factory(claimed), token

    def retry_inbound_message(self, message_id: str, lease_token: str, now: datetime) -> bool:
        return self._retry_queue("inbound_messages", message_id, lease_token, now)

    def retry_outbox_message(self, message_id: str, lease_token: str, now: datetime) -> bool:
        return self._retry_queue("outbox_messages", message_id, lease_token, now)

    def _retry_queue(self, table: str, message_id: str, lease_token: str, now: datetime) -> bool:
        now = require_utc_datetime(now, "now")
        with self._transaction():
            row = self.connection.execute(
                f"SELECT attempts FROM {table} WHERE id = ? AND status = 'claimed' AND lease_token = ? AND lease_until > ?",
                (message_id, lease_token, encode_datetime(now)),
            ).fetchone()
            if row is None:
                return False
            attempts = row["attempts"] + 1
            delay = min(2 ** attempts, RETRY_MAX_SECONDS)
            updated = self.connection.execute(
                f"UPDATE {table} SET status = 'pending', attempts = ?, next_attempt_at = ?, lease_token = NULL, lease_until = NULL "
                "WHERE id = ? AND lease_token = ?",
                (attempts, encode_datetime(now + timedelta(seconds=delay)), message_id, lease_token),
            )
            return updated.rowcount == 1

    def complete_inbound_message(self, message_id: str, lease_token: str, response_text: str) -> bool:
        response_text = require_text(response_text, "response")
        now = self._now()
        with self._transaction():
            row = self.connection.execute(
                "SELECT status, lease_until, channel, binding FROM inbound_messages WHERE id = ? AND lease_token = ?",
                (message_id, lease_token),
            ).fetchone()
            if row is None or row["status"] != "claimed" or row["lease_until"] is None or decode_datetime(row["lease_until"]) <= now:
                return False
            updated = self.connection.execute(
                "UPDATE inbound_messages SET status = 'completed', response_text = ?, lease_token = NULL, lease_until = NULL "
                "WHERE id = ? AND lease_token = ?", (response_text, message_id, lease_token)
            )
            if updated.rowcount != 1:
                return False
            channel = row["channel"] if "channel" in row.keys() else "evolution"
            binding = row["binding"] if "binding" in row.keys() else None
            self.connection.execute(
                """INSERT OR IGNORE INTO outbox_messages
                   (id, text, status, attempts, next_attempt_at, lease_token, lease_until, channel, binding)
                   VALUES (?, ?, 'pending', 0, ?, NULL, NULL, ?, ?)""",
                (message_id, response_text, encode_datetime(now), channel, binding),
            )
            return True

    def mark_outbox_sent(self, message_id: str, lease_token: str) -> bool:
        now = self._now()
        with self._transaction():
            updated = self.connection.execute(
                "UPDATE outbox_messages SET status = 'sent', lease_token = NULL, lease_until = NULL "
                "WHERE id = ? AND status = 'claimed' AND lease_token = ? AND lease_until > ?",
                (message_id, lease_token, encode_datetime(now)),
            )
            return updated.rowcount == 1

    def get_telegram_cursor(self, bot_namespace: str, binding: str) -> int | None:
        row = self.connection.execute(
            "SELECT next_offset FROM telegram_poll_cursors WHERE bot_namespace = ? AND binding = ?",
            (bot_namespace, binding),
        ).fetchone()
        return row["next_offset"] if row is not None else None

    def acquire_poll_lease(
        self,
        bot_namespace: str,
        binding: str,
        now: datetime,
        *,
        lease_seconds: int = 60,
    ) -> tuple[str, int | None] | None:
        now = require_utc_datetime(now, "now")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        lease_until = encode_datetime(now + timedelta(seconds=lease_seconds))
        with self._transaction():
            row = self.connection.execute(
                "SELECT next_offset, lease_token, lease_until FROM telegram_poll_cursors WHERE bot_namespace = ? AND binding = ?",
                (bot_namespace, binding),
            ).fetchone()
            if row is not None:
                if row["lease_until"] is not None and decode_datetime(row["lease_until"]) > now:
                    return None
                token = str(uuid4())
                self.connection.execute(
                    """UPDATE telegram_poll_cursors
                       SET lease_token = ?, lease_until = ?, updated_at = ?
                       WHERE bot_namespace = ? AND binding = ?""",
                    (token, lease_until, encode_datetime(now), bot_namespace, binding),
                )
                return token, row["next_offset"]
            token = str(uuid4())
            self.connection.execute(
                """INSERT INTO telegram_poll_cursors
                   (bot_namespace, binding, next_offset, lease_token, lease_until, updated_at)
                   VALUES (?, ?, 0, ?, ?, ?)""",
                (bot_namespace, binding, token, lease_until, encode_datetime(now)),
            )
            return token, None

    def renew_poll_lease(
        self,
        bot_namespace: str,
        binding: str,
        lease_token: str,
        now: datetime,
        *,
        lease_seconds: int = 60,
    ) -> bool:
        now = require_utc_datetime(now, "now")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        lease_until = encode_datetime(now + timedelta(seconds=lease_seconds))
        with self._transaction():
            updated = self.connection.execute(
                """UPDATE telegram_poll_cursors
                   SET lease_until = ?, updated_at = ?
                   WHERE bot_namespace = ? AND binding = ? AND lease_token = ? AND lease_until > ?""",
                (lease_until, encode_datetime(now), bot_namespace, binding, lease_token, encode_datetime(now)),
            )
            return updated.rowcount == 1

    def release_poll_lease(self, bot_namespace: str, binding: str, lease_token: str) -> bool:
        with self._transaction():
            updated = self.connection.execute(
                """UPDATE telegram_poll_cursors
                   SET lease_token = NULL, lease_until = NULL, updated_at = ?
                   WHERE bot_namespace = ? AND binding = ? AND lease_token = ?""",
                (encode_datetime(self._now()), bot_namespace, binding, lease_token),
            )
            return updated.rowcount == 1

    def commit_poll_batch(
        self,
        bot_namespace: str,
        binding: str,
        lease_token: str,
        next_offset: int,
        accepted_messages: list[tuple[str, str]],
        now: datetime,
    ) -> bool:
        now = require_utc_datetime(now, "now")
        if not isinstance(next_offset, int) or isinstance(next_offset, bool) or next_offset < 0:
            raise ValueError("next_offset must be a non-negative integer")
        with self._transaction():
            row = self.connection.execute(
                "SELECT lease_token, lease_until FROM telegram_poll_cursors WHERE bot_namespace = ? AND binding = ?",
                (bot_namespace, binding),
            ).fetchone()
            if (
                row is None
                or row["lease_token"] != lease_token
                or row["lease_until"] is None
                or decode_datetime(row["lease_until"]) <= now
            ):
                return False
            for message_id, text in accepted_messages:
                self._accept_inbound_message_locked(
                    message_id, text, channel="telegram", binding=binding, now=now
                )
            self.connection.execute(
                """UPDATE telegram_poll_cursors
                   SET next_offset = ?, updated_at = ?
                   WHERE bot_namespace = ? AND binding = ? AND lease_token = ?""",
                (next_offset, encode_datetime(now), bot_namespace, binding, lease_token),
            )
            return True

    def reserve_conversation(self, incoming_message_id: str, text: str) -> ConversationReservation:
        """Atomically acquire a recoverable, fenced lease for an incoming message."""
        message_id = require_message_id(incoming_message_id)
        text = require_text(text, "message")
        input_hash = fingerprint({"text": text})
        now = self._now()
        lease_until = encode_datetime(now + timedelta(seconds=CONVERSATION_LEASE_SECONDS))
        with self._transaction():
            row = self.connection.execute(
                "SELECT input_hash, state, response_text, lease_token, lease_until "
                "FROM conversation_messages WHERE incoming_message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                token = str(uuid4())
                self.connection.execute(
                    """INSERT INTO conversation_messages
                       (incoming_message_id, input_hash, state, response_text, updated_at, lease_token, lease_until)
                       VALUES (?, ?, 'processing', NULL, ?, ?, ?)""",
                    (message_id, input_hash, encode_datetime(now), token, lease_until),
                )
                return ConversationReservation("owned", lease_token=token)
            if row["input_hash"] != input_hash:
                raise ValueError("incoming message ID was already used for different input")
            if row["state"] == "completed":
                return ConversationReservation("completed", response_text=row["response_text"])
            if row["lease_until"] is not None and decode_datetime(row["lease_until"]) > now:
                return ConversationReservation("active")
            token = str(uuid4())
            replaced = self.connection.execute(
                """UPDATE conversation_messages SET lease_token = ?, lease_until = ?, updated_at = ?
                   WHERE incoming_message_id = ? AND input_hash = ? AND state = 'processing'
                     AND lease_token IS ? AND lease_until IS ?""",
                (token, lease_until, encode_datetime(now), message_id, input_hash, row["lease_token"], row["lease_until"]),
            )
            if replaced.rowcount != 1:
                return ConversationReservation("active")
            return ConversationReservation("owned", lease_token=token)

    def activate_conversation_lease(self, incoming_message_id: str, lease_token: str) -> None:
        """Bind this store instance's tool execution to an owned conversation lease."""
        message_id = require_message_id(incoming_message_id)
        with self._transaction():
            self._require_conversation_lease(message_id, lease_token)
        self._active_conversation_leases[message_id] = lease_token

    def deactivate_conversation_lease(self, incoming_message_id: str) -> None:
        self._active_conversation_leases.pop(require_message_id(incoming_message_id), None)

    def reserve_tool_action(self, incoming_message_id: str, operation: str, arguments: dict[str, object]) -> None:
        """Bind the first mutation plan while holding the current conversation lease."""
        message_id = require_message_id(incoming_message_id)
        lease_token = self._active_conversation_leases.get(message_id)
        if lease_token is None:
            raise ValueError("conversation reservation is unavailable")
        if not operation or len(operation) > 100:
            raise ValueError("operation must be a short non-empty string")
        arguments_hash = fingerprint(arguments)
        with self._transaction():
            self._require_conversation_lease(message_id, lease_token)
            row = self.connection.execute(
                "SELECT operation, arguments_hash FROM conversation_tools WHERE incoming_message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                self.connection.execute(
                    "INSERT INTO conversation_tools VALUES (?, ?, ?, ?)",
                    (message_id, operation, arguments_hash, encode_datetime(self._now())),
                )
                return
            if row["operation"] != operation or row["arguments_hash"] != arguments_hash:
                raise ValueError("incoming message ID was already used for a different tool plan")

    def assert_conversation_lease(self, incoming_message_id: str, lease_token: str) -> None:
        with self._transaction():
            self._require_conversation_lease(require_message_id(incoming_message_id), lease_token)

    def complete_conversation(self, incoming_message_id: str, text: str, response_text: str, *, lease_token: str) -> None:
        message_id = require_message_id(incoming_message_id)
        text = require_text(text, "message")
        response_text = require_text(response_text, "response")
        with self._transaction():
            now = self._now()
            updated = self.connection.execute(
                """UPDATE conversation_messages
                   SET state = 'completed', response_text = ?, updated_at = ?, lease_token = NULL, lease_until = NULL
                   WHERE incoming_message_id = ? AND input_hash = ? AND state = 'processing' AND lease_token = ?
                     AND lease_until IS NOT NULL AND lease_until > ?""",
                (
                    response_text,
                    encode_datetime(now),
                    message_id,
                    fingerprint({"text": text}),
                    lease_token,
                    encode_datetime(now),
                ),
            )
            if updated.rowcount != 1:
                raise ValueError("conversation reservation is unavailable")

    def abandon_conversation(self, incoming_message_id: str, text: str, *, lease_token: str) -> None:
        """Release only this owner's uncompleted lease after a provider failure."""
        message_id = require_message_id(incoming_message_id)
        text = require_text(text, "message")
        with self._transaction():
            self.connection.execute(
                """DELETE FROM conversation_messages
                   WHERE incoming_message_id = ? AND input_hash = ? AND state = 'processing' AND lease_token = ?""",
                (message_id, fingerprint({"text": text}), lease_token),
            )

    def _require_active_conversation_lease_for_mutation(self) -> None:
        for message_id, lease_token in self._active_conversation_leases.items():
            self._require_conversation_lease(message_id, lease_token)

    def _require_conversation_lease(self, message_id: str, lease_token: str) -> None:
        row = self.connection.execute(
            "SELECT state, lease_token, lease_until FROM conversation_messages WHERE incoming_message_id = ?",
            (message_id,),
        ).fetchone()
        if (
            row is None
            or row["state"] != "processing"
            or row["lease_token"] != lease_token
            or row["lease_until"] is None
            or decode_datetime(row["lease_until"]) <= self._now()
        ):
            raise ValueError("conversation reservation is unavailable")

    def _mutation_result(self, message_id: str, operation: str, request: dict[str, str]) -> str | None:
        row = self.connection.execute(
            "SELECT operation, result, request_hash FROM processed_messages WHERE incoming_message_id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            return None
        if row["operation"] != operation:
            raise ValueError("incoming message ID was already used for another operation")
        if row["request_hash"] != fingerprint(request):
            raise ValueError("incoming message ID was already used for different input")
        return row["result"]

    def _record_mutation(self, message_id: str, operation: str, request: dict[str, str], result: str) -> None:
        self.connection.execute(
            "INSERT INTO processed_messages (incoming_message_id, operation, result, processed_at, request_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            (message_id, operation, result, encode_datetime(self._now()), fingerprint(request)),
        )

    def _note_by_id(self, note_id: str) -> Note:
        row = self.connection.execute("SELECT id, text, created_at FROM notes WHERE id = ?", (note_id,)).fetchone()
        if row is None:
            raise RuntimeError("idempotency record points to a missing note")
        return note_from_row(row)

    def _now(self) -> datetime:
        return require_utc_datetime(self.clock.now(), "clock result")

    @contextmanager
    def _transaction(self):
        """Serialize mutations so a reminder cannot be claimed by two dispatchers."""
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()


class ReminderDispatcher:
    def __init__(
        self,
        store: BrainStore,
        transport: Transport,
        *,
        clock: Clock | None = None,
        claim_seconds: int = 60,
        channel: str | None = None,
        binding: str | None = None,
    ):
        self.store = store
        self.transport = transport
        self.clock = clock or SystemClock()
        self.claim_seconds = claim_seconds
        self.channel = channel
        self.binding = binding

    def dispatch_due(self, limit: int = 100) -> int:
        if limit <= 0:
            raise ValueError("limit must be positive")
        attempts = 0
        for _ in range(limit):
            claim = self.store.claim_due_reminder(
                self._now(),
                channel=self.channel,
                binding=self.binding,
                claim_seconds=self.claim_seconds,
            )
            if claim is None:
                break
            reminder, token = claim
            attempts += 1
            try:
                self.transport.send(reminder)
            except Exception:
                self.store.mark_delivery_failed(reminder.id, token, self._now())
            else:
                self.store.mark_sent(reminder.id, token)
        return attempts

    def _now(self) -> datetime:
        return require_utc_datetime(self.clock.now(), "clock result")


def fingerprint(value: dict[str, object]) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def require_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must not be empty")
    if len(value) > MAX_TEXT_LENGTH:
        raise ValueError(f"{field} must be at most {MAX_TEXT_LENGTH} characters")
    return value.strip()


def require_query(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("query must not be empty")
    if len(value) > MAX_QUERY_LENGTH:
        raise ValueError(f"query must be at most {MAX_QUERY_LENGTH} characters")
    return value.strip()


def require_message_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError("incoming message ID must contain 1 to 200 characters")
    return value


def require_utc_datetime(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be an aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        return value.astimezone(timezone.utc)
    return value


def encode_datetime(value: datetime) -> str:
    return require_utc_datetime(value, "timestamp").isoformat().replace("+00:00", "Z")


def decode_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def note_from_row(row: sqlite3.Row) -> Note:
    return Note(row["id"], row["text"], decode_datetime(row["created_at"]))


def inbound_from_row(row: sqlite3.Row) -> InboundMessage:
    return InboundMessage(
        row["id"],
        row["text"],
        row["status"],
        row["attempts"],
        decode_datetime(row["next_attempt_at"]),
        channel=row["channel"] if "channel" in row.keys() else "evolution",
        binding=row["binding"] if "binding" in row.keys() else None,
    )


def outbox_from_row(row: sqlite3.Row) -> OutboxMessage:
    return OutboxMessage(
        row["id"],
        row["text"],
        row["status"],
        row["attempts"],
        decode_datetime(row["next_attempt_at"]),
        channel=row["channel"] if "channel" in row.keys() else "evolution",
        binding=row["binding"] if "binding" in row.keys() else None,
    )


def reminder_from_row(row: sqlite3.Row) -> Reminder:
    return Reminder(
        row["id"],
        row["text"],
        decode_datetime(row["due_at"]),
        row["status"],
        row["attempts"],
        decode_datetime(row["next_attempt_at"]),
        channel=row["channel"] if "channel" in row.keys() else None,
        binding=row["binding"] if "binding" in row.keys() else None,
    )


def backup_database(source: Path | str, destination: Path | str) -> None:
    """Create a consistent SQLite backup without copying WAL files manually."""
    with sqlite3.connect(source) as source_connection, sqlite3.connect(destination) as destination_connection:
        source_connection.backup(destination_connection)


def restore_database(backup: Path | str, destination: Path | str) -> None:
    """Restore an SQLite backup through SQLite's backup API."""
    backup_database(backup, destination)
