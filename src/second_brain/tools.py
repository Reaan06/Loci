"""Strict allow-listed tools for deterministic note and reminder operations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

from .calendar import CalendarClient
from .core import BrainStore, MAX_QUERY_LENGTH, MAX_TEXT_LENGTH, require_message_id

MAX_TOOL_ARGUMENT_TEXT = MAX_TEXT_LENGTH


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    message: str
    mutated: bool = False


def tool_definitions(calendar: CalendarClient | None = None) -> list[dict[str, object]]:
    definitions = [
        _definition("save_note", {"text": {"type": "string", "maxLength": MAX_TOOL_ARGUMENT_TEXT}}, ["text"]),
        _definition("search_notes", {"query": {"type": "string", "maxLength": MAX_QUERY_LENGTH}}, ["query"]),
        _definition("delete_note", {"note_id": {"type": "string", "maxLength": 200}}, ["note_id"]),
        _definition("create_reminder", {
            "text": {"type": "string", "maxLength": MAX_TOOL_ARGUMENT_TEXT},
            "due_at": {"type": "string", "description": "ISO-8601 datetime with explicit offset"},
            "timezone": {"type": "string", "description": "Required IANA timezone only for explicit offset-free local time"},
        }, ["text", "due_at"]),
        _definition("list_reminders", {"include_finished": {"type": "boolean"}}, []),
        _definition("cancel_reminder", {"reminder_id": {"type": "string", "maxLength": 200}}, ["reminder_id"]),
    ]
    if calendar is not None:
        definitions.extend([
            _definition("calendar_create", {
                "summary": {"type": "string", "maxLength": 500},
                "description": {"type": "string", "maxLength": 4000},
                "start": {"type": "string", "description": "ISO-8601 datetime with offset, or unambiguous local time in configured timezone"},
                "end": {"type": "string", "description": "ISO-8601 datetime with offset, or unambiguous local time in configured timezone"},
            }, ["summary", "start", "end"]),
            _definition("calendar_list", {
                "time_min": {"type": "string", "description": "ISO-8601 datetime with offset, or unambiguous local time in configured timezone"},
                "time_max": {"type": "string", "description": "ISO-8601 datetime with offset, or unambiguous local time in configured timezone"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
            }, ["time_min", "time_max"]),
        ])
    return definitions


def dispatch_tool(
    store: BrainStore, name: str, arguments: object, *, message_id: str, calendar: CalendarClient | None = None
) -> ToolResult:
    try:
        require_message_id(message_id)
        if name == "calendar_create":
            if calendar is None:
                raise ValueError("Google Calendar is not configured")
            fields = calendar.normalize_create(arguments, calendar.config)
            frozen_plan = {"event_intent": fields, "config_binding": calendar.create_binding(fields)}
            store.reserve_tool_action(message_id, name, frozen_plan)
            event = calendar.create_event(fields, message_id)
            return ToolResult(True, f"Created calendar event: {event.summary} from {event.start} to {event.end}", True)
        if name == "calendar_list":
            if calendar is None:
                raise ValueError("Google Calendar is not configured")
            fields = calendar.normalize_list(arguments, calendar.config)
            listing = calendar.list_events(fields)
            if not listing.events:
                return ToolResult(True, "No calendar events in the requested window" + (" (results truncated)" if listing.truncated else ""))
            rendered = "; ".join(
                f"{event.summary} ({'all-day ' if event.all_day else ''}{event.start} to {event.end})"
                for event in listing.events
            )
            return ToolResult(True, "Calendar events: " + rendered + ("; results truncated" if listing.truncated else ""))
        fields = _validate_arguments(name, arguments)
        if name in {"save_note", "delete_note", "create_reminder", "cancel_reminder"}:
            store.reserve_tool_action(message_id, name, fields)
        return _execute(store, name, fields, message_id)
    except (ValueError, KeyError) as error:
        if name == "create_reminder" and ("explicit --timezone" in str(error) or "ambiguous" in str(error)):
            return ToolResult(False, f"Clarification required: {error}")
        return ToolResult(False, f"Error: {error}")


def _definition(name: str, properties: dict[str, object], required: list[str]) -> dict[str, object]:
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False,
    }}}


def _validate_arguments(name: str, arguments: object) -> dict[str, object]:
    schemas: dict[str, tuple[set[str], set[str], dict[str, Callable[[object], object]]]] = {
        "save_note": ({"text"}, {"text"}, {"text": _text}),
        "search_notes": ({"query"}, {"query"}, {"query": _query}),
        "delete_note": ({"note_id"}, {"note_id"}, {"note_id": _identifier}),
        "create_reminder": ({"text", "due_at", "timezone"}, {"text", "due_at"}, {"text": _text, "due_at": _datetime, "timezone": _timezone}),
        "list_reminders": ({"include_finished"}, set(), {"include_finished": _boolean}),
        "cancel_reminder": ({"reminder_id"}, {"reminder_id"}, {"reminder_id": _identifier}),
    }
    if name not in schemas:
        raise ValueError(f"tool '{name}' is unavailable")
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    allowed, required, validators = schemas[name]
    keys = set(arguments)
    if extra := keys - allowed:
        raise ValueError(f"unexpected tool argument: {sorted(extra)[0]}")
    if missing := required - keys:
        raise ValueError(f"missing tool argument: {sorted(missing)[0]}")
    return {key: validators[key](value) for key, value in arguments.items()}


def _execute(store: BrainStore, name: str, fields: dict[str, object], message_id: str) -> ToolResult:
    if name == "save_note":
        note = store.save_note(fields["text"], incoming_message_id=_operation_id(message_id, name))
        return ToolResult(True, f"Saved note: {note.text}", True)
    if name == "search_notes":
        notes = store.search_notes(fields["query"])
        return ToolResult(True, "No matching notes" if not notes else "Notes: " + "; ".join(note.text for note in notes))
    if name == "delete_note":
        deleted = store.delete_note(fields["note_id"], incoming_message_id=_operation_id(message_id, name))
        return ToolResult(True, "Deleted note" if deleted else "Note not found", deleted)
    if name == "create_reminder":
        # Imported lazily to keep the legacy CLI parser independent of the agent.
        from .__main__ import parse_schedule
        due_at = parse_schedule(fields["due_at"], fields.get("timezone"))
        reminder = store.create_reminder(fields["text"], due_at, incoming_message_id=_operation_id(message_id, name))
        return ToolResult(True, f"Created reminder: {reminder.text} at {reminder.due_at.isoformat()}", True)
    if name == "list_reminders":
        reminders = store.list_reminders(bool(fields.get("include_finished", False)))
        return ToolResult(True, "No reminders" if not reminders else "Reminders: " + "; ".join(reminder.text for reminder in reminders))
    cancelled = store.cancel_reminder(fields["reminder_id"], incoming_message_id=_operation_id(message_id, name))
    return ToolResult(True, "Cancelled reminder" if cancelled else "Reminder not found or already finished", cancelled)


def _operation_id(message_id: str, name: str) -> str:
    # Model call IDs are untrusted and unstable across retries; server-derived IDs are bounded.
    digest = hashlib.sha256(f"{message_id}:{name}".encode("utf-8")).hexdigest()
    return f"agent:{digest}"


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TOOL_ARGUMENT_TEXT:
        raise ValueError("text must be a non-empty string within the length limit")
    return value.strip()


def _query(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_QUERY_LENGTH:
        raise ValueError("query must be a non-empty string within the length limit")
    return value.strip()


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError("identifier must be a non-empty string within the length limit")
    return value


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("include_finished must be a boolean")
    return value


def _datetime(value: object) -> str:
    if not isinstance(value, str) or len(value) > 100:
        raise ValueError("due_at must be a short ISO-8601 string")
    return value


def _timezone(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 100:
        raise ValueError("timezone must be a short IANA name")
    return value
