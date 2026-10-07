"""Command-line interface for the local personal second brain."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import sys
import threading
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .core import BrainStore, ReminderDispatcher, backup_database, restore_database
from .environment import load_environment_file


class ConsoleTransport:
    def send(self, reminder) -> None:
        print(f"REMINDER {reminder.id}: {reminder.text}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        load_environment_file(arguments.env_file)
        return run(arguments)
    except (ValueError, KeyError, ZoneInfoNotFoundError, OSError) as error:
        parser.error(str(error))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local SQLite personal second brain")
    parser.add_argument("--db", type=Path, default=Path("second-brain.sqlite3"), help="SQLite database path")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to environment file (default: .env in current working directory)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    note = commands.add_parser("note", help="Manage notes")
    note_commands = note.add_subparsers(dest="note_command", required=True)
    add = note_commands.add_parser("save", help="Save a note")
    add.add_argument("text")
    add.add_argument("--message-id", required=True)
    search = note_commands.add_parser("search", help="Search notes")
    search.add_argument("query")
    delete = note_commands.add_parser("delete", help="Delete a note")
    delete.add_argument("note_id")
    delete.add_argument("--message-id", required=True)

    reminder = commands.add_parser("reminder", help="Manage reminders")
    reminder_commands = reminder.add_subparsers(dest="reminder_command", required=True)
    create = reminder_commands.add_parser("create", help="Create a reminder")
    create.add_argument("text")
    create.add_argument("--at", required=True, help="ISO-8601 datetime with offset, or local datetime with --timezone")
    create.add_argument("--timezone", help="Explicit IANA timezone for an offset-free local datetime")
    create.add_argument("--message-id", required=True)
    listing = reminder_commands.add_parser("list", help="List pending and claimed reminders")
    listing.add_argument("--all", action="store_true", help="Include sent and cancelled reminders")
    cancel = reminder_commands.add_parser("cancel", help="Cancel a pending reminder")
    cancel.add_argument("reminder_id")
    cancel.add_argument("--message-id", required=True)
    dispatch = reminder_commands.add_parser("dispatch", help="Deliver due reminders to the local console")
    dispatch.add_argument("--limit", type=int, default=100)

    backup = commands.add_parser("backup", help="Create a consistent SQLite backup")
    backup.add_argument("destination", type=Path)
    restore = commands.add_parser("restore", help="Restore a SQLite backup into --db")
    restore.add_argument("backup", type=Path)

    evolution = commands.add_parser("evolution", help="Run the existing-instance Evolution webhook and durable worker")
    evolution.add_argument("--bind", default="127.0.0.1", help="Loopback bind address only")
    evolution.add_argument("--port", type=int, default=8081)
    evolution.add_argument("--once", action="store_true", help="Process at most one inbox and one outbox item, then exit")
    evolution.add_argument("--allow-insecure-localhost-provider", action="store_true")

    telegram = commands.add_parser("telegram", help="Run the owner-only Telegram long poller, worker, and reminder dispatcher")
    telegram.add_argument("--once", action="store_true", help="Process at most one poll batch, inbox, outbox, and reminder pass, then exit")
    telegram.add_argument("--allow-insecure-localhost-provider", action="store_true")

    chat = commands.add_parser("chat", help="Handle one message using an explicitly configured provider")
    chat.add_argument("text", help="Single incoming message")
    chat.add_argument("--message-id", required=True, help="Stable upstream message ID")
    chat.add_argument(
        "--allow-insecure-localhost-provider", action="store_true",
        help="Allow HTTP only for a deliberately trusted localhost provider proxy",
    )
    return parser


def run(arguments: argparse.Namespace) -> int:
    if arguments.command == "backup":
        backup_database(arguments.db, arguments.destination)
        print(f"Backup created at {arguments.destination}")
        return 0
    if arguments.command == "restore":
        restore_database(arguments.backup, arguments.db)
        print(f"Database restored to {arguments.db}")
        return 0

    if arguments.command == "evolution":
        return run_evolution(arguments)
    if arguments.command == "telegram":
        return run_telegram(arguments)

    store = BrainStore(arguments.db)
    try:
        if arguments.command == "note":
            return run_note(store, arguments)
        if arguments.command == "reminder":
            return run_reminder(store, arguments)
        return run_chat(store, arguments)
    finally:
        store.close()


def run_evolution(arguments: argparse.Namespace) -> int:
    from .agent import Agent
    from .calendar import CalendarClient, CalendarConfig
    from .evolution import EvolutionConfig, EvolutionTransport
    from .model import OpenAICompatibleModel
    from .server import EvolutionWorker, create_webhook_server

    config = EvolutionConfig.from_environment(os.environ)
    model = OpenAICompatibleModel.from_environment(
        os.environ, allow_insecure_localhost=arguments.allow_insecure_localhost_provider
    )
    calendar_config = CalendarConfig.from_environment(os.environ)
    worker = EvolutionWorker(
        arguments.db,
        lambda store: Agent(store, model, calendar=CalendarClient(calendar_config) if calendar_config else None),
        EvolutionTransport(config),
    )
    if arguments.once:
        worker.process_once()
        return 0
    server = create_webhook_server(arguments.bind, arguments.port, arguments.db, config)
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            worker.process_once()
            stop.wait(1)

    polling_thread = threading.Thread(target=poll, name="evolution-worker", daemon=True)
    polling_thread.start()
    print(f"Evolution webhook listening on http://{arguments.bind}:{arguments.port}/evolution/webhook")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        stop.set()
        polling_thread.join(timeout=5)
        server.server_close()


def run_telegram(
    arguments: argparse.Namespace,
    *,
    stop: threading.Event | None = None,
    join_warn_seconds: float = 5.0,
) -> int:
    from .agent import Agent
    from .calendar import CalendarClient, CalendarConfig
    from .model import OpenAICompatibleModel
    from .server import TelegramWorkerProcess
    from .telegram import TelegramClient, TelegramConfig, TelegramConflictError, TelegramPoller, TelegramRequestError

    config = TelegramConfig.from_environment(os.environ)
    client = TelegramClient(config)
    model = OpenAICompatibleModel.from_environment(
        os.environ, allow_insecure_localhost=arguments.allow_insecure_localhost_provider
    )
    calendar_config = CalendarConfig.from_environment(os.environ)
    worker = TelegramWorkerProcess(
        arguments.db,
        lambda store_arg: Agent(store_arg, model, calendar=CalendarClient(calendar_config) if calendar_config else None),
        client,
        config,
    )
    store = BrainStore(arguments.db)
    worker_thread: threading.Thread | None = None
    try:
        poller = TelegramPoller(store, client, config)

        if arguments.once:
            if stop is None:
                stop = threading.Event()

            orig_sigint = None
            if threading.current_thread() is threading.main_thread():
                def _once_sigint_handler(signum, frame):
                    stop.set()
                try:
                    orig_sigint = signal.signal(signal.SIGINT, _once_sigint_handler)
                except (ValueError, AttributeError):
                    orig_sigint = None

            try:
                try:
                    poller.poll_once()
                except TelegramConflictError as e:
                    sys.stderr.write(
                        "telegram_conflict_error: HTTP 409 Conflict: webhook or another poller is active for this bot token. "
                        "Cannot poll while another instance or webhook is active.\n"
                    )
                    return 1
                except TelegramRequestError as e:
                    sys.stderr.write(f"telegram_request_error: {type(e).__name__}\n")
                    return 1
                except Exception as e:
                    sys.stderr.write(f"telegram_poller_error: {type(e).__name__}\n")
                    return 1

                try:
                    processed, cap_reached = worker.drain_due(max_iterations=100, stop=stop)
                    if cap_reached:
                        sys.stderr.write(
                            "telegram_drain_warning: Reached bounded iteration cap (100). Remaining work left intact in queue.\n"
                        )
                except KeyboardInterrupt:
                    stop.set()
                    sys.stderr.write("telegram_drain_warning: Interrupted by signal during drain. Work left intact in queue.\n")
                return 0
            finally:
                if orig_sigint is not None:
                    try:
                        signal.signal(signal.SIGINT, orig_sigint)
                    except (ValueError, AttributeError):
                        pass

        if stop is None:
            stop = threading.Event()

        def worker_loop() -> None:
            while not stop.is_set():
                try:
                    worker.process_once(stop=stop)
                except Exception as e:
                    sys.stderr.write(f"telegram_worker_error: worker_loop_exception: {type(e).__name__}\n")
                stop.wait(1.0)

        # Worker thread is non-daemon so inflight operations finish safely without abrupt truncation
        worker_thread = threading.Thread(target=worker_loop, name="telegram-worker", daemon=False)
        worker_thread.start()
        print("Telegram long poller and worker running (Press Ctrl+C to stop)")

        backoff = 1.0
        max_backoff = 60.0

        try:
            while not stop.is_set():
                try:
                    poller.poll_once()
                    backoff = 1.0
                except TelegramConflictError as e:
                    sys.stderr.write(
                        "telegram_conflict_error: HTTP 409 Conflict: webhook or another poller is active for this bot token. "
                        "Exiting.\n"
                    )
                    return 1
                except TelegramRequestError as e:
                    sys.stderr.write(f"telegram_request_error: {type(e).__name__}\n")
                    stop.wait(backoff)
                    backoff = min(backoff * 2.0, max_backoff)
                    continue
                except Exception as e:
                    sys.stderr.write(f"telegram_poller_error: {type(e).__name__}\n")
                    stop.wait(backoff)
                    backoff = min(backoff * 2.0, max_backoff)
                    continue

                stop.wait(1.0)
            return 0
        except KeyboardInterrupt:
            return 0
    finally:
        if stop is not None:
            stop.set()
        if worker_thread is not None and worker_thread.is_alive():
            # Wait up to join_warn_seconds for fast completion
            worker_thread.join(timeout=join_warn_seconds)
            if worker_thread.is_alive():
                sys.stderr.write(
                    "telegram_shutdown_warning: Worker still active; waiting for in-flight work to complete safely...\n"
                )
                # Keep waiting without timeout so in-flight operation finishes cleanly
                # Even repeated KeyboardInterrupt during shutdown wait will not kill or corrupt
                while worker_thread.is_alive():
                    try:
                        worker_thread.join(timeout=1.0)
                    except KeyboardInterrupt:
                        sys.stderr.write(
                            "telegram_shutdown_warning: Shutdown in progress; waiting for in-flight work to complete...\n"
                        )
        store.close()


def run_chat(store: BrainStore, arguments: argparse.Namespace) -> int:
    from .agent import Agent
    from .calendar import CalendarClient, CalendarConfig
    from .model import OpenAICompatibleModel

    model = OpenAICompatibleModel.from_environment(
        os.environ, allow_insecure_localhost=arguments.allow_insecure_localhost_provider
    )
    calendar_config = CalendarConfig.from_environment(os.environ)
    reply = Agent(store, model, calendar=CalendarClient(calendar_config) if calendar_config else None).handle(
        arguments.text, message_id=arguments.message_id
    )
    print(reply.text)
    return 0


def run_note(store: BrainStore, arguments: argparse.Namespace) -> int:
    if arguments.note_command == "save":
        note = store.save_note(arguments.text, incoming_message_id=arguments.message_id)
        print_note(note)
    elif arguments.note_command == "search":
        for note in store.search_notes(arguments.query):
            print_note(note)
    else:
        deleted = store.delete_note(arguments.note_id, incoming_message_id=arguments.message_id)
        print("Deleted" if deleted else "Note not found")
    return 0


def run_reminder(store: BrainStore, arguments: argparse.Namespace) -> int:
    if arguments.reminder_command == "create":
        reminder = store.create_reminder(
            arguments.text,
            parse_schedule(arguments.at, arguments.timezone),
            incoming_message_id=arguments.message_id,
        )
        print_reminder(reminder)
    elif arguments.reminder_command == "list":
        for reminder in store.list_reminders(include_finished=arguments.all):
            print_reminder(reminder)
    elif arguments.reminder_command == "cancel":
        cancelled = store.cancel_reminder(arguments.reminder_id, incoming_message_id=arguments.message_id)
        print("Cancelled" if cancelled else "Reminder not found or already finished")
    else:
        attempts = ReminderDispatcher(store, ConsoleTransport()).dispatch_due(arguments.limit)
        print(f"Attempted delivery for {attempts} reminder(s)")
    return 0


def parse_schedule(value: str, timezone_name: str | None) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("--at must be an ISO-8601 datetime") from error
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        if timezone_name:
            raise ValueError("--timezone is only valid with an offset-free local datetime")
        return parsed.astimezone(timezone.utc)
    if not timezone_name:
        raise ValueError("offset-free --at requires an explicit --timezone IANA name")
    zone = ZoneInfo(timezone_name)
    first = parsed.replace(tzinfo=zone, fold=0)
    second = parsed.replace(tzinfo=zone, fold=1)
    first_round_trip = first.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
    second_round_trip = second.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
    if first_round_trip != parsed and second_round_trip != parsed:
        raise ValueError("local datetime is invalid in the supplied timezone; use an explicit UTC offset")
    if first.utcoffset() != second.utcoffset():
        raise ValueError("local datetime is ambiguous; use an explicit UTC offset")
    return first.astimezone(timezone.utc)


def print_note(note) -> None:
    print(f"{note.id}\t{note.created_at.isoformat()}\t{note.text}")


def print_reminder(reminder) -> None:
    print(
        f"{reminder.id}\t{reminder.status}\t{reminder.due_at.isoformat()}\t"
        f"attempts={reminder.attempts}\t{reminder.text}"
    )


if __name__ == "__main__":
    sys.exit(main())
