# Loci

A standard-library-first, local-first Python package for durable notes, reminders, bounded model-directed chat, an offline-tested Telegram poller, an offline-tested Evolution WhatsApp adapter, and an optional Google Calendar slice. SQLite remains the single source of truth. Audio, briefing, passwords, and multi-user access are not implemented.

## Quick start

Python 3.11+ on Linux / POSIX is required. The package has no external runtime dependencies.

```bash
git clone https://github.com/Reaan06/Loci.git
cd Loci
PYTHONPATH=src python3 -m second_brain --help
```

Use `--db` on every command to select a database file (defaults to `second-brain.sqlite3` in the current working directory).

```bash
# Notes: mutations require a stable upstream message ID for replay safety
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 note save "Call the dentist" --message-id msg-001
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 note search dentist
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 note delete NOTE_ID --message-id msg-002

# Reminders with UTC offsets are normalized to UTC
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 reminder create "Take medication" --at 2026-06-01T09:00:00+02:00 --message-id msg-003

# Offset-free local timestamps require a valid IANA timezone; ambiguous DST transitions are rejected
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 reminder create "Pay rent" --at 2026-06-01T09:00:00 --timezone Europe/Berlin --message-id msg-004
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 reminder list
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 reminder cancel REMINDER_ID --message-id msg-005

# Local demonstration transport (prints due reminders to standard output)
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 reminder dispatch
```

## Configuration (.env)

The package includes a built-in literal environment file parser.

### File loading rules

- **Default location**: Looks strictly for `.env` in the current working directory (`Path.cwd() / ".env"`). The loader does not traverse parent directories.
- **Custom path**: Pass `--env-file <path>` as a global option before the subcommand:
  ```bash
  PYTHONPATH=src python3 -m second_brain --env-file /path/to/custom.env telegram
  ```
- **Precedence**: Operating system and service environment variables always take precedence, including empty strings (`KEY=""`). Values defined in the file only populate unset keys.
- **Missing file handling**: A missing default `./.env` is ignored. If `--env-file` is explicitly passed and the file does not exist, an `EnvironmentFileNotFoundError` is raised.
- **File constraints**: Only trusted regular files up to 64 KiB (65,536 bytes) with valid UTF-8 encoding are accepted. Symlinks (enforced via `O_NOFOLLOW`), FIFOs, sockets, character devices, and directories are rejected.
- **Platform limitation**: Linux / POSIX systems only. The loader requires POSIX file open flags (`O_NOFOLLOW`, `O_CLOEXEC`) and regular file descriptors. Windows is not supported.

### Grammar and syntax

- **Format**: Lines use `KEY=VALUE`. Keys must match `^[A-Za-z_][A-Za-z0-9_]*$`. Duplicate keys trigger a parse error.
- **Comments**: Full lines starting with `#` are comments. Blank lines are ignored.
- **No inline comments**: A `#` character inside a value is treated as part of the literal value.
- **Values**: Values may be unquoted or wrapped in matching single (`'...'`) or double (`"..."`) quotes.
- **Quotes**: Quoted values are read literally between the quote delimiters. Escape sequences (such as `\n` or `\t`) are not decoded, and characters after the closing quote are rejected. Unquoted values cannot contain single or double quotes.
- **Literal characters**: Dollar signs (`$`) are treated as literal characters. Shell expansions (`$VAR`, `${VAR}`) are not performed.
- **No shell execution**: The parser does not run code (`eval`) or parse `export` prefixes. Do not `source .env` in your shell.
- **Secret hygiene**: Set restrictive permissions (`chmod 600 .env`). An environment template is not shipped in the repository; create and populate your local `.env` manually. Never store credentials in shell command arguments or git commits.

### Runtime environment variables

| Variable | Required by | Description |
|---|---|---|
| `SECOND_BRAIN_PROVIDER_URL` | Chat, Telegram, Evolution | HTTPS OpenAI-compatible chat-completions endpoint |
| `SECOND_BRAIN_PROVIDER_KEY` | Chat, Telegram, Evolution | Bearer API authentication key |
| `SECOND_BRAIN_PROVIDER_MODEL` | Chat, Telegram, Evolution | Explicit model identifier |
| `SECOND_BRAIN_TELEGRAM_BOT_TOKEN` | Telegram | Bot authentication token issued by BotFather |
| `SECOND_BRAIN_TELEGRAM_OWNER_USER_ID` | Telegram | Positive integer Telegram user ID of the owner |
| `SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID` | Telegram | Positive integer Telegram chat ID of the owner |
| `SECOND_BRAIN_TELEGRAM_BOT_NAMESPACE` | Telegram (optional) | Cursor partition key; defaults to token numeric ID or `default` |
| `SECOND_BRAIN_EVOLUTION_URL` | Evolution | HTTPS base URL of the Evolution instance |
| `SECOND_BRAIN_EVOLUTION_API_KEY` | Evolution | API key for the Evolution instance |
| `SECOND_BRAIN_EVOLUTION_INSTANCE` | Evolution | Configured instance name |
| `SECOND_BRAIN_EVOLUTION_OWNER_JID` | Evolution | Owner WhatsApp JID (`<phone>@s.whatsapp.net`) |
| `SECOND_BRAIN_WEBHOOK_SECRET` | Evolution | Value verified against incoming `X-Webhook-Secret` |
| `SECOND_BRAIN_EVOLUTION_TEXT_DIALECT` | Evolution (optional) | Payload dialect: `textMessage` (default) or `text` |
| `SECOND_BRAIN_EVOLUTION_ALLOW_INSECURE_LOCAL_HTTP` | Evolution (optional) | Set to `true` to allow `http://` for local testing |
| `SECOND_BRAIN_GOOGLE_CLIENT_ID` | Calendar (optional) | Google OAuth client ID (all 4 Calendar vars required if enabled) |
| `SECOND_BRAIN_GOOGLE_CLIENT_SECRET` | Calendar (optional) | Google OAuth client secret |
| `SECOND_BRAIN_GOOGLE_REFRESH_TOKEN` | Calendar (optional) | Google OAuth refresh token |
| `SECOND_BRAIN_GOOGLE_TIMEZONE` | Calendar (optional) | Configured IANA timezone (e.g. `Europe/Berlin`) |

### Example .env configuration

Store configuration in a local `.env` file instead of shell command arguments to avoid leaking secrets into shell history:

```ini
# Model provider
SECOND_BRAIN_PROVIDER_URL="https://provider.example/v1/chat/completions"
SECOND_BRAIN_PROVIDER_KEY="provider-api-key-placeholder"
SECOND_BRAIN_PROVIDER_MODEL="model-name-placeholder"

# Telegram runner
SECOND_BRAIN_TELEGRAM_BOT_TOKEN="123456789:EXAMPLE_BOT_TOKEN_PLACEHOLDER"
SECOND_BRAIN_TELEGRAM_OWNER_USER_ID="123456789"
SECOND_BRAIN_TELEGRAM_OWNER_CHAT_ID="123456789"
SECOND_BRAIN_TELEGRAM_BOT_NAMESPACE="primary-bot"

# WhatsApp Evolution adapter (optional)
# SECOND_BRAIN_EVOLUTION_URL="https://evolution.example"
# SECOND_BRAIN_EVOLUTION_API_KEY="evolution-key-placeholder"
# SECOND_BRAIN_EVOLUTION_INSTANCE="instance-name"
# SECOND_BRAIN_EVOLUTION_OWNER_JID="15551234567@s.whatsapp.net"
# SECOND_BRAIN_WEBHOOK_SECRET="webhook-secret-placeholder"

# Google Calendar (optional: all four variables required to activate)
# SECOND_BRAIN_GOOGLE_CLIENT_ID="google-client-id-placeholder.apps.googleusercontent.com"
# SECOND_BRAIN_GOOGLE_CLIENT_SECRET="google-client-secret-placeholder"
# SECOND_BRAIN_GOOGLE_REFRESH_TOKEN="google-refresh-token-placeholder"
# SECOND_BRAIN_GOOGLE_TIMEZONE="Europe/Berlin"
```

## Chat provider and tool safety

The `chat` command handles one message. Supplying a stable `--message-id` ensures replay safety:

```bash
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 chat "Save a note to call the dentist" --message-id chat-001
```

Provider URLs require HTTPS. Plain HTTP is rejected unless `--allow-insecure-localhost-provider` is explicitly passed for a trusted loopback proxy. Endpoints and credentials are never guessed. The adapter sends standard chat-completions JSON using the Python standard library with bounded request and response limits. It is tested using offline deterministic transports; compatibility with live providers requires independent verification.

The model can invoke only allowlisted tools:
- Notes: save, search, delete.
- Reminders: create, list, cancel.
- Google Calendar (only when configured): create, list.

Action confirmations derive strictly from deterministic tool execution, never generated model prose. When no tool calls are returned, the assistant responds: `No action was performed. Please clarify what you need.` Stored notes are treated as untrusted data and never interpreted as system instructions.

A 60-second SQLite lease fences active conversations. If a worker crashes, the expired lease can be acquired by a new worker using an opaque lease token. Earlier workers fenced by takeover cannot mutate state. Redirects are rejected by the transport to prevent credential leakage.

## Google Calendar offline slice

Google Calendar remains disabled unless all four configuration variables (`SECOND_BRAIN_GOOGLE_CLIENT_ID`, `SECOND_BRAIN_GOOGLE_CLIENT_SECRET`, `SECOND_BRAIN_GOOGLE_REFRESH_TOKEN`, and `SECOND_BRAIN_GOOGLE_TIMEZONE`) are set.

The adapter uses Google's fixed HTTPS token refresh and Calendar v3 endpoints. It does not read local credential files, use service accounts, open interactive browsers, or accept arbitrary URLs. Tokens are held exclusively in memory.

Operations are restricted to the configured `primary` calendar:
- Creates require `summary`, `start`, and `end`, bounded to a 14-day window.
- Reads are limited to a 31-day window and at most 20 results.
- Timestamps require explicit UTC offsets or unambiguous local times under the configured IANA timezone.
- Attendees, recurring events, deletes, updates, and HTML rendering are not supported.

A configuration binding digest fences calendar creations. Rotating OAuth credentials blocks pending creations until operator review, preventing state mismatches across account changes. Offline tests do not communicate with live Google endpoints.

## Evolution WhatsApp adapter

This adapter connects to an existing, already-linked Evolution instance. It does not provision Evolution containers, generate QR codes, or activate phone numbers.

```bash
# Process one queued inbox/outbox item
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 evolution --once

# Run the authenticated loopback webhook and worker poller
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 evolution --bind 127.0.0.1 --port 8081
```

Configure incoming webhooks to post to `http://127.0.0.1:8081/evolution/webhook`. The endpoint binds only to loopback and verifies `X-Webhook-Secret` in constant time.

Only direct 1:1 text messages from `SECOND_BRAIN_EVOLUTION_OWNER_JID` are processed (`fromMe` must be `false`). Group messages, broadcasts, status updates, and media are ignored. Inbound and outbox records are partitioned in SQLite by channel and binding hash (`channel='evolution'`). Deliveries follow at-least-once semantics with exponential retry backoff.

## Telegram adapter

Private Telegram long polling offers an outbound HTTPS polling channel to `https://api.telegram.org` without exposing open inbound ports or requiring reverse proxies.

```bash
# Process at most one poll batch, inbox, outbox, and reminder pass, then exit
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 telegram --once

# Run continuous long poller, worker, and automatic reminder dispatcher
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 telegram
```

Key behaviors:
- **Owner-only**: Accepts only 1:1 private messages from `owner_user_id` inside `owner_chat_id`. Groups, channels, media, edits, and bot commands are ignored.
- **Durable cursor**: Updates are tracked using fenced SQLite leases (`telegram_poll_cursors`). Offsets and inbound queue items commit in a single transaction. Telegram retains updates up to 24 hours; unpolled updates older than that retention window expire upstream.
- **Namespace and token rotation**: By default, `bot_namespace` uses the numeric prefix of the bot token. To preserve cursor position and recipient bindings (`telegram:<hash>`) across bot token rotations, configure `SECOND_BRAIN_TELEGRAM_BOT_NAMESPACE` to a stable string. Conversely, changing the namespace intentionally creates a new partition, resetting cursors and allowing tool interactions to run within a fresh partition.
- **Binding isolation**: Messages are isolated under `channel='telegram'` and recipient hash `telegram:<hash>`.
- **Reminder routing**: The Telegram poller includes an integrated dispatcher routing due reminders to the owner chat. Unbound reminders are claimed by Telegram upon dispatch, while local console dispatch handles only unbound reminders.
- **Supervisors and shutdown**: Service supervisors (such as systemd) are not bundled or installed. Operators should configure process management and set termination grace periods (`TimeoutStopSec`) to allow in-flight turns and network requests to complete during graceful shutdown. An HTTP 409 Conflict stops with a non-zero exit code to alert operators to conflicting webhooks or parallel pollers.

## Delivery behavior

The dispatcher claims due reminders in SQLite using database transactions. Failed transports return items to `pending` with exponential backoff up to one hour. Expired claims are recovered on subsequent runs.

Delivery is at-least-once: if a process sends an external notification and terminates before updating SQLite, recovery may resend the item. Mutation commands are guarded by unique upstream message IDs to prevent duplicate actions.

## Backup and restore

Do not copy active SQLite database and WAL files directly. Use SQLite's online backup API:

```bash
PYTHONPATH=src python3 -m second_brain --db brain.sqlite3 backup brain-backup.sqlite3

# Stop running processes before restoring
PYTHONPATH=src python3 -m second_brain --db restored.sqlite3 restore brain-backup.sqlite3
```

## Verification

Run the test suite and bytecode compilation checks:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
PYTHONPATH=src python3 -m second_brain --help
```

## Repository hygiene and licensing

- **Public allowlist**: Only audited source files are tracked in version control: `README.md`, `.gitignore`, `pyproject.toml`, `src/**/*.py`, and `tests/**/*.py`. Never run `git add .` or `git add -A`.
- **License**: No open source license has been selected or granted yet; this repository is currently unlicensed for public redistribution.
- **Safety boundaries**: Application constraints (such as tool allowlists, argument limits, and lease tokens) provide defensive runtime bounds rather than absolute sandbox guarantees. External service adapters have been tested against offline mocks; live provider compatibility must be validated by the operator.
