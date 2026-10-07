"""Bounded Google Calendar read/create adapter with refresh-token OAuth.

Only the fixed Google OAuth and Calendar HTTPS endpoints are used.  The
transport is injectable so this module is exercised without live credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import time
from typing import Callable
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_CALENDAR_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
MAX_CALENDAR_REQUEST_BYTES = 20_000
MAX_CALENDAR_RESPONSE_BYTES = 100_000
MAX_CALENDAR_TIMEOUT_SECONDS = 60
MAX_SUMMARY_LENGTH = 500
MAX_DESCRIPTION_LENGTH = 4_000
MAX_EVENT_DURATION = timedelta(days=14)
MAX_LIST_WINDOW = timedelta(days=31)
MAX_LIST_RESULTS = 20
FINGERPRINT_KEY = "second_brain_fingerprint"


class CalendarConfigurationError(ValueError):
    pass


class CalendarRetryableError(RuntimeError):
    """A provider ambiguity/failure that must leave the durable conversation retryable."""


@dataclass(frozen=True)
class CalendarConfig:
    client_id: str = field(repr=False)
    client_secret: str = field(repr=False)
    refresh_token: str = field(repr=False)
    timezone_name: str
    timeout_seconds: float = 20

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (self.client_id, self.client_secret, self.refresh_token)):
            raise CalendarConfigurationError("Google Calendar OAuth client ID, client secret, and refresh token are required")
        try:
            ZoneInfo(self.timezone_name)
        except (ZoneInfoNotFoundError, TypeError) as error:
            raise CalendarConfigurationError("Google Calendar requires a configured IANA timezone") from error
        if not 0 < self.timeout_seconds <= MAX_CALENDAR_TIMEOUT_SECONDS:
            raise CalendarConfigurationError("Google Calendar timeout is out of range")

    @classmethod
    def from_environment(cls, environment: dict[str, str]) -> "CalendarConfig | None":
        keys = (
            "SECOND_BRAIN_GOOGLE_CLIENT_ID",
            "SECOND_BRAIN_GOOGLE_CLIENT_SECRET",
            "SECOND_BRAIN_GOOGLE_REFRESH_TOKEN",
            "SECOND_BRAIN_GOOGLE_TIMEZONE",
        )
        values = [environment.get(key) for key in keys]
        if not any(values):
            return None
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise CalendarConfigurationError("Google Calendar configuration must be complete or disabled")
        return cls(values[0], values[1], values[2], values[3])


@dataclass(frozen=True)
class CalendarResponse:
    status: int
    body: bytes


@dataclass(frozen=True)
class CalendarEvent:
    id: str
    summary: str
    description: str
    start: str
    end: str
    all_day: bool = False


@dataclass(frozen=True)
class CalendarList:
    events: tuple[CalendarEvent, ...]
    truncated: bool


CalendarTransport = Callable[[Request, float], CalendarResponse]


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _default_transport(request: Request, timeout: float) -> CalendarResponse:
    opener = build_opener(_RejectRedirects())
    try:
        with opener.open(request, timeout=timeout) as response:  # nosec B310: fixed Google endpoints only
            return CalendarResponse(response.status, response.read(MAX_CALENDAR_RESPONSE_BYTES + 1))
    except HTTPError as error:
        try:
            body = error.read(MAX_CALENDAR_RESPONSE_BYTES + 1)
        finally:
            error.close()
        return CalendarResponse(error.code, body)
    except Exception as error:
        raise CalendarRetryableError("Google Calendar request failed") from error


class CalendarClient:
    def __init__(
        self,
        config: CalendarConfig,
        *,
        transport: CalendarTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.transport = transport or _default_transport
        self.monotonic = monotonic
        self._access_token: str | None = None
        self._token_expires_at = 0.0

    @staticmethod
    def normalize_create(arguments: object, config: CalendarConfig) -> dict[str, object]:
        fields = _object(arguments)
        _only(fields, {"summary", "description", "start", "end"}, {"summary", "start", "end"})
        summary = _short_text(fields["summary"], "summary", MAX_SUMMARY_LENGTH)
        description = _optional_text(fields.get("description", ""), "description", MAX_DESCRIPTION_LENGTH)
        start = parse_calendar_time(_short_text(fields["start"], "start", 100), config.timezone_name)
        end = parse_calendar_time(_short_text(fields["end"], "end", 100), config.timezone_name)
        if end <= start:
            raise ValueError("calendar end must be after start")
        if end - start > MAX_EVENT_DURATION:
            raise ValueError("calendar event duration must not exceed 14 days")
        return {
            "calendar_id": "primary",
            "timezone": config.timezone_name,
            "summary": summary,
            "description": description,
            "start": _timestamp(start),
            "end": _timestamp(end),
        }

    @staticmethod
    def normalize_list(arguments: object, config: CalendarConfig) -> dict[str, object]:
        fields = _object(arguments)
        _only(fields, {"time_min", "time_max", "max_results"}, {"time_min", "time_max"})
        time_min = parse_calendar_time(_short_text(fields["time_min"], "time_min", 100), config.timezone_name)
        time_max = parse_calendar_time(_short_text(fields["time_max"], "time_max", 100), config.timezone_name)
        if time_max <= time_min:
            raise ValueError("calendar time_max must be after time_min")
        if time_max - time_min > MAX_LIST_WINDOW:
            raise ValueError("calendar list window must not exceed 31 days")
        max_results = fields.get("max_results", MAX_LIST_RESULTS)
        if type(max_results) is not int or not 1 <= max_results <= MAX_LIST_RESULTS:
            raise ValueError("calendar max_results must be an integer from 1 to 20")
        return {
            "calendar_id": "primary",
            "timezone": config.timezone_name,
            "time_min": _timestamp(time_min),
            "time_max": _timestamp(time_max),
            "max_results": max_results,
        }

    def provider_event_id(self, message_id: str) -> str:
        # SHA-256 hexadecimal uses Google's lowercase base32hex-compatible alphabet.
        return hashlib.sha256(f"{message_id}:calendar_create".encode("utf-8")).hexdigest()

    def create_binding(self, intent: dict[str, object]) -> str:
        """Return an opaque configuration fence for a frozen calendar creation plan."""
        calendar_id = intent.get("calendar_id")
        if not isinstance(calendar_id, str) or not calendar_id:
            raise ValueError("calendar creation intent is missing its calendar target")
        return _digest({
            "domain": "second_brain.calendar.config-binding.v1",
            "calendar_id": calendar_id,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "refresh_token": self.config.refresh_token,
        })

    def create_event(self, intent: dict[str, object], message_id: str) -> CalendarEvent:
        event_id = self.provider_event_id(message_id)
        fingerprint = _fingerprint(intent)
        body = {
            "id": event_id,
            "summary": intent["summary"],
            "description": intent["description"],
            "start": {"dateTime": intent["start"], "timeZone": intent["timezone"]},
            "end": {"dateTime": intent["end"], "timeZone": intent["timezone"]},
            "extendedProperties": {"private": {FINGERPRINT_KEY: fingerprint}},
        }
        result = self._api_request("POST", GOOGLE_CALENDAR_URL, body)
        if result.status == 409:
            lookup = self._api_request("GET", f"{GOOGLE_CALENDAR_URL}/{quote(event_id, safe='')}")
            return self._validated_created_event(lookup, intent, event_id, fingerprint)
        return self._validated_created_event(result, intent, event_id, fingerprint)

    def list_events(self, intent: dict[str, object]) -> CalendarList:
        query = urlencode({
            "timeMin": intent["time_min"], "timeMax": intent["time_max"], "maxResults": intent["max_results"],
            "singleEvents": "true", "orderBy": "startTime",
        })
        result = self._api_request("GET", f"{GOOGLE_CALENDAR_URL}?{query}")
        payload = self._success_json(result)
        items = payload.get("items")
        if not isinstance(items, list):
            raise CalendarRetryableError("Google Calendar response is malformed")
        events: list[CalendarEvent] = []
        for item in items:
            if not isinstance(item, dict) or item.get("status") == "cancelled":
                continue
            event = _read_event(item)
            if event is not None:
                events.append(event)
        requested_limit = intent["max_results"]
        if type(requested_limit) is not int or not 1 <= requested_limit <= MAX_LIST_RESULTS:
            raise ValueError("calendar list intent has an invalid result limit")
        return CalendarList(
            tuple(events[:requested_limit]),
            bool(payload.get("nextPageToken")) or len(events) > requested_limit,
        )

    def _validated_created_event(self, result: CalendarResponse, intent: dict[str, object], event_id: str, fingerprint: str) -> CalendarEvent:
        payload = self._success_json(result)
        event = _read_event(payload)
        marker = _marker(payload)
        if (
            event is None or event.id != event_id or payload.get("status") == "cancelled" or marker != fingerprint
            or event.summary != intent["summary"] or event.description != intent["description"]
            or event.all_day or event.start != intent["start"] or event.end != intent["end"]
        ):
            raise ValueError("Google Calendar event response does not match the requested event")
        return event

    def _api_request(self, method: str, url: str, payload: dict[str, object] | None = None) -> CalendarResponse:
        response = self._authorized_request(method, url, payload)
        if response.status == 401:
            self._access_token = None
            self._token_expires_at = 0
            response = self._authorized_request(method, url, payload)
        if response.status in {401, 429} or response.status >= 500:
            raise CalendarRetryableError("Google Calendar request should be retried")
        if response.status == 403:
            if _is_rate_limited(response.body):
                raise CalendarRetryableError("Google Calendar request should be retried")
            raise ValueError("Google Calendar permission was denied")
        if response.status == 409:
            return response
        if not 200 <= response.status < 300:
            raise ValueError("Google Calendar rejected the request")
        return response

    def _authorized_request(self, method: str, url: str, payload: dict[str, object] | None) -> CalendarResponse:
        token = self._token()
        data = None
        headers = {"Authorization": f"Bearer {token}"}
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            if len(data) > MAX_CALENDAR_REQUEST_BYTES:
                raise ValueError("Google Calendar request exceeds size limit")
            headers["Content-Type"] = "application/json"
        try:
            response = self.transport(Request(url, data=data, headers=headers, method=method), self.config.timeout_seconds)
        except CalendarRetryableError:
            raise
        except Exception as error:
            raise CalendarRetryableError("Google Calendar request failed") from error
        return _bounded_response(response)

    def _token(self) -> str:
        if self._access_token is not None and self.monotonic() < self._token_expires_at:
            return self._access_token
        encoded = urlencode({
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self.config.refresh_token,
        }).encode("ascii")
        request = Request(GOOGLE_TOKEN_URL, data=encoded, headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
        try:
            response = _bounded_response(self.transport(request, self.config.timeout_seconds))
        except CalendarRetryableError:
            raise
        except Exception as error:
            raise CalendarRetryableError("Google OAuth token refresh failed") from error
        if response.status in {401, 429} or response.status >= 500:
            raise CalendarRetryableError("Google OAuth token refresh should be retried")
        if not 200 <= response.status < 300:
            raise ValueError("Google OAuth credentials were rejected")
        payload = _json_object(response.body)
        token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        if not isinstance(token, str) or not token or type(expires_in) not in {int, float} or expires_in <= 0:
            raise CalendarRetryableError("Google OAuth token response is malformed")
        self._access_token = token
        self._token_expires_at = self.monotonic() + max(1, float(expires_in) - 30)
        return token

    @staticmethod
    def _success_json(response: CalendarResponse) -> dict[str, object]:
        if not 200 <= response.status < 300:
            raise CalendarRetryableError("Google Calendar response is not successful")
        return _json_object(response.body)


def parse_calendar_time(value: str, timezone_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("calendar time must be an ISO-8601 datetime") from error
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        return parsed.astimezone(timezone.utc)
    zone = ZoneInfo(timezone_name)
    first = parsed.replace(tzinfo=zone, fold=0)
    second = parsed.replace(tzinfo=zone, fold=1)
    first_round_trip = first.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
    second_round_trip = second.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
    if first_round_trip != parsed and second_round_trip != parsed:
        raise ValueError("calendar local time is invalid; use an explicit UTC offset")
    if first.utcoffset() != second.utcoffset():
        raise ValueError("calendar local time is ambiguous; use an explicit UTC offset")
    return first.astimezone(timezone.utc)


def _bounded_response(response: CalendarResponse) -> CalendarResponse:
    if not isinstance(response, CalendarResponse) or not isinstance(response.status, int) or not isinstance(response.body, bytes):
        raise CalendarRetryableError("Google Calendar transport response is malformed")
    if len(response.body) > MAX_CALENDAR_RESPONSE_BYTES:
        raise CalendarRetryableError("Google Calendar response exceeds size limit")
    return response


def _json_object(body: bytes) -> dict[str, object]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalendarRetryableError("Google Calendar response is malformed") from error
    if not isinstance(value, dict):
        raise CalendarRetryableError("Google Calendar response is malformed")
    return value


def _read_event(value: dict[str, object]) -> CalendarEvent | None:
    event_id = value.get("id")
    summary = value.get("summary", "")
    description = value.get("description", "")
    start = value.get("start")
    end = value.get("end")
    if not isinstance(event_id, str) or not event_id or not isinstance(summary, str) or not isinstance(description, str):
        return None
    if len(summary) > MAX_SUMMARY_LENGTH or len(description) > MAX_DESCRIPTION_LENGTH:
        return None
    if not isinstance(start, dict) or not isinstance(end, dict):
        return None
    start_date = start.get("date")
    end_date = end.get("date")
    if isinstance(start_date, str) and isinstance(end_date, str):
        start_day = _calendar_date(start_date)
        end_day = _calendar_date(end_date)
        if start_day is None or end_day is None or end_day <= start_day:
            return None
        return CalendarEvent(event_id, summary, description, start_date, end_date, True)
    start_time = start.get("dateTime")
    end_time = end.get("dateTime")
    if not isinstance(start_time, str) or not isinstance(end_time, str):
        return None
    try:
        return CalendarEvent(event_id, summary, description, _timestamp(parse_calendar_time(start_time, "UTC")), _timestamp(parse_calendar_time(end_time, "UTC")))
    except ValueError:
        return None


def _marker(value: dict[str, object]) -> str | None:
    extended = value.get("extendedProperties")
    if not isinstance(extended, dict):
        return None
    private = extended.get("private")
    marker = private.get(FINGERPRINT_KEY) if isinstance(private, dict) else None
    return marker if isinstance(marker, str) else None


def _fingerprint(intent: dict[str, object]) -> str:
    """Application event-intent marker; distinct from the configuration fence."""
    return _digest({"domain": "second_brain.calendar.event-intent.v1", "intent": intent})


def _digest(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _calendar_date(value: str) -> date | None:
    if len(value) != 10:
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be an object")
    return value


def _only(values: dict[str, object], allowed: set[str], required: set[str]) -> None:
    extra = set(values) - allowed
    missing = required - set(values)
    if extra:
        raise ValueError(f"unexpected tool argument: {sorted(extra)[0]}")
    if missing:
        raise ValueError(f"missing tool argument: {sorted(missing)[0]}")


def _short_text(value: object, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"calendar {field} must be a non-empty string within the length limit")
    return value.strip()


def _optional_text(value: object, field: str, limit: int) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"calendar {field} must be a string within the length limit")
    return value.strip()


def _is_rate_limited(body: bytes) -> bool:
    try:
        encoded = json.dumps(_json_object(body))
    except CalendarRetryableError:
        return True
    return "userRateLimitExceeded" in encoded or "rateLimitExceeded" in encoded
