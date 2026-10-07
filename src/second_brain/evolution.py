"""Bounded Evolution API adapter and strict webhook event parser.

This module supports an already-running Evolution instance only.  It never
links a QR account, provisions an instance, or attempts provider activation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import json
from typing import Callable
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .core import MAX_TEXT_LENGTH, Reminder

MAX_EVOLUTION_RESPONSE_BYTES = 100_000
MAX_EVOLUTION_REQUEST_BYTES = 20_000
MAX_EVOLUTION_TIMEOUT_SECONDS = 60


class EvolutionRequestError(RuntimeError):
    """The provider did not give a bounded, positive send acceptance."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _default_transport(request: Request, timeout: float) -> bytes:
    opener = build_opener(_RejectRedirects())
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.read(MAX_EVOLUTION_RESPONSE_BYTES + 1)
    except HTTPError as error:
        error.close()
        raise


@dataclass(frozen=True)
class EvolutionConfig:
    base_url: str
    api_key: str
    instance_name: str
    owner_jid: str
    webhook_secret: str
    text_dialect: str = "textMessage"
    timeout_seconds: float = 20
    allow_insecure_private_http: bool = False

    def __post_init__(self) -> None:
        try:
            parsed = urlparse(self.base_url)
            hostname = parsed.hostname
            parsed.port
        except ValueError as error:
            raise ValueError("Evolution base URL must include a valid authority") from error
        if not parsed.netloc or not hostname:
            raise ValueError("Evolution base URL must include a valid authority")
        if "?" in self.base_url or "#" in self.base_url or "@" in parsed.netloc:
            raise ValueError("Evolution base URL must not include query, fragment, or userinfo")
        private_http = parsed.scheme == "http" and _is_local_or_private_host(hostname)
        if parsed.scheme != "https" and not (self.allow_insecure_private_http and private_http):
            raise ValueError("Evolution base URL must use HTTPS (or an explicitly trusted local HTTP endpoint)")
        if not self.api_key.strip() or not self.webhook_secret.strip():
            raise ValueError("Evolution API key and webhook secret must be non-empty")
        if not self.instance_name.strip() or len(self.instance_name) > 100:
            raise ValueError("Evolution instance name must contain 1 to 100 characters")
        if not self.owner_jid.endswith("@s.whatsapp.net") or not self.owner_jid.split("@", 1)[0].isdigit():
            raise ValueError("owner JID must be a phone-number@s.whatsapp.net identity")
        if self.text_dialect not in {"textMessage", "text"}:
            raise ValueError("Evolution text dialect must be 'textMessage' or 'text'")
        if not 0 < self.timeout_seconds <= MAX_EVOLUTION_TIMEOUT_SECONDS:
            raise ValueError("Evolution timeout is out of range")

    @property
    def owner_number(self) -> str:
        return self.owner_jid.split("@", 1)[0]

    @property
    def binding(self) -> str:
        """Secret-free recipient/instance binding digest."""
        stable = f"evolution\0{self.instance_name}\0{self.owner_jid}".encode("utf-8")
        return hashlib.sha256(stable).hexdigest()

    @classmethod
    def from_environment(cls, environment: dict[str, str]) -> "EvolutionConfig":
        try:
            return cls(
                base_url=environment["SECOND_BRAIN_EVOLUTION_URL"],
                api_key=environment["SECOND_BRAIN_EVOLUTION_API_KEY"],
                instance_name=environment["SECOND_BRAIN_EVOLUTION_INSTANCE"],
                owner_jid=environment["SECOND_BRAIN_EVOLUTION_OWNER_JID"],
                webhook_secret=environment["SECOND_BRAIN_WEBHOOK_SECRET"],
                text_dialect=environment.get("SECOND_BRAIN_EVOLUTION_TEXT_DIALECT", "textMessage"),
                allow_insecure_private_http=environment.get("SECOND_BRAIN_EVOLUTION_ALLOW_INSECURE_LOCAL_HTTP") == "true",
            )
        except KeyError as error:
            raise ValueError("Evolution URL, API key, instance, owner JID, and webhook secret are required") from error


@dataclass(frozen=True)
class InboundEvolutionMessage:
    message_id: str
    text: str


def parse_evolution_message(payload: object, config: EvolutionConfig) -> InboundEvolutionMessage | None:
    """Return only a direct, inbound owner text message; discard everything else."""
    if not isinstance(payload, dict) or payload.get("event") != "messages.upsert" or payload.get("instance") != config.instance_name:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    key = data.get("key")
    message = data.get("message")
    if not isinstance(key, dict) or not isinstance(message, dict):
        return None
    if key.get("fromMe") is not False or key.get("remoteJid") != config.owner_jid:
        return None
    raw_id = key.get("id")
    if not isinstance(raw_id, str) or not raw_id or len(raw_id) > 2_000:
        return None
    text = message.get("conversation")
    if not isinstance(text, str):
        extended = message.get("extendedTextMessage")
        text = extended.get("text") if isinstance(extended, dict) else None
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_LENGTH:
        return None
    stable = f"{config.instance_name}\0{config.owner_jid}\0{raw_id}".encode("utf-8")
    return InboundEvolutionMessage(f"evolution:{hashlib.sha256(stable).hexdigest()}", text.strip())


class EvolutionTransport:
    """Evolution sendText client; dialect selection is explicit and never retried."""

    def __init__(self, config: EvolutionConfig, *, transport: Callable[[Request, float], bytes] | None = None):
        self.config = config
        self.transport = transport or _default_transport

    def send(self, reminder: Reminder) -> None:
        self.send_text(f"Reminder: {reminder.text}")

    def send_text(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_LENGTH:
            raise ValueError("outgoing Evolution text must be non-empty and bounded")
        if self.config.text_dialect == "textMessage":
            body = {"number": self.config.owner_number, "textMessage": {"text": text.strip()}}
        else:
            body = {"number": self.config.owner_number, "text": text.strip()}
        encoded = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_EVOLUTION_REQUEST_BYTES:
            raise ValueError("Evolution request exceeds size limit")
        endpoint = f"{self.config.base_url.rstrip('/')}/message/sendText/{self.config.instance_name}"
        request = Request(endpoint, data=encoded, headers={"apikey": self.config.api_key, "Content-Type": "application/json"}, method="POST")
        try:
            raw = self.transport(request, self.config.timeout_seconds)
        except Exception as error:
            raise EvolutionRequestError("Evolution send request failed") from error
        if len(raw) > MAX_EVOLUTION_RESPONSE_BYTES:
            raise EvolutionRequestError("Evolution response exceeds size limit")
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as error:
            raise EvolutionRequestError("Evolution response is invalid") from error
        if (
            not isinstance(response, dict)
            or not isinstance(response.get("key"), dict)
            or not response["key"].get("id")
            or not isinstance(response.get("message"), dict)
            or not isinstance(response.get("status"), str)
            or not response["status"].strip()
        ):
            raise EvolutionRequestError("Evolution response lacks positive acceptance")
        if response["status"].upper() in {"ERROR", "FAILED", "REJECTED"}:
            raise EvolutionRequestError("Evolution response does not accept the message")


def _is_local_or_private_host(hostname: str | None) -> bool:
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        address = ipaddress.ip_address(hostname or "")
    except ValueError:
        return False
    return address.is_private
