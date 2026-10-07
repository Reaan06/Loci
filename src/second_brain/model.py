"""Bounded, injectable OpenAI-compatible chat-completions adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Protocol
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_PROVIDER_RESPONSE_BYTES = 1_000_000
MAX_PROVIDER_REQUEST_BYTES = 100_000
MAX_TIMEOUT_SECONDS = 60


class ModelConfigurationError(ValueError):
    pass


class ModelRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ModelResponse:
    content: str | None
    tool_calls: tuple[ModelToolCall, ...]


class Model(Protocol):
    def complete(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> ModelResponse: ...


HttpTransport = Callable[[Request, float], bytes]


class OpenAICompatibleModel:
    """Minimal provider adapter; injected transport keeps tests offline."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 20,
        transport: HttpTransport | None = None,
        allow_insecure_localhost: bool = False,
    ):
        parsed = urlparse(endpoint)
        local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and not (allow_insecure_localhost and local_http):
            raise ModelConfigurationError("provider endpoint must use HTTPS (or an explicitly trusted localhost proxy)")
        if not api_key or not api_key.strip():
            raise ModelConfigurationError("provider API key is required")
        if not model or not model.strip():
            raise ModelConfigurationError("provider model is required")
        if not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
            raise ModelConfigurationError(f"provider timeout must be between 0 and {MAX_TIMEOUT_SECONDS} seconds")
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.transport = transport or _default_transport

    @classmethod
    def from_environment(cls, environment: dict[str, str], **kwargs: object) -> "OpenAICompatibleModel":
        try:
            endpoint = environment["SECOND_BRAIN_PROVIDER_URL"]
            api_key = environment["SECOND_BRAIN_PROVIDER_KEY"]
            model = environment["SECOND_BRAIN_PROVIDER_MODEL"]
        except KeyError as error:
            raise ModelConfigurationError("provider URL, key, and model must be configured explicitly") from error
        return cls(endpoint, api_key, model, **kwargs)

    def complete(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> ModelResponse:
        payload = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto"}
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_PROVIDER_REQUEST_BYTES:
            raise ValueError("provider request exceeds size limit")
        request = Request(
            self.endpoint,
            data=encoded,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            raw_response = self.transport(request, self.timeout_seconds)
        except Exception as error:
            raise ModelRequestError("provider request failed") from error
        if len(raw_response) > MAX_PROVIDER_RESPONSE_BYTES:
            raise ValueError("provider response exceeds size limit")
        try:
            message = json.loads(raw_response)["choices"][0]["message"]
            content = message.get("content")
            tool_calls = tuple(_parse_tool_call(item) for item in message.get("tool_calls", []))
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise ModelRequestError("provider response has an invalid chat-completions shape") from error
        if content is not None and not isinstance(content, str):
            raise ModelRequestError("provider response content must be text")
        return ModelResponse(content, tool_calls)


def _parse_tool_call(value: object) -> ModelToolCall:
    if not isinstance(value, dict):
        raise ModelRequestError("provider tool call must be an object")
    function = value.get("function")
    if not isinstance(function, dict) or not isinstance(value.get("id"), str) or not isinstance(function.get("name"), str):
        raise ModelRequestError("provider tool call is missing fields")
    try:
        arguments = json.loads(function.get("arguments", ""))
    except json.JSONDecodeError as error:
        raise ModelRequestError("provider tool arguments must be JSON") from error
    if not isinstance(arguments, dict):
        raise ModelRequestError("provider tool arguments must be an object")
    return ModelToolCall(value["id"], function["name"], arguments)


class _RejectRedirects(HTTPRedirectHandler):
    """Reject every redirect before urllib can follow it with credentials."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _default_transport(request: Request, timeout: float) -> bytes:
    opener = build_opener(_RejectRedirects())
    try:
        with opener.open(request, timeout=timeout) as response:  # nosec B310: endpoint is explicitly validated
            return response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
    except HTTPError as error:
        error.close()
        raise
