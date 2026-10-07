"""Bounded agent orchestration with durable replay protection."""

from __future__ import annotations

from dataclasses import dataclass

from .calendar import CalendarClient
from .core import BrainStore
from .model import Model, ModelToolCall
from .tools import ToolResult, dispatch_tool, tool_definitions

MAX_MESSAGE_LENGTH = 5_000
MAX_TOOL_ROUNDS = 3
MAX_TOOL_CALLS = 8

ToolCall = ModelToolCall


@dataclass(frozen=True)
class AgentReply:
    text: str
    action_performed: bool = False


class Agent:
    def __init__(self, store: BrainStore, model: Model, *, calendar: CalendarClient | None = None):
        self.store = store
        self.model = model
        self.calendar = calendar

    def handle(self, text: str, *, message_id: str) -> AgentReply:
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"message must contain 1 to {MAX_MESSAGE_LENGTH} characters")
        text = text.strip()
        reservation = self.store.reserve_conversation(message_id, text)
        if reservation.state == "completed":
            return AgentReply(reservation.response_text or "Error: completed response is unavailable")
        if reservation.state == "active":
            return AgentReply("Error: message is already being processed")
        if reservation.lease_token is None:
            raise RuntimeError("owned conversation is missing a lease token")
        self.store.activate_conversation_lease(message_id, reservation.lease_token)
        try:
            reply = self._run(text, message_id, reservation.lease_token)
            self.store.complete_conversation(message_id, text, reply.text, lease_token=reservation.lease_token)
            return reply
        except Exception:
            self.store.abandon_conversation(message_id, text, lease_token=reservation.lease_token)
            raise
        finally:
            self.store.deactivate_conversation_lease(message_id)

    def _run(self, text: str, message_id: str, lease_token: str) -> AgentReply:
        messages = [{"role": "user", "content": text}]
        calls_used = 0
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.model.complete(messages, tool_definitions(self.calendar))
            if not response.tool_calls:
                return AgentReply("No action was performed. Please clarify what you need.")
            results: list[ToolResult] = []
            for call in response.tool_calls:
                calls_used += 1
                if calls_used > MAX_TOOL_CALLS:
                    return AgentReply("Error: tool call limit exhausted")
                self.store.assert_conversation_lease(message_id, lease_token)
                results.append(dispatch_tool(self.store, call.name, call.arguments, message_id=message_id, calendar=self.calendar))
            # Action results, not model prose, are the confirmation boundary.
            actionable = next((result for result in results if result.ok and result.mutated), None)
            if actionable:
                return AgentReply(actionable.message, True)
            error = next((result for result in results if not result.ok), None)
            if error:
                return AgentReply(error.message)
            return AgentReply(results[-1].message)
        return AgentReply("Error: tool round limit exhausted")

