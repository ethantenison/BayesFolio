from __future__ import annotations

from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract
from bayesfolio.contracts.chat.intent import ParsedIntent


class ChatMessageUser(VersionedContract):
    """User chat message contract.

    Attributes:
        schema: Contract schema identifier; always ``SchemaName.CHAT_MESSAGE_USER``.
        schema_version: Semantic schema version string.
        content: Raw user message text.
    """

    schema: Literal[SchemaName.CHAT_MESSAGE_USER] = SchemaName.CHAT_MESSAGE_USER
    schema_version: Literal["0.1.0"] = "0.1.0"
    content: str


class ChatMessageAssistant(VersionedContract):
    """Assistant chat message contract.

    Attributes:
        schema: Contract schema identifier; always ``SchemaName.CHAT_MESSAGE_ASSISTANT``.
        schema_version: Semantic schema version string.
        content: Assistant response text.
    """

    schema: Literal[SchemaName.CHAT_MESSAGE_ASSISTANT] = SchemaName.CHAT_MESSAGE_ASSISTANT
    schema_version: Literal["0.1.0"] = "0.1.0"
    content: str


class ChatToolCall(VersionedContract):
    """Tool invocation request emitted by an assistant planner.

    Attributes:
        schema: Contract schema identifier; always ``SchemaName.CHAT_TOOL_CALL``.
        schema_version: Semantic schema version string.
        call_id: Stable identifier linking a tool result to this call.
        tool_name: Logical tool name to invoke.
        arguments: Tool arguments payload.
    """

    schema: Literal[SchemaName.CHAT_TOOL_CALL] = SchemaName.CHAT_TOOL_CALL
    schema_version: Literal["0.1.0"] = "0.1.0"
    call_id: str
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class ChatToolResult(VersionedContract):
    """Result payload for a previously issued tool call.

    Attributes:
        schema: Contract schema identifier; always ``SchemaName.CHAT_TOOL_RESULT``.
        schema_version: Semantic schema version string.
        call_id: Identifier of the originating ``ChatToolCall``.
        tool_name: Logical tool name that produced this result.
        success: Whether the tool invocation succeeded.
        payload: Tool output payload for successful calls.
        error_message: Optional error summary when ``success`` is ``False``.
    """

    schema: Literal[SchemaName.CHAT_TOOL_RESULT] = SchemaName.CHAT_TOOL_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    call_id: str
    tool_name: str
    success: bool
    payload: dict[str, object] = Field(default_factory=dict)
    error_message: str | None = None


class ChatTurn(VersionedContract):
    """Single end-to-end conversation turn with optional tool activity.

    Attributes:
        schema: Contract schema identifier; always ``SchemaName.CHAT_TURN``.
        schema_version: Semantic schema version string.
        user_message: Required user utterance for this turn.
        assistant_message: Optional assistant response for this turn.
        parsed_intent: Optional parsed intent extracted from ``user_message``.
        tool_calls: Ordered tool calls issued during the turn.
        tool_results: Ordered tool results captured for the turn.
        diagnostics: Optional debug metadata for traceability.
    """

    schema: Literal[SchemaName.CHAT_TURN] = SchemaName.CHAT_TURN
    schema_version: Literal["0.1.0"] = "0.1.0"
    user_message: ChatMessageUser
    assistant_message: ChatMessageAssistant | None = None
    parsed_intent: ParsedIntent | None = None
    tool_calls: list[ChatToolCall] = Field(default_factory=list)
    tool_results: list[ChatToolResult] = Field(default_factory=list)
    diagnostics: dict[str, object] = Field(default_factory=dict)
