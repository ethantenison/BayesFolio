from __future__ import annotations

import pytest
from pydantic import ValidationError

from bayesfolio.contracts.base import SchemaName
from bayesfolio.contracts.chat.intent import ParsedIntent
from bayesfolio.contracts.chat.protocol import (
    ChatMessageAssistant,
    ChatMessageUser,
    ChatToolCall,
    ChatToolResult,
    ChatTurn,
)


def test_chat_message_user_schema_identity_and_forbid_extra() -> None:
    message = ChatMessageUser(content="Build a monthly portfolio.")

    assert message.schema is SchemaName.CHAT_MESSAGE_USER
    assert message.schema_version == "0.1.0"

    with pytest.raises(ValidationError):
        ChatMessageUser(content="hello", unexpected_field="nope")


def test_chat_message_assistant_schema_identity_and_forbid_extra() -> None:
    message = ChatMessageAssistant(content="I can help with that.")

    assert message.schema is SchemaName.CHAT_MESSAGE_ASSISTANT
    assert message.schema_version == "0.1.0"

    with pytest.raises(ValidationError):
        ChatMessageAssistant(content="ok", extra_detail="nope")


def test_chat_tool_call_schema_identity_and_required_fields() -> None:
    call = ChatToolCall(
        call_id="call_001",
        tool_name="optimize_portfolio",
        arguments={"objective": "Sharpe"},
    )

    assert call.schema is SchemaName.CHAT_TOOL_CALL
    assert call.schema_version == "0.1.0"

    with pytest.raises(ValidationError):
        ChatToolCall(call_id="call_002", tool_name="optimize_portfolio", unexpected_field=True)


def test_chat_tool_result_schema_identity_and_required_fields() -> None:
    result = ChatToolResult(
        call_id="call_001",
        tool_name="optimize_portfolio",
        success=True,
        payload={"weights": {"SPY": 0.6, "TLT": 0.4}},
    )

    assert result.schema is SchemaName.CHAT_TOOL_RESULT
    assert result.schema_version == "0.1.0"

    with pytest.raises(ValidationError):
        ChatToolResult(call_id="call_001", tool_name="optimize_portfolio", success=True, extra_info="nope")


def test_chat_turn_schema_identity_and_nested_payloads() -> None:
    turn = ChatTurn(
        user_message=ChatMessageUser(content="Optimize for CVaR."),
        assistant_message=ChatMessageAssistant(content="Running optimization now."),
        parsed_intent=ParsedIntent(objective="Sharpe", risk_measure="CVaR"),
        tool_calls=[
            ChatToolCall(
                call_id="call_001",
                tool_name="optimize_portfolio",
                arguments={"risk_measure": "CVaR"},
            )
        ],
        tool_results=[
            ChatToolResult(
                call_id="call_001",
                tool_name="optimize_portfolio",
                success=True,
                payload={"status": "ok"},
            )
        ],
        diagnostics={"planner": "default"},
    )

    assert turn.schema is SchemaName.CHAT_TURN
    assert turn.schema_version == "0.1.0"
    assert turn.user_message.content == "Optimize for CVaR."
    assert turn.tool_calls[0].tool_name == "optimize_portfolio"

    with pytest.raises(ValidationError):
        ChatTurn(user_message=ChatMessageUser(content="hello"), unknown="nope")
