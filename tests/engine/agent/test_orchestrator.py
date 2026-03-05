from __future__ import annotations

from bayesfolio.contracts.chat.protocol import (
    ChatMessageAssistant,
    ChatMessageUser,
    ChatToolCall,
    ChatToolResult,
    ChatTurn,
)
from bayesfolio.engine.agent.orchestrator import OrchestratorAction, evaluate_turn


def _base_turn() -> ChatTurn:
    return ChatTurn(user_message=ChatMessageUser(content="Optimize my ETF portfolio."))


def test_evaluate_turn_requests_assistant_message_when_no_tools() -> None:
    decision = evaluate_turn(_base_turn())

    assert decision.action is OrchestratorAction.NEEDS_ASSISTANT_MESSAGE
    assert decision.pending_tool_calls == []


def test_evaluate_turn_requests_tool_execution_for_pending_calls() -> None:
    turn = _base_turn()
    turn.tool_calls = [
        ChatToolCall(call_id="call_001", tool_name="build_features", arguments={}),
        ChatToolCall(call_id="call_002", tool_name="optimize_portfolio", arguments={"objective": "Sharpe"}),
    ]
    turn.tool_results = [
        ChatToolResult(
            call_id="call_001",
            tool_name="build_features",
            success=True,
            payload={"artifact": "features.parquet"},
        )
    ]

    decision = evaluate_turn(turn)

    assert decision.action is OrchestratorAction.RUN_TOOLS
    assert [call.call_id for call in decision.pending_tool_calls] == ["call_002"]


def test_evaluate_turn_finalizes_when_tools_completed_and_assistant_present() -> None:
    turn = _base_turn()
    turn.assistant_message = ChatMessageAssistant(content="Portfolio generated.")
    turn.tool_calls = [
        ChatToolCall(call_id="call_001", tool_name="optimize_portfolio", arguments={}),
    ]
    turn.tool_results = [
        ChatToolResult(
            call_id="call_001",
            tool_name="optimize_portfolio",
            success=True,
            payload={"status": "ok"},
        )
    ]

    decision = evaluate_turn(turn)

    assert decision.action is OrchestratorAction.FINALIZE_TURN


def test_evaluate_turn_marks_invalid_when_result_call_id_unknown() -> None:
    turn = _base_turn()
    turn.tool_calls = [ChatToolCall(call_id="call_001", tool_name="optimize_portfolio", arguments={})]
    turn.tool_results = [
        ChatToolResult(
            call_id="unknown",
            tool_name="optimize_portfolio",
            success=False,
            payload={},
            error_message="missing call",
        )
    ]

    decision = evaluate_turn(turn)

    assert decision.action is OrchestratorAction.INVALID_TURN
    assert decision.diagnostics["unknown_result_call_ids"] == ["unknown"]


def test_evaluate_turn_marks_invalid_when_duplicate_tool_call_ids() -> None:
    turn = _base_turn()
    turn.tool_calls = [
        ChatToolCall(call_id="call_001", tool_name="build_features", arguments={}),
        ChatToolCall(call_id="call_001", tool_name="optimize_portfolio", arguments={}),
    ]

    decision = evaluate_turn(turn)

    assert decision.action is OrchestratorAction.INVALID_TURN
    assert decision.diagnostics["error"] == "Duplicate tool call IDs in turn."
