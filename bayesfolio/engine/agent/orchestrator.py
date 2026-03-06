from __future__ import annotations

from enum import StrEnum
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from bayesfolio.contracts.chat.protocol import ChatToolCall, ChatToolResult, ChatTurn


class OrchestratorAction(StrEnum):
    """Deterministic next-action labels for chat turn orchestration."""

    NEEDS_ASSISTANT_MESSAGE = "needs_assistant_message"
    RUN_TOOLS = "run_tools"
    FINALIZE_TURN = "finalize_turn"
    INVALID_TURN = "invalid_turn"


class OrchestratorDecision(BaseModel):
    """Decision payload returned by the agent orchestrator state machine.

    Attributes:
        action: Selected next step for the current turn.
        pending_tool_calls: Tool calls that still need execution.
        diagnostics: Debug metadata explaining why the action was selected.
    """

    action: OrchestratorAction
    pending_tool_calls: list[ChatToolCall] = Field(default_factory=list)
    diagnostics: dict[str, object] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


ToolExecutor = Callable[[ChatToolCall], ChatToolResult]


def evaluate_turn(turn: ChatTurn) -> OrchestratorDecision:
    """Compute the next deterministic action for a chat turn.

    Args:
        turn: Chat turn contract containing user message, optional assistant
            response, tool calls, and tool results.

    Returns:
        Orchestrator decision with action and any pending tool calls.
    """

    call_ids = {call.call_id for call in turn.tool_calls}
    duplicate_call_ids = len(call_ids) != len(turn.tool_calls)

    result_by_call_id: dict[str, bool] = {}
    unknown_result_call_ids: list[str] = []
    for result in turn.tool_results:
        if result.call_id not in call_ids:
            unknown_result_call_ids.append(result.call_id)
            continue
        result_by_call_id[result.call_id] = result.success

    pending_calls = [call for call in turn.tool_calls if call.call_id not in result_by_call_id]

    diagnostics: dict[str, object] = {
        "tool_call_count": len(turn.tool_calls),
        "tool_result_count": len(turn.tool_results),
        "pending_tool_call_ids": [call.call_id for call in pending_calls],
    }

    if duplicate_call_ids:
        diagnostics["error"] = "Duplicate tool call IDs in turn."
        return OrchestratorDecision(action=OrchestratorAction.INVALID_TURN, diagnostics=diagnostics)

    if unknown_result_call_ids:
        diagnostics["error"] = "Tool results include unknown call IDs."
        diagnostics["unknown_result_call_ids"] = unknown_result_call_ids
        return OrchestratorDecision(action=OrchestratorAction.INVALID_TURN, diagnostics=diagnostics)

    if pending_calls:
        return OrchestratorDecision(
            action=OrchestratorAction.RUN_TOOLS,
            pending_tool_calls=pending_calls,
            diagnostics=diagnostics,
        )

    if turn.assistant_message is None:
        diagnostics["reason"] = "Awaiting assistant message after tool execution or direct response path."
        return OrchestratorDecision(action=OrchestratorAction.NEEDS_ASSISTANT_MESSAGE, diagnostics=diagnostics)

    diagnostics["reason"] = "Assistant message present and no pending tool executions."
    return OrchestratorDecision(action=OrchestratorAction.FINALIZE_TURN, diagnostics=diagnostics)


def run_orchestration_cycle(turn: ChatTurn, tool_executor: ToolExecutor) -> ChatTurn:
    """Execute one deterministic orchestration cycle for a chat turn.

    The cycle evaluates turn state, runs any pending tool calls through the
    provided executor, and appends resulting ``ChatToolResult`` entries.

    Args:
        turn: Input turn state to evaluate.
        tool_executor: Callback used to execute one ``ChatToolCall``.

    Returns:
        Updated turn snapshot after one cycle.
    """

    updated_turn = turn.model_copy(deep=True)
    decision = evaluate_turn(updated_turn)

    updated_turn.diagnostics["orchestrator_action"] = decision.action.value
    updated_turn.diagnostics["orchestrator"] = decision.diagnostics

    if decision.action is not OrchestratorAction.RUN_TOOLS:
        return updated_turn

    for call in decision.pending_tool_calls:
        try:
            result = tool_executor(call)
        except Exception as exc:
            result = ChatToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                payload={},
                error_message=str(exc),
            )

        if result.call_id != call.call_id or result.tool_name != call.tool_name:
            result = ChatToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                payload={},
                error_message="Tool executor returned mismatched call identity.",
            )

        updated_turn.tool_results.append(result)

    return updated_turn
