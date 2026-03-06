from __future__ import annotations

from typing import cast

from bayesfolio.contracts.chat.protocol import ChatToolCall
from bayesfolio.io.agent_tool_executor import RegistryToolExecutor


def test_registry_tool_executor_runs_registered_tool() -> None:
    executor = RegistryToolExecutor()

    def _handler(arguments: dict[str, object]) -> dict[str, object]:
        return {"echo": arguments.get("value")}

    executor.register("echo", _handler)
    result = executor(ChatToolCall(call_id="call_001", tool_name="echo", arguments={"value": "ok"}))

    assert result.success is True
    assert result.payload == {"echo": "ok"}
    assert result.error_message is None


def test_registry_tool_executor_returns_failure_for_unknown_tool() -> None:
    executor = RegistryToolExecutor()

    result = executor(ChatToolCall(call_id="call_001", tool_name="missing_tool", arguments={}))

    assert result.success is False
    assert "No registered tool handler" in (result.error_message or "")


def test_registry_tool_executor_returns_failure_when_handler_raises() -> None:
    def _failing_handler(arguments: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("tool failed")

    executor = RegistryToolExecutor(handlers={"failing": _failing_handler})
    result = executor(ChatToolCall(call_id="call_001", tool_name="failing", arguments={}))

    assert result.success is False
    assert result.error_message == "tool failed"


def test_registry_tool_executor_returns_failure_when_payload_not_dict() -> None:
    def _bad_handler(arguments: dict[str, object]) -> dict[str, object]:
        return cast(dict[str, object], "not-a-dict")

    executor = RegistryToolExecutor(handlers={"bad": _bad_handler})
    result = executor(ChatToolCall(call_id="call_001", tool_name="bad", arguments={}))

    assert result.success is False
    assert result.error_message == "Tool handler must return dict[str, object] payload."


def test_registry_tool_executor_has_tool_reflects_registration() -> None:
    executor = RegistryToolExecutor()

    assert executor.has_tool("echo") is False
    executor.register("echo", lambda arguments: {"status": "ok"})
    assert executor.has_tool("echo") is True
