from __future__ import annotations

from collections.abc import Callable, Mapping

from bayesfolio.contracts.chat.protocol import ChatToolCall, ChatToolResult

ToolHandler = Callable[[dict[str, object]], dict[str, object]]


class RegistryToolExecutor:
    """Registry-backed tool executor for chat tool contracts.

    This adapter keeps runtime-specific tool implementations in IO while exposing
    a simple callable compatible with ``run_orchestration_cycle``.

    Attributes:
        _handlers: Mapping of tool names to handler callables.
    """

    def __init__(self, handlers: Mapping[str, ToolHandler] | None = None) -> None:
        """Initialize the executor with optional pre-registered handlers.

        Args:
            handlers: Optional mapping of tool name to callable that accepts a
                tool argument payload and returns a payload dictionary.
        """

        self._handlers: dict[str, ToolHandler] = dict(handlers or {})

    def register(self, tool_name: str, handler: ToolHandler) -> None:
        """Register or replace a tool handler.

        Args:
            tool_name: Logical tool name used by ``ChatToolCall.tool_name``.
            handler: Callable that receives ``arguments`` and returns a payload
                dictionary for ``ChatToolResult.payload``.
        """

        self._handlers[tool_name] = handler

    def has_tool(self, tool_name: str) -> bool:
        """Return whether the executor can handle a given tool name.

        Args:
            tool_name: Logical tool name.

        Returns:
            ``True`` when a handler is registered; otherwise ``False``.
        """

        return tool_name in self._handlers

    def __call__(self, call: ChatToolCall) -> ChatToolResult:
        """Execute one tool call and return a contract result.

        Args:
            call: Tool call contract containing tool name and arguments.

        Returns:
            ``ChatToolResult`` marked success or failure with diagnostic message.
        """

        handler = self._handlers.get(call.tool_name)
        if handler is None:
            return ChatToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                payload={},
                error_message=f"No registered tool handler for '{call.tool_name}'.",
            )

        try:
            payload = handler(call.arguments)
        except Exception as exc:
            return ChatToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                payload={},
                error_message=str(exc),
            )

        if not isinstance(payload, dict):
            return ChatToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                payload={},
                error_message="Tool handler must return dict[str, object] payload.",
            )

        return ChatToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            payload=payload,
        )
