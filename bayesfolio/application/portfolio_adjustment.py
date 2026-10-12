"""Application composition root for structured portfolio adjustment.

This module wires the pure adjustment engine to an injected model adapter and
provides the default local-Qwen transport. It does not retrieve research or
brokerage data; callers must provide the validated point-in-time command.
"""

from __future__ import annotations

from bayesfolio.contracts.commands.portfolio_adjustment import GeneratePortfolioAdjustmentCommand
from bayesfolio.contracts.results.portfolio_adjustment import PortfolioAdjustmentResult
from bayesfolio.engine.agent.portfolio_adjustment import StructuredAdjustmentModel, generate_portfolio_adjustment
from bayesfolio.io.llm.openai_compatible import OpenAICompatibleChatModel

DEFAULT_LOCAL_QWEN_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LOCAL_QWEN_MODEL = "/Users/et/local-llm/models/Qwen3.8-27B-4bit"


def make_local_qwen_adjustment_model(
    *,
    base_url: str = DEFAULT_LOCAL_QWEN_BASE_URL,
    model: str = DEFAULT_LOCAL_QWEN_MODEL,
    timeout_seconds: float = 120.0,
) -> OpenAICompatibleChatModel:
    """Construct the default local Qwen adapter with hidden reasoning disabled.

    Args:
        base_url: Local OpenAI-compatible API base URL.
        model: Local server model identifier or filesystem path.
        timeout_seconds: HTTP timeout in seconds for one generation attempt.

    Returns:
        Configured chat-completions transport suitable for the adjustment engine.
    """

    return OpenAICompatibleChatModel(
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        request_options={"enable_thinking": False},
    )


def run_portfolio_adjustment(
    command: GeneratePortfolioAdjustmentCommand,
    *,
    model: StructuredAdjustmentModel,
) -> PortfolioAdjustmentResult:
    """Run the fixed portfolio-adjustment procedure with an injected model.

    Args:
        command: Complete point-in-time evidence and portfolio packet.
        model: Structured-generation adapter for local or hosted inference.

    Returns:
        Validated adjustment decision and deterministic final weights.
    """

    return generate_portfolio_adjustment(command, model)
