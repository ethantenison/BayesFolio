"""Run the versioned LLM portfolio-adjustment procedure from a JSON packet.

The input file must validate as ``GeneratePortfolioAdjustmentCommand`` and must
already contain the fixed retrieval audit trail and point-in-time evidence.
The output manifest stores both command and result for later backtesting.

Usage:
    uv run --locked python scripts/run_portfolio_adjustment.py \
        --input adjustment-command.json --output adjustment-result.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bayesfolio.application.portfolio_adjustment import (
    DEFAULT_LOCAL_QWEN_BASE_URL,
    DEFAULT_LOCAL_QWEN_MODEL,
    run_portfolio_adjustment,
)
from bayesfolio.contracts.commands.portfolio_adjustment import GeneratePortfolioAdjustmentCommand
from bayesfolio.io.llm.openai_compatible import OpenAICompatibleChatModel


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a local or hosted compatible endpoint.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Validated adjustment command JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Destination audit-manifest JSON.")
    parser.add_argument("--base-url", default=DEFAULT_LOCAL_QWEN_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_LOCAL_QWEN_MODEL)
    parser.add_argument("--api-key-env", default=None, help="Optional environment variable containing a bearer token.")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable provider-specific hidden reasoning; disabled by default for repeatability.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the command, call the model, and write an auditable JSON manifest."""

    args = parse_args()
    command = GeneratePortfolioAdjustmentCommand.model_validate_json(args.input.read_text(encoding="utf-8"))
    if command.model_config_input.model != args.model:
        raise ValueError(
            "input model_config_input.model must match --model so the manifest records the endpoint model exactly"
        )
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None
    model = OpenAICompatibleChatModel(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        timeout_seconds=args.timeout_seconds,
        request_options={"enable_thinking": args.enable_thinking},
    )
    result = run_portfolio_adjustment(command, model=model)
    manifest = {
        "command": command.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
