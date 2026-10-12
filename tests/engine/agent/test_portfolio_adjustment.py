"""Pure engine tests for constrained LLM portfolio adjustments."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest

from bayesfolio.contracts.commands.portfolio_adjustment import (
    AdjustmentModelConfig,
    AssetForecastEvidence,
    GeneratePortfolioAdjustmentCommand,
    PortfolioRiskEvidence,
    ResearchEvidence,
    RetrievalRecord,
)
from bayesfolio.engine.agent.portfolio_adjustment import (
    PortfolioAdjustmentError,
    blend_portfolio_weights,
    build_adjustment_user_prompt,
    generate_portfolio_adjustment,
)


class _QueuedModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        seed: int,
        max_output_tokens: int,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "seed": seed,
                "max_output_tokens": max_output_tokens,
            }
        )
        return self.responses.pop(0)


def _command(*, max_validation_attempts: int = 2) -> GeneratePortfolioAdjustmentCommand:
    timestamp = datetime(2026, 10, 11, 16, 0, tzinfo=UTC)
    required_tools = [
        "schwab_account_snapshot",
        "federal_reserve_releases",
        "bls_releases",
        "treasury_yield_curve",
        "schwab_market_commentary",
        "financial_news_search",
    ]
    return GeneratePortfolioAdjustmentCommand(
        strategy_id="accepted_monthly_portfolio_v1",
        as_of_date=date(2026, 9, 30),
        decision_at=timestamp,
        current_weights={"BND": 0.4, "SPY": 0.6},
        candidate_weights={"BND": 0.2, "SPY": 0.8},
        forecasts=[
            AssetForecastEvidence(asset="BND", expected_excess_return=-0.002, predictive_std=0.015),
            AssetForecastEvidence(asset="SPY", expected_excess_return=0.006, predictive_std=0.04),
        ],
        portfolio_risk=[
            PortfolioRiskEvidence(
                label=f"blend_{blend:.2f}",
                blend_fraction=blend,
                expected_return=0.002 + 0.001 * blend,
                volatility=0.03 + 0.002 * blend,
                probability_negative=0.47 - 0.01 * blend,
                cvar_05=-0.06 - 0.002 * blend,
            )
            for blend in (0.0, 0.25, 0.5, 0.75, 1.0)
        ],
        evidence=[
            ResearchEvidence(
                evidence_id="fed-2026-09",
                source_type="central_bank",
                source_name="Federal Reserve",
                title="September policy statement",
                url="https://example.test/fed",
                published_at=datetime(2026, 9, 16, 18, 0, tzinfo=UTC),
                summary="Rates fell while inflation remained elevated.",
                affected_assets=["BND", "SPY"],
            )
        ],
        retrievals=[
            RetrievalRecord(
                tool_name=tool_name,
                arguments={"as_of": "2026-10-11"},
                requested_at=timestamp,
                completed_at=timestamp,
                success=True,
                evidence_ids=["fed-2026-09"] if tool_name == "federal_reserve_releases" else [],
            )
            for tool_name in required_tools
        ],
        account_tax_treatment="tax_deferred_ira",
        model_config_input=AdjustmentModelConfig(
            provider="local_qwen",
            model="/models/Qwen3.8-27B-4bit",
            max_validation_attempts=max_validation_attempts,
        ),
    )


def _valid_response(*, blend_fraction: float = 0.5) -> str:
    return json.dumps(
        {
            "selected_blend_fraction": blend_fraction,
            "timing": "rebalance_now",
            "confidence": "medium",
            "decision_summary": "Take a partial step toward the optimizer.",
            "model_signal_assessment": "Expected returns are small relative to predictive uncertainty.",
            "uncertainty_assessment": "The full target has worse posterior tail risk.",
            "evidence_assessments": [
                {
                    "evidence_id": "fed-2026-09",
                    "direction": "mixed",
                    "materiality": "high",
                    "affected_assets": ["BND", "SPY"],
                    "rationale": "Lower policy rates and elevated inflation pull in opposite directions.",
                }
            ],
            "cited_evidence_ids": ["fed-2026-09"],
            "key_risks": ["Forecast signal may not persist."],
            "defer_until": None,
        }
    )


def test_generate_adjustment_applies_only_the_selected_deterministic_blend() -> None:
    command = _command()
    model = _QueuedModel([_valid_response(blend_fraction=0.5)])

    result = generate_portfolio_adjustment(command, model)

    assert result.adjusted_weights == pytest.approx({"BND": 0.3, "SPY": 0.7})
    assert result.one_way_turnover == pytest.approx(0.1)
    assert result.decision.selected_blend_fraction == 0.5
    assert result.validation_attempts == 1
    assert result.retrieval_tool_names == command.policy.required_retrievals
    assert len(result.input_hash) == 64
    assert len(result.prompt_hash) == 64
    assert len(result.response_hash) == 64


def test_generate_adjustment_repairs_invalid_out_of_policy_blend() -> None:
    model = _QueuedModel([_valid_response(blend_fraction=0.6), _valid_response(blend_fraction=0.25)])

    result = generate_portfolio_adjustment(_command(), model)

    assert result.decision.selected_blend_fraction == 0.25
    assert result.validation_attempts == 2
    assert "failed validation" in str(model.calls[1]["user_prompt"])


def test_generate_adjustment_fails_closed_after_validation_budget() -> None:
    model = _QueuedModel([_valid_response(blend_fraction=0.6)])

    with pytest.raises(PortfolioAdjustmentError, match="failed structured"):
        generate_portfolio_adjustment(_command(max_validation_attempts=1), model)


def test_prompt_and_blend_are_deterministic() -> None:
    command = _command()

    assert build_adjustment_user_prompt(command) == build_adjustment_user_prompt(command)
    assert blend_portfolio_weights(
        current_weights={"BND": 0.4, "SPY": 0.6},
        candidate_weights={"BND": 0.2, "SPY": 0.8},
        blend_fraction=0.5,
    ) == pytest.approx({"BND": 0.3, "SPY": 0.7})
