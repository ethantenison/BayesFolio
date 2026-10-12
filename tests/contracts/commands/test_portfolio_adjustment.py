"""Contract tests for point-in-time portfolio-adjustment commands."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from bayesfolio.contracts.base import SchemaName
from bayesfolio.contracts.commands.portfolio_adjustment import GeneratePortfolioAdjustmentCommand


def _valid_payload() -> dict[str, object]:
    timestamp = datetime(2026, 10, 11, 16, 0, tzinfo=UTC)
    required_tools = [
        "schwab_account_snapshot",
        "federal_reserve_releases",
        "bls_releases",
        "treasury_yield_curve",
        "schwab_market_commentary",
        "financial_news_search",
    ]
    return {
        "strategy_id": "accepted_monthly_portfolio_v1",
        "as_of_date": "2026-09-30",
        "decision_at": timestamp.isoformat(),
        "current_weights": {"BND": 0.4, "SPY": 0.6},
        "candidate_weights": {"BND": 0.2, "SPY": 0.8},
        "forecasts": [
            {"asset": "BND", "expected_excess_return": -0.002, "predictive_std": 0.015},
            {"asset": "SPY", "expected_excess_return": 0.006, "predictive_std": 0.04},
        ],
        "portfolio_risk": [
            {
                "label": f"blend_{blend:.2f}",
                "blend_fraction": blend,
                "expected_return": 0.002 + 0.001 * blend,
                "volatility": 0.03 + 0.002 * blend,
                "probability_negative": 0.47 - 0.01 * blend,
                "cvar_05": -0.06 - 0.002 * blend,
            }
            for blend in (0.0, 0.25, 0.5, 0.75, 1.0)
        ],
        "evidence": [
            {
                "evidence_id": "fed-2026-09",
                "source_type": "central_bank",
                "source_name": "Federal Reserve",
                "title": "September policy statement",
                "url": "https://example.test/fed",
                "published_at": "2026-09-16T18:00:00Z",
                "summary": "Rates fell while inflation remained elevated.",
                "affected_assets": ["BND", "SPY"],
            }
        ],
        "retrievals": [
            {
                "tool_name": tool_name,
                "arguments": {"as_of": "2026-10-11"},
                "requested_at": timestamp.isoformat(),
                "completed_at": timestamp.isoformat(),
                "success": True,
                "evidence_ids": ["fed-2026-09"] if tool_name == "federal_reserve_releases" else [],
            }
            for tool_name in required_tools
        ],
        "account_tax_treatment": "tax_deferred_ira",
        "model_config_input": {
            "provider": "local_qwen",
            "model": "/models/Qwen3.8-27B-4bit",
        },
    }


def test_adjustment_command_validates_schema_and_forbids_extra_fields() -> None:
    command = GeneratePortfolioAdjustmentCommand.model_validate(_valid_payload())

    assert command.schema is SchemaName.PORTFOLIO_ADJUSTMENT_COMMAND
    assert command.schema_version == "0.1.0"
    assert command.policy.allowed_blend_fractions == [0.0, 0.25, 0.5, 0.75, 1.0]

    with pytest.raises(ValidationError, match="extra_forbidden"):
        GeneratePortfolioAdjustmentCommand.model_validate({**_valid_payload(), "unexpected": True})


def test_adjustment_command_rejects_future_evidence() -> None:
    payload = _valid_payload()
    evidence = payload["evidence"]
    assert isinstance(evidence, list)
    assert isinstance(evidence[0], dict)
    evidence[0]["published_at"] = "2026-10-12T00:00:00Z"

    with pytest.raises(ValidationError, match="published after decision_at"):
        GeneratePortfolioAdjustmentCommand.model_validate(payload)


def test_adjustment_command_rejects_changed_retrieval_sequence() -> None:
    payload = _valid_payload()
    retrievals = payload["retrievals"]
    assert isinstance(retrievals, list)
    retrievals[0], retrievals[1] = retrievals[1], retrievals[0]

    with pytest.raises(ValidationError, match="successful retrieval sequence"):
        GeneratePortfolioAdjustmentCommand.model_validate(payload)
