"""Tests for the explicit monthly portfolio review gate."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

import pytest
from pydantic import ValidationError

from bayesfolio.application import finalize_monthly_portfolio
from bayesfolio.contracts.commands.backtest import WalkForwardBacktestCommand, WalkForwardPeriod
from bayesfolio.contracts.commands.monthly_decision import FinalizeMonthlyPortfolioCommand
from bayesfolio.contracts.results.backtest import WalkForwardBacktestResult
from bayesfolio.contracts.results.monthly_portfolio import MonthlyPortfolioResult
from bayesfolio.contracts.results.optimize import OptimizeResult
from bayesfolio.engine.backtest import run_walk_forward_backtest


def _candidate(*, strategy_id: str = "strategy-v1") -> MonthlyPortfolioResult:
    return MonthlyPortfolioResult.model_construct(
        strategy_id=strategy_id,
        as_of_date=date(2026, 4, 30),
        run_label="april-2026-candidate",
        optimize_result=OptimizeResult(asset_order=["SPY", "BND"], weights=[0.6, 0.4]),
    )


def _backtest(
    *,
    strategy_id: str = "strategy-v1",
    realization_date: date = date(2026, 3, 31),
) -> WalkForwardBacktestResult:
    return run_walk_forward_backtest(
        WalkForwardBacktestCommand(
            strategy_id=strategy_id,
            periods=[
                WalkForwardPeriod(
                    strategy_id=strategy_id,
                    source_run_label="february-2026-history",
                    decision_date=date(2026, 2, 28),
                    realization_date=realization_date,
                    weights={"SPY": 0.5, "BND": 0.5},
                    realized_returns={"SPY": 0.02, "BND": 0.01},
                )
            ],
        )
    )


def _command(
    *,
    decision: Literal["approved", "rejected"] = "approved",
    strategy_id: str = "strategy-v1",
    realization_date: date = date(2026, 3, 31),
) -> FinalizeMonthlyPortfolioCommand:
    return FinalizeMonthlyPortfolioCommand(
        candidate=_candidate(),
        backtest=_backtest(strategy_id=strategy_id, realization_date=realization_date),
        benchmarks=[_backtest(strategy_id="equal-weight-v1", realization_date=realization_date)],
        decision=decision,
        reviewer="portfolio-owner",
        reviewed_at=datetime(2026, 4, 30, 15, 0, tzinfo=UTC),
        rationale="Reviewed the recorded walk-forward evidence and constraints.",
    )


def test_approval_releases_candidate_weights_through_application_entry_point() -> None:
    result = finalize_monthly_portfolio(_command())

    assert result.decision == "approved"
    assert result.finalized_weights == _candidate().optimize_result
    assert result.candidate_run_label == "april-2026-candidate"


def test_rejection_records_evidence_without_final_weights() -> None:
    result = finalize_monthly_portfolio(_command(decision="rejected"))

    assert result.decision == "rejected"
    assert result.candidate_weights == _candidate().optimize_result
    assert result.finalized_weights is None


def test_decision_rejects_strategy_mismatch() -> None:
    with pytest.raises(ValueError, match="strategy_id values must match"):
        finalize_monthly_portfolio(_command(strategy_id="different-strategy"))


@pytest.mark.parametrize("realization_date", [date(2026, 4, 30), date(2026, 5, 31)])
def test_decision_rejects_evidence_not_strictly_before_candidate(realization_date: date) -> None:
    with pytest.raises(ValueError, match="strictly before the candidate as_of_date"):
        finalize_monthly_portfolio(_command(realization_date=realization_date))


def test_decision_contract_rejects_review_before_candidate_date() -> None:
    with pytest.raises(ValueError, match="reviewed_at cannot precede"):
        FinalizeMonthlyPortfolioCommand(
            candidate=_candidate(),
            backtest=_backtest(),
            benchmarks=[_backtest(strategy_id="equal-weight-v1")],
            decision="approved",
            reviewer="portfolio-owner",
            reviewed_at=datetime(2026, 4, 29, 15, 0, tzinfo=UTC),
            rationale="Invalid early review.",
        )


def test_approved_decision_requires_a_benchmark() -> None:
    with pytest.raises(ValueError, match="require at least one benchmark"):
        FinalizeMonthlyPortfolioCommand(
            candidate=_candidate(),
            backtest=_backtest(),
            decision="approved",
            reviewer="portfolio-owner",
            reviewed_at=datetime(2026, 4, 30, 15, 0, tzinfo=UTC),
            rationale="Approval without comparative evidence is invalid.",
        )


def test_decision_rejects_mismatched_benchmark_grid() -> None:
    command = FinalizeMonthlyPortfolioCommand(
        candidate=_candidate(),
        backtest=_backtest(),
        benchmarks=[
            _backtest(
                strategy_id="equal-weight-v1",
                realization_date=date(2026, 4, 15),
            )
        ],
        decision="approved",
        reviewer="portfolio-owner",
        reviewed_at=datetime(2026, 4, 30, 15, 0, tzinfo=UTC),
        rationale="The grid mismatch should be rejected.",
    )

    with pytest.raises(ValueError, match="same evaluation grid"):
        finalize_monthly_portfolio(command)


def test_decision_result_rejects_final_weights_on_a_rejection() -> None:
    approved = finalize_monthly_portfolio(_command())
    payload = approved.model_dump(mode="python")
    payload["decision"] = "rejected"

    with pytest.raises(ValidationError, match="rejected decisions cannot contain"):
        type(approved).model_validate(payload)
