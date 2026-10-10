"""Tests for brokerage-neutral, non-executing rebalance planning."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from bayesfolio.contracts.commands.backtest import WalkForwardBacktestCommand, WalkForwardPeriod
from bayesfolio.contracts.commands.brokerage import BuildRebalancePlanCommand
from bayesfolio.contracts.results.brokerage import BrokerageAccountSnapshot, BrokeragePosition
from bayesfolio.contracts.results.monthly_decision import MonthlyPortfolioDecisionResult
from bayesfolio.contracts.results.optimize import OptimizeResult
from bayesfolio.engine.backtest import run_walk_forward_backtest
from bayesfolio.engine.brokerage import build_rebalance_plan


def _decision(*, approved: bool = True) -> MonthlyPortfolioDecisionResult:
    candidate = OptimizeResult(asset_order=["SPY", "BND"], weights=[0.5, 0.5])
    period_values = {
        "source_run_label": "february-2026-history",
        "decision_date": date(2026, 2, 28),
        "realization_date": date(2026, 3, 31),
        "weights": {"SPY": 0.5, "BND": 0.5},
        "realized_returns": {"SPY": 0.01, "BND": 0.005},
    }
    backtest = run_walk_forward_backtest(
        WalkForwardBacktestCommand(
            strategy_id="strategy-v1",
            periods=[WalkForwardPeriod(strategy_id="strategy-v1", **period_values)],
        )
    )
    benchmark = run_walk_forward_backtest(
        WalkForwardBacktestCommand(
            strategy_id="benchmark-v1",
            periods=[WalkForwardPeriod(strategy_id="benchmark-v1", **period_values)],
        )
    )
    return MonthlyPortfolioDecisionResult(
        strategy_id="strategy-v1",
        candidate_run_label="april-2026-candidate",
        as_of_date=date(2026, 4, 30),
        decision="approved" if approved else "rejected",
        reviewer="portfolio-owner",
        reviewed_at=datetime(2026, 4, 30, 15, 0, tzinfo=UTC),
        rationale="Reviewed evidence.",
        backtest=backtest,
        benchmarks=[benchmark] if approved else [],
        candidate_weights=candidate,
        finalized_weights=candidate if approved else None,
    )


def _account(*, as_of: datetime = datetime(2026, 5, 1, 15, 0, tzinfo=UTC)) -> BrokerageAccountSnapshot:
    return BrokerageAccountSnapshot(
        account_reference="ira-primary",
        as_of=as_of,
        net_liquidation_value=Decimal("10000"),
        cash_value=Decimal("1000"),
        positions=[
            BrokeragePosition(
                asset="SPY",
                quantity=Decimal("10"),
                market_price=Decimal("600"),
                market_value=Decimal("6000"),
            ),
            BrokeragePosition(
                asset="BND",
                quantity=Decimal("30"),
                market_price=Decimal("100"),
                market_value=Decimal("3000"),
            ),
        ],
    )


def test_build_rebalance_plan_uses_final_weights_and_current_account_value() -> None:
    result = build_rebalance_plan(
        BuildRebalancePlanCommand(
            plan_id="may-2026-rebalance",
            decision=_decision(),
            account=_account(),
        )
    )

    by_asset = {instruction.asset: instruction for instruction in result.instructions}
    assert result.account_reference == "ira-primary"
    assert result.decision_reviewed_at == _decision().reviewed_at
    assert by_asset["SPY"].action == "sell"
    assert by_asset["SPY"].trade_notional == Decimal("1000.0")
    assert by_asset["BND"].action == "buy"
    assert by_asset["BND"].trade_notional == Decimal("2000.0")
    assert result.estimated_cash_after_trades == Decimal("0.0")
    assert result.executable is False
    assert "no brokerage orders" in result.diagnostics[0]


def test_rebalance_threshold_cannot_create_an_unfunded_plan() -> None:
    with pytest.raises(ValueError, match="Planned buys exceed available cash"):
        build_rebalance_plan(
            BuildRebalancePlanCommand(
                plan_id="may-2026-threshold",
                decision=_decision(),
                account=_account(),
                min_trade_notional=Decimal("1500"),
            )
        )


def test_rebalance_plan_sells_existing_asset_outside_final_weights() -> None:
    account = BrokerageAccountSnapshot(
        account_reference="ira-primary",
        as_of=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
        net_liquidation_value=Decimal("10000"),
        cash_value=Decimal("1000"),
        positions=[
            BrokeragePosition(
                asset="SPY",
                quantity=Decimal("10"),
                market_price=Decimal("500"),
                market_value=Decimal("5000"),
            ),
            BrokeragePosition(
                asset="VTI",
                quantity=Decimal("10"),
                market_price=Decimal("400"),
                market_value=Decimal("4000"),
            ),
        ],
    )

    result = build_rebalance_plan(
        BuildRebalancePlanCommand(
            plan_id="may-2026-exit-vti",
            decision=_decision(),
            account=account,
        )
    )

    by_asset = {instruction.asset: instruction for instruction in result.instructions}
    assert by_asset["VTI"].action == "sell"
    assert by_asset["VTI"].target_weight == Decimal("0")
    assert by_asset["VTI"].trade_notional == Decimal("4000")
    assert by_asset["BND"].action == "buy"
    assert by_asset["BND"].trade_notional == Decimal("5000.0")
    assert result.estimated_cash_after_trades == Decimal("0.0")


def test_rebalance_plan_rejects_rejected_decision_and_stale_snapshot() -> None:
    with pytest.raises(ValueError, match="requires an approved decision"):
        build_rebalance_plan(
            BuildRebalancePlanCommand(
                plan_id="rejected",
                decision=_decision(approved=False),
                account=_account(),
            )
        )

    with pytest.raises(ValueError, match="at or after portfolio review"):
        build_rebalance_plan(
            BuildRebalancePlanCommand(
                plan_id="stale",
                decision=_decision(),
                account=_account(as_of=datetime(2026, 4, 30, 14, 0, tzinfo=UTC)),
            )
        )


def test_account_snapshot_requires_reconciled_unique_positions() -> None:
    with pytest.raises(ValueError, match="reconcile"):
        BrokerageAccountSnapshot(
            account_reference="ira-primary",
            as_of=datetime(2026, 5, 1, tzinfo=UTC),
            net_liquidation_value=Decimal("10000"),
            cash_value=Decimal("1000"),
            positions=[],
        )
