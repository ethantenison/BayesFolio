from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from bayesfolio.contracts.commands.backtest import WalkForwardBacktestCommand, WalkForwardPeriod
from bayesfolio.contracts.results.monthly_portfolio import MonthlyPortfolioResult
from bayesfolio.contracts.results.optimize import OptimizeResult
from bayesfolio.engine.backtest.runner import (
    build_walk_forward_period,
    run_walk_forward_backtest,
    run_weighted_backtest,
)


def test_run_weighted_backtest_includes_new_metrics_and_non_positive_drawdown() -> None:
    realized_returns = pd.DataFrame(
        {
            "SPY": [0.02, -0.01, 0.015, -0.02, 0.01],
            "QQQ": [0.01, -0.015, 0.02, -0.01, 0.005],
        }
    )
    optimization = OptimizeResult(asset_order=["SPY", "QQQ"], weights=[0.6, 0.4])

    result = run_weighted_backtest(realized_returns=realized_returns, optimization=optimization, periods_per_year=12)

    assert isinstance(result.max_drawdown, float)
    assert isinstance(result.calmar_ratio, float)
    assert isinstance(result.sortino_ratio, float)
    assert result.max_drawdown <= 0.0


def test_run_walk_forward_backtest_tracks_alignment_turnover_and_costs() -> None:
    command = WalkForwardBacktestCommand(
        strategy_id="accepted_monthly_portfolio_v1",
        starting_value=1_000.0,
        transaction_cost_bps=10.0,
        periods=[
            WalkForwardPeriod(
                strategy_id="accepted_monthly_portfolio_v1",
                source_run_label="january-2026",
                decision_date=date(2026, 1, 31),
                realization_date=date(2026, 2, 28),
                weights={"SPY": 0.5, "BND": 0.5},
                realized_returns={"SPY": 0.10, "BND": 0.0},
            ),
            WalkForwardPeriod(
                strategy_id="accepted_monthly_portfolio_v1",
                source_run_label="february-2026",
                decision_date=date(2026, 2, 28),
                realization_date=date(2026, 3, 31),
                weights={"SPY": 0.75, "BND": 0.25},
                realized_returns={"SPY": -0.02, "BND": 0.04},
            ),
        ],
    )

    result = run_walk_forward_backtest(command)

    assert result.strategy_id == command.strategy_id
    assert result.periods[0].gross_return == pytest.approx(0.05)
    assert result.periods[0].turnover == pytest.approx(1.0)
    assert result.periods[0].transaction_cost_rate == pytest.approx(0.001)
    assert result.periods[0].net_return == pytest.approx(0.049)
    assert result.periods[0].ending_value == pytest.approx(1_049.0)
    assert result.periods[1].gross_return == pytest.approx(-0.005)
    assert result.periods[1].turnover == pytest.approx(0.25)
    assert result.periods[1].transaction_cost_rate == pytest.approx(0.00025)
    assert result.periods[1].net_return == pytest.approx(-0.00525)
    assert result.terminal_value == pytest.approx(1_043.49275)
    assert result.average_turnover == pytest.approx(0.625)
    assert result.hit_rate == pytest.approx(0.5)


def test_build_walk_forward_period_uses_candidate_date_asset_order_and_weights() -> None:
    candidate = MonthlyPortfolioResult.model_construct(
        strategy_id="strategy-v1",
        as_of_date=date(2026, 1, 31),
        run_label="january-2026",
        optimize_result=OptimizeResult(asset_order=["SPY", "BND"], weights=[0.6, 0.4]),
    )

    period = build_walk_forward_period(
        candidate,
        realization_date=date(2026, 2, 28),
        realized_returns={"SPY": 0.02, "BND": 0.01},
    )

    assert period.decision_date == candidate.as_of_date
    assert period.strategy_id == candidate.strategy_id
    assert period.source_run_label == candidate.run_label
    assert period.weights == {"SPY": 0.6, "BND": 0.4}
    assert period.realized_returns == {"SPY": 0.02, "BND": 0.01}
