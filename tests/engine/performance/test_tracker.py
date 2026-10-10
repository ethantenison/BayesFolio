"""Tests for realized performance and benchmark-relative measurement."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from bayesfolio.contracts.commands.performance import PerformancePeriod, TrackPortfolioPerformanceCommand
from bayesfolio.engine.performance import track_portfolio_performance


def _period(
    start_date: date,
    end_date: date,
    portfolio_return: float,
    benchmark_return: float,
    *,
    realized_gain_loss: float = 0.0,
    unrealized_gain_loss_change: float = 0.0,
    fees: float = 0.0,
) -> PerformancePeriod:
    return PerformancePeriod(
        start_date=start_date,
        end_date=end_date,
        portfolio_return=portfolio_return,
        benchmark_return=benchmark_return,
        realized_gain_loss=realized_gain_loss,
        unrealized_gain_loss_change=unrealized_gain_loss_change,
        fees=fees,
    )


def test_track_portfolio_performance_reports_active_risk_beta_alpha_and_pl() -> None:
    command = TrackPortfolioPerformanceCommand(
        account_reference="ira-primary",
        strategy_id="strategy-v1",
        benchmark_id="balanced-benchmark-v1",
        return_measurement_method="broker_time_weighted_return",
        periods=[
            _period(
                date(2026, 1, 31),
                date(2026, 2, 28),
                0.02,
                0.01,
                realized_gain_loss=100.0,
                unrealized_gain_loss_change=50.0,
                fees=1.0,
            ),
            _period(
                date(2026, 2, 28),
                date(2026, 3, 31),
                0.01,
                0.005,
                realized_gain_loss=-20.0,
                unrealized_gain_loss_change=25.0,
                fees=2.0,
            ),
            _period(
                date(2026, 3, 31),
                date(2026, 4, 30),
                -0.01,
                -0.005,
                unrealized_gain_loss_change=-40.0,
                fees=1.5,
            ),
        ],
    )

    result = track_portfolio_performance(command)

    assert result.account_reference == "ira-primary"
    assert result.return_measurement_method == "broker_time_weighted_return"
    expected_portfolio = float(np.prod([1.02, 1.01, 0.99]) - 1.0)
    expected_benchmark = float(np.prod([1.01, 1.005, 0.995]) - 1.0)
    assert result.cumulative_portfolio_return == pytest.approx(expected_portfolio)
    assert result.cumulative_benchmark_return == pytest.approx(expected_benchmark)
    assert result.periods[-1].cumulative_portfolio_return == pytest.approx(expected_portfolio)
    assert result.beta == pytest.approx(2.0)
    assert result.annualized_jensen_alpha == pytest.approx(0.0, abs=1e-15)
    assert result.annualized_tracking_error > 0.0
    assert result.information_ratio is not None
    assert result.total_realized_gain_loss == pytest.approx(80.0)
    assert result.total_unrealized_gain_loss_change == pytest.approx(35.0)
    assert result.total_fees == pytest.approx(4.5)


def test_single_period_does_not_claim_beta_or_alpha() -> None:
    result = track_portfolio_performance(
        TrackPortfolioPerformanceCommand(
            account_reference="ira-primary",
            strategy_id="strategy-v1",
            benchmark_id="benchmark-v1",
            return_measurement_method="broker_time_weighted_return",
            periods=[_period(date(2026, 1, 31), date(2026, 2, 28), 0.02, 0.01)],
        )
    )

    assert result.beta is None
    assert result.annualized_jensen_alpha is None
    assert result.information_ratio is None


def test_performance_command_rejects_overlapping_periods() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        TrackPortfolioPerformanceCommand(
            account_reference="ira-primary",
            strategy_id="strategy-v1",
            benchmark_id="benchmark-v1",
            return_measurement_method="broker_time_weighted_return",
            periods=[
                _period(date(2026, 1, 1), date(2026, 2, 15), 0.01, 0.01),
                _period(date(2026, 2, 1), date(2026, 3, 1), 0.01, 0.01),
            ],
        )
