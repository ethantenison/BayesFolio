"""Tests for explicit walk-forward backtest contracts."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from bayesfolio.contracts.commands.backtest import WalkForwardBacktestCommand, WalkForwardPeriod


def _period(**overrides: object) -> WalkForwardPeriod:
    values = {
        "strategy_id": "strategy-v1",
        "source_run_label": "january-2026",
        "decision_date": date(2026, 1, 31),
        "realization_date": date(2026, 2, 28),
        "weights": {"SPY": 0.6, "BND": 0.4},
        "realized_returns": {"SPY": 0.02, "BND": 0.01},
        **overrides,
    }
    return WalkForwardPeriod.model_validate(values)


def test_walk_forward_period_requires_later_realization_and_exact_assets() -> None:
    with pytest.raises(ValidationError, match="realization_date must be later"):
        _period(realization_date=date(2026, 1, 31))

    with pytest.raises(ValidationError, match="same assets"):
        _period(realized_returns={"SPY": 0.02})


def test_walk_forward_period_requires_long_only_fully_invested_weights() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        _period(weights={"SPY": 1.1, "BND": -0.1})

    with pytest.raises(ValidationError, match="sum to 1.0"):
        _period(weights={"SPY": 0.5, "BND": 0.4})


def test_walk_forward_command_requires_unique_increasing_periods() -> None:
    later = _period(
        decision_date=date(2026, 2, 28),
        realization_date=date(2026, 3, 31),
    )

    with pytest.raises(ValidationError, match="decision dates must be unique and increasing"):
        WalkForwardBacktestCommand(
            strategy_id="strategy-v1",
            periods=[later, _period()],
        )


def test_walk_forward_command_rejects_overlapping_periods() -> None:
    first = _period(realization_date=date(2026, 3, 31))
    overlapping = _period(
        decision_date=date(2026, 2, 28),
        realization_date=date(2026, 4, 30),
    )

    with pytest.raises(ValidationError, match="must not overlap"):
        WalkForwardBacktestCommand(
            strategy_id="strategy-v1",
            periods=[first, overlapping],
        )


def test_walk_forward_command_rejects_period_from_another_strategy() -> None:
    with pytest.raises(ValidationError, match="period strategy_id must match"):
        WalkForwardBacktestCommand(
            strategy_id="strategy-v1",
            periods=[_period(strategy_id="other-strategy")],
        )
