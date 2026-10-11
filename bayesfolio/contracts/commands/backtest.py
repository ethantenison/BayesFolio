from __future__ import annotations

import math
from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from bayesfolio.contracts.base import ContractModel, SchemaName, VersionedContract
from bayesfolio.contracts.results.optimize import OptimizeResult


class BacktestCommand(VersionedContract):
    """Command to run a portfolio backtest given optimization weights."""

    schema: Literal[SchemaName.BACKTEST_COMMAND] = SchemaName.BACKTEST_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    optimize_result: OptimizeResult
    start_date: str
    end_date: str


class WalkForwardPeriod(ContractModel):
    """One strictly out-of-sample portfolio evaluation period.

    Attributes:
        strategy_id: Stable strategy identifier used to create the weights.
        source_run_label: Historical monthly run that produced the weights.
        decision_date: Date on which the portfolio weights were fixed.
        realization_date: Later date on which the subsequent return became
            observable.
        weights: Long-only decimal portfolio weights keyed by asset.
        realized_returns: Subsequent decimal returns keyed by the same assets.
    """

    strategy_id: str = Field(min_length=1)
    source_run_label: str = Field(min_length=1)
    decision_date: date
    realization_date: date
    weights: dict[str, float] = Field(min_length=1)
    realized_returns: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_period(self) -> "WalkForwardPeriod":
        """Reject ambiguous alignment, invalid returns, and invalid weights."""

        if self.realization_date <= self.decision_date:
            raise ValueError("realization_date must be later than decision_date")
        if set(self.weights) != set(self.realized_returns):
            raise ValueError("weights and realized_returns must contain the same assets")
        if any(not math.isfinite(value) or value < 0.0 for value in self.weights.values()):
            raise ValueError("weights must be finite and non-negative")
        if not math.isclose(sum(self.weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("weights must sum to 1.0 within 1e-8")
        if any(not math.isfinite(value) or value <= -1.0 for value in self.realized_returns.values()):
            raise ValueError("realized returns must be finite and greater than -1.0")
        return self


class WalkForwardBacktestCommand(VersionedContract):
    """Configuration and aligned observations for walk-forward evaluation.

    Attributes:
        schema: Stable schema identifier.
        schema_version: Semantic version of this contract.
        strategy_id: Stable identifier shared with the monthly candidate being
            evaluated.
        periods: Chronologically ordered out-of-sample portfolio periods.
        periods_per_year: Annualization factor for the rebalance cadence.
        starting_value: Initial hypothetical portfolio value.
        transaction_cost_bps: One-way transaction cost in basis points applied
            to measured turnover.
        charge_initial_trade: Whether first-period deployment from cash incurs
            turnover and transaction cost.
    """

    schema: Literal[SchemaName.WALK_FORWARD_BACKTEST_COMMAND] = SchemaName.WALK_FORWARD_BACKTEST_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    strategy_id: str = Field(min_length=1)
    periods: list[WalkForwardPeriod] = Field(min_length=1)
    periods_per_year: float = Field(default=12.0, gt=0.0)
    starting_value: float = Field(default=10_000.0, gt=0.0)
    transaction_cost_bps: float = Field(default=0.0, ge=0.0, le=10_000.0)
    charge_initial_trade: bool = True

    @model_validator(mode="after")
    def validate_period_order(self) -> "WalkForwardBacktestCommand":
        """Require deterministic chronological evaluation order."""

        decision_dates = [period.decision_date for period in self.periods]
        realization_dates = [period.realization_date for period in self.periods]
        mismatched_periods = [
            period.source_run_label for period in self.periods if period.strategy_id != self.strategy_id
        ]
        if mismatched_periods:
            raise ValueError(
                f"every period strategy_id must match the backtest strategy_id; mismatched_runs={mismatched_periods}"
            )
        if decision_dates != sorted(decision_dates) or len(set(decision_dates)) != len(decision_dates):
            raise ValueError("period decision dates must be unique and increasing")
        if realization_dates != sorted(realization_dates) or len(set(realization_dates)) != len(realization_dates):
            raise ValueError("period realization dates must be unique and increasing")
        for previous, current in zip(self.periods, self.periods[1:], strict=False):
            if current.decision_date < previous.realization_date:
                raise ValueError("walk-forward periods must not overlap")
        return self
