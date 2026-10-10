"""Commands for realized portfolio and benchmark performance tracking."""

from __future__ import annotations

import math
from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from bayesfolio.contracts.base import ContractModel, SchemaName, VersionedContract


class PerformancePeriod(ContractModel):
    """One externally measured portfolio-performance period.

    Returns must already account for external cash flows using the measurement
    method chosen by the account data provider. BayesFolio does not infer
    time-weighted returns from cash-flow timing.

    Attributes:
        start_date: First date of the measurement period.
        end_date: Later date of the measurement period.
        portfolio_return: Net portfolio return as a decimal.
        benchmark_return: Comparable benchmark return as a decimal.
        risk_free_return: Comparable risk-free return as a decimal.
        realized_gain_loss: Realized gain or loss amount during the period.
        unrealized_gain_loss_change: Change in unrealized gain or loss.
        fees: Fees charged during the period as a non-negative amount.
    """

    start_date: date
    end_date: date
    portfolio_return: float
    benchmark_return: float
    risk_free_return: float = 0.0
    realized_gain_loss: float = 0.0
    unrealized_gain_loss_change: float = 0.0
    fees: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_period(self) -> "PerformancePeriod":
        """Reject invalid dates, non-finite values, and total losses below -100%."""

        if self.end_date <= self.start_date:
            raise ValueError("performance end_date must be later than start_date")
        return_values = (self.portfolio_return, self.benchmark_return, self.risk_free_return)
        if any(not math.isfinite(value) or value <= -1.0 for value in return_values):
            raise ValueError("performance returns must be finite and greater than -1.0")
        amount_values = (self.realized_gain_loss, self.unrealized_gain_loss_change, self.fees)
        if any(not math.isfinite(value) for value in amount_values):
            raise ValueError("gain, loss, and fee amounts must be finite")
        return self


class TrackPortfolioPerformanceCommand(VersionedContract):
    """Configure benchmark-relative realized performance measurement.

    Attributes:
        schema: Stable schema identifier.
        schema_version: Semantic version of this contract.
        account_reference: Opaque non-sensitive account reference.
        strategy_id: Strategy whose live performance is being measured.
        benchmark_id: Benchmark used for active return and alpha estimates.
        currency: Currency of gain, loss, and fee amounts.
        return_measurement_method: Provider method used to account for external
            cash-flow timing.
        periods: Unique, ordered, non-overlapping measured periods.
        periods_per_year: Annualization factor for the measurement cadence.
    """

    schema: Literal[SchemaName.PERFORMANCE_TRACKING_COMMAND] = SchemaName.PERFORMANCE_TRACKING_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    account_reference: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    return_measurement_method: str = Field(min_length=1)
    periods: list[PerformancePeriod] = Field(min_length=1)
    periods_per_year: float = Field(default=12.0, gt=0.0)

    @model_validator(mode="after")
    def validate_period_order(self) -> "TrackPortfolioPerformanceCommand":
        """Require deterministic chronological periods without overlap."""

        starts = [period.start_date for period in self.periods]
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            raise ValueError("performance periods must have unique increasing start dates")
        for previous, current in zip(self.periods, self.periods[1:], strict=False):
            if current.start_date < previous.end_date:
                raise ValueError("performance periods must not overlap")
        return self
