"""Results for realized portfolio and benchmark performance tracking."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from bayesfolio.contracts.base import ContractModel, SchemaName, VersionedContract


class PerformancePeriodResult(ContractModel):
    """One measured period with active and cumulative performance.

    Attributes:
        start_date: First date of the measurement period.
        end_date: Last date of the measurement period.
        portfolio_return: Net portfolio return as a decimal.
        benchmark_return: Benchmark return as a decimal.
        active_return: Portfolio minus benchmark period return.
        cumulative_portfolio_return: Compounded portfolio return through period.
        cumulative_benchmark_return: Compounded benchmark return through period.
        realized_gain_loss: Realized gain or loss amount during the period.
        unrealized_gain_loss_change: Change in unrealized gain or loss.
        fees: Fees charged during the period.
    """

    start_date: date
    end_date: date
    portfolio_return: float
    benchmark_return: float
    active_return: float
    cumulative_portfolio_return: float
    cumulative_benchmark_return: float
    realized_gain_loss: float
    unrealized_gain_loss_change: float
    fees: float


class PortfolioPerformanceResult(VersionedContract):
    """Benchmark-relative realized performance and P/L summary.

    Jensen alpha here is a descriptive single-benchmark estimate, not proof of
    causal skill or future outperformance.

    Attributes:
        schema: Stable schema identifier.
        schema_version: Semantic version of this contract.
        account_reference: Opaque non-sensitive account reference.
        strategy_id: Measured strategy identifier.
        benchmark_id: Comparison benchmark identifier.
        currency: Currency of gain, loss, and fee amounts.
        return_measurement_method: Cash-flow-aware measurement method reported
            by the provider.
        periods: Period-level realized performance.
        period_count: Number of measured periods.
        cumulative_portfolio_return: Compounded portfolio return.
        cumulative_benchmark_return: Compounded benchmark return.
        annualized_portfolio_return: Annualized portfolio return.
        annualized_benchmark_return: Annualized benchmark return.
        annualized_excess_return: Difference of annualized returns.
        annualized_tracking_error: Annualized standard deviation of active returns.
        information_ratio: Annualized mean active return divided by tracking
            error, or ``None`` when tracking error is zero.
        beta: Single-benchmark realized beta, or ``None`` when unidentified.
        annualized_jensen_alpha: Arithmetic annualized Jensen alpha estimate, or
            ``None`` when benchmark variance is zero or fewer than two periods exist.
        total_realized_gain_loss: Sum of realized gain/loss amounts.
        total_unrealized_gain_loss_change: Sum of unrealized gain/loss changes.
        total_fees: Sum of reported fees.
    """

    schema: Literal[SchemaName.PERFORMANCE_TRACKING_RESULT] = SchemaName.PERFORMANCE_TRACKING_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    account_reference: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    return_measurement_method: str = Field(min_length=1)
    periods: list[PerformancePeriodResult] = Field(min_length=1)
    period_count: int = Field(ge=1)
    cumulative_portfolio_return: float
    cumulative_benchmark_return: float
    annualized_portfolio_return: float
    annualized_benchmark_return: float
    annualized_excess_return: float
    annualized_tracking_error: float
    information_ratio: float | None
    beta: float | None
    annualized_jensen_alpha: float | None
    total_realized_gain_loss: float
    total_unrealized_gain_loss_change: float
    total_fees: float

    @model_validator(mode="after")
    def validate_period_count(self) -> "PortfolioPerformanceResult":
        """Keep the summary count aligned with serialized period evidence."""

        if self.period_count != len(self.periods):
            raise ValueError("period_count must equal the number of period results")
        return self
