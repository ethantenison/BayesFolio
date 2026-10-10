from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class BacktestResult(VersionedContract):
    """Backtest summary with key portfolio performance metrics.

    Attributes:
        schema: Stable schema identifier.
        schema_version: Semantic version of this contract.
        cumulative_return: Total return as decimal (0.10 = 10%).
        annualized_return: Compound annual growth rate (CAGR) as decimal (0.10 = 10%).
        annualized_volatility: Annualized volatility as decimal (0.15 = 15%).
        sharpe_ratio: Sharpe ratio (dimensionless).
        max_drawdown: Maximum drawdown as decimal (negative or zero).
        calmar_ratio: Calmar ratio (dimensionless).
        sortino_ratio: Sortino ratio (dimensionless).
    """

    schema: Literal[SchemaName.BACKTEST_RESULT] = SchemaName.BACKTEST_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0


class WalkForwardPeriodResult(VersionedContract):
    """Auditable realized outcome for one walk-forward period.

    Attributes:
        schema: Stable schema identifier.
        schema_version: Semantic version of this contract.
        source_run_label: Historical monthly run that produced the weights.
        decision_date: Date on which weights were fixed.
        realization_date: Date on which the subsequent return was observed.
        gross_return: Portfolio return before transaction costs.
        turnover: One-way portfolio turnover as a decimal.
        transaction_cost_rate: Return drag applied for transaction costs.
        transaction_cost_amount: Hypothetical currency cost at period start.
        net_return: Portfolio return after transaction costs.
        ending_value: Hypothetical portfolio value after the period.
    """

    schema: Literal[SchemaName.WALK_FORWARD_PERIOD_RESULT] = SchemaName.WALK_FORWARD_PERIOD_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    source_run_label: str
    decision_date: date
    realization_date: date
    gross_return: float
    turnover: float
    transaction_cost_rate: float
    transaction_cost_amount: float
    net_return: float
    ending_value: float


class WalkForwardBacktestResult(VersionedContract):
    """Walk-forward performance summary with period-level audit evidence.

    Attributes:
        schema: Stable schema identifier.
        schema_version: Semantic version of this contract.
        strategy_id: Stable identifier for the evaluated monthly strategy.
        summary: Standard return, volatility, and drawdown metrics.
        periods: Chronological period-level realized outcomes.
        periods_per_year: Annualization factor used for summary metrics.
        starting_value: Initial hypothetical portfolio value.
        terminal_value: Ending hypothetical portfolio value.
        transaction_cost_bps: Cost assumption applied to turnover.
        average_turnover: Mean one-way turnover across periods.
        total_transaction_cost_amount: Sum of hypothetical currency costs.
        hit_rate: Fraction of periods with positive net returns.
    """

    schema: Literal[SchemaName.WALK_FORWARD_BACKTEST_RESULT] = SchemaName.WALK_FORWARD_BACKTEST_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    strategy_id: str = Field(min_length=1)
    summary: BacktestResult
    periods: list[WalkForwardPeriodResult] = Field(min_length=1)
    periods_per_year: float
    starting_value: float
    terminal_value: float
    transaction_cost_bps: float
    average_turnover: float
    total_transaction_cost_amount: float
    hit_rate: float
