from __future__ import annotations

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class BacktestResult(VersionedContract):
    """Backtest summary with key portfolio performance metrics.

    Attributes:
        cumulative_return: Total return as decimal (0.10 = 10%).
        annualized_return: Annualized return as decimal (0.10 = 10%).
        annualized_volatility: Annualized volatility as decimal (0.15 = 15%).
        sharpe_ratio: Sharpe ratio (dimensionless).
    """

    schema: SchemaName = Field(default=SchemaName.BACKTEST_RESULT, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
