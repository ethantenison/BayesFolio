from __future__ import annotations

from typing import Literal

from bayesfolio.contracts.base import SchemaName, VersionedContract


class BacktestResult(VersionedContract):
    """Backtest summary with key portfolio performance metrics.

    Attributes:
        cumulative_return: Total return as decimal (0.10 = 10%).
        annualized_return: Annualized return as decimal (0.10 = 10%).
        annualized_volatility: Annualized volatility as decimal (0.15 = 15%).
        sharpe_ratio: Sharpe ratio (dimensionless).
    """

    schema: Literal[SchemaName.BACKTEST_RESULT] = SchemaName.BACKTEST_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
