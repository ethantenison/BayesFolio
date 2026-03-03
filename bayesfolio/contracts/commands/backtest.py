from __future__ import annotations

from typing import Literal

from bayesfolio.contracts.base import SchemaName, VersionedContract
from bayesfolio.contracts.results.optimize import OptimizeResult


class BacktestCommand(VersionedContract):
    """Command to run a portfolio backtest given optimization weights."""

    schema: Literal[SchemaName.BACKTEST_COMMAND] = SchemaName.BACKTEST_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    optimize_result: OptimizeResult
    start_date: str
    end_date: str
