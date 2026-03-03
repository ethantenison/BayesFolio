from __future__ import annotations

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract
from bayesfolio.contracts.results.optimize import OptimizeResult


class BacktestCommand(VersionedContract):
    """Command to run a portfolio backtest given optimization weights."""

    schema: SchemaName = Field(default=SchemaName.BACKTEST_COMMAND, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
    optimize_result: OptimizeResult
    start_date: str
    end_date: str
