from __future__ import annotations

from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class ForecastCommand(VersionedContract):
    """Command to generate return forecasts for a universe of assets."""

    schema: Literal[SchemaName.FORECAST_COMMAND] = SchemaName.FORECAST_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    tickers: list[str]
    horizon_days: int = Field(ge=1)
    seed: int = 42
