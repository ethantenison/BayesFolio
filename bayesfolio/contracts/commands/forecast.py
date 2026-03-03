from __future__ import annotations

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class ForecastCommand(VersionedContract):
    """Command to generate return forecasts for a universe of assets."""

    schema: SchemaName = Field(default=SchemaName.FORECAST_COMMAND, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
    tickers: list[str]
    horizon_days: int = Field(ge=1)
    seed: int = 42
