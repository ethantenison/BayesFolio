from __future__ import annotations

from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class UniverseUiInput(VersionedContract):
    """UI input for universe selection.

    Attributes:
        tickers: List of asset ticker symbols to include.
        start_date: ISO 8601 start date (YYYY-MM-DD).
        end_date: ISO 8601 end date (YYYY-MM-DD).
    """

    schema: Literal[SchemaName.UNIVERSE_UI_INPUT] = SchemaName.UNIVERSE_UI_INPUT
    schema_version: Literal["0.1.0"] = "0.1.0"
    tickers: list[str] = Field(default_factory=list)
    start_date: str
    end_date: str


class UniverseRecord(VersionedContract):
    """Stable snapshot of the universe used by the pipeline.

    Attributes:
        asset_order: Ticker symbols in canonical order.
        n_observations: Number of time observations.
        return_unit: Unit of returns; 'decimal' or 'percent_points'.
    """

    schema: Literal[SchemaName.UNIVERSE_RECORD] = SchemaName.UNIVERSE_RECORD
    schema_version: Literal["0.1.0"] = "0.1.0"
    asset_order: list[str]
    n_observations: int = Field(ge=0)
    return_unit: str = Field(default="decimal")
