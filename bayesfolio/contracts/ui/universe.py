from __future__ import annotations

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class UniverseUiInput(VersionedContract):
    """UI input for universe selection.

    Attributes:
        tickers: List of asset ticker symbols to include.
        start_date: ISO 8601 start date (YYYY-MM-DD).
        end_date: ISO 8601 end date (YYYY-MM-DD).
    """

    schema: SchemaName = Field(default=SchemaName.UNIVERSE_UI_INPUT, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
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

    schema: SchemaName = Field(default=SchemaName.UNIVERSE_RECORD, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
    asset_order: list[str]
    n_observations: int = Field(ge=0)
    return_unit: str = Field(default="decimal")
