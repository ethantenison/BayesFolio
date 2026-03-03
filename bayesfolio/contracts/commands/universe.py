from __future__ import annotations

from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class UniverseCommand(VersionedContract):
    """Command to build an investable universe and aligned feature panel.

    Attributes:
        tickers: List of asset ticker symbols.
        start_date: ISO 8601 start date (YYYY-MM-DD).
        end_date: ISO 8601 end date (YYYY-MM-DD).
        return_unit: Unit of returns; 'decimal' or 'percent_points'.
    """

    schema: Literal[SchemaName.UNIVERSE_COMMAND] = SchemaName.UNIVERSE_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    tickers: list[str] = Field(default_factory=list)
    start_date: str
    end_date: str
    return_unit: str = Field(default="decimal")
