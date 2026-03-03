from __future__ import annotations

from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class ForecastResult(VersionedContract):
    """Forecast outputs from the GP model stage.

    Attributes:
        asset_order: List of ticker symbols.
        mean: Predicted mean returns as decimals (0.02 = 2%).
        covariance: Predicted covariance matrix as decimals squared.
        return_unit: Unit of returns; always 'decimal'.
    """

    schema: Literal[SchemaName.FORECAST_RESULT] = SchemaName.FORECAST_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    asset_order: list[str]
    mean: list[float]
    covariance: list[list[float]]
    return_unit: str = Field(default="decimal")
