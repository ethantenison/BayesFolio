from __future__ import annotations

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

    schema: SchemaName = Field(default=SchemaName.FORECAST_RESULT, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
    asset_order: list[str]
    mean: list[float]
    covariance: list[list[float]]
    return_unit: str = Field(default="decimal")
