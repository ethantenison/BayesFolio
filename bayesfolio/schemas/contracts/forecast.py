from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bayesfolio.schemas.common import SchemaMetadata


class ForecastPayload(BaseModel):
    """Forecast outputs passed from model stage to scenario/optimization stages."""

    metadata: SchemaMetadata = Field(default_factory=SchemaMetadata)
    asset_order: list[str]
    mean: list[float]
    covariance: list[list[float]]
    return_unit: str = Field(default="decimal")

    model_config = ConfigDict(extra="forbid")
