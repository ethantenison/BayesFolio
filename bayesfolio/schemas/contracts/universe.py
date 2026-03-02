from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bayesfolio.schemas.common import SchemaMetadata


class UniverseRequest(BaseModel):
    """Inputs required to build an investable universe and aligned feature panel."""

    metadata: SchemaMetadata = Field(default_factory=SchemaMetadata)
    tickers: list[str] = Field(default_factory=list)
    start_date: str
    end_date: str
    return_unit: str = Field(default="decimal", description="Return unit: decimal or percent_points")

    model_config = ConfigDict(extra="forbid")


class UniverseSnapshot(BaseModel):
    """Stable description of the universe used by the pipeline."""

    metadata: SchemaMetadata = Field(default_factory=SchemaMetadata)
    asset_order: list[str]
    n_observations: int = Field(ge=0)
    return_unit: str = Field(default="decimal")

    model_config = ConfigDict(extra="forbid")
