from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bayesfolio.schemas.common import SchemaMetadata


class OptimizationRequest(BaseModel):
    """Optimization payload with constraints and objective settings."""

    metadata: SchemaMetadata = Field(default_factory=SchemaMetadata)
    objective: str
    risk_measure: str
    min_weight: float = 0.0
    max_weight: float = 0.35
    hist: bool = True

    model_config = ConfigDict(extra="forbid")


class OptimizationResult(BaseModel):
    """Optimized portfolio weights in stable asset order."""

    metadata: SchemaMetadata = Field(default_factory=SchemaMetadata)
    asset_order: list[str]
    weights: list[float]

    model_config = ConfigDict(extra="forbid")
