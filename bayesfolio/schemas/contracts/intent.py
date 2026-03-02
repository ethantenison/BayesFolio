from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bayesfolio.schemas.common import SchemaMetadata


class OptimizationIntent(BaseModel):
    """High-level optimization request to drive the engine pipeline."""

    metadata: SchemaMetadata = Field(default_factory=SchemaMetadata)
    objective: str = Field(default="Sharpe")
    risk_measure: str = Field(default="CVaR")
    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float = 0.35

    model_config = ConfigDict(extra="forbid")
