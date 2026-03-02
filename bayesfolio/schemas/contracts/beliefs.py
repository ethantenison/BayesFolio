from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bayesfolio.schemas.common import SchemaMetadata


class PriorBeliefs(BaseModel):
    """Structured prior beliefs over expected return, volatility, and correlation."""

    metadata: SchemaMetadata = Field(default_factory=SchemaMetadata)
    expected_return: dict[str, float] = Field(default_factory=dict)
    volatility: dict[str, float] = Field(default_factory=dict)
    pairwise_correlation: dict[str, dict[str, float]] = Field(default_factory=dict)
    return_unit: str = Field(default="decimal")

    model_config = ConfigDict(extra="forbid")
