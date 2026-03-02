from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bayesfolio.schemas.common import SchemaMetadata


class ScenarioPanel(BaseModel):
    """Scenario return panel used by scenario-based risk optimization."""

    metadata: SchemaMetadata = Field(default_factory=SchemaMetadata)
    asset_order: list[str]
    n_scenarios: int = Field(ge=1)
    values: list[list[float]]
    return_unit: str = Field(default="decimal")

    model_config = ConfigDict(extra="forbid")
