from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bayesfolio.schemas.common import ArtifactRef, SchemaMetadata


class ReportBundle(BaseModel):
    """Final report payload with metrics and optional artifact references."""

    metadata: SchemaMetadata = Field(default_factory=SchemaMetadata)
    headline_metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
