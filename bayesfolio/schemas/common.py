from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SchemaMetadata(BaseModel):
    """Cross-boundary metadata for reproducible artifacts and payloads."""

    schema_version: str = Field("1.0.0", description="Semantic version of the transport schema")
    created_at_utc: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field("bayesfolio")
    run_id: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ArtifactFingerprint(BaseModel):
    """Deterministic fingerprint for persisted artifacts."""

    algorithm: str = Field(default="sha256")
    digest: str
    byte_size: int = Field(ge=0)

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactRef(BaseModel):
    """Pointer to a persisted artifact with integrity metadata."""

    path: str
    format: str
    fingerprint: ArtifactFingerprint

    model_config = ConfigDict(extra="forbid", frozen=True)
