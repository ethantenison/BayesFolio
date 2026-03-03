from __future__ import annotations

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class ArtifactPointer(VersionedContract):
    """Pointer to a persisted artifact.

    Attributes:
        path: Storage path or URI for the artifact.
        artifact_format: File format (e.g. 'parquet', 'png').
        digest: SHA-256 hex digest for integrity.
        byte_size: File size in bytes.
    """

    schema: SchemaName = Field(default=SchemaName.ARTIFACT_POINTER, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
    path: str
    artifact_format: str
    digest: str
    byte_size: int = Field(ge=0)


class ReportResult(VersionedContract):
    """Final report payload with metrics and artifact pointers.

    Attributes:
        headline_metrics: Key performance metrics as decimals.
        artifacts: List of artifact pointers for charts and data exports.
    """

    schema: SchemaName = Field(default=SchemaName.REPORT_RESULT, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
    headline_metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: list[ArtifactPointer] = Field(default_factory=list)
