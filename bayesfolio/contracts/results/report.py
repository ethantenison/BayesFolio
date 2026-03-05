from __future__ import annotations

from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import ContractModel, SchemaName, VersionedContract
from bayesfolio.contracts.results.features import MarketStructureDiagnostics


class ArtifactPointer(VersionedContract):
    """Pointer to a persisted artifact.

    Attributes:
        path: Storage path or URI for the artifact.
        artifact_format: File format (e.g. 'parquet', 'png').
        digest: SHA-256 hex digest for integrity.
        byte_size: File size in bytes.
    """

    schema: Literal[SchemaName.ARTIFACT_POINTER] = SchemaName.ARTIFACT_POINTER
    schema_version: Literal["0.1.0"] = "0.1.0"
    path: str
    artifact_format: str
    digest: str
    byte_size: int = Field(ge=0)


class DiagnosticFigureSummary(ContractModel):
    """Metadata summary for a generated report diagnostic figure.

    Attributes:
        name: Stable figure identifier key.
        title: Human-readable figure title.
        trace_count: Number of traces in the generated figure.
    """

    name: str
    title: str
    trace_count: int = Field(ge=0)


class ReportResult(VersionedContract):
    """Final report payload with metrics and artifact pointers.

    Attributes:
        headline_metrics: Key performance metrics as decimals.
        artifacts: List of artifact pointers for charts and data exports.
        market_structure: Optional model-free feature-dataset diagnostics.
        diagnostic_figures: Metadata for generated diagnostic visualizations.
    """

    schema: Literal[SchemaName.REPORT_RESULT] = SchemaName.REPORT_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    headline_metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: list[ArtifactPointer] = Field(default_factory=list)
    market_structure: MarketStructureDiagnostics | None = None
    diagnostic_figures: list[DiagnosticFigureSummary] = Field(default_factory=list)
