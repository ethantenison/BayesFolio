from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import ContractModel, SchemaName, VersionedContract
from bayesfolio.core.settings import Horizon, Interval


class ArtifactPointer(ContractModel):
    """Pointer metadata for a persisted features dataset artifact.

    Attributes:
        uri: Storage URI or local path to the parquet artifact.
        format: Artifact serialization format. Always ``"parquet"``.
        fingerprint: SHA-256 hex digest of the persisted parquet bytes.
        row_count: Number of rows in the persisted dataset.
        column_count: Number of columns in the persisted dataset.
    """

    uri: str
    format: Literal["parquet"] = "parquet"
    fingerprint: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)


class FeatureColumnSpec(ContractModel):
    """Metadata describing a single dataset column.

    Attributes:
        name: Column name in the persisted dataset.
        kind: Logical column kind.
        unit: Unit convention for values (for example ``"decimal"``).
        lag: Optional lag in periods relative to target time index.
        description: Human-readable column description.
    """

    name: str
    kind: Literal["id", "target", "etf", "macro"]
    unit: str
    lag: int | None = None
    description: str


class IndexInfo(ContractModel):
    """Dataset indexing metadata.

    Attributes:
        interval: Raw source data interval used for retrieval.
        horizon: Forecast horizon frequency for labels/features.
        start_date: Earliest date present in the persisted dataset.
        end_date: Latest date present in the persisted dataset.
        timezone_note: Note describing timezone normalization assumptions.
    """

    interval: Interval
    horizon: Horizon
    start_date: date
    end_date: date
    timezone_note: str


class FeaturesDatasetResult(VersionedContract):
    """Result contract for a persisted features dataset build.

    Attributes:
        schema: Contract schema identifier.
        schema_version: Semantic schema version.
        return_unit: Return unit convention. Always ``"decimal"``
            where ``0.02`` means ``2%``.
        artifact: Persisted artifact pointer and shape metadata.
        columns: Column-level metadata for downstream consumers.
        index_info: Time index metadata for dataset interpretation.
        diagnostics: Build-time informational diagnostics and warnings.
    """

    schema: Literal[SchemaName.FEATURES_DATASET_RESULT] = SchemaName.FEATURES_DATASET_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    return_unit: Literal["decimal"] = "decimal"
    artifact: ArtifactPointer
    columns: list[FeatureColumnSpec] = Field(default_factory=list)
    index_info: IndexInfo
    diagnostics: list[str] = Field(default_factory=list)
