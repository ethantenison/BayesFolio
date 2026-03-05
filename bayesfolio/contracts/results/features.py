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


class FeatureTargetAssociation(ContractModel):
    """Univariate feature-to-target association summary.

    Attributes:
        feature_name: Feature column name.
        pearson_corr: Pearson correlation with target (if computable).
        abs_pearson_corr: Absolute Pearson correlation for ranking.
        sample_size: Non-null sample size used in the correlation.
    """

    feature_name: str
    pearson_corr: float | None = None
    abs_pearson_corr: float
    sample_size: int = Field(ge=0)


class TargetSummaryDiagnostics(ContractModel):
    """Distribution diagnostics for the target return column.

    All return values are decimal units (`0.02 = 2%`).
    """

    count: int = Field(ge=0)
    mean: float = 0.0
    std: float = 0.0
    p01: float = 0.0
    p50: float = 0.0
    p99: float = 0.0
    positive_share: float = 0.0
    target_missing_rate_before_drop: float = 0.0


class FeatureQualityDiagnostics(ContractModel):
    """Coverage and stability diagnostics for feature columns."""

    feature_count: int = Field(ge=0)
    features_with_missing_count: int = Field(ge=0)
    constant_feature_names: list[str] = Field(default_factory=list)
    worst_missing_features: dict[str, float] = Field(default_factory=dict)


class CrossSectionalBreadthDiagnostics(ContractModel):
    """Cross-sectional panel breadth diagnostics by date."""

    date_count: int = Field(ge=0)
    min_assets_per_date: int = Field(ge=0)
    median_assets_per_date: float = 0.0
    max_assets_per_date: int = Field(ge=0)


class MarketStructureDiagnostics(ContractModel):
    """Model-free market structure diagnostics for the built dataset."""

    row_count: int = Field(ge=0)
    asset_count: int = Field(ge=0)
    date_count: int = Field(ge=0)
    rows_per_asset_min: int = Field(ge=0)
    rows_per_asset_median: float = 0.0
    rows_per_asset_max: int = Field(ge=0)
    target_summary: TargetSummaryDiagnostics
    feature_quality: FeatureQualityDiagnostics
    cross_sectional_breadth: CrossSectionalBreadthDiagnostics
    top_feature_target_correlations: list[FeatureTargetAssociation] = Field(default_factory=list)


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
        market_structure: Model-free market structure diagnostics.
        diagnostics: Build-time informational diagnostics and warnings.
    """

    schema: Literal[SchemaName.FEATURES_DATASET_RESULT] = SchemaName.FEATURES_DATASET_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    return_unit: Literal["decimal"] = "decimal"
    artifact: ArtifactPointer
    columns: list[FeatureColumnSpec] = Field(default_factory=list)
    index_info: IndexInfo
    market_structure: MarketStructureDiagnostics | None = None
    diagnostics: list[str] = Field(default_factory=list)
