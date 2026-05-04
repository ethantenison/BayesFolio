"""Feature dataset build command contracts.

This module defines command-layer boundary schemas used to request feature
dataset construction. It is data-only and carries no business logic or I/O.
Financial return fields are interpreted in decimal units unless documented
otherwise.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract
from bayesfolio.core.settings import Horizon, Interval


class BuildFeaturesDatasetCommand(VersionedContract):
    """Command contract for building the features dataset panel.

    Attributes:
        schema: Contract schema identifier.
        schema_version: Semantic schema version.
        tickers: Asset tickers to include in source data fetches.
        drop_assets: Assets removed from ETF features and return labels.
        lookback_date: Earliest date used for fetching source histories.
        start_date: Start date for final output rows after preprocessing.
        end_date: End date for source data fetches and final output filtering.
        interval: Raw vendor interval used for source retrieval.
        horizon: Target horizon frequency for labels and features.
        macro_cols: Optional explicit macro feature columns to keep.
        etf_cols: Optional explicit ETF feature columns to keep.
        drop_macro_cols: Macro columns to drop after selection/defaulting.
        drop_etf_cols: ETF columns to drop after selection/defaulting.
        clip_quantile: Upper quantile used for log-liquidity clipping.
        seed: Optional deterministic seed used by downstream components.
        artifact_name: Optional base artifact name for persisted parquet output.
        include_unlabeled_tail: When True, includes the final period with NaN
            return labels for forecasting. When False (default), drops rows
            with missing y_excess_lead (training-only behavior).
    """

    schema: Literal[SchemaName.FEATURES_DATASET_COMMAND] = SchemaName.FEATURES_DATASET_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    tickers: list[str] = Field(min_length=1)
    drop_assets: list[str] = Field(default_factory=list)
    lookback_date: date
    start_date: date
    end_date: date
    interval: Interval
    horizon: Horizon
    macro_cols: list[str] | None = None
    etf_cols: list[str] | None = None
    drop_macro_cols: list[str] = Field(default_factory=list)
    drop_etf_cols: list[str] = Field(default_factory=list)
    clip_quantile: float = Field(default=0.99, gt=0.0, le=1.0)
    seed: int | None = None
    artifact_name: str | None = None
    include_unlabeled_tail: bool = Field(
        default=False,
        description="Include final period with NaN returns for forecasting.",
    )
