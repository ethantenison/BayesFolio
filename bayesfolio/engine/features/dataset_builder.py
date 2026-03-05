from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.contracts.results.features import (
    ArtifactPointer,
    CrossSectionalBreadthDiagnostics,
    FeatureColumnSpec,
    FeatureQualityDiagnostics,
    FeaturesDatasetResult,
    FeatureTargetAssociation,
    IndexInfo,
    MarketStructureDiagnostics,
    TargetSummaryDiagnostics,
)
from bayesfolio.core.settings import Horizon
from bayesfolio.engine.features.engineering import (
    add_cross_sectional_momentum_rank,
    add_log_liquidity_features,
    add_target_lags,
    build_t_index,
)


class ReturnsProviderProtocol(Protocol):
    """Protocol for return-label providers.

    Implementations return long-format labels in decimal units with columns
    ``date``, ``asset_id``, and ``y_excess_lead``.
    """

    def get_y_excess_lead_long(
        self,
        tickers: list[str],
        start: str,
        end: str,
        horizon: Horizon,
    ) -> pd.DataFrame: ...


class MacroProviderProtocol(Protocol):
    """Protocol for macro feature providers returning date-indexed features."""

    def get_macro_features(self, start: str, end: str, horizon: Horizon) -> pd.DataFrame: ...


class EtfFeaturesProviderProtocol(Protocol):
    """Protocol for ETF feature providers returning long-format features."""

    def get_etf_features_long(
        self,
        tickers: list[str],
        start: str,
        end: str,
        horizon: Horizon,
    ) -> pd.DataFrame: ...


class ArtifactStoreProtocol(Protocol):
    """Protocol for persisted dataset storage backends."""

    def save_parquet(
        self,
        frame: pd.DataFrame,
        artifact_name: str,
        metadata: dict[str, object],
    ) -> ArtifactPointer: ...


@dataclass(frozen=True)
class FeatureProviders:
    """Dependency container for external data providers."""

    returns_provider: ReturnsProviderProtocol
    macro_provider: MacroProviderProtocol
    etf_features_provider: EtfFeaturesProviderProtocol


def build_features_dataset(
    command: BuildFeaturesDatasetCommand,
    providers: FeatureProviders,
    artifact_store: ArtifactStoreProtocol,
) -> FeaturesDatasetResult:
    """Build and persist the long-format features dataset.

    Pipeline behavior:
    1) Fetch return labels, macro features, and ETF features from providers.
    2) Apply ETF engineering transforms (liquidity logs and momentum rank).
    3) Merge into a long panel, align predictors to ``t-1`` to avoid look-ahead,
       and add lagged targets.
    4) Build ``t_index``, select output columns, and persist parquet artifact.

    Args:
        command: Build command contract containing date/ticker selectors.
        providers: External source providers container.
        artifact_store: Persistence backend for parquet output.

    Returns:
        FeaturesDatasetResult containing decimal-unit metadata, artifact pointer,
        index metadata, and diagnostics.

    Raises:
        ValueError: If required columns are missing or date bounds are invalid.
    """

    _validate_command(command)

    diagnostics: list[str] = []
    returns_frame = providers.returns_provider.get_y_excess_lead_long(
        tickers=command.tickers,
        start=command.lookback_date.isoformat(),
        end=command.end_date.isoformat(),
        horizon=command.horizon,
    )
    macro_frame = providers.macro_provider.get_macro_features(
        start=command.lookback_date.isoformat(),
        end=command.end_date.isoformat(),
        horizon=command.horizon,
    )
    etf_frame = providers.etf_features_provider.get_etf_features_long(
        tickers=command.tickers,
        start=command.lookback_date.isoformat(),
        end=command.end_date.isoformat(),
        horizon=command.horizon,
    )

    returns_panel = _prepare_returns(returns_frame, command.drop_assets)
    etf_panel = _prepare_etf_features(etf_frame, command.drop_assets, command.clip_quantile)
    macro_panel = _prepare_macro_features(macro_frame)

    merged = (
        returns_panel.merge(etf_panel, on=["date", "asset_id"], how="left")
        .merge(macro_panel, on="date", how="left")
        .sort_values(["date", "asset_id"])
        .reset_index(drop=True)
    )

    selected_etf_cols = _resolve_selected_columns(
        available_cols=[col for col in etf_panel.columns if col not in {"date", "asset_id"}],
        selected_cols=command.etf_cols,
        dropped_cols=command.drop_etf_cols,
        label="etf",
    )
    selected_macro_cols = _resolve_selected_columns(
        available_cols=[col for col in macro_panel.columns if col != "date"],
        selected_cols=command.macro_cols,
        dropped_cols=command.drop_macro_cols,
        label="macro",
    )

    merged = _apply_lookahead_alignment(merged, selected_etf_cols, selected_macro_cols)
    merged = add_target_lags(merged, target_col="y_excess_lead", lags=[1, 2])

    merged["date"] = pd.to_datetime(merged["date"])
    start_ts = pd.Timestamp(command.start_date)
    end_ts = pd.Timestamp(command.end_date)
    merged_window = merged[(merged["date"] >= start_ts) & (merged["date"] <= end_ts)].copy()
    target_missing_rate_before_drop = (
        float(merged_window["y_excess_lead"].isna().mean()) if not merged_window.empty else 0.0
    )
    merged = merged_window.dropna(subset=["y_excess_lead"]).reset_index(drop=True)

    lag_cols = ["lag_y_excess_lead", "lag2_y_excess_lead"]
    output_cols = [
        "date",
        "asset_id",
        *selected_etf_cols,
        *selected_macro_cols,
        *[col for col in lag_cols if col in merged.columns],
        "y_excess_lead",
    ]
    dataset = merged.loc[:, output_cols]
    dataset = build_t_index(dataset, date_col="date")

    if dataset.empty:
        diagnostics.append("Dataset is empty after date filtering and target NA drop.")

    diagnostics.append(
        "Look-ahead policy applied: ETF and macro predictors are shifted by one period per asset before model use."
    )
    diagnostics.append("Return unit is decimal for y_excess_lead and lagged target columns.")

    artifact_name = command.artifact_name or (
        f"features_dataset_{command.start_date.isoformat()}_{command.end_date.isoformat()}.parquet"
    )
    metadata = {
        "command": command.model_dump(mode="json"),
        "diagnostics": diagnostics,
        "columns": output_cols,
    }
    artifact = artifact_store.save_parquet(dataset, artifact_name=artifact_name, metadata=metadata)

    index_info = _build_index_info(dataset, command)
    column_specs = _build_column_specs(dataset.columns.tolist(), selected_etf_cols, selected_macro_cols)
    feature_cols = [
        *selected_etf_cols,
        *selected_macro_cols,
        *[col for col in lag_cols if col in dataset.columns],
    ]
    market_structure = _build_market_structure_diagnostics(
        dataset=dataset,
        feature_cols=feature_cols,
        target_missing_rate_before_drop=target_missing_rate_before_drop,
    )

    return FeaturesDatasetResult(
        artifact=artifact,
        columns=column_specs,
        index_info=index_info,
        market_structure=market_structure,
        diagnostics=diagnostics,
    )


def _validate_command(command: BuildFeaturesDatasetCommand) -> None:
    if command.lookback_date > command.start_date:
        msg = "lookback_date must be on or before start_date."
        raise ValueError(msg)

    if command.start_date > command.end_date:
        msg = "start_date must be on or before end_date."
        raise ValueError(msg)


def _prepare_returns(frame: pd.DataFrame, drop_assets: list[str]) -> pd.DataFrame:
    required = {"date", "asset_id", "y_excess_lead"}
    missing = required - set(frame.columns)
    if missing:
        msg = f"Returns provider missing columns: {sorted(missing)}"
        raise ValueError(msg)

    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"])
    if drop_assets:
        output = output[~output["asset_id"].isin(drop_assets)]
    return output.sort_values(["date", "asset_id"]).reset_index(drop=True)


def _prepare_etf_features(frame: pd.DataFrame, drop_assets: list[str], clip_quantile: float) -> pd.DataFrame:
    required = {"date", "asset_id"}
    missing = required - set(frame.columns)
    if missing:
        msg = f"ETF provider missing columns: {sorted(missing)}"
        raise ValueError(msg)

    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"])
    if drop_assets:
        output = output[~output["asset_id"].isin(drop_assets)]

    output = add_log_liquidity_features(output, ill_col="ill", dolvol_col="dolvol", q=clip_quantile)
    if "mom12m" in output.columns:
        output = add_cross_sectional_momentum_rank(output, momentum_col="mom12m", out_col="cs_mom_rank")

    return output.sort_values(["date", "asset_id"]).reset_index(drop=True)


def _prepare_macro_features(frame: pd.DataFrame) -> pd.DataFrame:
    if "date" not in frame.columns:
        msg = "Macro provider must return a date column."
        raise ValueError(msg)

    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"])
    return output.sort_values("date").reset_index(drop=True)


def _resolve_selected_columns(
    available_cols: list[str],
    selected_cols: list[str] | None,
    dropped_cols: list[str],
    label: str,
) -> list[str]:
    base = available_cols if selected_cols is None else selected_cols
    selected = [col for col in base if col not in set(dropped_cols)]
    missing = sorted(set(selected) - set(available_cols))
    if missing:
        msg = f"Requested {label} columns are unavailable: {missing}"
        raise ValueError(msg)
    return selected


def _apply_lookahead_alignment(
    frame: pd.DataFrame,
    etf_cols: list[str],
    macro_cols: list[str],
) -> pd.DataFrame:
    output = frame.sort_values(["asset_id", "date"]).copy()
    if etf_cols:
        output[etf_cols] = output.groupby("asset_id", sort=False)[etf_cols].shift(1)
    if macro_cols:
        output[macro_cols] = output.groupby("asset_id", sort=False)[macro_cols].shift(1)

    return output.sort_values(["date", "asset_id"]).reset_index(drop=True)


def _build_index_info(dataset: pd.DataFrame, command: BuildFeaturesDatasetCommand) -> IndexInfo:
    if dataset.empty:
        start_date = command.start_date
        end_date = command.end_date
    else:
        start_date = dataset["date"].min().date()
        end_date = dataset["date"].max().date()

    return IndexInfo(
        interval=command.interval,
        horizon=command.horizon,
        start_date=start_date,
        end_date=end_date,
        timezone_note="Dates are normalized to market-close period timestamps without timezone offsets.",
    )


def _build_column_specs(
    column_names: list[str],
    etf_cols: list[str],
    macro_cols: list[str],
) -> list[FeatureColumnSpec]:
    specs: list[FeatureColumnSpec] = []
    etf_set = set(etf_cols)
    macro_set = set(macro_cols)

    for name in column_names:
        if name in {"t_index", "date", "asset_id"}:
            specs.append(
                FeatureColumnSpec(
                    name=name,
                    kind="id",
                    unit="index",
                    lag=None,
                    description="Identifier/index column used for long-panel alignment.",
                )
            )
            continue

        if name == "y_excess_lead":
            specs.append(
                FeatureColumnSpec(
                    name=name,
                    kind="target",
                    unit="decimal",
                    lag=0,
                    description="Lead next-period excess return target in decimal units.",
                )
            )
            continue

        if name.startswith("lag") and name.endswith("_y_excess_lead"):
            lag = 1 if name.startswith("lag_") else int(name.split("_", maxsplit=1)[0].replace("lag", ""))
            specs.append(
                FeatureColumnSpec(
                    name=name,
                    kind="target",
                    unit="decimal",
                    lag=lag,
                    description="Lagged target return in decimal units.",
                )
            )
            continue

        if name in etf_set:
            specs.append(
                FeatureColumnSpec(
                    name=name,
                    kind="etf",
                    unit="feature",
                    lag=1,
                    description="ETF-level predictor aligned to t-1.",
                )
            )
            continue

        if name in macro_set:
            specs.append(
                FeatureColumnSpec(
                    name=name,
                    kind="macro",
                    unit="feature",
                    lag=1,
                    description="Macro predictor aligned to t-1.",
                )
            )

    return specs


def _build_market_structure_diagnostics(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    target_missing_rate_before_drop: float,
) -> MarketStructureDiagnostics:
    if dataset.empty:
        return MarketStructureDiagnostics(
            row_count=0,
            asset_count=0,
            date_count=0,
            rows_per_asset_min=0,
            rows_per_asset_median=0.0,
            rows_per_asset_max=0,
            target_summary=TargetSummaryDiagnostics(target_missing_rate_before_drop=target_missing_rate_before_drop),
            feature_quality=FeatureQualityDiagnostics(feature_count=len(feature_cols)),
            cross_sectional_breadth=CrossSectionalBreadthDiagnostics(
                date_count=0,
                min_assets_per_date=0,
                median_assets_per_date=0.0,
                max_assets_per_date=0,
            ),
            top_feature_target_correlations=[],
        )

    rows_per_asset = dataset.groupby("asset_id", sort=False).size()
    breadth = dataset.groupby("date", sort=False)["asset_id"].nunique()
    target = dataset["y_excess_lead"].astype(float)

    quantiles = target.quantile([0.01, 0.50, 0.99])
    missing_rates = {col: float(dataset[col].isna().mean()) for col in feature_cols if col in dataset.columns}
    worst_missing = dict(sorted(missing_rates.items(), key=lambda item: item[1], reverse=True)[:5])
    constant_features = [
        col for col in feature_cols if col in dataset.columns and dataset[col].nunique(dropna=True) <= 1
    ]

    associations: list[FeatureTargetAssociation] = []
    for col in feature_cols:
        if col not in dataset.columns:
            continue
        pair = dataset[[col, "y_excess_lead"]].dropna()
        if len(pair) < 3 or pair[col].nunique() <= 1 or pair["y_excess_lead"].nunique() <= 1:
            continue
        corr_value = float(pair[col].corr(pair["y_excess_lead"]))
        if np.isnan(corr_value):
            continue
        associations.append(
            FeatureTargetAssociation(
                feature_name=col,
                pearson_corr=corr_value,
                abs_pearson_corr=abs(corr_value),
                sample_size=int(len(pair)),
            )
        )

    associations = sorted(associations, key=lambda item: item.abs_pearson_corr, reverse=True)[:10]

    return MarketStructureDiagnostics(
        row_count=int(len(dataset)),
        asset_count=int(dataset["asset_id"].nunique()),
        date_count=int(dataset["date"].nunique()),
        rows_per_asset_min=int(rows_per_asset.min()),
        rows_per_asset_median=float(rows_per_asset.median()),
        rows_per_asset_max=int(rows_per_asset.max()),
        target_summary=TargetSummaryDiagnostics(
            count=int(target.count()),
            mean=float(target.mean()),
            std=float(target.std(ddof=1)) if target.count() > 1 else 0.0,
            p01=float(quantiles.loc[0.01]),
            p50=float(quantiles.loc[0.50]),
            p99=float(quantiles.loc[0.99]),
            positive_share=float((target > 0).mean()),
            target_missing_rate_before_drop=target_missing_rate_before_drop,
        ),
        feature_quality=FeatureQualityDiagnostics(
            feature_count=len(feature_cols),
            features_with_missing_count=sum(rate > 0.0 for rate in missing_rates.values()),
            constant_feature_names=constant_features,
            worst_missing_features=worst_missing,
        ),
        cross_sectional_breadth=CrossSectionalBreadthDiagnostics(
            date_count=int(len(breadth)),
            min_assets_per_date=int(breadth.min()),
            median_assets_per_date=float(breadth.median()),
            max_assets_per_date=int(breadth.max()),
        ),
        top_feature_target_correlations=associations,
    )
