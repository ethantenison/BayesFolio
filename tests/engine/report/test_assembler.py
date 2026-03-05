from __future__ import annotations

from datetime import date

from bayesfolio.contracts.results.backtest import BacktestResult
from bayesfolio.contracts.results.features import (
    ArtifactPointer,
    CrossSectionalBreadthDiagnostics,
    FeatureQualityDiagnostics,
    FeaturesDatasetResult,
    IndexInfo,
    MarketStructureDiagnostics,
    TargetSummaryDiagnostics,
)
from bayesfolio.core.settings import Horizon, Interval
from bayesfolio.engine.report.assembler import assemble_report


def _sample_backtest_result() -> BacktestResult:
    return BacktestResult(
        cumulative_return=0.10,
        annualized_return=0.08,
        annualized_volatility=0.12,
        sharpe_ratio=0.67,
    )


def _sample_features_result() -> FeaturesDatasetResult:
    market_structure = MarketStructureDiagnostics(
        row_count=6,
        asset_count=2,
        date_count=3,
        rows_per_asset_min=3,
        rows_per_asset_median=3.0,
        rows_per_asset_max=3,
        target_summary=TargetSummaryDiagnostics(count=6, mean=0.01, std=0.02),
        feature_quality=FeatureQualityDiagnostics(feature_count=4, features_with_missing_count=0),
        cross_sectional_breadth=CrossSectionalBreadthDiagnostics(
            date_count=3,
            min_assets_per_date=2,
            median_assets_per_date=2.0,
            max_assets_per_date=2,
        ),
    )

    return FeaturesDatasetResult(
        artifact=ArtifactPointer(
            uri="memory://features/example.parquet",
            fingerprint="abc123",
            row_count=6,
            column_count=7,
        ),
        columns=[],
        index_info=IndexInfo(
            interval=Interval.DAILY,
            horizon=Horizon.MONTHLY,
            start_date=date(2020, 1, 31),
            end_date=date(2020, 3, 31),
            timezone_note="UTC-normalized market close dates",
        ),
        market_structure=market_structure,
        diagnostics=[],
    )


def test_assemble_report_headline_metrics_only() -> None:
    report = assemble_report(_sample_backtest_result())

    assert report.headline_metrics["cumulative_return"] == 0.10
    assert report.headline_metrics["annualized_return"] == 0.08
    assert report.headline_metrics["annualized_volatility"] == 0.12
    assert report.headline_metrics["sharpe_ratio"] == 0.67
    assert report.market_structure is None


def test_assemble_report_includes_market_structure_when_available() -> None:
    report = assemble_report(
        backtest_result=_sample_backtest_result(),
        features_result=_sample_features_result(),
    )

    assert report.market_structure is not None
    assert report.market_structure.asset_count == 2
    assert report.market_structure.date_count == 3
