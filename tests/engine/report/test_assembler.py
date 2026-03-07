from __future__ import annotations

from datetime import date

from bayesfolio.contracts.results.backtest import BacktestResult
from bayesfolio.contracts.results.features import (
    ArtifactPointer,
    CrossSectionalBreadthDiagnostics,
    FeatureQualityDiagnostics,
    FeaturesDatasetResult,
    HistogramDiagnostics,
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
        max_drawdown=-0.21,
        calmar_ratio=0.38,
        sortino_ratio=0.74,
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
        feature_target_correlation_matrix={
            "y_excess_lead": {"y_excess_lead": 1.0, "mom12m": 0.25},
            "mom12m": {"y_excess_lead": 0.25, "mom12m": 1.0},
        },
        pivoted_returns_correlation_matrix={
            "AAA": {"AAA": 1.0, "BBB": 0.15},
            "BBB": {"AAA": 0.15, "BBB": 1.0},
        },
        target_histogram=HistogramDiagnostics(
            bin_edges=[-0.02, -0.01, 0.0, 0.01],
            counts=[1, 2, 3],
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
    assert report.headline_metrics["max_drawdown"] == -0.21
    assert report.headline_metrics["calmar_ratio"] == 0.38
    assert report.headline_metrics["sortino_ratio"] == 0.74
    assert report.market_structure is None
    assert report.diagnostic_figures == []


def test_assemble_report_includes_market_structure_when_available() -> None:
    report = assemble_report(
        backtest_result=_sample_backtest_result(),
        features_result=_sample_features_result(),
    )

    assert report.market_structure is not None
    assert report.market_structure.asset_count == 2
    assert report.market_structure.date_count == 3
    figure_names = {item.name for item in report.diagnostic_figures}
    assert "feature_target_correlation_heatmap" in figure_names
    assert "pivoted_returns_correlation_heatmap" in figure_names
    assert "target_histogram" in figure_names
