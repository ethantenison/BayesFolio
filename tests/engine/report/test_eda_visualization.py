from __future__ import annotations

from datetime import date

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
from bayesfolio.engine.report.visualization.eda import build_feature_diagnostic_figures


def _sample_features_result(with_market_structure: bool) -> FeaturesDatasetResult:
    market_structure = None
    if with_market_structure:
        market_structure = MarketStructureDiagnostics(
            row_count=6,
            asset_count=2,
            date_count=3,
            rows_per_asset_min=3,
            rows_per_asset_median=3.0,
            rows_per_asset_max=3,
            target_summary=TargetSummaryDiagnostics(count=6, mean=0.01, std=0.02),
            feature_quality=FeatureQualityDiagnostics(feature_count=3, features_with_missing_count=0),
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


def test_build_feature_diagnostic_figures_returns_expected_figures() -> None:
    result = _sample_features_result(with_market_structure=True)

    figures = build_feature_diagnostic_figures(result)

    assert "feature_target_correlation_heatmap" in figures
    assert "pivoted_returns_correlation_heatmap" in figures
    assert "target_histogram" in figures
    assert figures["feature_target_correlation_heatmap"].data
    assert figures["pivoted_returns_correlation_heatmap"].data
    assert figures["target_histogram"].data


def test_build_feature_diagnostic_figures_returns_empty_without_market_structure() -> None:
    result = _sample_features_result(with_market_structure=False)

    figures = build_feature_diagnostic_figures(result)

    assert figures == {}
