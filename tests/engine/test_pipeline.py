from __future__ import annotations

from datetime import date

import pandas as pd

from bayesfolio.contracts.results.features import ArtifactPointer, FeaturesDatasetResult, IndexInfo
from bayesfolio.contracts.results.report import ReportResult
from bayesfolio.core.settings import Horizon, Interval
from bayesfolio.engine import pipeline


def _sample_features_result() -> FeaturesDatasetResult:
    return FeaturesDatasetResult(
        artifact=ArtifactPointer(
            uri="memory://features/example.parquet",
            fingerprint="abc123",
            row_count=2,
            column_count=4,
        ),
        columns=[],
        index_info=IndexInfo(
            interval=Interval.DAILY,
            horizon=Horizon.MONTHLY,
            start_date=date(2020, 1, 31),
            end_date=date(2020, 2, 29),
            timezone_note="UTC-normalized market close dates",
        ),
        diagnostics=[],
    )


def test_run_schema_first_pipeline_forwards_features_result(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "build_forecast_payload", lambda asset_order, mean, covariance: "forecast")
    monkeypatch.setattr(pipeline, "sample_joint_scenarios", lambda forecast, n_scenarios, seed: "scenarios")
    monkeypatch.setattr(pipeline, "optimize_from_scenarios", lambda scenarios, request: "optimization")
    monkeypatch.setattr(
        pipeline,
        "run_weighted_backtest",
        lambda realized_returns, optimization: "backtest-result",
    )

    def _fake_assemble_report(backtest_result, features_result=None) -> ReportResult:
        captured["backtest_result"] = backtest_result
        captured["features_result"] = features_result
        return ReportResult(headline_metrics={"sharpe_ratio": 0.5})

    monkeypatch.setattr(pipeline, "assemble_report", _fake_assemble_report)

    features_result = _sample_features_result()
    result = pipeline.run_schema_first_pipeline(
        asset_order=["SPY", "VTV"],
        mean=[0.01, 0.02],
        covariance=[[0.01, 0.0], [0.0, 0.02]],
        realized_returns=pd.DataFrame({"SPY": [0.01], "VTV": [0.0]}),
        features_result=features_result,
    )

    assert result.headline_metrics["sharpe_ratio"] == 0.5
    assert captured["backtest_result"] == "backtest-result"
    assert captured["features_result"] is features_result


def test_run_schema_first_pipeline_defaults_features_result_to_none(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "build_forecast_payload", lambda asset_order, mean, covariance: "forecast")
    monkeypatch.setattr(pipeline, "sample_joint_scenarios", lambda forecast, n_scenarios, seed: "scenarios")
    monkeypatch.setattr(pipeline, "optimize_from_scenarios", lambda scenarios, request: "optimization")
    monkeypatch.setattr(
        pipeline,
        "run_weighted_backtest",
        lambda realized_returns, optimization: "backtest-result",
    )

    def _fake_assemble_report(backtest_result, features_result=None) -> ReportResult:
        captured["features_result"] = features_result
        return ReportResult(headline_metrics={"sharpe_ratio": 0.4})

    monkeypatch.setattr(pipeline, "assemble_report", _fake_assemble_report)

    result = pipeline.run_schema_first_pipeline(
        asset_order=["SPY"],
        mean=[0.01],
        covariance=[[0.02]],
        realized_returns=pd.DataFrame({"SPY": [0.01]}),
    )

    assert result.headline_metrics["sharpe_ratio"] == 0.4
    assert captured["features_result"] is None
