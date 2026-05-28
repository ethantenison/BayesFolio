from __future__ import annotations

from contextlib import nullcontext
from datetime import date
from pathlib import Path

import pandas as pd
import torch

from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.contracts.commands.monthly_portfolio import (
    MonthlyForecastConfig,
    MonthlyPortfolioArtifactsConfig,
    MonthlyPortfolioCommand,
)
from bayesfolio.contracts.results.features import (
    ArtifactPointer as FeaturesArtifactPointer,
)
from bayesfolio.contracts.results.features import (
    CrossSectionalBreadthDiagnostics,
    FeatureQualityDiagnostics,
    FeaturesDatasetResult,
    HistogramDiagnostics,
    IndexInfo,
    MarketStructureDiagnostics,
    TargetSummaryDiagnostics,
)
from bayesfolio.contracts.results.gp_workflow import (
    GPFitValidationSummary,
    GPPlannerResponse,
    GPWorkflowResult,
    NormalizationStats,
)
from bayesfolio.contracts.results.optimize import OptimizeResult
from bayesfolio.core.settings import Horizon, Interval, RiskfolioConfig
from bayesfolio.engine.automation.monthly_portfolio import (
    MonthlyPortfolioRuntimeDependencies,
    run_monthly_portfolio,
)


class _FakePosterior:
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor, samples: torch.Tensor) -> None:
        self.mean = mean
        self.variance = variance
        self._samples = samples

    def rsample(self, shape: torch.Size) -> torch.Tensor:
        assert shape[0] == self._samples.shape[0]
        return self._samples


class _FakeModel:
    def __init__(self, posterior: _FakePosterior) -> None:
        self._posterior = posterior

    def posterior(self, x: torch.Tensor, observation_noise: bool = True) -> _FakePosterior:
        return self._posterior


class _StubArtifactStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir


class _StubProviders:
    pass


def test_monthly_portfolio_runner_outputs_result(monkeypatch, tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    frame = pd.DataFrame(
        {
            "date": [
                "2026-04-30",
                "2026-04-30",
                "2026-04-30",
                "2026-05-31",
                "2026-05-31",
                "2026-05-31",
            ],
            "asset_id": ["SPY", "MGK", "VTV", "SPY", "MGK", "VTV"],
            "t_index": [0, 0, 0, 1, 1, 1],
            "f1": [0.1, 0.2, 0.15, 0.3, 0.4, 0.35],
            "f2": [1.0, 2.0, 1.5, 3.0, 4.0, 3.5],
            "y_excess_lead": [0.01, 0.02, 0.015, 0.03, 0.04, 0.025],
        }
    )
    frame.to_parquet(features_path, index=False)

    features_result = FeaturesDatasetResult(
        artifact=FeaturesArtifactPointer(uri=str(features_path), fingerprint="abc", row_count=6, column_count=6),
        index_info=IndexInfo(
            interval=Interval.DAILY,
            horizon=Horizon.MONTHLY,
            start_date=date(2026, 4, 30),
            end_date=date(2026, 5, 31),
            timezone_note="UTC",
        ),
        market_structure=MarketStructureDiagnostics(
            row_count=6,
            asset_count=3,
            date_count=2,
            rows_per_asset_min=2,
            rows_per_asset_median=2,
            rows_per_asset_max=2,
            target_summary=TargetSummaryDiagnostics(count=6),
            feature_quality=FeatureQualityDiagnostics(feature_count=3, features_with_missing_count=0),
            cross_sectional_breadth=CrossSectionalBreadthDiagnostics(
                date_count=2,
                min_assets_per_date=3,
                median_assets_per_date=3,
                max_assets_per_date=3,
            ),
            target_histogram=HistogramDiagnostics(),
        ),
    )

    mean = torch.tensor([[0.06], [0.02], [0.05]], dtype=torch.float64)
    variance = torch.tensor([[0.01], [0.04], [0.02]], dtype=torch.float64)
    samples = torch.tensor(
        [
            [[0.05], [0.01], [0.04]],
            [[0.06], [0.02], [0.05]],
            [[0.07], [0.03], [0.06]],
        ],
        dtype=torch.float64,
    )
    fake_model = _FakeModel(_FakePosterior(mean=mean, variance=variance, samples=samples))
    gp_result = GPWorkflowResult(
        planner_client_status="deterministic_default",
        planner_response=GPPlannerResponse(
            planner_status="ok",
            instruction_mode="explicit_constraints",
            selected_design=None,
        ),
        target_column="y_excess_lead",
        task_column="asset_id",
        feature_columns=["t_index", "f1", "f2"],
        normalization=NormalizationStats(
            feature_names=["t_index", "f1", "f2"],
            mins={"t_index": 0.0, "f1": 0.1, "f2": 1.0},
            maxs={"t_index": 1.0, "f1": 0.4, "f2": 4.0},
            ranges={"t_index": 1.0, "f1": 0.3, "f2": 3.0},
        ),
        fit_validation=GPFitValidationSummary(
            build_success=True,
            fit_success=True,
            prediction_success=True,
            attempt_count=1,
        ),
        final_status="ok",
    )
    gp_artifacts = type(
        "StubArtifacts",
        (),
        {
            "result": gp_result,
            "model": fake_model,
            "likelihood": None,
            "normalized_train_x": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 2.0]],
                dtype=torch.float64,
            ),
            "train_y": torch.tensor([[0.01], [0.02], [0.015]], dtype=torch.float64),
            "task_map": {"SPY": 0, "MGK": 1, "VTV": 2},
            "outcome_transform": None,
        },
    )()

    command = MonthlyPortfolioCommand(
        as_of_date=date(2026, 5, 31),
        training_universe=["SPY", "MGK", "VTV"],
        helper_assets_only=["MGK"],
        final_portfolio_universe=["SPY", "VTV"],
        feature_dataset=BuildFeaturesDatasetCommand(
            tickers=["SPY", "MGK", "VTV"],
            drop_assets=[],
            lookback_date=date(2026, 1, 1),
            start_date=date(2026, 4, 30),
            end_date=date(2026, 5, 31),
            interval=Interval.DAILY,
            horizon=Horizon.MONTHLY,
        ),
        forecast=MonthlyForecastConfig(
            time_feature_columns=["t_index"],
            etf_feature_columns=["f1"],
            macro_feature_columns=["f2"],
            posterior_scenario_count=3,
            rank=2,
        ),
        riskfolio=RiskfolioConfig(nea=1),
        artifacts=MonthlyPortfolioArtifactsConfig(artifact_root=str(tmp_path), save_summary_json=True),
    )

    monkeypatch.setattr(
        "bayesfolio.engine.automation.monthly_portfolio.build_features_dataset",
        lambda command, providers, artifact_store: features_result,
    )
    monkeypatch.setattr(
        "bayesfolio.engine.automation.monthly_portfolio._run_fixed_monthly_gp",
        lambda training_df, command: gp_artifacts,
    )
    monkeypatch.setattr(
        "bayesfolio.engine.automation.monthly_portfolio.optimize_from_historical_returns",
        lambda returns, request: OptimizeResult(asset_order=returns.columns.tolist(), weights=[0.55, 0.45]),
    )
    monkeypatch.setattr("mlflow.start_run", lambda run_name=None: nullcontext())
    monkeypatch.setattr("mlflow.log_dict", lambda *args, **kwargs: None)
    monkeypatch.setattr("mlflow.log_param", lambda *args, **kwargs: None)

    result = run_monthly_portfolio(
        command,
        dependencies=MonthlyPortfolioRuntimeDependencies(
            feature_providers=_StubProviders(),
            artifact_store=_StubArtifactStore(tmp_path),
        ),
    )

    assert result.run_label == "monthly_portfolio_2026-05-31"
    assert result.final_portfolio_universe == ["SPY", "VTV"]
    assert result.optimize_result.asset_order == ["SPY", "VTV"]
    assert result.top_predictions[0].asset == "SPY"
    assert result.top_predictions[0].eligible_for_final_portfolio is True
    helper_prediction = next(record for record in result.top_predictions if record.asset == "MGK")
    assert helper_prediction.eligible_for_final_portfolio is False
    assert result.artifacts.summary_json is not None
    assert result.gp_workflow_result.planner_client_status == "deterministic_default"
