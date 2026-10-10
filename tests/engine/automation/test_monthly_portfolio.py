"""Tests for the monthly portfolio automation workflow."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import date

import pandas as pd
import torch
from pydantic import BaseModel

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
from bayesfolio.contracts.results.report import ArtifactPointer
from bayesfolio.core.settings import Horizon, Interval, RiskfolioConfig
from bayesfolio.engine.automation.monthly_portfolio import (
    MonthlyPortfolioRuntimeDependencies,
    _prepare_as_of_feature_frames,
    execute_monthly_portfolio,
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
    pass


class _StubProviders:
    pass


class _FakeRunIO:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.logged_contracts: list[str] = []
        self.loaded_uris: list[str] = []
        self.saved_frames: dict[str, pd.DataFrame] = {}
        self.saved_contracts: dict[str, BaseModel] = {}
        self.source_revision_logged = False

    def start_run(self, *, run_name: str):
        assert run_name == "monthly_portfolio_2026-05-31"
        return nullcontext(object())

    def log_contract(self, name: str, payload: BaseModel) -> None:
        self.logged_contracts.append(name)

    def log_source_revision(self) -> None:
        self.source_revision_logged = True

    def load_parquet(self, uri: str) -> pd.DataFrame:
        self.loaded_uris.append(uri)
        return self.frame.copy()

    def save_csv(self, frame: pd.DataFrame, relative_path: str) -> ArtifactPointer:
        self.saved_frames[relative_path] = frame.copy()
        content = frame.to_csv(index=False).encode("utf-8")
        return ArtifactPointer(
            path=f"memory://{relative_path}",
            artifact_format="csv",
            digest="digest",
            byte_size=len(content),
        )

    def save_json_contract(self, payload: BaseModel, relative_path: str) -> ArtifactPointer:
        self.saved_contracts[relative_path] = payload
        return ArtifactPointer(
            path=f"memory://{relative_path}",
            artifact_format="json",
            digest="digest",
            byte_size=1,
        )


def test_monthly_portfolio_runner_outputs_result(monkeypatch) -> None:
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
    features_result = FeaturesDatasetResult(
        artifact=FeaturesArtifactPointer(
            uri="memory://features.parquet",
            fingerprint="abc",
            row_count=6,
            column_count=6,
        ),
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
            "train_x": torch.tensor(
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
        riskfolio=RiskfolioConfig(alpha=0.5, nea=1),
        artifacts=MonthlyPortfolioArtifactsConfig(artifact_root="unused", save_summary_json=True),
    )

    monkeypatch.setattr(
        "bayesfolio.engine.automation.monthly_portfolio.build_features_dataset",
        lambda command, providers, artifact_store: features_result,
    )
    monkeypatch.setattr(
        "bayesfolio.engine.automation.monthly_portfolio._run_fixed_monthly_gp",
        lambda training_df, command: gp_artifacts,
    )
    captured: dict[str, object] = {}

    def _fake_optimize(returns, request):
        captured["request"] = request
        return OptimizeResult(asset_order=returns.columns.tolist(), weights=[0.55, 0.45])

    monkeypatch.setattr(
        "bayesfolio.engine.automation.monthly_portfolio.optimize_from_historical_returns",
        _fake_optimize,
    )
    run_io = _FakeRunIO(frame)

    result = execute_monthly_portfolio(
        command,
        dependencies=MonthlyPortfolioRuntimeDependencies(
            feature_providers=_StubProviders(),
            artifact_store=_StubArtifactStore(),
            run_io=run_io,
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
    assert captured["request"].alpha == 0.5
    assert run_io.loaded_uris == ["memory://features.parquet"]
    assert run_io.source_revision_logged is True
    assert run_io.logged_contracts == [
        "monthly_portfolio_command",
        "features_result",
        "gp_workflow_result",
        "forecast_result",
        "optimize_result",
        "monthly_portfolio_result",
    ]
    run_prefix = "monthly_portfolio_2026-05-31/monthly_portfolio_2026-05-31"
    assert set(run_io.saved_frames) == {
        f"{run_prefix}_posterior_scenarios.csv",
        f"{run_prefix}_predictions.csv",
        f"{run_prefix}_weights.csv",
    }
    assert list(run_io.saved_frames[f"{run_prefix}_weights.csv"].columns) == ["asset", "weight"]
    assert set(run_io.saved_contracts) == {f"{run_prefix}_summary.json"}


def test_monthly_engine_requires_runtime_io() -> None:
    command = MonthlyPortfolioCommand.model_construct()

    try:
        execute_monthly_portfolio(
            command,
            dependencies=MonthlyPortfolioRuntimeDependencies(
                feature_providers=_StubProviders(),
                artifact_store=_StubArtifactStore(),
            ),
        )
    except ValueError as exc:
        assert str(exc) == "Monthly engine execution requires an injected run_io adapter."
    else:
        raise AssertionError("Expected missing run_io to be rejected.")


def test_monthly_feature_alignment_excludes_as_of_labels_and_future_rows() -> None:
    frame = pd.DataFrame(
        {
            "date": [
                "2026-04-30",
                "2026-04-30",
                "2026-05-31",
                "2026-05-31",
                "2026-06-30",
                "2026-06-30",
            ],
            "asset_id": ["SPY", "VTV", "SPY", "VTV", "SPY", "VTV"],
            "t_index": [0, 0, 1, 1, 2, 2],
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "f2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y_excess_lead": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )
    command = MonthlyPortfolioCommand.model_construct(
        as_of_date=date(2026, 5, 31),
        training_universe=["SPY", "VTV"],
        forecast=MonthlyForecastConfig(
            time_feature_columns=["t_index"],
            etf_feature_columns=["f1"],
            macro_feature_columns=["f2"],
        ),
    )

    training, scoring = _prepare_as_of_feature_frames(features_df=frame, command=command)

    assert training["date"].dt.date.unique().tolist() == [date(2026, 4, 30)]
    assert scoring["date"].dt.date.unique().tolist() == [date(2026, 5, 31)]
    assert training["asset_id"].astype(str).tolist() == ["SPY", "VTV"]
    assert scoring["asset_id"].astype(str).tolist() == ["SPY", "VTV"]


def test_monthly_forecast_defaults_capture_accepted_model_controls() -> None:
    forecast = MonthlyForecastConfig()

    assert forecast.maxiter == 75
    assert forecast.input_transform == "botorch_normalize"
    assert forecast.time_modulation == "lengthscale_only"
    assert forecast.task_covar_prior == "none"
