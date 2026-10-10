"""Regression gate for July 2026 feature-artifact-to-portfolio parity.

The test replaces external feature providers and persistence with the frozen
historical parquet artifact. It executes the maintained application entry point,
as-of alignment, multitask GP fit, posterior sampling, and Riskfolio optimization
without network access. Returns and weights use decimal units.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from bayesfolio.application import run_monthly_portfolio
from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.contracts.commands.monthly_portfolio import (
    MonthlyForecastConfig,
    MonthlyPortfolioArtifactsConfig,
    MonthlyPortfolioCommand,
)
from bayesfolio.contracts.results.features import ArtifactPointer as FeaturesArtifactPointer
from bayesfolio.contracts.results.features import FeaturesDatasetResult, IndexInfo
from bayesfolio.contracts.results.report import ArtifactPointer
from bayesfolio.core.settings import Horizon, Interval, RiskfolioConfig
from bayesfolio.engine.automation import monthly_portfolio as monthly_engine
from bayesfolio.engine.automation.monthly_portfolio import MonthlyPortfolioRuntimeDependencies
from bayesfolio.engine.features.dataset_builder import ArtifactStoreProtocol, FeatureProviders

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "regression" / "july_2026"

TRAINING_UNIVERSE = [
    "SPY",
    "MGK",
    "VTV",
    "IJR",
    "IWM",
    "VNQ",
    "VNQI",
    "VEA",
    "VWO",
    "VSS",
    "BND",
    "IEF",
    "BNDX",
    "LQD",
    "HYG",
    "EWX",
    "VWOB",
    "HYEM",
]
HELPER_ASSETS = ["MGK", "BND"]
FINAL_UNIVERSE = [asset for asset in TRAINING_UNIVERSE if asset not in HELPER_ASSETS]
ETF_FEATURES = [
    "baspread",
    "ret_kurt",
    "chmom",
    "mom12m",
    "mom36m",
    "cs_mom_rank",
    "max_dd_6m",
    "ma_signal",
    "ret_autocorr",
    "vol_z",
]
MODEL_ETF_FEATURES = ["lag_y_excess_lead", *ETF_FEATURES]
MACRO_FEATURES = [
    "hy_spread",
    "hy_spread_chg_1m",
    "hy_spread_z_12m",
    "vix_slope",
    "vix_ts_z_12m",
    "vix",
    "spy_flow_z_12m",
    "spy_ret",
    "erp",
    "cpi_yoy",
    "cpi_mom",
    "copper_ret",
    "oil_ret",
    "gold_crude_ratio",
    "pct_above_50dma",
    "em_fx_ret",
]


class _FrozenFeatureRunIO:
    """In-memory runtime adapter that exposes the frozen parity feature panel."""

    def __init__(self, feature_path: Path) -> None:
        self.feature_path = feature_path
        self.saved_frames: dict[str, pd.DataFrame] = {}

    def start_run(self, *, run_name: str):
        assert run_name == "july_2026_full_pipeline_parity"
        return nullcontext(object())

    def log_contract(self, name: str, payload: BaseModel) -> None:
        del name, payload

    def log_source_revision(self) -> None:
        return None

    def load_parquet(self, uri: str) -> pd.DataFrame:
        assert uri == str(self.feature_path)
        return pd.read_parquet(self.feature_path)

    def save_csv(self, frame: pd.DataFrame, relative_path: str) -> ArtifactPointer:
        self.saved_frames[relative_path] = frame.copy()
        return ArtifactPointer(
            path=f"memory://{relative_path}",
            artifact_format="csv",
            digest="parity-fixture",
            byte_size=len(frame.to_csv(index=False).encode("utf-8")),
        )

    def save_json_contract(self, payload: BaseModel, relative_path: str) -> ArtifactPointer:
        del payload
        return ArtifactPointer(
            path=f"memory://{relative_path}",
            artifact_format="json",
            digest="parity-fixture",
            byte_size=0,
        )


def _features_result() -> FeaturesDatasetResult:
    feature_path = FIXTURE_DIR / "features.parquet"
    return FeaturesDatasetResult(
        artifact=FeaturesArtifactPointer(
            uri=str(feature_path),
            fingerprint="83aee76cfb979f56b48dbfd362464af8f431334023063604c55ee3be83b92e63",
            row_count=1152,
            column_count=32,
        ),
        index_info=IndexInfo(
            interval=Interval.DAILY,
            horizon=Horizon.MONTHLY,
            start_date=date(2021, 3, 31),
            end_date=date(2026, 6, 30),
            timezone_note="Historical month-end panel; timezone-naive dates.",
        ),
    )


def _command() -> MonthlyPortfolioCommand:
    return MonthlyPortfolioCommand(
        as_of_date=date(2026, 6, 30),
        training_universe=TRAINING_UNIVERSE,
        final_portfolio_universe=FINAL_UNIVERSE,
        helper_assets_only=HELPER_ASSETS,
        feature_dataset=BuildFeaturesDatasetCommand(
            tickers=TRAINING_UNIVERSE,
            drop_assets=[],
            lookback_date=date(2019, 3, 1),
            start_date=date(2021, 3, 1),
            end_date=date(2026, 7, 2),
            interval=Interval.DAILY,
            horizon=Horizon.MONTHLY,
            etf_cols=ETF_FEATURES,
            macro_cols=MACRO_FEATURES,
            seed=27,
            include_unlabeled_tail=True,
        ),
        forecast=MonthlyForecastConfig(
            etf_feature_columns=MODEL_ETF_FEATURES,
            macro_feature_columns=MACRO_FEATURES,
            rank=5,
            min_inferred_noise_level=0.0025,
            seed=16638,
            maxiter=75,
            posterior_scenario_count=5000,
        ),
        riskfolio=RiskfolioConfig(
            model="Classic",
            rm="CVaR",
            rf=0.0,
            alpha=0.5,
            hist=True,
            obj="Sharpe",
            method_mu="hist",
            method_cov="hist",
            upperlng=0.25,
            nea=10,
        ),
        artifacts=MonthlyPortfolioArtifactsConfig(
            artifact_root="unused",
            run_label="july_2026_full_pipeline_parity",
            save_summary_json=False,
        ),
    )


def test_july_2026_feature_gp_and_optimizer_match_golden_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    features_result = _features_result()
    monkeypatch.setattr(
        monthly_engine,
        "build_features_dataset",
        lambda command, providers, artifact_store: features_result,
    )
    run_io = _FrozenFeatureRunIO(FIXTURE_DIR / "features.parquet")

    result = run_monthly_portfolio(
        _command(),
        dependencies=MonthlyPortfolioRuntimeDependencies(
            feature_providers=cast(FeatureProviders, object()),
            artifact_store=cast(ArtifactStoreProtocol, object()),
            run_io=run_io,
        ),
    )

    assert result.strategy_id == "accepted_monthly_portfolio_v1"
    scenario_path = "july_2026_full_pipeline_parity/july_2026_full_pipeline_parity_posterior_scenarios.csv"
    actual_scenarios = run_io.saved_frames[scenario_path].loc[:, FINAL_UNIVERSE]
    expected_scenarios = pd.read_csv(FIXTURE_DIR / "scenarios.csv")
    assert actual_scenarios.columns.tolist() == expected_scenarios.columns.tolist()
    np.testing.assert_allclose(
        actual_scenarios.to_numpy(),
        expected_scenarios.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )

    expected_predictions = pd.read_csv(FIXTURE_DIR / "expected_predictions.csv").sort_values("asset_id")
    actual_predictions = pd.DataFrame(
        [record.model_dump(mode="python") for record in result.top_predictions]
    ).sort_values("asset")
    assert actual_predictions["asset"].tolist() == expected_predictions["asset_id"].tolist()
    np.testing.assert_allclose(
        actual_predictions[["prediction", "uncertainty", "score"]].to_numpy(),
        expected_predictions[["y_pred", "y_std", "score"]].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )

    expected_weights = pd.read_csv(FIXTURE_DIR / "expected_weights.csv")
    assert result.optimize_result.asset_order == expected_weights["asset"].tolist()
    actual_weights = np.asarray(result.optimize_result.weights, dtype=float)
    expected_weight_values = expected_weights["weight"].to_numpy(dtype=float)
    np.testing.assert_allclose(actual_weights, expected_weight_values, rtol=0.0, atol=1e-10)
    assert float(np.abs(actual_weights - expected_weight_values).sum()) <= 1e-9
