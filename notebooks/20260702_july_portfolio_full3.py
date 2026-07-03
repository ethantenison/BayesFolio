"""Create the July 2026 portfolio with the selected full3 June-start GP.

This script builds a July-ready feature artifact, fits the requested live GP
construction window, and writes scenario-method Riskfolio weights plus a small
parameter sensitivity grid.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand  # noqa: E402
from bayesfolio.core.settings import Horizon, Interval  # noqa: E402
from bayesfolio.engine.features import build_features_dataset, make_default_feature_providers  # noqa: E402
from bayesfolio.io import ParquetArtifactStore  # noqa: E402

RUNNER_PATH = REPO_ROOT / "experiments" / "2026-06-portfolio-optimization" / "run_monthly_optimization_walkforward.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("portfolio_walkforward_runner", RUNNER_PATH)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise RuntimeError(f"Unable to import portfolio runner from {RUNNER_PATH}")
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)

RUN_ID = "20260702_july_full3_june_start_noise0025_tvls_riskon25_10"
OUTPUT_DIR = REPO_ROOT / "notebooks" / "outputs" / "20260702_july_portfolio_full3"
FEATURE_ARTIFACT_NAME = "portfolio_etf_macro_features_2026_07.parquet"
FEATURE_ARTIFACT_ROOT = Path("/Users/et/.bayesfolio/artifacts/features")

ETF_TICKERS = [
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

SELECTED_ETF_COLS = [
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

SELECTED_MACRO_COLS = [
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

RISKFOLIO_GRID = [
    {"label": "old_conservative_20_10", "upperlng": 0.20, "nea": 10},
    {"label": "recommended_25_10", "upperlng": 0.25, "nea": 10},
    {"label": "previous_25_8", "upperlng": 0.25, "nea": 8},
    {"label": "assertive_30_6", "upperlng": 0.30, "nea": 6},
]


def build_july_features() -> Path:
    command = BuildFeaturesDatasetCommand.model_validate(
        {
            "schema": "bayesfolio.features_dataset.command",
            "tickers": ETF_TICKERS,
            "drop_assets": [],
            "lookback_date": date(2019, 3, 1),
            "start_date": date(2021, 3, 1),
            "end_date": date(2026, 7, 2),
            "interval": Interval.DAILY,
            "horizon": Horizon.MONTHLY,
            "etf_cols": SELECTED_ETF_COLS,
            "macro_cols": SELECTED_MACRO_COLS,
            "drop_etf_cols": [],
            "drop_macro_cols": [],
            "clip_quantile": 0.99,
            "seed": 27,
            "artifact_name": FEATURE_ARTIFACT_NAME,
            "include_unlabeled_tail": True,
        }
    )
    providers = make_default_feature_providers(cache_root="artifacts/cache")
    artifact_store = ParquetArtifactStore(base_dir=str(FEATURE_ARTIFACT_ROOT))
    result = build_features_dataset(command=command, providers=providers, artifact_store=artifact_store)
    print(f"Feature artifact: {result.artifact.uri}")
    print(f"Rows: {result.artifact.row_count}; columns: {result.artifact.column_count}")
    for note in result.diagnostics:
        print(f"Diagnostic: {note}")
    uri = str(result.artifact.uri)
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        return Path(parsed.path)
    return Path(uri)


def model_args(feature_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        feature_path=feature_path,
        output_dir=OUTPUT_DIR,
        run_id=RUN_ID,
        max_windows=11,
        maxiter=75,
        seed=27,
        posterior_scenarios=5000,
        train_months=None,
        include_live_window=True,
        drop_incomplete_feature_dates=False,
        min_feature_date=None,
        min_inferred_noise_level=0.0025,
        upperlng=0.25,
        nea=8,
        gp_experiment="positive_no_prior",
        input_transform_mode="botorch_normalize",
        time_modulation_mode="lengthscale_only",
        kernel_proposal="none",
        kernel_half_life_months=36.0,
        kernel_changepoint_date="2021-03-31",
        kernel_changepoint_width_months=6.0,
        kernel_composition_proposal="e_plus_m_plus_t_plus_et_plus_mt_plus_emt",
        mean_kind="multitask_constant",
        scenario_mean_scale=1.0,
        turnover_blend=0.50,
        lengthscale_floor=0.02,
        outputscale_floor=0.01,
        outputscale_prior_median=0.05,
        outputscale_prior_sigma=0.75,
        recency_half_life_months=None,
        recency_base_noise_level=5e-3,
        recency_max_noise_multiplier=100.0,
        task_noise_floor_raw_std=0.005,
    )


def scenario_summary(label: str, scenarios: pd.DataFrame, weights: pd.Series) -> dict[str, Any]:
    aligned = scenarios.reindex(columns=weights.index).fillna(0.0)
    portfolio_returns = aligned.to_numpy() @ weights.to_numpy()
    return {
        "setting": label,
        "scenario_mean": float(np.mean(portfolio_returns)),
        "scenario_std": float(np.std(portfolio_returns, ddof=0)),
        "scenario_p05": float(np.quantile(portfolio_returns, 0.05)),
        "scenario_p50": float(np.quantile(portfolio_returns, 0.50)),
        "scenario_p95": float(np.quantile(portfolio_returns, 0.95)),
        "scenario_cvar05": float(np.mean(portfolio_returns[portfolio_returns <= np.quantile(portfolio_returns, 0.05)])),
        "max_weight": float(weights.max()),
        "effective_n": float(1.0 / np.square(weights.to_numpy()).sum()),
        "nonzero_weight_count": int((weights > 1e-8).sum()),
    }


def write_manifest(args: argparse.Namespace, live_date: pd.Timestamp, train_df: pd.DataFrame) -> None:
    manifest = {
        "schema": "bayesfolio.july_portfolio_full3.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": RUN_ID,
        "feature_path": str(args.feature_path),
        "feature_sha256": runner.sha256_file(args.feature_path),
        "git_sha": runner.git_sha(short=False),
        "git_dirty_summary_at_start": runner.git_dirty_summary(),
        "portfolio_construction_date": live_date.date().isoformat(),
        "training_rows": int(len(train_df)),
        "training_date_min": train_df["date"].min().date().isoformat(),
        "training_date_max": train_df["date"].max().date().isoformat(),
        "modeling": {
            "experiment": args.gp_experiment,
            "variant": runner.gp_variant_name(args.gp_experiment),
            "task_kernel": "PositiveIndexKernel",
            "task_covar_prior": None,
            "min_inferred_noise_level": args.min_inferred_noise_level,
            "time_modulation_mode": args.time_modulation_mode,
            "kernel_composition_proposal": args.kernel_composition_proposal,
            "input_transform": runner.input_transform_description(args.input_transform_mode),
            "outcome_transform": "StratifiedStandardize by ETF task",
            "maxiter": args.maxiter,
            "seed": args.seed,
            "posterior_scenarios": args.posterior_scenarios,
        },
        "riskfolio_recommendation": {
            "scenario_method": "GP posterior scenarios with method_mu=hist and method_cov=hist",
            "risk_measure": "CVaR",
            "objective": "Sharpe",
            "recommended_upperlng": 0.25,
            "recommended_nea": 10,
            "sensitivity_grid": RISKFOLIO_GRID,
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def create_portfolio() -> None:
    torch.set_default_dtype(torch.float64)
    feature_path = build_july_features()
    args = model_args(feature_path)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = runner.task_exp.load_features(args.feature_path)
    scored_dates, live_date = runner.task_exp.scored_and_live_dates(df, args.max_windows)
    if live_date is None:
        live_date = pd.Timestamp(df["date"].max())
        labeled_dates = sorted(
            pd.Timestamp(date) for date in df.loc[df[runner.task_exp.TARGET_COL].notna(), "date"].unique()
        )
        scored_dates = [date for date in labeled_dates if date < live_date][-args.max_windows :]
        print(
            "No unlabeled tail found; treating max feature date "
            f"{live_date.date()} as the live construction date and excluding it from training."
        )

    train_df = df[(df["date"] < live_date) & df[runner.task_exp.TARGET_COL].notna()].copy()
    eval_df = df[df["date"] == live_date].copy()
    final_universe = [asset for asset in runner.task_exp.ETF_TICKERS if asset not in runner.HELPER_ASSETS]
    window_index = len(scored_dates)
    seed = runner.task_exp.stable_seed(args.seed, runner.gp_variant_name(args.gp_experiment), window_index)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios, predictions, model_diagnostics, task_diag = runner.fit_gp_window(
        train_df,
        eval_df,
        args=args,
        seed=seed,
        maxiter=args.maxiter,
        posterior_scenarios=args.posterior_scenarios,
    )
    final_scenarios = scenarios.loc[:, final_universe]
    final_scenarios.to_csv(OUTPUT_DIR / "july_gp_posterior_scenarios.csv", index=False)
    predictions.sort_values("score", ascending=False).to_csv(OUTPUT_DIR / "july_gameday_predictions.csv", index=False)
    pd.DataFrame(model_diagnostics).to_csv(OUTPUT_DIR / "july_model_diagnostics.csv", index=False)
    if task_diag is not None:
        if isinstance(task_diag, pd.DataFrame):
            task_diag.to_csv(OUTPUT_DIR / "july_task_covariance_diagnostics.csv", index=False)
        else:
            pd.DataFrame([task_diag]).to_csv(OUTPUT_DIR / "july_task_covariance_diagnostics.csv", index=False)

    weight_frames = []
    summary_rows = []
    for spec in RISKFOLIO_GRID:
        weights = (
            runner.optimize_riskfolio(
                final_scenarios,
                method_mu="hist",
                method_cov="hist",
                upperlng=spec["upperlng"],
                nea=spec["nea"],
            )
            .reindex(final_universe)
            .fillna(0.0)
        )
        weight_frames.append(weights.rename(spec["label"]))
        summary_rows.append(scenario_summary(spec["label"], final_scenarios, weights))

    weights_df = pd.DataFrame(weight_frames)
    weights_df.index.name = "setting"
    weights_df.to_csv(OUTPUT_DIR / "july_riskfolio_weight_sensitivity.csv")
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "july_riskfolio_sensitivity_summary.csv", index=False)
    weights_df.loc[["recommended_25_10"]].T.rename(columns={"recommended_25_10": "weight"}).to_csv(
        OUTPUT_DIR / "july_recommended_weights.csv"
    )
    write_manifest(args, live_date, train_df)

    print(f"Portfolio construction date: {live_date.date()}")
    print(f"Training window: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"Wrote outputs to {OUTPUT_DIR}")
    print(weights_df.T.to_string())
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    create_portfolio()
