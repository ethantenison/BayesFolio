"""Create the July 2026 portfolio with the TVLS E*T + M*T + E*M*T setup.

This notebook-script mirrors the monthly portfolio notebooks while preserving the
lineage from the tested run:

    20260701_positive_et_plus_mt_plus_emt_11realized_plus_live_tvls_screen

It fits only the live construction window and writes scenarios, predictions,
weights, and a small manifest under:

    notebooks/outputs/20260701_portfolio/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - only used outside notebooks
    display = print


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RUNNER_PATH = REPO_ROOT / "experiments" / "2026-06-portfolio-optimization" / "run_monthly_optimization_walkforward.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("portfolio_walkforward_runner", RUNNER_PATH)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise RuntimeError(f"Unable to import portfolio runner from {RUNNER_PATH}")
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)


RUN_ID = "20260701_portfolio_tvls_et_plus_mt_plus_emt"
OUTPUT_DIR = REPO_ROOT / "notebooks" / "outputs" / "20260701_portfolio"
FEATURE_PATH = Path(
    "/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_18etf_common_history_201306_202605.parquet"
)
CONTROL_RUN_DIR = (
    REPO_ROOT
    / "experiments"
    / "2026-06-portfolio-optimization"
    / "outputs"
    / "runs"
    / "20260701_positive_et_plus_mt_plus_emt_11realized_plus_live_tvls_screen"
)


def config() -> argparse.Namespace:
    """Return the exact model/optimizer configuration used for the TVLS control."""

    return argparse.Namespace(
        feature_path=FEATURE_PATH,
        output_dir=OUTPUT_DIR,
        run_id=RUN_ID,
        max_windows=11,
        maxiter=75,
        seed=27,
        posterior_scenarios=5000,
        include_live_window=True,
        drop_incomplete_feature_dates=True,
        min_inferred_noise_level=5e-3,
        upperlng=0.20,
        nea=10,
        gp_experiment="positive_no_prior",
        input_transform_mode="botorch_normalize",
        time_modulation_mode="lengthscale_only",
        kernel_proposal="none",
        kernel_half_life_months=36.0,
        kernel_changepoint_date="2021-03-31",
        kernel_changepoint_width_months=6.0,
        kernel_composition_proposal="et_plus_mt_plus_emt",
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


def build_manifest(args: argparse.Namespace, live_date: pd.Timestamp, train_df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact notebook manifest for the single live portfolio run."""

    return {
        "schema": "bayesfolio.monthly_portfolio_notebook.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": RUN_ID,
        "source_control_run": str(CONTROL_RUN_DIR),
        "source_control_manifest": str(CONTROL_RUN_DIR / "manifest.json"),
        "feature_path": str(args.feature_path),
        "feature_sha256": runner.sha256_file(args.feature_path),
        "git_sha": runner.git_sha(short=False),
        "git_dirty_summary_at_start": runner.git_dirty_summary(),
        "portfolio_construction_date": live_date.date().isoformat(),
        "training_rows": int(len(train_df)),
        "training_date_min": pd.Timestamp(train_df["date"].min()).date().isoformat(),
        "training_date_max": pd.Timestamp(train_df["date"].max()).date().isoformat(),
        "helper_assets_fit_but_excluded": sorted(runner.HELPER_ASSETS),
        "final_portfolio_universe": [
            asset for asset in runner.task_exp.ETF_TICKERS if asset not in runner.HELPER_ASSETS
        ],
        "modeling": {
            "experiment": args.gp_experiment,
            "variant": runner.gp_variant_name(args.gp_experiment),
            "task_kernel": "PositiveIndexKernel",
            "task_covar_prior": None,
            "rank": runner.task_exp.RANK,
            "min_inferred_noise_level": args.min_inferred_noise_level,
            "input_transform": runner.input_transform_description(args.input_transform_mode),
            "outcome_transform": "StratifiedStandardize by ETF task",
            "time_modulation_mode": args.time_modulation_mode,
            "kernel_proposal": args.kernel_proposal,
            "kernel_composition_proposal": {
                "name": args.kernel_composition_proposal,
                "components": "E=Matern(0.5), M=Linear+Matern(0.5)+RQ, T=Matern(0.5)",
            },
            "maxiter": args.maxiter,
            "seed": args.seed,
            "posterior_scenarios": args.posterior_scenarios,
        },
        "portfolio": {
            "method_mu": "hist",
            "method_cov": "hist",
            "model": "Classic",
            "rm": "CVaR",
            "obj": "Sharpe",
            "upperlng": args.upperlng,
            "nea": args.nea,
            "alpha": 0.5,
            "excluded_from_final_weights": sorted(runner.HELPER_ASSETS),
        },
    }


def create_portfolio() -> dict[str, pd.DataFrame | pd.Series | float]:
    """Fit the live TVLS GP and create the GP-scenario Riskfolio portfolio."""

    torch.set_default_dtype(torch.float64)
    args = config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = runner.task_exp.load_features(args.feature_path)
    df = runner.drop_incomplete_feature_dates(df)
    scored_dates, live_date = runner.task_exp.scored_and_live_dates(df, args.max_windows)
    if live_date is None:
        raise RuntimeError("Expected an unlabeled live construction date, but none was found.")

    train_df = df[(df["date"] < live_date) & df[runner.task_exp.TARGET_COL].notna()].copy()
    eval_df = df[df["date"] == live_date].copy()
    final_universe = [asset for asset in runner.task_exp.ETF_TICKERS if asset not in runner.HELPER_ASSETS]

    window_index = len(scored_dates)
    seed = runner.task_exp.stable_seed(args.seed, runner.gp_variant_name(args.gp_experiment), window_index)
    scenarios, predictions, model_diagnostics, task_diag = runner.fit_gp_window(
        train_df,
        eval_df,
        args=args,
        seed=seed,
        maxiter=args.maxiter,
        posterior_scenarios=args.posterior_scenarios,
    )
    final_scenarios = scenarios.loc[:, final_universe]
    weights = (
        runner.optimize_riskfolio(
            final_scenarios,
            method_mu="hist",
            method_cov="hist",
            upperlng=args.upperlng,
            nea=args.nea,
        )
        .reindex(final_universe)
        .fillna(0.0)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args, live_date, train_df)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    final_scenarios.to_csv(OUTPUT_DIR / "20260701_gp_posterior_scenarios.csv", index=False)
    predictions.sort_values("score", ascending=False).to_csv(
        OUTPUT_DIR / "20260701_gameday_predictions.csv",
        index=False,
    )
    weights.rename("weight").to_frame().to_csv(OUTPUT_DIR / "20260701_gp_weights.csv")
    pd.DataFrame(model_diagnostics).to_csv(OUTPUT_DIR / "20260701_model_diagnostics.csv", index=False)
    if task_diag is not None:
        task_diag.to_csv(OUTPUT_DIR / "20260701_task_covariance_diagnostics.csv", index=False)

    print(f"Portfolio construction date: {live_date.date()}")
    print(f"Training window: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"Wrote outputs to {OUTPUT_DIR}")
    display(weights.sort_values(ascending=False).to_frame("weight").T)
    return {
        "scenarios": final_scenarios,
        "predictions": predictions,
        "weights": weights,
        "live_date": live_date,
    }


if __name__ == "__main__":
    create_portfolio()
