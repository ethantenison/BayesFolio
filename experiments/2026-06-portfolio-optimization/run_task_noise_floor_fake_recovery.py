"""Fake-data recovery check for the signed task-noise-floor portfolio GP.

This script simulates ETF monthly excess returns on the real feature/date/task
panel, then fits the same GP paths used by the portfolio walk-forward script.

The recovery target is the known latent mean and ETF rank ordering, not the
noisy realized return.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PORTFOLIO_EXPERIMENT = REPO_ROOT / "experiments" / "2026-06-portfolio-optimization" / (
    "run_monthly_optimization_walkforward.py"
)
SPEC = importlib.util.spec_from_file_location("portfolio_walkforward", PORTFOLIO_EXPERIMENT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load portfolio experiment from {PORTFOLIO_EXPERIMENT}")
portfolio_exp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = portfolio_exp
SPEC.loader.exec_module(portfolio_exp)
task_exp = portfolio_exp.task_exp

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_PATH = portfolio_exp.DEFAULT_FEATURE_PATH
OUTPUT_ROOT = EXPERIMENT_DIR / "outputs" / "fake_data_recovery"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    gp_experiment: str
    task_noise_floor_raw_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--train-months", type=int, default=36)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--task-noise-floor-raw-std", type=float, default=0.005)
    parser.add_argument("--posterior-scenarios", type=int, default=1)
    return parser.parse_args()


def git_sha(*, short: bool = False) -> str:
    cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
    return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()


def git_dirty_summary() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    run_id = args.run_id or f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{git_sha(short=True)}"
    return OUTPUT_ROOT / "runs" / run_id


def model_args(spec: ModelSpec, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        gp_experiment=spec.gp_experiment,
        scenario_mean_scale=1.0,
        turnover_blend=0.5,
        lengthscale_floor=0.02,
        outputscale_floor=0.01,
        outputscale_prior_median=0.05,
        outputscale_prior_sigma=0.75,
        task_noise_floor_raw_std=spec.task_noise_floor_raw_std,
    )


def simulate_targets(df: pd.DataFrame, *, seed: int, raw_noise_floor: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sim = df.copy()
    feature_frame = sim.loc[:, task_exp.INPUT_COLUMNS].astype(float)
    normalized = (feature_frame - feature_frame.mean()) / feature_frame.std(ddof=0).replace(0.0, 1.0)
    normalized = normalized.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    coeffs = pd.Series(0.0, index=task_exp.INPUT_COLUMNS)
    coeffs.update(
        {
            "lag_y_excess_lead": 0.006,
            "mom12m": 0.005,
            "cs_mom_rank": 0.004,
            "vol_z": -0.003,
            "hy_spread_z_12m": -0.004,
            "vix_ts_z_12m": -0.003,
            "erp": 0.003,
            "pct_above_50dma": 0.003,
            "em_fx_ret": 0.002,
        }
    )
    feature_signal = normalized.to_numpy() @ coeffs.to_numpy(dtype=float)

    groups = pd.Series(task_exp.ASSET_GROUPS)
    group_effect = {
        "us_equity": 0.005,
        "real_estate": 0.002,
        "intl_equity": 0.001,
        "fixed_income": -0.001,
        "credit": 0.0015,
    }
    asset_offsets = pd.Series(
        rng.normal(0.0, 0.0025, size=len(task_exp.ETF_TICKERS)),
        index=task_exp.ETF_TICKERS,
    )
    asset = sim["asset_id"].astype(str)
    task_signal = asset.map(groups.map(group_effect)).astype(float).to_numpy()
    task_signal += asset.map(asset_offsets).astype(float).to_numpy()

    month_index = pd.factorize(sim["date"])[0].astype(float)
    cyclical_signal = 0.003 * np.sin(month_index / 5.0) + 0.0015 * np.cos(month_index / 11.0)
    latent_mean = feature_signal + task_signal + cyclical_signal

    multipliers = np.linspace(1.0, 2.8, len(task_exp.ETF_TICKERS))
    noise_std_by_asset = pd.Series(raw_noise_floor * multipliers, index=task_exp.ETF_TICKERS)
    noise_std = asset.map(noise_std_by_asset).astype(float).to_numpy()
    observed = latent_mean + rng.normal(0.0, noise_std)

    sim[task_exp.TARGET_COL] = observed
    sim["latent_mean"] = latent_mean
    sim["true_noise_raw_std"] = noise_std
    return sim


def restrict_train_months(train_df: pd.DataFrame, *, months: int) -> pd.DataFrame:
    if months <= 0:
        return train_df
    dates = sorted(pd.to_datetime(train_df["date"]).unique())
    keep_dates = dates[-months:]
    return train_df[train_df["date"].isin(keep_dates)].copy()


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return math.nan
    return float(a.corr(b, method="spearman"))


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    residual_latent = predictions["y_pred"] - predictions["latent_mean"]
    residual_observed = predictions["y_pred"] - predictions["y_true"]
    final = predictions[predictions["asset_id"].isin(portfolio_exp.HELPER_ASSETS) == False]  # noqa: E712
    within_80 = (predictions["y_true"] - predictions["y_pred"]).abs() <= 1.2816 * predictions["y_std"]
    within_95 = (predictions["y_true"] - predictions["y_pred"]).abs() <= 1.96 * predictions["y_std"]
    return {
        "n_predictions": float(len(predictions)),
        "latent_mae": float(residual_latent.abs().mean()),
        "latent_rmse": float(np.sqrt(np.mean(np.square(residual_latent)))),
        "observed_mae": float(residual_observed.abs().mean()),
        "observed_rmse": float(np.sqrt(np.mean(np.square(residual_observed)))),
        "mean_pred_std": float(predictions["y_std"].mean()),
        "spearman_pred_vs_latent_all": safe_spearman(predictions["y_pred"], predictions["latent_mean"]),
        "spearman_pred_vs_observed_all": safe_spearman(predictions["y_pred"], predictions["y_true"]),
        "spearman_pred_vs_latent_final_universe": safe_spearman(final["y_pred"], final["latent_mean"]),
        "spearman_pred_vs_observed_final_universe": safe_spearman(final["y_pred"], final["y_true"]),
        "observed_80pct_interval_coverage": float(within_80.mean()),
        "observed_95pct_interval_coverage": float(within_95.mean()),
    }


def write_plots(predictions: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    for model_name, frame in predictions.groupby("model"):
        axes[0].scatter(frame["latent_mean"], frame["y_pred"], label=model_name, alpha=0.75)
    lo = float(min(predictions["latent_mean"].min(), predictions["y_pred"].min()))
    hi = float(max(predictions["latent_mean"].max(), predictions["y_pred"].max()))
    axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
    axes[0].set_title("Latent Mean Recovery")
    axes[0].set_xlabel("Known latent mean")
    axes[0].set_ylabel("Predicted mean")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    pivot = metrics.pivot(index="window_date", columns="model", values="spearman_pred_vs_latent_final_universe")
    pivot.plot(ax=axes[1], marker="o")
    axes[1].set_title("Final-Universe Rank Recovery")
    axes[1].set_xlabel("Window")
    axes[1].set_ylabel("Spearman vs latent mean")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fake_data_recovery_diagnostics.png", dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = [str(column) for column in formatted.columns]
    rows = formatted.astype(str).values.tolist()
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def write_manifest(args: argparse.Namespace, output_dir: Path, scored_dates: list[pd.Timestamp]) -> None:
    manifest = {
        "schema": "bayesfolio.task_noise_floor_fake_recovery.manifest.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git_sha": git_sha(short=False),
        "git_dirty_summary_at_start": git_dirty_summary(),
        "feature_path": str(args.feature_path),
        "feature_sha256": sha256_file(args.feature_path),
        "source_portfolio_experiment": str(PORTFOLIO_EXPERIMENT),
        "output_dir": str(output_dir),
        "simulation": {
            "seed": args.seed,
            "target_col": task_exp.TARGET_COL,
            "raw_noise_floor": args.task_noise_floor_raw_std,
            "true_noise_raw_std": "raw_noise_floor times linearly spaced ETF multiplier from 1.0 to 2.8",
            "latent_mean": "linear feature signal + group/task offsets + smooth monthly cycle",
        },
        "fit": {
            "models": ["signed_lkj_eta_2", "signed_lkj_eta_2_task_noise_floor"],
            "max_windows": args.max_windows,
            "train_months": args.train_months,
            "maxiter": args.maxiter,
            "posterior_scenarios": args.posterior_scenarios,
            "rebalance_dates": [date.date().isoformat() for date in scored_dates],
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    real_df = task_exp.load_features(args.feature_path)
    scored_dates, _ = task_exp.scored_and_live_dates(real_df, args.max_windows)
    sim_df = simulate_targets(real_df, seed=args.seed, raw_noise_floor=args.task_noise_floor_raw_std)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_manifest(args, output_dir, scored_dates)

    specs = [
        ModelSpec("signed_lkj_eta_2", "signed_lkj_eta_2", args.task_noise_floor_raw_std),
        ModelSpec(
            "signed_lkj_eta_2_task_noise_floor",
            "signed_lkj_eta_2_task_noise_floor",
            args.task_noise_floor_raw_std,
        ),
    ]
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    task_diag_rows: list[dict[str, Any]] = []

    for window_index, window_date in enumerate(scored_dates):
        train_df = sim_df[(sim_df["date"] < window_date) & sim_df[task_exp.TARGET_COL].notna()].copy()
        train_df = restrict_train_months(train_df, months=args.train_months)
        eval_df = sim_df[sim_df["date"] == window_date].copy()
        print(f"window {window_date.date()} train_rows={len(train_df)} eval_rows={len(eval_df)}", flush=True)

        for spec in specs:
            print(f"  fitting {spec.name}", flush=True)
            fit_args = model_args(spec, args)
            _, predictions, model_diagnostics, task_diag = portfolio_exp.fit_gp_window(
                train_df,
                eval_df,
                args=fit_args,
                seed=task_exp.stable_seed(args.seed, spec.name, window_index),
                maxiter=args.maxiter,
                posterior_scenarios=args.posterior_scenarios,
            )
            eval_truth = eval_df.loc[:, ["asset_id", "latent_mean", "true_noise_raw_std"]].copy()
            eval_truth["asset_id"] = eval_truth["asset_id"].astype(str)
            predictions = predictions.merge(eval_truth, on="asset_id", how="left")
            predictions["model"] = spec.name
            predictions["window_date"] = window_date.date().isoformat()
            prediction_rows.append(predictions)
            metric_rows.append(
                {
                    "window_date": window_date.date().isoformat(),
                    "model": spec.name,
                    **prediction_metrics(predictions),
                }
            )
            for row in model_diagnostics:
                row["model"] = spec.name
                diagnostic_rows.append(row)
            if task_diag is not None:
                task_diag["model"] = spec.name
                task_diag_rows.append(task_diag)

    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    summary_df = metrics_df.groupby("model", as_index=False).mean(numeric_only=True)
    predictions_df.to_csv(output_dir / "fake_data_predictions.csv", index=False)
    metrics_df.to_csv(output_dir / "window_recovery_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "recovery_summary.csv", index=False)
    if diagnostic_rows:
        pd.DataFrame(diagnostic_rows).to_csv(output_dir / "model_diagnostics.csv", index=False)
    if task_diag_rows:
        pd.DataFrame(task_diag_rows).to_csv(output_dir / "task_covariance_diagnostics.csv", index=False)
    true_noise = (
        predictions_df[["asset_id", "true_noise_raw_std"]]
        .drop_duplicates()
        .sort_values("asset_id")
        .reset_index(drop=True)
    )
    true_noise.to_csv(output_dir / "true_task_noise.csv", index=False)
    write_plots(predictions_df, metrics_df, output_dir)

    report = [
        "# Task Noise Floor Fake-Data Recovery",
        "",
        f"Run directory: `{output_dir}`",
        f"Windows: `{len(scored_dates)}`; training history per window: last `{args.train_months}` months.",
        f"Raw monthly task-noise floor: `{args.task_noise_floor_raw_std}`.",
        "",
        "## Recovery Summary",
        "",
        markdown_table(summary_df),
        "",
        "## Interpretation Guardrails",
        "",
        (
            "- Targets were synthetic, but the feature/date/task panel and walk-forward geometry came from the "
            "real June 2026 feature artifact."
        ),
        (
            "- Recovery is scored against the known latent mean and final-universe ETF ranking, not only noisy "
            "observed returns."
        ),
        (
            "- This checks the implementation path and calibration behavior; it does not prove the real-data "
            "backtest improvement is stable."
        ),
        (
            "- The noise-floor candidate earns more confidence only if it improves or preserves latent recovery "
            "without pathological coverage/noise diagnostics."
        ),
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")

    print(summary_df.to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    run(parse_args())
