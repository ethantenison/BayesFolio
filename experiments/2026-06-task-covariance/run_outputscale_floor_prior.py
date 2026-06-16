"""Roll-forward run with outputscale floor and prior for the task-covariance GP."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import run_task_covariance_rollforward as exp
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import ScaleKernel
from gpytorch.priors import LogNormalPrior

RUN_ID = "20260614_positive_beta_both_outputscale_floor_prior"
VARIANT_NAME = "positive_beta_prior"
TIME_MODULATION_MODE = "both"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=exp.DEFAULT_FEATURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--outputscale-floor", type=float, default=0.01)
    parser.add_argument("--outputscale-prior-median", type=float, default=0.05)
    parser.add_argument("--outputscale-prior-sigma", type=float, default=0.75)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = exp.OUTPUT_ROOT / "runs" / RUN_ID
    return args


def apply_outputscale_floor_prior(
    model: torch.nn.Module,
    *,
    floor: float,
    prior_median: float,
    prior_sigma: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prior = LogNormalPrior(loc=float(np.log(prior_median)), scale=prior_sigma)
    constraint = GreaterThan(floor)
    for index, (path, module) in enumerate(model.named_modules()):
        if not isinstance(module, ScaleKernel):
            continue
        module.register_constraint("raw_outputscale", constraint)
        module.register_prior("floor_lognormal_outputscale_prior", prior, "outputscale")
        module.initialize(outputscale=max(prior_median, floor * 1.01))
        rows.append(
            {
                "scale_index": index,
                "module_path": path,
                "base_kernel_type": module.base_kernel.__class__.__name__,
                "outputscale_floor": floor,
                "outputscale_prior_median": prior_median,
                "outputscale_prior_sigma": prior_sigma,
            }
        )
    return rows


def collect_outputscale_diagnostics(
    *,
    model: torch.nn.Module,
    window_date: pd.Timestamp | None,
    window_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    date = window_date.date().isoformat() if window_date is not None else "live"
    for index, (path, module) in enumerate(model.named_modules()):
        if not isinstance(module, ScaleKernel):
            continue
        values = module.outputscale.detach().cpu().reshape(-1)
        rows.append(
            {
                "variant": VARIANT_NAME,
                "date": date,
                "window_index": window_index,
                "scale_index": index,
                "module_path": path,
                "base_kernel_type": module.base_kernel.__class__.__name__,
                "learned_outputscale_mean": float(values.mean()),
                "learned_outputscale_min": float(values.min()),
                "learned_outputscale_max": float(values.max()),
            }
        )
    return rows


def prediction_records(
    *,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> list[dict[str, Any]]:
    train_means = exp.training_asset_means(train_df, eval_df)
    rows: list[dict[str, Any]] = []
    for row, mean, std, train_mean in zip(eval_df.itertuples(index=False), y_pred, y_std, train_means, strict=True):
        rows.append(
            {
                "variant": VARIANT_NAME,
                "date": pd.Timestamp(row.date).date().isoformat(),
                "asset_id": str(row.asset_id),
                "train_mean": float(train_mean),
                "y_true": float(row.y_excess_lead),
                "y_pred": float(mean),
                "y_std": float(std),
                "y_true_resid": float(row.y_excess_lead - train_mean),
                "y_pred_resid": float(mean - train_mean),
            }
        )
    return rows


def fit_window(
    *,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    window_index: int,
    window_date: pd.Timestamp | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    variant = exp.VARIANTS[VARIANT_NAME]
    torch.manual_seed(exp.stable_seed(args.seed, VARIANT_NAME, window_index))
    train_x, train_y, eval_x, _, _, _ = exp.prepare_window_tensors(train_df, eval_df)
    model = exp.build_model(
        train_x,
        train_y,
        variant,
        time_modulation_mode=TIME_MODULATION_MODE,
    )
    applied_rows = apply_outputscale_floor_prior(
        model,
        floor=args.outputscale_floor,
        prior_median=args.outputscale_prior_median,
        prior_sigma=args.outputscale_prior_sigma,
    )
    y_pred, y_std = exp.fit_and_predict(model, eval_x, maxiter=args.maxiter)
    diagnostics = collect_outputscale_diagnostics(
        model=model,
        window_date=window_date,
        window_index=window_index,
    )
    task_diag = (
        exp.covariance_diagnostics(
            exp.task_correlation(model),
            variant=VARIANT_NAME,
            window_date=window_date,
        )
        if window_date is not None
        else {}
    )
    return y_pred, y_std, applied_rows, diagnostics, task_diag


def run(args: argparse.Namespace) -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = exp.load_features(args.feature_path)
    scored_dates, live_date = exp.scored_and_live_dates(df, args.max_windows)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema": "bayesfolio.task_covariance_outputscale_floor_prior.manifest.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git_sha": exp.git_sha(short=False),
        "git_dirty_summary_at_start": exp.git_dirty_summary(),
        "feature_path": str(args.feature_path),
        "feature_sha256": exp.sha256_file(args.feature_path),
        "variant": VARIANT_NAME,
        "time_modulation_mode": TIME_MODULATION_MODE,
        "maxiter": args.maxiter,
        "seed": args.seed,
        "outputscale_floor": args.outputscale_floor,
        "outputscale_prior_median": args.outputscale_prior_median,
        "outputscale_prior_sigma": args.outputscale_prior_sigma,
        "scored_dates": [date.date().isoformat() for date in scored_dates],
        "live_date": live_date.date().isoformat() if live_date is not None else None,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    pred_rows: list[dict[str, Any]] = []
    window_metric_rows: list[dict[str, Any]] = []
    applied_rows: list[dict[str, Any]] = []
    outputscale_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    live_rows: list[dict[str, Any]] = []

    for window_index, window_date in enumerate(scored_dates):
        print(f"window {window_date.date()}", flush=True)
        train_df = df[(df["date"] < window_date) & df[exp.TARGET_COL].notna()].copy()
        eval_df = df[df["date"] == window_date].copy()
        y_pred, y_std, applied, outputscale_diag, task_diag = fit_window(
            train_df=train_df,
            eval_df=eval_df,
            window_index=window_index,
            window_date=window_date,
            args=args,
        )
        y_true = eval_df[exp.TARGET_COL].to_numpy(dtype=float)
        window_metric_rows.append(
            {
                "variant": VARIANT_NAME,
                "date": window_date.date().isoformat(),
                **exp.window_scalar_metrics(y_true, y_pred, y_std),
            }
        )
        applied_rows.extend(applied)
        outputscale_rows.extend(outputscale_diag)
        diag_rows.append(task_diag)
        pred_rows.extend(prediction_records(train_df=train_df, eval_df=eval_df, y_pred=y_pred, y_std=y_std))

    if live_date is not None:
        print(f"live {live_date.date()}", flush=True)
        train_df = df[(df["date"] < live_date) & df[exp.TARGET_COL].notna()].copy()
        eval_df = df[df["date"] == live_date].copy()
        y_pred, y_std, applied, outputscale_diag, _ = fit_window(
            train_df=train_df,
            eval_df=eval_df,
            window_index=len(scored_dates),
            window_date=None,
            args=args,
        )
        applied_rows.extend(applied)
        outputscale_rows.extend(outputscale_diag)
        train_means = exp.training_asset_means(train_df, eval_df)
        for row, mean, std, train_mean in zip(eval_df.itertuples(index=False), y_pred, y_std, train_means, strict=True):
            live_rows.append(
                {
                    "variant": VARIANT_NAME,
                    "date": pd.Timestamp(row.date).date().isoformat(),
                    "asset_id": str(row.asset_id),
                    "train_mean": float(train_mean),
                    "y_pred": float(mean),
                    "y_std": float(std),
                    "score": float(mean / max(float(std), 1e-12)),
                    "y_pred_resid": float(mean - train_mean),
                    "resid_score": float((mean - train_mean) / max(float(std), 1e-12)),
                }
            )

    summary = pd.DataFrame([exp.summarize_variant(VARIANT_NAME, pred_rows)])
    pd.DataFrame(pred_rows).to_csv(args.output_dir / "window_predictions.csv", index=False)
    pd.DataFrame(window_metric_rows).to_csv(args.output_dir / "window_metrics.csv", index=False)
    pd.DataFrame(applied_rows).drop_duplicates().to_csv(
        args.output_dir / "outputscale_prior_application.csv",
        index=False,
    )
    pd.DataFrame(outputscale_rows).to_csv(args.output_dir / "outputscale_diagnostics.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(args.output_dir / "task_covariance_diagnostics.csv", index=False)
    pd.DataFrame(live_rows).to_csv(args.output_dir / "live_june_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "variant_summary.csv", index=False)
    summary.to_csv(args.output_dir / f"summary_{VARIANT_NAME}.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    run(parse_args())
