"""Evaluate Student-t predictive tails for momentum_plus_macro MTGP forecasts."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PANEL = (
    EXPERIMENT_DIR / "runs/residual_scale_calibration_20260705T1739Z/momentum_plus_macro_oos_residual_panel.csv"
)
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "runs"
DEFAULT_RUN_ID = "student_t_likelihood_probe_20260705T1744Z"
OUTER_DATES = [
    "2026-01-30",
    "2026-02-27",
    "2026-03-31",
    "2026-04-30",
    "2026-05-29",
    "2026-06-30",
]
COVERAGE_LEVELS = [0.50, 0.80, 0.95]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--df-grid", type=float, nargs="+", default=[3, 4, 5, 7, 10, 15, 30])
    parser.add_argument("--scale-grid", type=float, nargs="+", default=[0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument(
        "--outer-last-n",
        type=int,
        default=None,
        help="Use the last N unique dates from the residual panel as outer evaluation dates.",
    )
    parser.add_argument(
        "--outer-dates",
        nargs="*",
        default=None,
        help="Explicit YYYY-MM-DD outer evaluation dates. Overrides --outer-last-n.",
    )
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def git_dirty_summary() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip()


def student_t_nll(residual: np.ndarray, scale: np.ndarray, df: float) -> np.ndarray:
    scale = np.clip(scale, 1e-12, None)
    return -t.logpdf(residual / scale, df=df) + np.log(scale)


def gaussian_nll(residual: np.ndarray, std: np.ndarray) -> np.ndarray:
    variance = np.square(np.clip(std, 1e-12, None))
    return 0.5 * (np.log(2.0 * math.pi * variance) + np.square(residual) / variance)


def choose_student_t_params(
    train_frame: pd.DataFrame,
    df_grid: list[float],
    scale_grid: list[float],
) -> dict[str, float]:
    residual = train_frame["residual"].to_numpy(dtype=float)
    base_std = train_frame["baseline_y_std"].to_numpy(dtype=float)
    best: dict[str, float] | None = None
    for df in df_grid:
        for scale_multiplier in scale_grid:
            nll = float(np.mean(student_t_nll(residual, base_std * scale_multiplier, df)))
            if best is None or nll < best["train_nll"]:
                best = {
                    "df": float(df),
                    "scale_multiplier": float(scale_multiplier),
                    "train_nll": nll,
                }
    if best is None:
        raise RuntimeError("Empty Student-t parameter grid")
    return best


def resolve_outer_dates(panel: pd.DataFrame, args: argparse.Namespace) -> list[pd.Timestamp]:
    available_dates = [pd.Timestamp(date) for date in sorted(panel["date"].dt.normalize().unique())]
    if args.outer_dates:
        requested = [pd.Timestamp(date).normalize() for date in args.outer_dates]
        missing = sorted(set(requested) - {date.normalize() for date in available_dates})
        if missing:
            missing_text = ", ".join(date.strftime("%Y-%m-%d") for date in missing)
            raise ValueError(f"Requested outer dates are missing from the residual panel: {missing_text}")
        return requested
    if args.outer_last_n is not None:
        if args.outer_last_n <= 0:
            raise ValueError("--outer-last-n must be positive")
        if args.outer_last_n > len(available_dates):
            raise ValueError(f"--outer-last-n={args.outer_last_n} exceeds available dates ({len(available_dates)})")
        return available_dates[-args.outer_last_n :]
    return [pd.Timestamp(date) for date in OUTER_DATES]


def coverage(frame: pd.DataFrame, candidate: str, level: float) -> float:
    residual = np.abs(frame["residual"].to_numpy(dtype=float))
    if candidate == "gaussian_baseline":
        threshold = norm.ppf((1.0 + level) / 2.0) * frame["y_std"].to_numpy(dtype=float)
    else:
        threshold = t.ppf((1.0 + level) / 2.0, df=frame["student_t_df"].to_numpy(dtype=float)) * frame[
            "student_t_scale"
        ].to_numpy(dtype=float)
    return float(np.mean(residual <= threshold))


def metric_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, frame in predictions.groupby("candidate", observed=True):
        y_true = frame["y_true"].to_numpy(dtype=float)
        y_pred = frame["y_pred"].to_numpy(dtype=float)
        residual = y_true - y_pred
        row: dict[str, Any] = {
            "candidate": candidate,
            "n": int(len(frame)),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(y_pred - y_true)),
            "mean_pred_std": float(np.mean(frame["y_std"].to_numpy(dtype=float))),
            "mean_nll": float(np.mean(frame["nll"].to_numpy(dtype=float))),
            "standardized_residual_std": float(
                np.std(residual / np.clip(frame["y_std"].to_numpy(dtype=float), 1e-12, None), ddof=1)
            ),
        }
        for level in COVERAGE_LEVELS:
            row[f"coverage_{int(level * 100)}"] = coverage(frame, candidate, level)
        rows.append(row)
    return rows


def build_predictions(panel: pd.DataFrame, outer_dates: list[pd.Timestamp], args: argparse.Namespace) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for outer_date in outer_dates:
        train_frame = panel[panel["date"] < outer_date].copy()
        eval_frame = panel[panel["date"].dt.normalize().eq(outer_date.normalize())].copy()
        gaussian = eval_frame.copy()
        gaussian["candidate"] = "gaussian_baseline"
        gaussian["y_std"] = gaussian["baseline_y_std"]
        gaussian["nll"] = gaussian_nll(gaussian["residual"].to_numpy(float), gaussian["y_std"].to_numpy(float))
        gaussian["student_t_df"] = np.nan
        gaussian["student_t_scale"] = np.nan
        rows.append(gaussian)

        tuned = choose_student_t_params(train_frame, args.df_grid, args.scale_grid)
        student = eval_frame.copy()
        student["candidate"] = "student_t_tuned_prior_residuals"
        student["student_t_df"] = tuned["df"]
        student["student_t_scale"] = student["baseline_y_std"] * tuned["scale_multiplier"]
        student["y_std"] = np.where(
            tuned["df"] > 2.0,
            student["student_t_scale"] * math.sqrt(tuned["df"] / (tuned["df"] - 2.0)),
            np.inf,
        )
        student["nll"] = student_t_nll(
            student["residual"].to_numpy(float),
            student["student_t_scale"].to_numpy(float),
            tuned["df"],
        )
        student["tuned_train_nll"] = tuned["train_nll"]
        student["scale_multiplier"] = tuned["scale_multiplier"]
        rows.append(student)

        for df in [3.0, 5.0, 7.0]:
            fixed = eval_frame.copy()
            fixed["candidate"] = f"student_t_df{int(df)}_variance_matched"
            fixed["student_t_df"] = df
            fixed["student_t_scale"] = fixed["baseline_y_std"] * math.sqrt((df - 2.0) / df)
            fixed["y_std"] = fixed["baseline_y_std"]
            fixed["nll"] = student_t_nll(
                fixed["residual"].to_numpy(float),
                fixed["student_t_scale"].to_numpy(float),
                df,
            )
            rows.append(fixed)
    return pd.concat(rows, ignore_index=True)


def write_plot(output_dir: Path, metrics: pd.DataFrame) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot = metrics.sort_values("mean_nll")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(plot["candidate"], plot["mean_nll"], color="#4f46e5")
    ax.invert_yaxis()
    ax.set_xlabel("Mean predictive NLL (lower is better)")
    ax.set_title("Student-t predictive tails vs Gaussian baseline")
    for i, value in enumerate(plot["mean_nll"]):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "student_t_nll_leaderboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_predicted_vs_actual_plot(output_dir: Path, predictions: pd.DataFrame, candidate: str) -> str:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    frame = predictions[predictions["candidate"].eq(candidate)].copy()
    if frame.empty:
        raise ValueError(f"No predictions found for candidate {candidate}")
    frame = frame.sort_values(["asset_id", "date"])
    assets = sorted(frame["asset_id"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, asset in zip(axes_flat, assets, strict=False):
        asset_frame = frame[frame["asset_id"].eq(asset)].sort_values("date")
        x = asset_frame["date"]
        y_pred = asset_frame["y_pred"].to_numpy(dtype=float)
        if candidate == "gaussian_baseline":
            q80 = norm.ppf(0.90) * asset_frame["y_std"].to_numpy(dtype=float)
            q95 = norm.ppf(0.975) * asset_frame["y_std"].to_numpy(dtype=float)
        else:
            df = asset_frame["student_t_df"].to_numpy(dtype=float)
            scale = asset_frame["student_t_scale"].to_numpy(dtype=float)
            q80 = t.ppf(0.90, df=df) * scale
            q95 = t.ppf(0.975, df=df) * scale
        ax.fill_between(x, y_pred - q95, y_pred + q95, color="#93c5fd", alpha=0.22, label="95% band")
        ax.fill_between(x, y_pred - q80, y_pred + q80, color="#2563eb", alpha=0.20, label="80% band")
        ax.plot(x, asset_frame["y_true"], color="#111827", marker="o", linewidth=1.6, label="Actual")
        ax.plot(x, y_pred, color="#dc2626", marker="s", linewidth=1.4, label="Predicted mean")
        ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
        ax.set_title(str(asset))
        ax.grid(True, alpha=0.20)
    for ax in axes_flat[len(assets) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"{candidate}: predicted vs actual by ETF", y=0.98)
    fig.supxlabel("Forecast date")
    fig.supylabel("One-month excess return")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    filename = f"predicted_vs_actual_{candidate}_by_etf.png"
    fig.savefig(fig_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{filename}"


def write_summary(
    output_dir: Path, metrics: pd.DataFrame, outer_dates: list[pd.Timestamp], plot_paths: list[str]
) -> None:
    metric_columns = [
        "candidate",
        "n",
        "rmse",
        "mean_nll",
        "coverage_50",
        "coverage_80",
        "coverage_95",
        "mean_pred_std",
    ]
    metric_table = metrics.loc[:, metric_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
    lines = [
        "# Student-t Likelihood Probe",
        "",
        "This run evaluates post-fit Student-t predictive tails on top of the existing "
        "`momentum_plus_macro` MTGP predictions. It does not refit the GP with a Student-t likelihood.",
        "",
        f"- Outer evaluation dates: {outer_dates[0].date()} through {outer_dates[-1].date()}",
        f"- Number of monthly dates: {len(outer_dates)}",
        f"- Number of forecast observations per candidate: {int(metrics['n'].iloc[0])}",
        "",
        "## Metrics",
        "",
        "```text",
        metric_table,
        "```",
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in plot_paths)
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "This is a predictive likelihood/tail wrapper. A true Student-t GP would require fitting the "
            "model with a Student-t likelihood or an equivalent robust observation model.",
            "",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.panel_path, parse_dates=["date"])
    outer_dates = resolve_outer_dates(panel, args)
    predictions = build_predictions(panel, outer_dates, args)
    metrics = pd.DataFrame(metric_rows(predictions)).sort_values("mean_nll")
    predictions.to_csv(output_dir / "student_t_predictions.csv", index=False)
    metrics.to_csv(output_dir / "student_t_metrics.csv", index=False)
    write_plot(output_dir, metrics)
    plot_paths = ["figures/student_t_nll_leaderboard.png"]
    plot_paths.append(write_predicted_vs_actual_plot(output_dir, predictions, "gaussian_baseline"))
    plot_paths.append(write_predicted_vs_actual_plot(output_dir, predictions, "student_t_df5_variance_matched"))
    plot_paths.append(write_predicted_vs_actual_plot(output_dir, predictions, "student_t_df7_variance_matched"))
    write_summary(output_dir, metrics, outer_dates, plot_paths)
    manifest = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "experiment_type": "student_t_predictive_likelihood_probe",
        "panel_path": str(args.panel_path),
        "outer_date_selection": {
            "outer_last_n": args.outer_last_n,
            "outer_dates_arg": args.outer_dates,
        },
        "outer_dates": [date.strftime("%Y-%m-%d") for date in outer_dates],
        "df_grid": args.df_grid,
        "scale_grid": args.scale_grid,
        "metrics": "student_t_metrics.csv",
        "predictions": "student_t_predictions.csv",
        "plots": plot_paths,
        "summary": "SUMMARY.md",
        "caveat": (
            "This is a predictive likelihood layer on top of MTGP forecasts, not a jointly fitted "
            "Student-t GP. Tuned Student-t df/scale are selected from prior residual rows only."
        ),
        "git_sha": git_sha(),
        "git_dirty_summary_at_start": git_dirty_summary(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(output_dir)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
