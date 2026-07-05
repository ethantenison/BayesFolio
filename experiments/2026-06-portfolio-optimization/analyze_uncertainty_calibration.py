"""Compare GP uncertainty calibration across portfolio experiment runs.

Usage:
    poetry run python experiments/2026-06-portfolio-optimization/analyze_uncertainty_calibration.py
"""

from __future__ import annotations

import argparse
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
from scipy.special import erf
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "calibration"

DEFAULT_RUNS = {
    "signed_lkj_eta_2": Path(
        "experiments/2026-06-portfolio-optimization/outputs/runs/20260616_signed_lkj_eta_2_portfolio"
    ),
    "task_noise_floor_0.0025": Path(
        "experiments/2026-06-portfolio-optimization/outputs/runs/"
        "20260616_signed_lkj_eta_2_task_noise_floor_00025_portfolio"
    ),
    "task_noise_floor_0.005": Path(
        "experiments/2026-06-portfolio-optimization/outputs/runs/"
        "20260616_signed_lkj_eta_2_task_noise_floor_0005_portfolio"
    ),
    "task_noise_floor_0.0075": Path(
        "experiments/2026-06-portfolio-optimization/outputs/runs/"
        "20260616_signed_lkj_eta_2_task_noise_floor_00075_portfolio"
    ),
}

INTERVAL_LEVELS = (0.50, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class RunInput:
    label: str
    run_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=RUN_DIR",
        help="Run directory to evaluate. Defaults to the signed baseline and three task-noise floors.",
    )
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def git_dirty_summary() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return OUTPUT_ROOT / f"{stamp}_task_noise_floor_calibration"


def parse_runs(args: argparse.Namespace) -> list[RunInput]:
    if not args.run:
        return [RunInput(label, REPO_ROOT / run_dir) for label, run_dir in DEFAULT_RUNS.items()]
    runs: list[RunInput] = []
    for raw in args.run:
        if "=" not in raw:
            raise ValueError(f"--run must be LABEL=RUN_DIR, got {raw!r}")
        label, run_dir = raw.split("=", 1)
        runs.append(RunInput(label.strip(), Path(run_dir).expanduser().resolve()))
    return runs


def normal_crps(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (y_true - mean) / std
    phi = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    return std * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def interval_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    width = upper - lower
    below = np.maximum(lower - y_true, 0.0)
    above = np.maximum(y_true - upper, 0.0)
    return width + (2.0 / alpha) * (below + above)


def load_predictions(run: RunInput) -> pd.DataFrame:
    path = run.run_dir / "gp_window_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions for {run.label}: {path}")
    frame = pd.read_csv(path)
    required = {"date", "asset_id", "y_true", "y_pred", "y_std"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["run_label"] = run.label
    frame["run_dir"] = str(run.run_dir)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[np.isfinite(frame["y_true"]) & np.isfinite(frame["y_pred"]) & np.isfinite(frame["y_std"])]
    frame = frame[frame["y_std"] > 0]
    return frame


def summarize_run(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    mean = frame["y_pred"].to_numpy(dtype=float)
    std = frame["y_std"].to_numpy(dtype=float)
    err = y_true - mean
    z = err / std
    pit = norm.cdf(z)
    nll = 0.5 * np.log(2.0 * math.pi * std**2) + 0.5 * z**2
    crps = normal_crps(y_true, mean, std)
    row: dict[str, Any] = {
        "run_label": frame["run_label"].iloc[0],
        "n": float(len(frame)),
        "mean_nll": float(np.mean(nll)),
        "mean_crps": float(np.mean(crps)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mean_pred_std": float(np.mean(std)),
        "mean_abs_z": float(np.mean(np.abs(z))),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z, ddof=0)),
        "pit_ks": float(np.max(np.abs(np.sort(pit) - (np.arange(1, len(pit) + 1) - 0.5) / len(pit)))),
    }
    for level in INTERVAL_LEVELS:
        alpha = 1.0 - level
        zcrit = norm.ppf(1.0 - alpha / 2.0)
        lower = mean - zcrit * std
        upper = mean + zcrit * std
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        row[f"coverage_{int(level * 100)}"] = coverage
        row[f"coverage_error_{int(level * 100)}"] = coverage - level
        row[f"interval_score_{int(level * 100)}"] = float(np.mean(interval_score(y_true, lower, upper, alpha)))
    row["mean_abs_coverage_error"] = float(
        np.mean([abs(row[f"coverage_error_{int(level * 100)}"]) for level in INTERVAL_LEVELS])
    )
    return row


def summarize_by_asset(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (run_label, asset_id), group in frame.groupby(["run_label", "asset_id"], observed=True):
        row = summarize_run(group)
        row["run_label"] = run_label
        row["asset_id"] = asset_id
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_date(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (run_label, date), group in frame.groupby(["run_label", "date"], observed=True):
        row = summarize_run(group)
        row["run_label"] = run_label
        row["date"] = pd.Timestamp(date).date().isoformat()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_reliability(frame: pd.DataFrame, output_dir: Path) -> None:
    levels = np.array(INTERVAL_LEVELS, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(levels, levels, color="black", linewidth=1.0, linestyle="--", label="ideal")
    for run_label, group in frame.groupby("run_label", observed=True):
        y_true = group["y_true"].to_numpy(dtype=float)
        mean = group["y_pred"].to_numpy(dtype=float)
        std = group["y_std"].to_numpy(dtype=float)
        observed = []
        for level in levels:
            alpha = 1.0 - level
            zcrit = norm.ppf(1.0 - alpha / 2.0)
            observed.append(float(np.mean((y_true >= mean - zcrit * std) & (y_true <= mean + zcrit * std))))
        ax.plot(levels, observed, marker="o", label=run_label)
    ax.set_title("Predictive Interval Calibration")
    ax.set_xlabel("Nominal central interval")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.35, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "interval_reliability.png", dpi=160)
    plt.close(fig)


def plot_z_histogram(frame: pd.DataFrame, output_dir: Path) -> None:
    labels = list(frame["run_label"].drop_duplicates())
    ncols = 2
    nrows = math.ceil(len(labels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), squeeze=False)
    x = np.linspace(-4.0, 4.0, 300)
    for ax, label in zip(axes.ravel(), labels, strict=False):
        group = frame[frame["run_label"] == label]
        z = ((group["y_true"] - group["y_pred"]) / group["y_std"]).to_numpy(dtype=float)
        ax.hist(z, bins=np.linspace(-4.0, 4.0, 25), density=True, alpha=0.65)
        ax.plot(x, norm.pdf(x), color="black", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("standardized residual")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(labels) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "standardized_residuals.png", dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df.loc[:, columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = [str(column) for column in view.columns]
    rows = view.astype(str).values.tolist()
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


def write_report(
    output_dir: Path,
    *,
    summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    baseline = summary[summary["run_label"] == "signed_lkj_eta_2"].iloc[0]
    best_nll = summary.sort_values("mean_nll").iloc[0]
    best_crps = summary.sort_values("mean_crps").iloc[0]
    best_coverage = summary.sort_values("mean_abs_coverage_error").iloc[0]
    candidate = summary[summary["run_label"] == "task_noise_floor_0.005"].iloc[0]

    report = [
        "# Task Noise Floor Uncertainty Calibration",
        "",
        f"Run directory: `{output_dir}`",
        f"Created UTC: `{manifest['created_at_utc']}`",
        "",
        "## Question",
        "",
        (
            "Does the calibrated raw-return per-task noise floor improve posterior predictive uncertainty "
            "calibration versus the current `signed_lkj_eta_2` baseline?"
        ),
        "",
        "## Overall Metrics",
        "",
        markdown_table(
            summary,
            [
                "run_label",
                "n",
                "mean_nll",
                "mean_crps",
                "rmse",
                "mean_pred_std",
                "z_std",
                "mean_abs_coverage_error",
                "coverage_50",
                "coverage_80",
                "coverage_90",
                "coverage_95",
            ],
        ),
        "",
        "## Readout",
        "",
        (
            f"- Best mean NLL: `{best_nll['run_label']}` at `{best_nll['mean_nll']:.4f}` "
            f"versus baseline `{baseline['mean_nll']:.4f}`."
        ),
        (
            f"- Best mean CRPS: `{best_crps['run_label']}` at `{best_crps['mean_crps']:.4f}` "
            f"versus baseline `{baseline['mean_crps']:.4f}`."
        ),
        (
            f"- Best average interval calibration error: `{best_coverage['run_label']}` at "
            f"`{best_coverage['mean_abs_coverage_error']:.4f}` versus baseline "
            f"`{baseline['mean_abs_coverage_error']:.4f}`."
        ),
        (
            f"- The previously best portfolio floor, `task_noise_floor_0.005`, has mean NLL "
            f"`{candidate['mean_nll']:.4f}`, mean CRPS `{candidate['mean_crps']:.4f}`, "
            f"and z-score std `{candidate['z_std']:.4f}`."
        ),
        "",
        "## Window Stability",
        "",
        markdown_table(
            window_summary.groupby("run_label", observed=True)
            .agg(
                mean_window_nll=("mean_nll", "mean"),
                sd_window_nll=("mean_nll", "std"),
                mean_window_crps=("mean_crps", "mean"),
                sd_window_crps=("mean_crps", "std"),
            )
            .reset_index(),
            ["run_label", "mean_window_nll", "sd_window_nll", "mean_window_crps", "sd_window_crps"],
        ),
        "",
        "## Visuals",
        "",
        "- `interval_reliability.png`: empirical versus nominal central interval coverage.",
        "- `standardized_residuals.png`: standardized residual histograms versus a standard normal density.",
        "",
        "## Caveats",
        "",
        "- This is an artifact-level calibration audit using the existing 12 rebalance windows and 18 fitted tasks.",
        "- It assumes each GP `y_std` is a Normal predictive scale for `y_excess_lead` after inverse transforms.",
        (
            "- The sample is small for asset-level calibration; window and asset slices are diagnostics, "
            "not definitive proof."
        ),
        (
            "- No transaction costs or portfolio utility are included here; this isolates forecast uncertainty "
            "calibration."
        ),
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")


def run(args: argparse.Namespace) -> None:
    runs = parse_runs(args)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=False)

    frames = [load_predictions(run) for run in runs]
    all_predictions = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame([summarize_run(frame) for frame in frames]).sort_values("run_label")
    asset_summary = summarize_by_asset(all_predictions).sort_values(["asset_id", "run_label"])
    window_summary = summarize_by_date(all_predictions).sort_values(["date", "run_label"])

    manifest = {
        "schema": "bayesfolio.task_noise_floor_calibration_report.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git_sha": git_sha(),
        "git_dirty_summary_at_start": git_dirty_summary(),
        "runs": [
            {
                "label": run.label,
                "run_dir": str(run.run_dir),
                "manifest_path": str(run.run_dir / "manifest.json"),
            }
            for run in runs
        ],
        "metrics": {
            "proper_scores": ["mean_nll", "mean_crps"],
            "coverage_levels": list(INTERVAL_LEVELS),
            "residual_diagnostics": ["z_mean", "z_std", "pit_ks"],
        },
        "output_dir": str(output_dir),
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    all_predictions.to_csv(output_dir / "calibration_predictions.csv", index=False)
    summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    asset_summary.to_csv(output_dir / "calibration_by_asset.csv", index=False)
    window_summary.to_csv(output_dir / "calibration_by_window.csv", index=False)
    plot_reliability(all_predictions, output_dir)
    plot_z_histogram(all_predictions, output_dir)
    write_report(output_dir, summary=summary, window_summary=window_summary, manifest=manifest)

    print(summary.to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    run(parse_args())
