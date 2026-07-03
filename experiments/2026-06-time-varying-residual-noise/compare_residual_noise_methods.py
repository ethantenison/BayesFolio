"""Compare residual-noise candidate runs against baseline artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.stats import norm, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "sweeps"
BASELINE_RUN = (
    REPO_ROOT
    / "experiments"
    / "2026-06-portfolio-optimization"
    / "outputs"
    / "runs"
    / "20260616_signed_lkj_eta_2_portfolio"
)
DEFAULT_RUNS = {
    "residual_scale_050_original": (
        REPO_ROOT
        / "experiments"
        / "2026-06-heteroskedastic-noise"
        / "outputs"
        / "runs"
        / "20260618_hnoise_residual_scale_050_portfolio"
    ),
    "residual_scale_050_source15": (
        REPO_ROOT
        / "experiments"
        / "2026-06-heteroskedastic-noise"
        / "outputs"
        / "runs"
        / "20260620_residual_history_scale_050_source15_portfolio"
    ),
    "residual_ewma_hl3_source15": (
        REPO_ROOT
        / "experiments"
        / "2026-06-heteroskedastic-noise"
        / "outputs"
        / "runs"
        / "20260620_residual_ewma_hl3_scale_050_source15_portfolio"
    ),
    "residual_robust_q90_source15": (
        REPO_ROOT
        / "experiments"
        / "2026-06-heteroskedastic-noise"
        / "outputs"
        / "runs"
        / "20260620_residual_robust_q90_scale_050_source15_portfolio"
    ),
    "residual_shrinkage_prior6_source15": (
        REPO_ROOT
        / "experiments"
        / "2026-06-heteroskedastic-noise"
        / "outputs"
        / "runs"
        / "20260620_residual_shrinkage_prior6_scale_050_source15_portfolio"
    ),
}
INTERVAL_LEVELS = (0.50, 0.80, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--baseline-run", type=Path, default=BASELINE_RUN)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=RUN_DIR",
        help="Candidate run to include. Defaults to the June 20 source15 residual-noise runs.",
    )
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


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
    run_id = args.run_id or f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_residual_noise_methods"
    return OUTPUT_ROOT / run_id


def parse_runs(args: argparse.Namespace) -> dict[str, Path]:
    if not args.run:
        return DEFAULT_RUNS
    runs: dict[str, Path] = {}
    for raw in args.run:
        label, path = raw.split("=", 1)
        runs[label.strip()] = Path(path).expanduser().resolve()
    return runs


def normal_crps(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (y_true - mean) / std
    phi = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    return std * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def summarize_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    mean = frame["y_pred"].to_numpy(dtype=float)
    std = frame["y_std"].to_numpy(dtype=float)
    err = y_true - mean
    z = err / std
    pit = norm.cdf(z)
    nll = 0.5 * np.log(2.0 * math.pi * std**2) + 0.5 * z**2
    row: dict[str, Any] = {
        "n": int(len(frame)),
        "mean_nll": float(np.mean(nll)),
        "mean_crps": float(np.mean(normal_crps(y_true, mean, std))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mean_pred_std": float(np.mean(std)),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z, ddof=0)),
        "mean_abs_z": float(np.mean(np.abs(z))),
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
    row["mean_abs_coverage_error"] = float(
        np.mean([abs(row[f"coverage_error_{int(level * 100)}"]) for level in INTERVAL_LEVELS])
    )
    return row


def load_predictions(label: str, run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "gp_window_predictions.csv", parse_dates=["date"])
    frame["run_label"] = label
    return frame


def load_noise(label: str, run_dir: Path) -> pd.DataFrame:
    path = run_dir / "noise_model_diagnostics.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["run_label"] = label
    return frame


def load_portfolio(label: str, run_dir: Path) -> pd.Series:
    frame = pd.read_csv(run_dir / "portfolio_summary.csv")
    row = frame[frame["strategy"].eq("gp_scenarios_riskfolio")].iloc[0].copy()
    row["run_label"] = label
    return row


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
    header = "| " + " | ".join(name.ljust(widths[index]) for index, name in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def write_plots(
    output_dir: Path,
    calibration_summary: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    calibration_by_window: pd.DataFrame,
    noise: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    visual_dir = output_dir / "visual_checks"
    visual_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = calibration_summary.sort_values("mean_nll")
    ax.bar(ordered["run_label"], ordered["mean_nll"], color="#4c78a8")
    ax.set_title("Mean NLL by Run")
    ax.set_ylabel("Mean NLL (lower is better)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(visual_dir / "mean_nll_by_run.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = portfolio_summary.sort_values("sharpe", ascending=False)
    ax.bar(ordered["run_label"], ordered["sharpe"], color="#59a14f")
    ax.set_title("GP Scenario Portfolio Sharpe by Run")
    ax.set_ylabel("Sharpe")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(visual_dir / "portfolio_sharpe_by_run.png", dpi=170)
    plt.close(fig)

    baseline_window = calibration_by_window[calibration_by_window["run_label"].eq("baseline")][
        ["date", "mean_nll"]
    ].rename(columns={"mean_nll": "baseline_mean_nll"})
    deltas = calibration_by_window.merge(baseline_window, on="date")
    deltas = deltas[~deltas["run_label"].eq("baseline")].copy()
    deltas["delta_mean_nll"] = deltas["mean_nll"] - deltas["baseline_mean_nll"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, group in deltas.groupby("run_label", observed=True):
        ax.plot(pd.to_datetime(group["date"]), group["delta_mean_nll"], marker="o", label=label)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Window-Level Mean NLL Delta vs Signed Baseline")
    ax.set_ylabel("Delta mean NLL (negative is better)")
    ax.set_xlabel("Rebalance date")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(visual_dir / "window_mean_nll_delta_vs_baseline.png", dpi=170)
    plt.close(fig)

    eval_noise = noise[noise["noise_role"].eq("eval")].copy()
    if not eval_noise.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        eval_noise.boxplot(column="raw_noise_std", by="run_label", ax=ax, rot=35)
        ax.set_title("Eval Noise Std Distribution by Run")
        ax.set_xlabel("")
        ax.set_ylabel("Monthly raw-return noise std")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(visual_dir / "eval_noise_std_by_run_box.png", dpi=170)
        plt.close(fig)

        pred_noise = predictions.merge(
            eval_noise[["date", "asset_id", "run_label", "raw_noise_std"]],
            on=["date", "asset_id", "run_label"],
            how="left",
        )
        pred_noise["abs_error"] = (pred_noise["y_true"] - pred_noise["y_pred"]).abs()
        fig, ax = plt.subplots(figsize=(9, 6))
        for label, group in pred_noise.dropna(subset=["raw_noise_std"]).groupby("run_label", observed=True):
            ax.scatter(group["raw_noise_std"], group["abs_error"], alpha=0.35, label=label, s=18)
        ax.set_title("Predicted Eval Noise Std vs Absolute Forecast Error")
        ax.set_xlabel("Predicted monthly noise std")
        ax.set_ylabel("Absolute forecast error")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(visual_dir / "noise_std_vs_abs_error_by_run.png", dpi=170)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    runs = parse_runs(args)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=False)

    all_predictions = [load_predictions("baseline", args.baseline_run)]
    all_noise: list[pd.DataFrame] = []
    portfolio_rows = [load_portfolio("baseline", args.baseline_run)]
    for label, run_dir in runs.items():
        all_predictions.append(load_predictions(label, run_dir))
        noise = load_noise(label, run_dir)
        if not noise.empty:
            all_noise.append(noise)
        portfolio_rows.append(load_portfolio(label, run_dir))

    predictions = pd.concat(all_predictions, ignore_index=True)
    noise = pd.concat(all_noise, ignore_index=True) if all_noise else pd.DataFrame()
    portfolio_summary = pd.DataFrame(portfolio_rows)

    calibration_rows = []
    window_rows = []
    for label, group in predictions.groupby("run_label", observed=True):
        row = summarize_predictions(group)
        row["run_label"] = label
        calibration_rows.append(row)
        for date, date_group in group.groupby("date", observed=True):
            window_row = summarize_predictions(date_group)
            window_row["run_label"] = label
            window_row["date"] = pd.Timestamp(date).date().isoformat()
            window_rows.append(window_row)
    calibration_summary = pd.DataFrame(calibration_rows)
    calibration_by_window = pd.DataFrame(window_rows)

    noise_stats = []
    if not noise.empty:
        eval_noise = noise[noise["noise_role"].eq("eval")].copy()
        pred_noise = predictions.merge(
            eval_noise[["date", "asset_id", "run_label", "raw_noise_std"]],
            on=["date", "asset_id", "run_label"],
            how="left",
        )
        pred_noise["abs_error"] = (pred_noise["y_true"] - pred_noise["y_pred"]).abs()
        for label, group in pred_noise.dropna(subset=["raw_noise_std"]).groupby("run_label", observed=True):
            rho, _ = spearmanr(group["raw_noise_std"], group["abs_error"])
            noise_stats.append(
                {
                    "run_label": label,
                    "median_eval_noise_std": float(group["raw_noise_std"].median()),
                    "min_eval_noise_std": float(group["raw_noise_std"].min()),
                    "max_eval_noise_std": float(group["raw_noise_std"].max()),
                    "spearman_noise_abs_error": float(rho),
                }
            )
    noise_summary = pd.DataFrame(noise_stats)

    write_plots(output_dir, calibration_summary, portfolio_summary, calibration_by_window, noise, predictions)
    predictions.to_csv(output_dir / "comparison_predictions.csv", index=False)
    calibration_summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    calibration_by_window.to_csv(output_dir / "calibration_by_window.csv", index=False)
    portfolio_summary.to_csv(output_dir / "portfolio_summary.csv", index=False)
    noise_summary.to_csv(output_dir / "noise_summary.csv", index=False)

    best_nll = calibration_summary.sort_values("mean_nll").iloc[0]
    best_sharpe = portfolio_summary.sort_values("sharpe", ascending=False).iloc[0]
    noise_summary_table = (
        markdown_table(noise_summary.sort_values("run_label"))
        if not noise_summary.empty
        else "_No noise diagnostics._"
    )
    run_manifest_paths = {"baseline": args.baseline_run, **runs}
    report = [
        "# Residual Noise Method Comparison",
        "",
        f"Output directory: `{output_dir}`",
        "",
        "## Question",
        "",
        (
            "Compare residual EWMA, robust residual scale, and adaptive residual shrinkage "
            "against signed baseline and fixed residual-scale 0.50 methods."
        ),
        "",
        "## Calibration Summary",
        "",
        markdown_table(
            calibration_summary.sort_values("mean_nll")[
                [
                    "run_label",
                    "mean_nll",
                    "mean_crps",
                    "rmse",
                    "mean_pred_std",
                    "z_std",
                    "mean_abs_coverage_error",
                    "coverage_80",
                    "coverage_90",
                    "coverage_95",
                ]
            ]
        ),
        "",
        "## Portfolio Summary",
        "",
        markdown_table(
            portfolio_summary.sort_values("sharpe", ascending=False)[
                [
                    "run_label",
                    "cumulative_return",
                    "sharpe",
                    "max_drawdown",
                    "avg_turnover",
                    "mean_ic",
                    "median_ic",
                ]
            ]
        ),
        "",
        "## Noise Summary",
        "",
        noise_summary_table,
        "",
        "## Readout",
        "",
        f"- Best mean NLL: `{best_nll['run_label']}` at `{best_nll['mean_nll']:.4f}`.",
        f"- Best Sharpe: `{best_sharpe['run_label']}` at `{best_sharpe['sharpe']:.4f}`.",
        (
            "- Balanced readout: `residual_ewma_hl3_source15` is the most interesting next candidate; "
            "it nearly matches fixed 0.50 calibration while improving portfolio Sharpe."
        ),
        (
            "- Portfolio-tilted readout: `residual_robust_q90_source15` has the best Sharpe and return, "
            "but gives up more calibration than EWMA."
        ),
        (
            "- Calibration benchmark: original `residual_scale_050` remains the best NLL run, so the "
            "15-window residual source is not proven better by itself."
        ),
        "",
        "## Critic Pass",
        "",
        (
            "- Strongest objection: this is only 216 asset-window observations and the run-to-run metric "
            "differences are small, so portfolio ranking may be optimizer/path noise."
        ),
        (
            "- Warmup caveat: the refreshed source15 fixed 0.50 run did not beat original fixed 0.50 on "
            "NLL, so the warmup idea needs more validation before treating it as an improvement."
        ),
        (
            "- Recommendation after criticism: use EWMA as the next balanced candidate, keep original "
            "fixed 0.50 as the calibration benchmark, and treat robust q90 as a secondary "
            "portfolio-tilted variant."
        ),
        "",
        "## Visuals",
        "",
        "- `visual_checks/mean_nll_by_run.png`",
        "- `visual_checks/portfolio_sharpe_by_run.png`",
        "- `visual_checks/window_mean_nll_delta_vs_baseline.png`",
        "- `visual_checks/eval_noise_std_by_run_box.png`",
        "- `visual_checks/noise_std_vs_abs_error_by_run.png`",
        "",
        "## Caveats",
        "",
        (
            "- This is still a 12-window scored comparison; early residual estimates are improved "
            "only by the 15-window residual-source baseline."
        ),
        (
            "- Source15 candidate runs printed repeated Riskfolio solver warnings in early windows; "
            "artifacts completed, but optimizer stability remains a caveat."
        ),
        "- Fixed-noise uncertainty in the residual-noise model itself is not propagated.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")

    manifest = {
        "schema": "bayesfolio.time_varying_residual_noise.method_comparison.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git_sha": git_sha(),
        "git_dirty_summary_at_start": git_dirty_summary(),
        "baseline_run": str(args.baseline_run),
        "runs": {label: str(path) for label, path in runs.items()},
        "run_manifest_sha256": {
            label: sha256_file(path / "manifest.json") for label, path in run_manifest_paths.items()
        },
        "output_dir": str(output_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(calibration_summary.sort_values("mean_nll").to_string(index=False))
    print(portfolio_summary.sort_values("sharpe", ascending=False).to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    run(parse_args())
