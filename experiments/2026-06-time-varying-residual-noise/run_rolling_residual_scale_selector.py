"""Assemble a leakage-aware rolling selector over residual-noise scales.

This script does not refit the GP. It uses the already completed fixed-scale
residual-history runs and, for each rebalance date, selects the scale with the
best prior-window calibration evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import erf
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_DIR / "outputs"
PORTFOLIO_EXPERIMENT = (
    REPO_ROOT
    / "experiments"
    / "2026-06-portfolio-optimization"
    / "run_monthly_optimization_walkforward.py"
)
PORT_SPEC = importlib.util.spec_from_file_location("portfolio_walkforward", PORTFOLIO_EXPERIMENT)
if PORT_SPEC is None or PORT_SPEC.loader is None:
    raise RuntimeError(f"Unable to load portfolio experiment from {PORTFOLIO_EXPERIMENT}")
portfolio_exp = importlib.util.module_from_spec(PORT_SPEC)
sys.modules[PORT_SPEC.name] = portfolio_exp
PORT_SPEC.loader.exec_module(portfolio_exp)

BASELINE_RUN = (
    REPO_ROOT
    / "experiments"
    / "2026-06-portfolio-optimization"
    / "outputs"
    / "runs"
    / "20260616_signed_lkj_eta_2_portfolio"
)
HNOISE_ROOT = REPO_ROOT / "experiments" / "2026-06-heteroskedastic-noise" / "outputs" / "runs"
DEFAULT_SCALE_RUNS = {
    "residual_scale_025": HNOISE_ROOT / "20260618_hnoise_residual_scale_025_portfolio",
    "residual_scale_050": HNOISE_ROOT / "20260618_hnoise_residual_scale_050_portfolio",
    "residual_scale_075": HNOISE_ROOT / "20260618_hnoise_residual_scale_075_portfolio",
    "residual_scale_100": HNOISE_ROOT / "20260618_hnoise_residual_scale_100_portfolio",
}
DEFAULT_FALLBACK_SCALE = "residual_scale_050"
INTERVAL_LEVELS = (0.50, 0.80, 0.90, 0.95)
GP_STRATEGY = "gp_scenarios_riskfolio"


@dataclass(frozen=True)
class RunData:
    label: str
    run_dir: Path
    predictions: pd.DataFrame
    returns: pd.DataFrame
    weights: pd.DataFrame
    ic: pd.DataFrame
    noise: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--baseline-run", type=Path, default=BASELINE_RUN)
    parser.add_argument("--fallback-scale", type=str, default=DEFAULT_FALLBACK_SCALE)
    parser.add_argument(
        "--scale-run",
        action="append",
        default=[],
        metavar="LABEL=RUN_DIR",
        help="Fixed residual-scale run directory. Defaults to the four completed residual scale runs.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["mean_nll", "mean_abs_coverage_error", "z_std_error"],
        default="mean_nll",
        help="Prior-window calibration metric used for rolling scale selection.",
    )
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


def parse_scale_runs(args: argparse.Namespace) -> dict[str, Path]:
    if not args.scale_run:
        return DEFAULT_SCALE_RUNS
    parsed: dict[str, Path] = {}
    for raw in args.scale_run:
        if "=" not in raw:
            raise ValueError(f"--scale-run must be LABEL=RUN_DIR, got {raw!r}")
        label, run_dir = raw.split("=", 1)
        parsed[label.strip()] = Path(run_dir).expanduser().resolve()
    return parsed


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


def summarize_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    mean = frame["y_pred"].to_numpy(dtype=float)
    std = frame["y_std"].to_numpy(dtype=float)
    err = y_true - mean
    z = err / std
    pit = norm.cdf(z)
    nll = 0.5 * np.log(2.0 * math.pi * std**2) + 0.5 * z**2
    crps = normal_crps(y_true, mean, std)
    row: dict[str, Any] = {
        "n": float(len(frame)),
        "mean_nll": float(np.mean(nll)),
        "mean_crps": float(np.mean(crps)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mean_pred_std": float(np.mean(std)),
        "mean_abs_z": float(np.mean(np.abs(z))),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z, ddof=0)),
        "z_std_error": float(abs(np.std(z, ddof=0) - 1.0)),
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


def summarize_by_date(predictions: pd.DataFrame, *, run_label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in predictions.groupby("date", observed=True):
        row = summarize_predictions(group)
        row["run_label"] = run_label
        row["date"] = pd.Timestamp(date).date().isoformat()
        rows.append(row)
    return pd.DataFrame(rows)


def load_run(label: str, run_dir: Path) -> RunData:
    for filename in [
        "gp_window_predictions.csv",
        "portfolio_returns.csv",
        "portfolio_weights.csv",
        "gp_window_ic.csv",
        "noise_model_diagnostics.csv",
    ]:
        if not (run_dir / filename).exists():
            raise FileNotFoundError(run_dir / filename)
    predictions = pd.read_csv(run_dir / "gp_window_predictions.csv", parse_dates=["date"])
    predictions["run_label"] = label
    returns = pd.read_csv(run_dir / "portfolio_returns.csv", parse_dates=["date"])
    returns["run_label"] = label
    weights = pd.read_csv(run_dir / "portfolio_weights.csv", parse_dates=["date"])
    weights["run_label"] = label
    ic = pd.read_csv(run_dir / "gp_window_ic.csv", parse_dates=["date"])
    ic["run_label"] = label
    noise = pd.read_csv(run_dir / "noise_model_diagnostics.csv")
    noise["date"] = pd.to_datetime(noise["date"])
    noise["window_date"] = pd.to_datetime(noise["window_date"])
    noise["run_label"] = label
    return RunData(label, run_dir, predictions, returns, weights, ic, noise)


def load_baseline_predictions(path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(path / "gp_window_predictions.csv", parse_dates=["date"])
    predictions["run_label"] = "baseline"
    return predictions


def select_scales(
    *,
    run_data: dict[str, RunData],
    fallback_scale: str,
    selection_metric: str,
) -> pd.DataFrame:
    if fallback_scale not in run_data:
        raise ValueError(f"fallback scale {fallback_scale!r} not in scale runs: {sorted(run_data)}")
    all_dates = sorted(
        set.intersection(*(set(data.predictions["date"]) for data in run_data.values()))
    )
    prior_scores: dict[str, pd.DataFrame] = {
        label: summarize_by_date(data.predictions, run_label=label) for label, data in run_data.items()
    }
    rows: list[dict[str, Any]] = []
    for index, date in enumerate(all_dates):
        prior_date_mask = lambda frame: pd.to_datetime(frame["date"]) < date
        if index == 0:
            selected = fallback_scale
            reason = "fallback_no_prior_window"
            score = math.nan
        else:
            candidates: list[dict[str, Any]] = []
            for label, score_frame in prior_scores.items():
                prior = score_frame[prior_date_mask(score_frame)]
                if prior.empty:
                    continue
                metric_value = float(prior[selection_metric].mean())
                candidates.append(
                    {
                        "scale_label": label,
                        "selection_metric_value": metric_value,
                        "prior_windows": int(len(prior)),
                        "prior_mean_nll": float(prior["mean_nll"].mean()),
                        "prior_mean_abs_coverage_error": float(prior["mean_abs_coverage_error"].mean()),
                        "prior_z_std_error": float(prior["z_std_error"].mean()),
                    }
                )
            if not candidates:
                selected = fallback_scale
                reason = "fallback_no_candidate_prior"
                score = math.nan
            else:
                ranked = sorted(candidates, key=lambda row: (row["selection_metric_value"], row["scale_label"]))
                selected = ranked[0]["scale_label"]
                reason = f"best_prior_{selection_metric}"
                score = ranked[0]["selection_metric_value"]
        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "selected_scale": selected,
                "selection_reason": reason,
                "selection_metric": selection_metric,
                "selection_metric_value": score,
                "n_prior_windows": index,
            }
        )
    return pd.DataFrame(rows)


def assemble_selected_run(
    *,
    run_data: dict[str, RunData],
    selections: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    return_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []
    ic_frames: list[pd.DataFrame] = []
    noise_frames: list[pd.DataFrame] = []
    for row in selections.itertuples(index=False):
        date = pd.Timestamp(row.date)
        selected = str(row.selected_scale)
        data = run_data[selected]
        prediction_frames.append(
            data.predictions[data.predictions["date"].eq(date)].copy().assign(selected_scale=selected)
        )
        return_frames.append(data.returns[data.returns["date"].eq(date)].copy().assign(selected_scale=selected))
        weight_frames.append(data.weights[data.weights["date"].eq(date)].copy().assign(selected_scale=selected))
        ic_frames.append(data.ic[data.ic["date"].eq(date)].copy().assign(selected_scale=selected))
        noise_frames.append(data.noise[data.noise["window_date"].eq(date)].copy().assign(selected_scale=selected))
        scenario_name = f"gp_scenarios_{date.date().isoformat()}.csv"
        source_scenario = data.run_dir / scenario_name
        if source_scenario.exists():
            shutil.copyfile(source_scenario, output_dir / scenario_name)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    returns = pd.concat(return_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)
    ic = pd.concat(ic_frames, ignore_index=True)
    noise = pd.concat(noise_frames, ignore_index=True)
    return predictions, returns, weights, ic, noise


def summarize_portfolio(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, group in returns.groupby("strategy", observed=True):
        series = group.sort_values("date").set_index("date")["return"].astype(float)
        wide_weights = (
            weights[weights["strategy"].eq(strategy)]
            .pivot(index="date", columns="asset_id", values="weight")
            .sort_index()
            .fillna(0.0)
        )
        summary = {
            "strategy": strategy,
            **portfolio_exp.performance_stats(series, wide_weights, starting_value=portfolio_exp.STARTING_VALUE),
        }
        if strategy == GP_STRATEGY:
            gp_ic = group["gp_ic"].dropna().astype(float)
            summary["mean_ic"] = float(gp_ic.mean())
            summary["median_ic"] = float(gp_ic.median())
        else:
            summary["mean_ic"] = math.nan
            summary["median_ic"] = math.nan
        rows.append(summary)
    return pd.DataFrame(rows)


def summarize_all_calibration(
    baseline_predictions: pd.DataFrame,
    run_data: dict[str, RunData],
    selected_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, Any]] = []
    window_rows: list[pd.DataFrame] = []
    inputs = {"baseline": baseline_predictions, **{k: v.predictions for k, v in run_data.items()}}
    inputs["rolling_selector"] = selected_predictions
    for label, predictions in inputs.items():
        row = summarize_predictions(predictions)
        row["run_label"] = label
        overall_rows.append(row)
        window_rows.append(summarize_by_date(predictions, run_label=label))
    return pd.DataFrame(overall_rows), pd.concat(window_rows, ignore_index=True)


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


def write_visual_checks(
    *,
    output_dir: Path,
    predictions: pd.DataFrame,
    noise: pd.DataFrame,
    calibration_by_window: pd.DataFrame,
    returns: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    visual_dir = output_dir / "visual_checks"
    visual_dir.mkdir(exist_ok=True)

    pred = predictions.copy()
    pred["abs_error"] = (pred["y_true"] - pred["y_pred"]).abs()
    pred["noise_variance_share"] = np.square(pred["noise_y_std"]) / np.square(pred["y_std"])

    pivot = pred.pivot(index="asset_id", columns="date", values="noise_y_std")
    asset_order = pivot.mean(axis=1).sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(pivot.loc[asset_order], cmap="viridis", ax=ax, cbar_kws={"label": "Predicted monthly noise std"})
    ax.set_title("Rolling Selector: Eval Noise Std by Asset and Rebalance Date")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Asset")
    ax.set_xticklabels([pd.to_datetime(t.get_text()).strftime("%Y-%m") for t in ax.get_xticklabels()], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(visual_dir / "noise_std_asset_date_heatmap.png", dpi=170)
    plt.close(fig)

    eval_noise = noise[noise["noise_role"].eq("eval")].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=eval_noise, x="asset_group", y="raw_noise_std", ax=ax, color="#91c4f2")
    sns.stripplot(data=eval_noise, x="asset_group", y="raw_noise_std", ax=ax, color="black", alpha=0.45, size=3)
    ax.set_title("Rolling Selector: Eval Noise Std Distribution by Asset Group")
    ax.set_xlabel("Asset group")
    ax.set_ylabel("Predicted monthly noise std")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(visual_dir / "noise_std_by_asset_group_box.png", dpi=170)
    plt.close(fig)

    source = eval_noise.copy()
    source["source"] = source["noise_source_detail"].fillna(source["noise_model_fallback"]).fillna("unknown")
    counts = source.groupby(["window_date", "source"]).size().unstack(fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind="bar", stacked=True, ax=ax, width=0.85, colormap="tab20")
    ax.set_title("Rolling Selector: Eval Noise Source Mix by Rebalance")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Asset count")
    ax.set_xticklabels([pd.to_datetime(str(t.get_text())).strftime("%Y-%m") for t in ax.get_xticklabels()], rotation=45, ha="right")
    ax.legend(title="Noise source", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(visual_dir / "noise_source_mix_by_window.png", dpi=170)
    plt.close(fig)

    pred["noise_bin"] = pd.qcut(pred["noise_y_std"], q=5, duplicates="drop")
    binned = pred.groupby("noise_bin", observed=True).agg(
        mean_noise=("noise_y_std", "mean"),
        mean_abs_error=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
        n=("abs_error", "size"),
    ).reset_index()
    binned.to_csv(visual_dir / "noise_error_quintiles.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pred["noise_y_std"], pred["abs_error"], alpha=0.35, label="Asset-window obs")
    ax.plot(binned["mean_noise"], binned["mean_abs_error"], marker="o", linewidth=2.5, label="Mean abs error by quintile")
    ax.plot(binned["mean_noise"], binned["median_abs_error"], marker="s", linewidth=2.0, label="Median abs error by quintile")
    ax.set_title("Rolling Selector: Does Higher Noise Track Larger Errors?")
    ax.set_xlabel("Predicted monthly noise std")
    ax.set_ylabel("Absolute forecast error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(visual_dir / "noise_std_vs_abs_error_binned.png", dpi=170)
    plt.close(fig)

    share = pred.pivot(index="asset_id", columns="date", values="noise_variance_share").loc[asset_order]
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(share, cmap="mako", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Noise variance / total predictive variance"})
    ax.set_title("Rolling Selector: Observation Noise Share of Predictive Variance")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Asset")
    ax.set_xticklabels([pd.to_datetime(t.get_text()).strftime("%Y-%m") for t in ax.get_xticklabels()], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(visual_dir / "noise_variance_share_asset_date_heatmap.png", dpi=170)
    plt.close(fig)

    by_window = calibration_by_window.pivot(index="date", columns="run_label", values="mean_nll")
    if {"baseline", "rolling_selector"}.issubset(by_window.columns):
        deltas = (by_window["rolling_selector"] - by_window["baseline"]).rename("delta_mean_nll").reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = np.where(deltas["delta_mean_nll"] <= 0, "#2ca25f", "#de2d26")
        ax.bar(pd.to_datetime(deltas["date"]), deltas["delta_mean_nll"], color=colors, width=20)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title("Rolling Selector vs Baseline: Window Mean NLL Delta")
        ax.set_xlabel("Rebalance date")
        ax.set_ylabel("candidate - baseline")
        fig.tight_layout()
        fig.savefig(visual_dir / "window_level_calibration_deltas.png", dpi=170)
        plt.close(fig)

    ret = returns[returns["strategy"].eq(GP_STRATEGY)].sort_values("date")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(ret["date"], ret["return"], width=20)
    axes[0].set_title("Rolling Selector GP Strategy Return by Rebalance")
    axes[0].set_ylabel("Return")
    axes[1].bar(ret["date"], ret["gp_ic"], width=20, color="#756bb1")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Rolling Selector GP IC by Rebalance")
    axes[1].set_ylabel("IC")
    axes[1].set_xlabel("Rebalance date")
    fig.tight_layout()
    fig.savefig(visual_dir / "window_level_portfolio_deltas.png", dpi=170)
    plt.close(fig)

    stats = {
        "n_eval_predictions": int(len(pred)),
        "n_assets": int(pred["asset_id"].nunique()),
        "n_windows": int(pred["date"].nunique()),
        "noise_y_std_min": float(pred["noise_y_std"].min()),
        "noise_y_std_median": float(pred["noise_y_std"].median()),
        "noise_y_std_max": float(pred["noise_y_std"].max()),
        "noise_y_std_cv": float(pred["noise_y_std"].std(ddof=1) / pred["noise_y_std"].mean()),
        "noise_variance_share_median": float(pred["noise_variance_share"].median()),
        "noise_variance_share_p10": float(pred["noise_variance_share"].quantile(0.10)),
        "noise_variance_share_p90": float(pred["noise_variance_share"].quantile(0.90)),
        "spearman_noise_abs_error": float(pred[["noise_y_std", "abs_error"]].corr(method="spearman").iloc[0, 1]),
        "source_counts_eval": {str(k): int(v) for k, v in source["source"].value_counts().to_dict().items()},
    }
    (visual_dir / "visual_check_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")


def write_manifest(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    scale_runs: dict[str, Path],
    selections: pd.DataFrame,
) -> None:
    selected_windows = json.loads(selections.to_json(orient="records"))
    manifest = {
        "schema": "bayesfolio.time_varying_residual_noise.rolling_selector.manifest.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git_sha": git_sha(short=False),
        "git_dirty_summary_at_start": git_dirty_summary(),
        "output_dir": str(output_dir),
        "baseline_run": str(args.baseline_run),
        "scale_runs": {label: str(path) for label, path in scale_runs.items()},
        "scale_run_prediction_hashes": {
            label: sha256_file(path / "gp_window_predictions.csv") for label, path in scale_runs.items()
        },
        "selection": {
            "kind": "rolling_prior_window_scale_selector",
            "selection_metric": args.selection_metric,
            "fallback_scale": args.fallback_scale,
            "available_scales": sorted(scale_runs),
            "leakage_rule": "For date T, select scale using only calibration summaries from dates < T.",
        },
        "selected_scale_counts": selections["selected_scale"].value_counts().sort_index().to_dict(),
        "selected_windows": selected_windows,
        "artifact_note": "This run assembles existing fixed-scale artifacts; it does not refit the GP.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_report(
    *,
    output_dir: Path,
    selections: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    visual_stats: dict[str, Any],
) -> None:
    selector = calibration_summary[calibration_summary["run_label"].eq("rolling_selector")].iloc[0]
    baseline = calibration_summary[calibration_summary["run_label"].eq("baseline")].iloc[0]
    report = [
        "# Rolling Residual-Scale Selector",
        "",
        f"Run directory: `{output_dir}`",
        "",
        "## Question",
        "",
        (
            "Can the residual-history fixed-noise scale be selected using only prior-window "
            "calibration evidence, instead of choosing one global scale after seeing the full sweep?"
        ),
        "",
        "## Method",
        "",
        "- Inputs: completed fixed-scale residual-history runs for scales 0.25, 0.50, 0.75, and 1.00.",
        "- For the first rebalance, use `residual_scale_050` as the no-prior fallback.",
        "- For each later rebalance date, select the scale with the best mean prior-window NLL.",
        "- Assemble the selected window predictions, weights, returns, scenarios, and noise diagnostics.",
        "- No GP refit is performed in this artifact-level selector run.",
        "",
        "## Selected Scales",
        "",
        markdown_table(
            selections,
            ["date", "selected_scale", "selection_reason", "selection_metric_value", "n_prior_windows"],
        ),
        "",
        "## Portfolio Summary",
        "",
        markdown_table(
            portfolio_summary,
            [
                "strategy",
                "cumulative_return",
                "sharpe",
                "max_drawdown",
                "avg_turnover",
                "hit_rate",
                "mean_ic",
                "median_ic",
            ],
        ),
        "",
        "## Calibration Summary",
        "",
        markdown_table(
            calibration_summary.sort_values("run_label"),
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
            ],
        ),
        "",
        "## Readout",
        "",
        (
            f"- Rolling selector mean NLL: `{selector['mean_nll']:.4f}` versus baseline "
            f"`{baseline['mean_nll']:.4f}`."
        ),
        (
            f"- Rolling selector z-score std: `{selector['z_std']:.4f}` versus baseline "
            f"`{baseline['z_std']:.4f}`."
        ),
        (
            f"- Rolling selector 90% coverage: `{selector['coverage_90']:.4f}` versus baseline "
            f"`{baseline['coverage_90']:.4f}`."
        ),
        (
            f"- Predicted noise std range: `{visual_stats['noise_y_std_min']:.4f}` to "
            f"`{visual_stats['noise_y_std_max']:.4f}`; median `{visual_stats['noise_y_std_median']:.4f}`."
        ),
        (
            f"- Spearman correlation between predicted noise std and absolute error: "
            f"`{visual_stats['spearman_noise_abs_error']:.4f}`."
        ),
        "",
        "## Visuals",
        "",
        "- `visual_checks/noise_std_asset_date_heatmap.png`",
        "- `visual_checks/noise_std_by_asset_group_box.png`",
        "- `visual_checks/noise_source_mix_by_window.png`",
        "- `visual_checks/noise_std_vs_abs_error_binned.png`",
        "- `visual_checks/noise_variance_share_asset_date_heatmap.png`",
        "- `visual_checks/window_level_calibration_deltas.png`",
        "- `visual_checks/window_level_portfolio_deltas.png`",
        "",
        "## Caveats",
        "",
        "- This tests a selector policy from existing fixed-scale artifacts; it does not rerun the GP.",
        "- Prior-window selection is very data-poor early in the sequence.",
        "- The selected path still inherits the residual-noise model's fallback behavior in sparse early windows.",
        "- A stronger next version should evaluate rolling selection across more windows or through a nested backtest.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")


def run(args: argparse.Namespace) -> None:
    scale_runs = parse_scale_runs(args)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=False)

    run_data = {label: load_run(label, path) for label, path in scale_runs.items()}
    baseline_predictions = load_baseline_predictions(args.baseline_run)
    selections = select_scales(
        run_data=run_data,
        fallback_scale=args.fallback_scale,
        selection_metric=args.selection_metric,
    )
    predictions, returns, weights, ic, noise = assemble_selected_run(
        run_data=run_data,
        selections=selections,
        output_dir=output_dir,
    )
    portfolio_summary = summarize_portfolio(returns, weights)
    calibration_summary, calibration_by_window = summarize_all_calibration(
        baseline_predictions,
        run_data,
        predictions,
    )

    selections.to_csv(output_dir / "selected_scale_by_window.csv", index=False)
    predictions.to_csv(output_dir / "gp_window_predictions.csv", index=False)
    returns.to_csv(output_dir / "portfolio_returns.csv", index=False)
    weights.to_csv(output_dir / "portfolio_weights.csv", index=False)
    ic.to_csv(output_dir / "gp_window_ic.csv", index=False)
    noise.to_csv(output_dir / "noise_model_diagnostics.csv", index=False)
    portfolio_summary.to_csv(output_dir / "portfolio_summary.csv", index=False)
    calibration_summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    calibration_by_window.to_csv(output_dir / "calibration_by_window.csv", index=False)

    strategy_returns = {
        strategy: group.sort_values("date").set_index("date")["return"].astype(float)
        for strategy, group in returns.groupby("strategy", observed=True)
    }
    portfolio_exp.plot_equity_and_drawdown(strategy_returns, output_dir)
    write_visual_checks(
        output_dir=output_dir,
        predictions=predictions,
        noise=noise,
        calibration_by_window=calibration_by_window,
        returns=returns,
    )
    visual_stats = json.loads((output_dir / "visual_checks" / "visual_check_stats.json").read_text())
    write_manifest(args=args, output_dir=output_dir, scale_runs=scale_runs, selections=selections)
    write_report(
        output_dir=output_dir,
        selections=selections,
        portfolio_summary=portfolio_summary,
        calibration_summary=calibration_summary,
        visual_stats=visual_stats,
    )
    print(portfolio_summary.to_string(index=False))
    print(calibration_summary.sort_values("run_label").to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    run(parse_args())
