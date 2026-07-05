"""Rolling out-of-sample selection test for task-noise floors.

Usage:
    poetry run python experiments/2026-06-portfolio-optimization/run_rolling_floor_selection.py
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

import analyze_uncertainty_calibration as calib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "rolling_floor_selection"
BASELINE_LABEL = "signed_lkj_eta_2"
FLOOR_LABELS = ["task_noise_floor_0.0025", "task_noise_floor_0.005", "task_noise_floor_0.0075"]
SELECTOR_METRICS = {
    "rolling_select_nll": "mean_nll",
    "rolling_select_crps": "mean_crps",
    "rolling_select_coverage": "mean_abs_coverage_error",
}


@dataclass(frozen=True)
class RunBundle:
    label: str
    run_dir: Path
    predictions: pd.DataFrame
    portfolio_returns: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-history", type=int, default=4, help="Prior windows required before selecting a floor.")
    parser.add_argument(
        "--include-baseline-in-selector",
        action="store_true",
        help="Allow rolling selectors to choose the signed baseline as well as calibrated floors.",
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
    return OUTPUT_ROOT / f"{stamp}_task_noise_floor_rolling_selection"


def load_bundle(run: calib.RunInput) -> RunBundle:
    predictions = calib.load_predictions(run)
    returns_path = run.run_dir / "portfolio_returns.csv"
    if not returns_path.exists():
        raise FileNotFoundError(f"Missing portfolio returns for {run.label}: {returns_path}")
    returns = pd.read_csv(returns_path)
    required = {"date", "strategy", "return", "gp_ic"}
    missing = required.difference(returns.columns)
    if missing:
        raise ValueError(f"{returns_path} missing required columns: {sorted(missing)}")
    returns = returns.copy()
    returns["date"] = pd.to_datetime(returns["date"])
    returns["run_label"] = run.label
    return RunBundle(run.label, run.run_dir, predictions, returns)


def prediction_with_window_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (run_label, date), group in predictions.groupby(["run_label", "date"], observed=True):
        row = calib.summarize_run(group)
        row["run_label"] = run_label
        row["date"] = pd.Timestamp(date)
        rows.append(row)
    return predictions, pd.DataFrame(rows).sort_values(["date", "run_label"])


def portfolio_stats(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return {
            "n_rebalances": 0.0,
            "cumulative_return": math.nan,
            "annualized_vol": math.nan,
            "sharpe": math.nan,
            "max_drawdown": math.nan,
            "terminal_value": math.nan,
            "mean_monthly_return": math.nan,
            "hit_rate": math.nan,
        }
    equity = (1.0 + returns).cumprod()
    ann_vol = float(returns.std(ddof=0) * np.sqrt(12.0))
    cumulative_return = float(equity.iloc[-1] - 1.0)
    years = len(returns) / 12.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else math.nan
    drawdown = equity / equity.cummax() - 1.0
    return {
        "n_rebalances": float(len(returns)),
        "cumulative_return": cumulative_return,
        "annualized_vol": ann_vol,
        "sharpe": float(cagr / ann_vol) if ann_vol > 0 else math.nan,
        "max_drawdown": float(drawdown.min()),
        "terminal_value": float(10_000.0 * equity.iloc[-1]),
        "mean_monthly_return": float(returns.mean()),
        "hit_rate": float((returns > 0).mean()),
    }


def choose_floor(
    window_metrics: pd.DataFrame,
    *,
    date: pd.Timestamp,
    metric: str,
    candidates: list[str],
) -> str:
    past = window_metrics[(window_metrics["date"] < date) & window_metrics["run_label"].isin(candidates)]
    scores = past.groupby("run_label", observed=True)[metric].mean().sort_values()
    if scores.empty:
        raise ValueError(f"No prior scores available before {date.date()} for {metric}")
    return str(scores.index[0])


def build_selector_rows(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    window_metrics: pd.DataFrame,
    *,
    min_history: int,
    candidates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.Timestamp(date) for date in predictions["date"].drop_duplicates())
    test_dates = dates[min_history:]
    selection_rows: list[dict[str, Any]] = []
    selected_prediction_rows: list[pd.DataFrame] = []
    selected_return_rows: list[dict[str, Any]] = []

    for selector_name, metric in SELECTOR_METRICS.items():
        for date in test_dates:
            chosen = choose_floor(window_metrics, date=date, metric=metric, candidates=candidates)
            selection_rows.append(
                {
                    "selector": selector_name,
                    "selection_metric": metric,
                    "date": date.date().isoformat(),
                    "chosen_run_label": chosen,
                    "candidate_set": ",".join(candidates),
                }
            )
            pred_slice = predictions[(predictions["date"] == date) & (predictions["run_label"] == chosen)].copy()
            pred_slice["run_label"] = selector_name
            pred_slice["chosen_run_label"] = chosen
            selected_prediction_rows.append(pred_slice)

            return_slice = returns[
                (returns["date"] == date)
                & (returns["run_label"] == chosen)
                & (returns["strategy"] == "gp_scenarios_riskfolio")
            ]
            if return_slice.empty:
                raise ValueError(f"No GP portfolio return for {chosen} on {date.date()}")
            selected_return_rows.append(
                {
                    "date": date.date().isoformat(),
                    "strategy": "gp_scenarios_riskfolio",
                    "run_label": selector_name,
                    "chosen_run_label": chosen,
                    "return": float(return_slice["return"].iloc[0]),
                    "gp_ic": float(return_slice["gp_ic"].iloc[0]),
                }
            )

    return (
        pd.DataFrame(selection_rows),
        pd.concat(selected_prediction_rows, ignore_index=True),
        pd.DataFrame(selected_return_rows),
    )


def summarize_rolling(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    labels: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in labels:
        pred = predictions[predictions["run_label"] == label]
        ret = returns[(returns["run_label"] == label) & (returns["strategy"] == "gp_scenarios_riskfolio")]
        row = calib.summarize_run(pred)
        row.update(portfolio_stats(ret.sort_values("date")["return"]))
        row["run_label"] = label
        row["mean_ic"] = float(ret["gp_ic"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_selection_timeline(selection: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    marker_map = {label: index for index, label in enumerate(FLOOR_LABELS)}
    for selector, group in selection.groupby("selector", observed=True):
        x = pd.to_datetime(group["date"])
        y = group["chosen_run_label"].map(marker_map)
        ax.plot(x, y, marker="o", linewidth=1.5, label=selector)
    ax.set_title("Rolling Floor Selection")
    ax.set_ylabel("Selected floor")
    ax.set_yticks(list(marker_map.values()), list(marker_map.keys()))
    ax.set_xlabel("Evaluation rebalance date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "rolling_selection_timeline.png", dpi=160)
    plt.close(fig)


def plot_out_of_sample_reliability(predictions: pd.DataFrame, labels: list[str], output_dir: Path) -> None:
    calib.plot_reliability(predictions[predictions["run_label"].isin(labels)].copy(), output_dir)
    (output_dir / "interval_reliability.png").replace(output_dir / "rolling_interval_reliability.png")


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    return calib.markdown_table(df, columns)


def write_report(
    output_dir: Path,
    *,
    summary: pd.DataFrame,
    selection: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    baseline = summary[summary["run_label"] == BASELINE_LABEL].iloc[0]
    best_nll = summary.sort_values("mean_nll").iloc[0]
    best_terminal = summary.sort_values("terminal_value", ascending=False).iloc[0]
    selector_counts = (
        selection.groupby(["selector", "chosen_run_label"], observed=True)
        .size()
        .rename("n_selected")
        .reset_index()
        .sort_values(["selector", "chosen_run_label"])
    )
    report = [
        "# Rolling Task-Noise Floor Selection",
        "",
        f"Run directory: `{output_dir}`",
        f"Created UTC: `{manifest['created_at_utc']}`",
        "",
        "## Question",
        "",
        (
            "If the calibrated floor is selected using only prior rebalance-window calibration, does it beat the "
            "fixed `signed_lkj_eta_2` baseline on later windows?"
        ),
        "",
        "## Design",
        "",
        (
            f"- First `{manifest['min_history_windows']}` windows are selection history only; "
            f"the remaining `{manifest['n_test_windows']}` windows are scored out-of-sample."
        ),
        "- Candidate floors are selected by rolling mean NLL, CRPS, or average interval coverage error.",
        "- This reuses already generated GP artifacts; it tests the floor-selection layer, not a new GP refit.",
        "",
        "## Out-Of-Sample Metrics",
        "",
        markdown_table(
            summary,
            [
                "run_label",
                "n_rebalances",
                "terminal_value",
                "sharpe",
                "max_drawdown",
                "mean_ic",
                "mean_nll",
                "mean_crps",
                "z_std",
                "mean_abs_coverage_error",
                "coverage_80",
                "coverage_90",
                "coverage_95",
            ],
        ),
        "",
        "## Selection Counts",
        "",
        markdown_table(selector_counts, ["selector", "chosen_run_label", "n_selected"]),
        "",
        "## Readout",
        "",
        (
            f"- Best out-of-sample mean NLL: `{best_nll['run_label']}` at `{best_nll['mean_nll']:.4f}` "
            f"versus baseline `{baseline['mean_nll']:.4f}`."
        ),
        (
            f"- Best out-of-sample terminal value: `{best_terminal['run_label']}` at "
            f"`${best_terminal['terminal_value']:.2f}` versus baseline `${baseline['terminal_value']:.2f}`."
        ),
        "",
        "## Visuals",
        "",
        "- `rolling_selection_timeline.png`: chosen floor for each out-of-sample rebalance.",
        "- `rolling_interval_reliability.png`: predictive interval calibration on scored test windows.",
        "",
        "## Caveats",
        "",
        "- Only eight out-of-sample rebalance windows remain with the default four-window calibration warmup.",
        "- The candidate set itself was motivated by earlier experiments, so this is not a fully untouched holdout.",
        (
            "- Existing run artifacts are reused; this avoids refitting cost but does not test run-to-run "
            "optimizer variance."
        ),
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")


def run(args: argparse.Namespace) -> None:
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    run_inputs = [calib.RunInput(label, REPO_ROOT / run_dir) for label, run_dir in calib.DEFAULT_RUNS.items()]
    bundles = [load_bundle(run) for run in run_inputs]
    predictions = pd.concat([bundle.predictions for bundle in bundles], ignore_index=True)
    returns = pd.concat([bundle.portfolio_returns for bundle in bundles], ignore_index=True)
    _, window_metrics = prediction_with_window_metrics(predictions)

    candidates = [BASELINE_LABEL, *FLOOR_LABELS] if args.include_baseline_in_selector else FLOOR_LABELS
    selection, selector_predictions, selector_returns = build_selector_rows(
        predictions,
        returns,
        window_metrics,
        min_history=args.min_history,
        candidates=candidates,
    )

    test_dates = sorted(pd.Timestamp(date) for date in predictions["date"].drop_duplicates())[args.min_history :]
    fixed_predictions = predictions[predictions["date"].isin(test_dates)].copy()
    fixed_returns = returns[returns["date"].isin(test_dates)].copy()
    all_eval_predictions = pd.concat([fixed_predictions, selector_predictions], ignore_index=True)
    all_eval_returns = pd.concat([fixed_returns, selector_returns], ignore_index=True)
    labels = [BASELINE_LABEL, *FLOOR_LABELS, *SELECTOR_METRICS]
    summary = summarize_rolling(all_eval_predictions, all_eval_returns, labels=labels)

    manifest = {
        "schema": "bayesfolio.task_noise_floor_rolling_selection.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git_sha": git_sha(),
        "git_dirty_summary_at_start": git_dirty_summary(),
        "min_history_windows": args.min_history,
        "n_test_windows": len(test_dates),
        "test_dates": [date.date().isoformat() for date in test_dates],
        "candidate_floor_labels": FLOOR_LABELS,
        "selector_candidates": candidates,
        "selector_metrics": SELECTOR_METRICS,
        "runs": [{"label": bundle.label, "run_dir": str(bundle.run_dir)} for bundle in bundles],
        "output_dir": str(output_dir),
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    window_metrics.to_csv(output_dir / "window_calibration_metrics.csv", index=False)
    selection.to_csv(output_dir / "rolling_floor_selection.csv", index=False)
    all_eval_predictions.to_csv(output_dir / "rolling_eval_predictions.csv", index=False)
    all_eval_returns.to_csv(output_dir / "rolling_eval_portfolio_returns.csv", index=False)
    summary.to_csv(output_dir / "rolling_eval_summary.csv", index=False)
    plot_selection_timeline(selection, output_dir)
    plot_out_of_sample_reliability(all_eval_predictions, labels, output_dir)
    write_report(output_dir, summary=summary, selection=selection, manifest=manifest)

    print(summary.to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    run(parse_args())
