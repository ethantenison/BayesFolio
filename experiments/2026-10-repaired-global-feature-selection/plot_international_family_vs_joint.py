"""Compare saved VEA/VSS family-only and latest joint forecasts, without fitting.

Experiment-only scientific postprocessing. Selects the international family's
highest three-seed mean portfolio Sharpe, matching the earlier U.S. comparison.
The joint comparator is the completed HY-change, rank-two/maxiter-fifteen run.
Both use the same 84+k training months and 24 forecast origins. Source returns
are decimals; chart axes display percentages. Masks and task universes differ,
so these plots do not identify a causal feature or transfer mechanism.

Usage:
    MPLCONFIGDIR=/private/tmp/bayesfolio-mpl .venv/bin/python \
        experiments/2026-10-repaired-global-feature-selection/plot_international_family_vs_joint.py
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import run_month_parallel as pipeline
import workflow
from matplotlib.ticker import PercentFormatter

HERE = pipeline.data.HERE
JOINT = HERE / "month_parallel_hy_rank2_iter15"
OUT = HERE / "results/international_equity_joint_forecast_plots"
SEEDS = [27, 127, 227]
TICKERS = ["VEA", "VSS"]


def paired_panels(family: pd.DataFrame, joint: pd.DataFrame, ticker: str,
                  dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and align one ETF's two unsmoothed decimal forecast panels.

    Args:
        family: Saved international-equity-only forecasts for one seed.
        joint: Saved all-fifteen-ETF forecasts for the same seed.
        ticker: ETF to display, without selecting a model using its forecast score.
        dates: Complete frozen forecast-origin date grid.

    Returns:
        Date-sorted family and joint panels, each with identical realized labels.

    Raises:
        ValueError: Missing, duplicate, nonfinite or invalid predictive observations.
        AssertionError: Targets or scored dates differ across the models.
    """
    panels = []
    for frame in [family, joint]:
        panel = frame[frame.asset_id.eq(ticker)].sort_values("date").copy()
        panel["date"] = pd.to_datetime(panel.date)
        if len(panel) != len(dates) or panel.date.nunique() != len(dates):
            raise ValueError("Incomplete or duplicated ETF/date grid")
        if not np.isfinite(panel[["y_true", "y_pred", "y_std"]].to_numpy()).all() or panel.y_std.le(0).any():
            raise ValueError("Invalid saved predictive observations")
        np.testing.assert_array_equal(panel.date.to_numpy(), pd.to_datetime(dates).to_numpy())
        panels.append(panel)
    # Only decimal CSV serialization tolerance; no target shifts or rescaling.
    np.testing.assert_allclose(panels[0].y_true, panels[1].y_true, atol=1e-14, rtol=0)
    return panels[0], panels[1]


def forecast_metrics(panel: pd.DataFrame) -> dict[str, float]:
    """Compute raw-return forecast errors and mean variability, not portfolio scores.

    Args:
        panel: Valid finite monthly predictions with positive predictive SDs.

    Returns:
        Decimal RMSE, marginal Gaussian NLL, time-series correlation and SDs.
        Forecast time SD describes variation of means, not predictive uncertainty.
    """
    error = panel.y_true - panel.y_pred
    actual_sd, forecast_sd = panel.y_true.std(ddof=1), panel.y_pred.std(ddof=1)
    return {"rmse": float(np.sqrt(np.square(error).mean())),
            "nll": float((0.5 * np.log(2 * np.pi * panel.y_std**2) + 0.5 * (error / panel.y_std)**2).mean()),
            "actual_time_sd": float(actual_sd), "forecast_time_sd": float(forecast_sd),
            "forecast_to_actual_time_sd": float(forecast_sd / actual_sd),
            "time_series_correlation_not_cross_sectional_ic": float(panel.y_true.corr(panel.y_pred)),
            "mean_predictive_sd": float(panel.y_std.mean())}


def main() -> None:
    """Validate completed evidence, render seed-by-model figures and seal diagnostics."""
    final = json.loads((HERE / "results/final_recommendations.json").read_text())
    joint_contract = json.loads((JOINT / "contract.json").read_text())
    assert json.loads((JOINT / "status.json").read_text())["state"] == "completed"
    pipeline._fresh(joint_contract)
    for name in ["completed.json", "RESULTS_VERIFICATION.json"]:
        for path, digest in json.loads((JOINT / name).read_text())["artifact_sha256"].items():
            assert pipeline.data.digest(pipeline.data.ROOT / path) == digest, path
    for path, digest in final["source_sha256"].items():
        assert pipeline.data.digest(pipeline.data.ROOT / path) == digest, path
    result = HERE / "results/cumulative/international_equity"
    aggregate_path = result / "configuration_aggregates.csv"
    aggregate = pd.read_csv(aggregate_path)
    assert aggregate.validation_passed.all() and aggregate.seed_count.eq(3).all()
    family_id = str(aggregate.loc[aggregate.sharpe.idxmax(), "configuration"])
    plans = [(path, json.loads(path.read_text())) for path in (HERE / "plans").glob("*.json")]
    matches = [(path, plan) for path, plan in plans if plan["family"] == "international_equity"
               and any(item["id"] == family_id for item in plan["configurations"])]
    assert len(matches) == 1
    plan_path, plan = matches[0]
    workflow.validate_plan(plan, plan_path, workflow.validate_frozen_data())
    assert workflow.completed_batch_is_valid(plan_path, plan)
    family_inputs = json.loads((result / "exact_inputs.json").read_text())[family_id]
    joint_inputs = joint_contract["variants"]["rank2_iter15"]["inputs"]
    assert "t_index" in family_inputs and "t_index" in joint_inputs
    dates = joint_contract["dates"]
    assert len(dates) == 24 and joint_contract["rank"] == 2 and joint_contract["maxiter"] == 15
    sources = [Path(__file__).resolve(), plan_path, aggregate_path, result / "exact_inputs.json",
               HERE / "results/final_recommendations.json", JOINT / "contract.json",
               JOINT / "completed.json", JOINT / "RESULTS_VERIFICATION.json"]
    panels, rows = {}, []
    for seed in SEEDS:
        family_run = HERE / "runs" / f"{family_id}_s{seed}"
        joint_run = JOINT / "rank2_iter15/runs" / f"s{seed}"
        fm, jm = [json.loads((run / "manifest.json").read_text()) for run in [family_run, joint_run]]
        assert fm["modeling"]["rank"] == jm["rank"] == 2
        assert fm["modeling"]["maxiter"] == jm["maxiter"] == 15
        command = shlex.split(fm["command"])
        actual_inputs = ["t_index", *command[command.index("--custom-etf-cols") + 1].split(","),
                         *command[command.index("--custom-macro-cols") + 1].split(",")]
        assert actual_inputs == family_inputs and jm["input_columns"] == joint_inputs
        frames = []
        for run, size in [(family_run, 4), (joint_run, 15)]:
            frame = pd.read_csv(run / "gp_window_predictions.csv", parse_dates=["date"])
            assert frame.asset_id.nunique() == size and len(frame) == 24 * size
            training = pd.read_csv(run / "window_training_history.csv")
            np.testing.assert_array_equal(training.unique_train_dates, np.arange(84, 108))
            np.testing.assert_array_equal(training.train_rows, np.arange(84, 108) * size)
            frames.append(frame)
            sources.extend(run / name for name in ["gp_window_predictions.csv", "manifest.json",
                                                    "window_training_history.csv"])
        saved_metrics = pd.read_csv(family_run / "forecast_metrics_by_asset.csv").set_index("asset_id")
        sources.extend([family_run / "forecast_metrics_by_asset.csv",
                        family_run / "repaired_execution_fingerprint.json"])
        for ticker in TICKERS:
            family, joint = paired_panels(*frames, ticker, dates)
            for model, panel in [("international_family", family), ("joint", joint)]:
                panels[ticker, seed, model] = panel
                metrics = forecast_metrics(panel)
                if model == "international_family":
                    np.testing.assert_allclose([metrics["rmse"], metrics["nll"]],
                        saved_metrics.loc[ticker, ["rmse", "mean_gaussian_nll"]].to_numpy(dtype=float),
                        atol=1e-10, rtol=0)
                rows.append({"ETF": ticker, "seed": seed, "model": model, **metrics})
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics_by_seed.csv", index=False)
    means = metrics.drop(columns="seed").groupby(["ETF", "model"]).mean().reset_index()
    means.to_csv(OUT / "mean_metrics.csv", index=False)
    figures = []
    for ticker in TICKERS:
        values = np.concatenate([panel[["y_true", "y_pred"]].to_numpy().ravel()
                                 for (etf, _, _), panel in panels.items() if etf == ticker])
        low, high = min(values.min(), 0), max(values.max(), 0)
        margin = max((high - low) * 0.12, 0.001)
        fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True, sharey=True)
        for row, seed in enumerate(SEEDS):
            for col, (model, label) in enumerate([("international_family", "International equities only"),
                                                 ("joint", "Joint — all 15 ETFs")]):
                axis, panel = axes[row, col], panels[ticker, seed, model]
                axis.plot(panel.date, panel.y_true, color="#1764c0", linewidth=1.8,
                          marker="o", markersize=3.4, label="Actual excess return")
                axis.plot(panel.date, panel.y_pred, color="#d62728", linewidth=2,
                          marker="o", markersize=3.4, label="Predicted excess return")
                axis.axhline(0, color="#808b98", linewidth=0.8, linestyle="--")
                axis.set_ylim(low - margin, high + margin)
                axis.set_title(f"{label} | Seed {seed}", loc="left", fontsize=11)
                axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
                axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                axis.grid(alpha=0.18)
                axis.spines[["top", "right"]].set_visible(False)
                if col == 0:
                    axis.set_ylabel("Excess return")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.908),
                   frameon=False, ncol=2, fontsize=11)
        fig.suptitle(f"{ticker} | International-family versus joint forecast", fontsize=16, y=0.98)
        fig.text(0.5, 0.94, "Both rank 2 / maxiter 15 | 84 initial training months | Same 24 forecast dates",
                 ha="center", fontsize=11, color="#465160")
        fig.text(0.5, 0.055, "Forecast date (month-end; the excess-return target is the following month)",
                 ha="center", fontsize=10)
        fig.text(0.5, 0.022, "Family: highest mean portfolio Sharpe, 16 inputs + time | "
                 "Joint: latest HY-change model, 15 inputs + time | No averaging or smoothing",
                 ha="center", fontsize=10, color="#465160")
        fig.tight_layout(rect=(0.01, 0.075, 0.995, 0.90), h_pad=1.1, w_pad=1.6)
        path = OUT / f"{ticker}_family_vs_joint.png"
        fig.savefig(path, dpi=160, facecolor="white")
        plt.close(fig)
        figures.append(path)
    summary = {"family_configuration": family_id,
        "family_selection": "Highest validated three-seed mean portfolio Sharpe; not chosen per displayed ETF",
        "joint_configuration": "global_e6c41f2020fd + hy_spread_chg_1p, rank 2/maxiter 15",
        "rank": 2, "maxiter": 15, "seeds": SEEDS, "forecast_dates": dates,
        "training_months": "84+k, expanding", "new_fits": 0,
        "exact_inputs": {"family": family_inputs, "joint": joint_inputs},
        "family_only_inputs": sorted(set(family_inputs) - set(joint_inputs)),
        "joint_only_inputs": sorted(set(joint_inputs) - set(family_inputs)),
        "mean_metrics": means.to_dict("records"),
        "boundary": "Saved adaptive research forecasts. Masks AND universes differ; "
                    "not a causal test of irrelevant inputs, negative transfer, or convergence.",
        "source_sha256": pipeline._hashes(sources)}
    pipeline.data.write_json(OUT / "comparison.json", summary)
    # Confirm that source evidence was not modified while producing figures.
    for path, digest in summary["source_sha256"].items():
        assert pipeline.data.digest(pipeline.data.ROOT / path) == digest, path
    pipeline.data.write_json(OUT / "RESULTS_VERIFICATION.json", {"passed": True,
        "source_sha256": summary["source_sha256"], "artifact_sha256": pipeline._hashes([
            *figures, OUT / "comparison.json", OUT / "metrics_by_seed.csv", OUT / "mean_metrics.csv"]),
        "new_fits": 0, "paired_panels": 6, "matching_targets_and_dates": True})
    print(means.to_string(index=False))
    print(f"Saved figures and verified diagnostics: {OUT}")


if __name__ == "__main__":
    main()
