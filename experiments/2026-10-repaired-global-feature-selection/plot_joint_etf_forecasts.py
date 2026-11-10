"""Plot saved joint, rank-five or highest-Sharpe U.S. equity forecasts.

Experiment-only visualization: reads three frozen seed panels in decimal excess
returns and writes PNGs plus a JSON variability summary. Axes display percent
returns at the saved forecast-origin dates; each label is next month's excess
return. No seed averaging, smoothing, source repair or new fits are performed.

Usage:
    export MPLCONFIGDIR=/private/tmp/bayesfolio-mpl
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/plot_joint_etf_forecasts.py
    Add --us-equity for the seven-input highest-Sharpe U.S.-equity-only model.
    Add --rank5 for the completed rank-five sensitivity with rank-two overlays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = "global_e6c41f2020fd"
SEEDS = [27, 127, 227]
TICKERS = ["SPY", "IWM", "VEA", "HYG"]


def main() -> None:
    """Verify saved evidence and render actual-blue/predicted-red monthly line graphs."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--us-equity", action="store_true", help="Use highest mean Sharpe U.S.-equity-only mask")
    mode.add_argument("--rank5", action="store_true", help="Use rank-five joint forecasts with rank-two overlays")
    args = parser.parse_args()
    result = HERE / "followup/results/all_joint"
    model, tickers = MODEL, TICKERS
    selection = "14-input SPY-proxy joint model + t_index"
    source_hashes = {}
    if args.rank5:
        result = HERE / "rank5_followup"
        status = json.loads((result / "status.json").read_text())
        audit = json.loads((result / "independent_audit.json").read_text())
        contract = json.loads((result / "contract.json").read_text())
        assert status["state"] == "completed" and audit["passed"] and audit["seed_runs"] == 6
        assert hashlib.sha256((result / "contract.json").read_bytes()).hexdigest() == audit["contract_sha256"]
        for relative, expected in audit["source_sha256"].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
        model = MODEL + "_rank5"
        selection = "14-input SPY-proxy joint model + t_index | Rank 5 vs rank 2"
        out = result / "etf_forecast_plots"
        exact = contract["expected_inputs"]
    elif args.us_equity:
        import workflow

        final = json.loads((HERE / "results/final_recommendations.json").read_text())
        for relative, expected in final["source_sha256"].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
        result = HERE / "results/cumulative/us_equity"
        aggregates = pd.read_csv(result / "configuration_aggregates.csv")
        assert aggregates.validation_passed.all() and aggregates.seed_count.eq(3).all()
        model = str(aggregates.loc[aggregates.sharpe.idxmax(), "configuration"])
        plans = [(path, json.loads(path.read_text())) for path in (HERE / "plans").glob("*.json")]
        matched = [(path, plan) for path, plan in plans if plan["family"] == "us_equity"
                   and any(item["id"] == model for item in plan["configurations"])]
        assert len(matched) == 1
        path, plan = matched[0]
        workflow.validate_plan(plan, path, workflow.validate_frozen_data())
        assert workflow.completed_batch_is_valid(path, plan)
        tickers = ["SPY", "IWM"]
        selection = "7-input U.S.-equity-only model + t_index | Highest mean portfolio Sharpe"
        out = HERE / "results/us_equity_forecast_plots"
    else:
        audit = json.loads((result / "independent_audit.json").read_text())
        assert audit["passed"] and audit["checked_seed_runs"] == 111
        source_hashes = audit["source_sha256"]
        out = result / "etf_forecast_plots"
    if not args.rank5:
        exact = json.loads((result / "exact_inputs.json").read_text())[model]
    assert "t_index" in exact
    if args.us_equity:
        assert len(exact) == 8
    else:
        assert len(exact) == 15 and "spy_ret" in exact
    frames = {}
    hashes = {}
    for seed in SEEDS:
        run_root = HERE / "runs" if args.us_equity else HERE / "followup/runs"
        if args.rank5:
            run_root = result / "runs"
        path = run_root / f"{model}_s{seed}/gp_window_predictions.csv"
        relative = str(path.relative_to(ROOT))
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not args.us_equity and not args.rank5:
            assert hashes[relative] == source_hashes[relative]
        frames[seed] = pd.read_csv(path, parse_dates=["date"])
        assert "IWM" in frames[seed].asset_id.unique() and "IJR" not in frames[seed].asset_id.unique()
        if args.us_equity:
            assert set(frames[seed].asset_id) == {"SPY", "MGK", "VTV", "IWM"}
    out.mkdir(exist_ok=True)
    diagnostics = []
    for ticker in tickers:
        panels = {seed: frame[frame.asset_id.eq(ticker)].sort_values("date")
                  for seed, frame in frames.items()}
        for panel in panels.values():
            assert len(panel) == 24 and panel.date.nunique() == 24
            assert np.isfinite(panel[["y_true", "y_pred"]].to_numpy()).all()
            np.testing.assert_array_equal(panel.date.to_numpy(), panels[27].date.to_numpy())
            np.testing.assert_array_equal(panel.y_true.to_numpy(), panels[27].y_true.to_numpy())
        baselines = {}
        if args.rank5:
            baseline_audit = json.loads((HERE / "followup/results/all_joint/independent_audit.json").read_text())
            metrics = pd.read_csv(result / "etf_metrics_by_seed.csv")
            for seed, panel in panels.items():
                path = HERE / f"followup/runs/{MODEL}_s{seed}/gp_window_predictions.csv"
                assert hashlib.sha256(path.read_bytes()).hexdigest() == baseline_audit["source_sha256"][
                    str(path.relative_to(ROOT))]
                base = pd.read_csv(path, parse_dates=["date"])
                baselines[seed] = base[base.asset_id.eq(ticker)].sort_values("date")
                np.testing.assert_array_equal(panel.date.to_numpy(), baselines[seed].date.to_numpy())
                np.testing.assert_array_equal(panel.y_true.to_numpy(), baselines[seed].y_true.to_numpy())
                saved = metrics[metrics["rank"].eq(5) & metrics.seed.eq(seed) & metrics.ETF.eq(ticker)].iloc[0]
                nll = (0.5 * np.log(2 * np.pi * panel.y_std**2)
                       + 0.5 * ((panel.y_true - panel.y_pred) / panel.y_std)**2).mean()
                np.testing.assert_allclose(nll, saved.nll, atol=1e-10, rtol=0)
        values = np.concatenate([panel[["y_true", "y_pred"]].to_numpy().ravel() for panel in panels.values()])
        if args.rank5:
            values = np.concatenate([values, *[base.y_pred.to_numpy() for base in baselines.values()]])
        low, high = min(values.min(), 0), max(values.max(), 0)
        margin = max((high - low) * 0.12, 0.001)
        fig, axes = plt.subplots(3, 1, figsize=(12, 8.4), sharex=True, sharey=True)
        for axis, (seed, panel) in zip(axes, panels.items(), strict=True):
            if args.rank5:
                axis.plot(baselines[seed].date, baselines[seed].y_pred, color="#72767c", linewidth=1.7,
                          linestyle="--", label="Rank 2 prediction", zorder=1)
            axis.plot(panel.date, panel.y_true, color="#1764c0", linewidth=1.8,
                      marker="o", markersize=3.5, label="Actual excess return", zorder=2)
            axis.plot(panel.date, panel.y_pred, color="#d62728", linewidth=2,
                      marker="o", markersize=3.5,
                      label="Rank 5 prediction" if args.rank5 else "Predicted excess return", zorder=3)
            axis.axhline(0, color="#808b98", linewidth=0.8, linestyle="--", zorder=1)
            axis.set_ylim(low - margin, high + margin)
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
            axis.set_ylabel("Excess return")
            axis.set_title(f"Seed {seed}", loc="left", fontsize=11)
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
            actual_sd, predicted_sd = panel.y_true.std(ddof=1), panel.y_pred.std(ddof=1)
            diagnostics.append({"etf": ticker, "seed": seed, "months": len(panel),
                "actual_time_sd_decimal": float(actual_sd), "predicted_time_sd_decimal": float(predicted_sd),
                "predicted_to_actual_time_sd": float(predicted_sd / actual_sd),
                "predicted_min_decimal": float(panel.y_pred.min()),
                "predicted_max_decimal": float(panel.y_pred.max())})
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.907),
                   frameon=False, ncol=3 if args.rank5 else 2, fontsize=10)
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axes[-1].set_xlabel("Forecast date (month-end; return target is the following month)")
        fig.suptitle(f"{ticker} | Actual vs predicted next-month excess return", fontsize=16, y=0.98)
        fig.text(0.5, 0.935, selection,
                 ha="center", fontsize=10, color="#465160")
        fig.text(0.5, 0.018, "24 forecasts per seed; same y-axis scale. No smoothing, averaging or refitting.",
                 ha="center", fontsize=9, color="#465160")
        fig.tight_layout(rect=(0, 0.04, 1, 0.92), h_pad=1.0)
        fig.savefig(out / f"{ticker}_actual_vs_predicted.png", dpi=160, facecolor="white")
        plt.close(fig)
    summary = {"configuration": model, "selection": selection, "inputs": exact, "source_sha256": hashes,
               "x_axis": "Saved forecast-origin date, not following-month realization date",
               "units": "Source decimals; chart axes percent", "diagnostics": diagnostics}
    (out / "variability.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
