"""Compare cold month-parallel replay and the HY-change addition; plot SPY.

Read-only scientific postprocessing of completed frozen-model outputs. Writes
seed/ETF diagnostics, conventional excess-return portfolio statistics and three
SPY actual/predicted panels. Does not fit, refresh sources, or select a new model.
All return data are decimals; chart axes display percentages. Turnover excluded.

Usage:
    MPLCONFIGDIR=/private/tmp/bayesfolio-mpl .venv/bin/python \
        experiments/2026-10-repaired-global-feature-selection/report_month_parallel.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import run_month_parallel as pipeline
from matplotlib.ticker import PercentFormatter


def conventional_stats(returns: pd.Series) -> dict[str, float]:
    """Calculate date-sorted excess-return path diagnostics without a second RF subtraction.

    Args:
        returns: Complete finite monthly decimal excess returns in chronological order.

    Returns:
        Conventional annualized sample-SD Sharpe and diagnostic compounded excess path.

    Raises:
        ValueError: Returns are nonfinite or have zero sample standard deviation.
    """
    sd = float(returns.std(ddof=1))
    if not np.isfinite(returns.to_numpy()).all() or not np.isfinite(sd) or sd <= 0:
        raise ValueError("Invalid excess-return path for conventional Sharpe")
    path = (1 + returns).cumprod()
    # Include initial unit capital in the running peak, including an initial losing month.
    peaks = np.maximum.accumulate(np.r_[1.0, path.to_numpy()])[1:]
    return {"mean_monthly_excess_return": float(returns.mean()),
            "annualized_sample_vol": sd * np.sqrt(12), "sharpe": float(returns.mean() / sd * np.sqrt(12)),
            "cumulative_excess_return_diagnostic_only": float(path.iloc[-1] - 1),
            "max_drawdown_excess_path_diagnostic_only": float((path.to_numpy() / peaks - 1).min())}


def main() -> None:
    """Verify completed runs, report all metrics and render the requested SPY comparison."""
    root = pipeline.HERE
    comparison = json.loads((root / "comparison.json").read_text())
    contract = json.loads((root / "contract.json").read_text())
    assert comparison["completed"] and json.loads((root / "status.json").read_text())["state"] == "completed"
    pipeline._fresh(contract)
    for phase in ["baseline", "plus_hy_change"]:
        saved = json.loads((root / phase / "completed.json").read_text())
        assert saved["signature"] == pipeline.data.digest(root / "contract.json")
        for name, digest in saved["artifact_sha256"].items():
            assert pipeline.data.digest(pipeline.data.ROOT / name) == digest
        for stamp in (root / phase / "checkpoints").glob("*.json"):
            job = json.loads(stamp.read_text())
            for name, digest in job["artifact_sha256"].items():
                assert pipeline.data.digest(pipeline.data.ROOT / name) == digest
    frames, objectives, etfs, portfolio_rows = {}, [], [], []
    sources = [Path(__file__).resolve(), root / "contract.json", root / "comparison.json"]
    for phase in ["baseline", "plus_hy_change"]:
        audit = json.loads((root / phase / "independent_audit.json").read_text())
        assert audit["passed"] and audit["seed_runs"] == 3 and audit["fit_windows"] == 72
        for seed in pipeline.SEEDS:
            run = root / phase / "runs" / f"s{seed}"
            independent = pipeline.reconstruct(run, pipeline.data.TICKERS, contract["dates"])
            objectives.append({"phase": phase, "seed": seed, **{key: independent[key] for key in
                ["mean_ic", "mean_gaussian_nll", "sharpe", "cumulative_return_diagnostic_only",
                 "rmse_diagnostic_only"]}})
            frame = pd.read_csv(run / "gp_window_predictions.csv", parse_dates=["date"])
            frames[phase, seed] = frame
            for ticker, panel in frame.groupby("asset_id"):
                error = panel.y_true - panel.y_pred
                etfs.append({"phase": phase, "seed": seed, "ETF": ticker,
                    "nll": (0.5 * np.log(2 * np.pi * panel.y_std**2) + 0.5 * (error / panel.y_std)**2).mean(),
                    "rmse": np.sqrt(np.square(error).mean()), "predicted_time_sd": panel.y_pred.std(ddof=1),
                    "mean_predictive_sd": panel.y_std.mean(),
                    "coverage95": (error.abs() <= 1.959963984540054 * panel.y_std).mean()})
            paths = pd.read_csv(run / "portfolio_returns.csv")
            for strategy, panel in paths.groupby("strategy"):
                ordered = panel.sort_values("date")
                assert ordered.date.tolist() == contract["dates"]
                portfolio_rows.append({"phase": phase, "seed": seed, "strategy": strategy,
                                       **conventional_stats(ordered["return"])})
            sources.extend(run / name for name in ["gp_window_predictions.csv", "portfolio_returns.csv",
                                                   "portfolio_weights.csv", "manifest.json"])
    objective_rows, etf_rows = pd.DataFrame(objectives), pd.DataFrame(etfs)
    objective_rows.to_csv(root / "objectives_by_seed.csv", index=False)
    etf_rows.to_csv(root / "etf_metrics_by_seed.csv", index=False)
    pd.DataFrame(portfolio_rows).to_csv(root / "portfolio_statistics_by_seed.csv", index=False)
    means = objective_rows.drop(columns="seed").groupby("phase").mean()
    delta = objective_rows[objective_rows.phase.eq("plus_hy_change")].set_index("seed").drop(columns="phase") - (
        objective_rows[objective_rows.phase.eq("baseline")].set_index("seed").drop(columns="phase"))
    result = {**comparison, "means": means.to_dict("index"),
              "paired_mean_deltas_added_minus_base": delta.mean().to_dict(),
              "added_seed_wins": {key: int((delta[key] * sign > 0).sum()) for key, sign in
                  [("mean_ic", 1), ("mean_gaussian_nll", -1), ("sharpe", 1)]},
              "exact_inputs": {name: value["inputs"] for name, value in contract["variants"].items()},
              "interpretation_boundary": "Adaptive reused-date research; no untouched confirmation or promotion"}
    observation = root / "runtime_observation.json"
    if observation.exists():
        result["runtime_observation"] = json.loads(observation.read_text())
        sources.append(observation)
    pipeline.data.write_json(root / "final_comparison.json", result)
    out = root / "SPY_hy_change_comparison.png"
    panels = {(phase, seed): frame[frame.asset_id.eq("SPY")].sort_values("date")
              for (phase, seed), frame in frames.items()}
    values = np.concatenate([frame[["y_true", "y_pred"]].to_numpy().ravel() for frame in panels.values()])
    low, high = min(values.min(), 0), max(values.max(), 0)
    margin = max((high - low) * 0.12, 0.001)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.4), sharex=True, sharey=True)
    for axis, seed in zip(axes, pipeline.SEEDS, strict=True):
        base, added = panels["baseline", seed], panels["plus_hy_change", seed]
        np.testing.assert_array_equal(base[["date", "y_true"]].to_numpy(), added[["date", "y_true"]].to_numpy())
        axis.plot(base.date, base.y_pred, color="#72767c", linestyle="--", linewidth=1.8, label="14-input baseline")
        axis.plot(added.date, added.y_true, color="#1764c0", marker="o", markersize=3.5, linewidth=1.8,
                  label="Actual excess return")
        axis.plot(added.date, added.y_pred, color="#d62728", marker="o", markersize=3.5, linewidth=2,
                  label="Baseline + HY spread change")
        axis.axhline(0, color="#808b98", linestyle="--", linewidth=0.8)
        axis.set_ylim(low - margin, high + margin)
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        axis.set_ylabel("Excess return")
        axis.set_title(f"Seed {seed}", loc="left", fontsize=11)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.907), frameon=False, ncol=3, fontsize=10)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].set_xlabel("Forecast date (month-end; return target is the following month)")
    fig.suptitle("SPY | Does adding HY spread change improve the joint forecast?", fontsize=15, y=0.98)
    fig.text(0.5, 0.935, "Rank 5 | maxiter 36 | Only hy_spread_chg_1p added | Same data and portfolio constraints",
             ha="center", fontsize=10, color="#465160")
    fig.text(0.5, 0.018, "24 forecasts per seed; shared y-axis scale. No smoothing or seed averaging.",
             ha="center", fontsize=9, color="#465160")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92), h_pad=1)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    sources.extend([out, root / "final_comparison.json", root / "etf_metrics_by_seed.csv",
                    root / "objectives_by_seed.csv", root / "portfolio_statistics_by_seed.csv"])
    pipeline.data.write_json(root / "RESULTS_VERIFICATION.json", {"passed": True,
        "source_signature": pipeline.data.digest(root / "contract.json"), "new_seed_runs": 6,
        "new_gp_windows": 144, "strict_no_fallback": True, "artifact_sha256": pipeline._hashes(sources),
        "verification_commands": [
            ".venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection",
            ".venv/bin/ruff check experiments/2026-10-repaired-global-feature-selection",
            "MPLCONFIGDIR=/private/tmp/bayesfolio-mpl .venv/bin/python "
            "experiments/2026-10-repaired-global-feature-selection/report_month_parallel.py"]})
    print(json.dumps({key: value for key, value in result.items() if key != "exact_inputs"}, indent=2))


if __name__ == "__main__":
    main()
