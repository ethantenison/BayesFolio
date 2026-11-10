"""Plot rank-five maxiter150 forecasts against frozen maxiter15 predictions.

Research-only postprocessing of saved decimal excess returns for SPY, IWM, VEA
and HYG. Writes blue actuals, red larger-budget predictions and gray dashed
short-budget predictions, one panel per seed. No fits or protected-source edits.

Usage:
    MPLCONFIGDIR=/private/tmp/bayesfolio-mpl .venv/bin/python \
        experiments/2026-10-repaired-global-feature-selection/plot_rank5_budget150.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import run_rank5_budget150 as budget
from matplotlib.ticker import PercentFormatter


def main() -> None:
    """Validate completed evidence, render seed-specific comparisons and report stopping status."""
    root = budget.HERE
    audit = json.loads((root / "independent_audit.json").read_text())
    assert audit["passed"] and audit["seed_runs"] == 6
    assert json.loads((root / "status.json").read_text())["state"] == "completed"
    assert budget.data.digest(root / "contract.json") == audit["contract_sha256"]
    for relative, digest in audit["source_sha256"].items():
        assert budget.data.digest(budget.data.ROOT / relative) == digest
    out = root / "etf_forecast_plots"
    out.mkdir(exist_ok=True)
    metrics = pd.read_csv(root / "etf_metrics_by_seed.csv")
    frames = {}
    for seed in budget.prior.SEEDS:
        for cap, folder, identifier in [(15, budget.prior.HERE, budget.prior.IDENTIFIER),
                                        (150, root, budget.IDENTIFIER)]:
            frames[seed, cap] = pd.read_csv(folder / f"runs/{identifier}_s{seed}/gp_window_predictions.csv",
                                           parse_dates=["date"])
    for ticker in ["SPY", "IWM", "VEA", "HYG"]:
        panels = {key: frame[frame.asset_id.eq(ticker)].sort_values("date") for key, frame in frames.items()}
        values = []
        for (seed, cap), panel in panels.items():
            assert len(panel) == 24 and panel.date.nunique() == 24
            np.testing.assert_array_equal(panel[["date", "y_true"]].to_numpy(),
                                          panels[seed, 15][["date", "y_true"]].to_numpy())
            expected = metrics[metrics.seed.eq(seed) & metrics.maxiter.eq(cap) & metrics.ETF.eq(ticker)].iloc[0]
            error = panel.y_true - panel.y_pred
            nll = (0.5 * np.log(2 * np.pi * panel.y_std**2) + 0.5 * (error / panel.y_std)**2).mean()
            np.testing.assert_allclose(nll, expected.nll, atol=1e-10, rtol=0)
            values.extend(panel[["y_true", "y_pred"]].to_numpy().ravel().tolist())
        low, high = min(min(values), 0), max(max(values), 0)
        margin = max((high - low) * 0.12, 0.001)
        fig, axes = plt.subplots(3, 1, figsize=(12, 8.4), sharex=True, sharey=True)
        for axis, seed in zip(axes, budget.prior.SEEDS, strict=True):
            old, new = panels[seed, 15], panels[seed, 150]
            axis.plot(old.date, old.y_pred, color="#72767c", linestyle="--", linewidth=1.7,
                      label="Rank 5: maxiter 15")
            axis.plot(new.date, new.y_true, color="#1764c0", linewidth=1.8, marker="o", markersize=3.5,
                      label="Actual excess return")
            axis.plot(new.date, new.y_pred, color="#d62728", linewidth=2, marker="o", markersize=3.5,
                      label="Rank 5: maxiter 150")
            axis.axhline(0, color="#808b98", linestyle="--", linewidth=0.8)
            axis.set_ylim(low - margin, high + margin)
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
            axis.set_ylabel("Excess return")
            axis.set_title(f"Seed {seed}", loc="left", fontsize=11)
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.907), frameon=False,
                   ncol=3, fontsize=10)
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axes[-1].set_xlabel("Forecast date (month-end; return target is the following month)")
        fig.suptitle(f"{ticker} | Rank-five fitting-budget comparison", fontsize=16, y=0.98)
        fig.text(0.5, 0.935, "Same 14 inputs + t_index, rank, data, priors and scenarios | Only fitting cap changes",
                 ha="center", fontsize=10, color="#465160")
        fig.text(0.5, 0.018, "24 forecasts per seed; shared y-axis scale. No smoothing or seed averaging.",
                 ha="center", fontsize=9, color="#465160")
        fig.tight_layout(rect=(0, 0.04, 1, 0.92), h_pad=1)
        fig.savefig(out / f"{ticker}_budget15_vs150.png", dpi=160, facecolor="white")
        plt.close(fig)
    diag = pd.read_csv(root / "fitting_diagnostics.csv")
    returns = diag[diag.event.eq("optimizer_return")]
    final = returns.groupby(["seed_run", "date"], sort=False).tail(1)
    assert len(final) == 72
    summary = {"windows": len(final), "optimizer_attempts": len(returns),
               "final_status_counts": final.status.value_counts().to_dict(),
               "final_iterations_min": int(final.iterations.min()),
               "final_iterations_median": float(final.iterations.median()),
               "final_iterations_max": int(final.iterations.max()),
               "windows_using_full_cap": int(final.iterations.ge(150).sum()),
               "caveat": "SUCCESS is optimizer stopping-criterion convergence, not proof of a global optimum."}
    budget.data.write_json(root / "convergence_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(pd.read_csv(root / "objectives_by_seed.csv").groupby("maxiter").mean().to_string())


if __name__ == "__main__":
    main()
