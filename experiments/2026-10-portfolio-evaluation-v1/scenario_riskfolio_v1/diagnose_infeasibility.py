"""Explain incomplete old-Riskfolio allocations without substituting portfolios.

Research-only diagnostic for a failed comparison: certify maximum feasible means,
retain all failed scenarios, and plot complete controls plus a truncated Riskfolio
wealth path. No missing month is omitted or backfilled. Returns are decimals and
wealth is dollars. Writes only new diagnostic artifacts inside this run's outputs.

Usage:
    uv run --locked --no-sync python <this_file>
"""

from __future__ import annotations

import io
import json

import cvxpy as cp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import run_comparison as run
from matplotlib.ticker import FuncFormatter
from scenario_allocator import allocate_scenarios, sample_joint


def _maximum_mean(mean: np.ndarray) -> dict:
    weights = cp.Variable(len(mean))
    problem = cp.Problem(
        cp.Maximize(mean @ weights),
        [weights >= 0, cp.sum(weights) == 1, weights <= 0.25, cp.sum_squares(weights) <= 0.1],
    )
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    if problem.status != cp.OPTIMAL or weights.value is None:
        raise ValueError("Failed to establish maximum feasible portfolio mean")
    return {
        "status": problem.status,
        "maximum_expected_return": float(problem.value),
        "maximizing_weights": weights.value.tolist(),
        "solver": "CLARABEL",
    }


def _diagnose() -> None:
    contract = json.loads((run.OUTPUT / "contract.json").read_text())
    run._verify_hashes(contract["input_sha256"])
    with np.load(run.OUTPUT / "input_panel.npz", allow_pickle=False) as source:
        panel = {key: source[key].copy() for key in source.files}
    failures, prefixes, completed = [], [], 0
    for seed_index, seed in enumerate(panel["seeds"]):
        prefix, interrupted = [], False
        for month_index, origin in enumerate(panel["dates"]):
            stem = run.OUTPUT / "cases" / f"s{seed}_{origin}"
            if stem.with_suffix(".json").exists():
                record = json.loads(stem.with_suffix(".json").read_text())
                if record["scenario_artifact_sha256"] != run._hash(stem.with_suffix(".npz")):
                    raise ValueError("Successful monthly evidence changed")
                completed += 1
                with np.load(stem.with_suffix(".npz"), allow_pickle=False) as case:
                    if not interrupted:
                        prefix.append(case["weights"].copy())
                continue
            interrupted = True
            mean, covariance = panel["mean"][seed_index, month_index], panel["covariance"][seed_index, month_index]
            scenarios = sample_joint(mean, covariance, seed=int(seed), month=str(origin))
            try:
                allocate_scenarios(scenarios, panel["assets"].tolist())
            except RuntimeError as error:
                message = str(error)
            else:
                raise ValueError("Previously failed allocation unexpectedly solved; resume the main runner")
            population, sample = _maximum_mean(mean), _maximum_mean(scenarios.mean(axis=0))
            if sample["maximum_expected_return"] >= 0 or population["maximum_expected_return"] >= 0:
                raise ValueError("Negative-feasible-mean explanation does not apply to this failure")
            failure = {
                "seed": int(seed),
                "forecast_origin": str(origin),
                "month_index": month_index,
                "population_mean_certificate": population,
                "sample_mean_certificate": sample,
                "sample_individual_mean_max": float(scenarios.mean(axis=0).max()),
                "Riskfolio_error": message,
                "fallback_applied": False,
            }
            failure_path = run.OUTPUT / "failures" / f"s{seed}_{origin}.npz"
            run._snapshot(failure_path, {"scenarios": scenarios, "mean": mean, "covariance": covariance})
            failure["scenario_sha256"] = run._hash(failure_path)
            failures.append(failure)
        prefix_weights = np.asarray(prefix)
        prefix_returns = run.portfolio_returns(
            panel["actual"][: len(prefix)],
            panel["risk_free"][: len(prefix)],
            np.column_stack([prefix_weights, np.zeros(len(prefix))]),
        )
        prefixes.append(run.wealth_path(prefix_returns))
    lengths = {len(path) for path in prefixes}
    if len(lengths) != 1:
        raise ValueError("Partial plot requires the same first missing month across seeds")
    origins = panel["dates"].astype("datetime64[D]")
    ends = (origins.astype("datetime64[M]") + 2).astype("datetime64[D]") - 1
    timeline = np.r_[origins[:1], ends]
    figure, axis = plt.subplots(figsize=(14, 8.5), layout="constrained")
    for name, color in [("gp_hard_rank", "#2455a4"), ("schwab_aggressive", "#b17c26"), ("momentum_top7", "#368275")]:
        paths = panel[f"{name}_wealth"]
        if paths.ndim == 2:
            axis.fill_between(timeline, paths.min(axis=0), paths.max(axis=0), color=color, alpha=0.1)
            paths = paths.mean(axis=0)
        axis.plot(timeline, paths, color=color, linewidth=2.7, label=f"{run.LABELS[name]} | ${paths[-1]:,.0f}")
    partial = np.asarray(prefixes)
    length = partial.shape[1]
    axis.fill_between(timeline[:length], partial.min(axis=0), partial.max(axis=0), color="#ba4536", alpha=0.1)
    axis.plot(
        timeline[:length],
        partial.mean(axis=0),
        color="#ba4536",
        linewidth=2.7,
        label="GP scenarios + old Riskfolio | INCOMPLETE: through July 2025 only",
    )
    axis.scatter(timeline[length - 1], partial[:, -1].mean(), color="#ba4536", marker="X", s=80, zorder=6)
    axis.axvline(ends[length - 1], color="#ba4536", linestyle=":", alpha=0.6)
    axis.annotate(
        "August 2025 allocation fails\nfor all three seeds; no fallback",
        xy=(ends[length - 1], partial[:, -1].mean()),
        xytext=(24, -58),
        textcoords="offset points",
        color="#8e3027",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#fff4ee", "edgecolor": "#dba59a"},
    )
    axis.set_title(
        "Same current GP forecasts: old Riskfolio comparison is INCOMPLETE\n"
        "Growth of $10,000 | June 2024–May 2026 returns | Monthly rebalancing",
        pad=14,
    )
    axis.set_xlabel("Time — month-end investment value")
    axis.set_ylabel("Investment value ($)")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.legend(loc="upper left", fontsize=10, framealpha=0.96)
    axis.grid(alpha=0.18)
    axis.set_xlim(timeline[0], timeline[-1] + np.timedelta64(12, "D"))
    figure.supxlabel(
        "No GP refits. GP curves average three saved model seeds; shading is their range, not confidence.\n"
        "Old Riskfolio: 5,000 joint scenarios/month, CVaR alpha 0.5, effective assets ≥10, ETF cap 25%.\n"
        "The red curve stops at its first missing allocation; no full-period Riskfolio return is reported.\n"
        "Gross of explicit costs/taxes. Schwab is a frozen ETF proxy; momentum holds its top seven equally.",
        fontsize=9,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    run._write(run.OUTPUT / "partial_growth_comparison.png", buffer.getvalue())
    report = {
        "status": "blocked_on_explicit_negative_forecast_policy",
        "complete_comparison": False,
        "GP_refits": 0,
        "successful_allocations": completed,
        "failed_allocations": failures,
        "partial_return_months": length - 1,
        "missing_months_omitted": False,
        "fallbacks": 0,
        "finding": "Some ETF sample means are positive, but no portfolio satisfying the old constraints "
        "has positive expected return. Riskfolio imposes expected return normalization to one "
        "when any ETF mean is positive, making its Sharpe homogeneous problem infeasible.",
        "requires_user_decision": "A new negative-forecast-month allocation rule is needed for a complete curve; "
        "no economic objective or diversification rule was changed automatically.",
    }
    run._write(run.OUTPUT / "infeasibility.json", report)
    run._verify_hashes(contract["input_sha256"])
    print(
        json.dumps(
            {
                "status": report["status"],
                "successful": completed,
                "failed": len(failures),
                "partial_return_months": length - 1,
                "plot": str(run.OUTPUT / "partial_growth_comparison.png"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _diagnose()
