"""Compare the cached Sharpe leader with monthly rebalanced Schwab-style proxies.

Usage: .venv/bin/python <this_file> --execute; then --seal or --verify.
Research-only orchestration writes only this folder's outputs. Source weights,
forecasts and prior evidence stay read-only; no GP refits or portfolio optimizer
runs occur. Returns are decimals and investment values are dollars.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-schwab-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-schwab-cache")
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
FRAMEWORK = HERE.parent
sys.path.extend([str(FRAMEWORK / "blend_sleeves_v1")])

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import run_blends as base  # noqa: E402
from benchmark_math import portfolio_returns, scheduled_weights, wealth_path  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from metrics import paired_block_bootstrap, portfolio_performance  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: object) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Only new benchmark outputs are writable")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Conflicting immutable result: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def _audit_wealth(actual: np.ndarray, risk_free: np.ndarray, weights: np.ndarray, expected: np.ndarray) -> None:
    # Independent dollar-holdings recursion; every rebalance is self-financing.
    account = 10000.0
    observed = [account]
    for index in range(len(actual)):
        positions = account * weights[index]
        asset_total = np.r_[(1 + actual[index]) * (1 + risk_free[index]) - 1, risk_free[index]]
        account = float(np.sum(positions * (1 + asset_total)))
        observed.append(account)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-7)


def _plot(arrays: dict, results: dict) -> None:
    origins = np.asarray(arrays["dates"], dtype="datetime64[D]")
    targets = (origins.astype("datetime64[M]") + 2).astype("datetime64[D]") - 1
    timeline = np.r_[origins[:1], targets]
    figure, axis = plt.subplots(figsize=(12.8, 7), layout="constrained")
    gp = arrays["GP_wealth"]
    for seed_path in gp:
        axis.plot(timeline, seed_path, color="#245bb2", alpha=0.2, linewidth=0.9)
    axis.fill_between(
        timeline,
        gp.min(axis=0),
        gp.max(axis=0),
        color="#245bb2",
        alpha=0.12,
        label="GP initialization range (3 seeds; not a confidence interval)",
    )
    axis.plot(
        timeline, gp.mean(axis=0), color="#245bb2", linewidth=2.6, label="Sharpe leader — mean of 3 seed wealth paths"
    )
    for name, label, color in [
        ("aggressive", "Schwab-style aggressive", "#bd752d"),
        ("moderately_aggressive", "Schwab-style moderately aggressive", "#218775"),
    ]:
        axis.plot(timeline, arrays[f"{name}_wealth"], color=color, linewidth=2.2, label=label)
    labels = [
        (gp[:, -1].mean(), "#245bb2"),
        (arrays["aggressive_wealth"][-1], "#bd752d"),
        (arrays["moderately_aggressive_wealth"][-1], "#218775"),
    ]
    for value, color in labels:
        axis.annotate(
            f"${value:,.0f}",
            (timeline[-1], value),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            va="center",
            fontweight="bold",
        )
    axis.set_xlim(timeline[0], timeline[-1] + np.timedelta64(70, "D"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.set_xlabel("Time — month-end investment value")
    axis.set_ylabel("Investment value ($)")
    axis.set_title(
        "Growth of $10,000: Sharpe leader versus Schwab-style allocations\n"
        "June 2024–May 2026 returns | Monthly rebalancing | Dividends reinvested | Gross of trading costs"
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", fontsize=9)
    figure.supxlabel(
        "Published classic target weights carried forward; no dated weight revision verified. "
        "Cash earns the shared RF proxy. Not actual Schwab account performance.",
        fontsize=9,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=170)
    path = OUTPUT / "growth_10000.png"
    if path.exists() and path.read_bytes() != buffer.getvalue():
        raise ValueError("Conflicting immutable chart")
    if not path.exists():
        path.write_bytes(buffer.getvalue())
    plt.close(figure)


def _execute() -> None:
    if (OUTPUT / "COMPLETED.json").exists():
        _verify_completed()
        return
    prior_path = FRAMEWORK / "uncertainty_rank_v1/outputs/COMPLETED.json"
    prior = _read(prior_path)
    if prior["passed"] is not True:
        raise ValueError("Prior evidence incomplete")
    inputs = {**prior["input_sha256"], **prior["artifact_sha256"], str(prior_path): base._hash(prior_path)}
    manifest_path = FRAMEWORK / "CURRENT_BASELINES.json"
    for path in [
        manifest_path,
        HERE / "weights_schedule.json",
        HERE / "README.md",
        *HERE.glob("*.py"),
        *HERE.glob("sources/*"),
    ]:
        if path.is_file():
            inputs[str(path)] = base._hash(path)
    base._verify(inputs)
    manifest, schedule, context = _read(manifest_path), _read(HERE / "weights_schedule.json"), base._context()
    selected = next(b for b in manifest["baselines"] if b["id"] == "sharpe_leader")
    npz_path = FRAMEWORK / selected["evidence"]["npz"]
    if base._hash(npz_path) != selected["evidence"]["npz_sha256"]:
        raise ValueError("Selected baseline evidence changed")
    with np.load(npz_path, allow_pickle=False) as source:
        gp_weights = source["weights"].copy()
    arrays = {key: context[key] for key in ["actual", "risk_free"]}
    arrays.update(dates=np.array(context["dates"]), assets=np.array(context["assets"]), GP_weights=gp_weights)
    outcomes, paths = [], []
    for offset, seed in enumerate(manifest["evaluation_reference"]["seeds"]):
        weights = np.column_stack([gp_weights[offset], np.zeros(24)])
        total = portfolio_returns(context["actual"], context["risk_free"], weights)
        wealth = wealth_path(total)
        _audit_wealth(context["actual"], context["risk_free"], weights, wealth)
        performance = portfolio_performance(total, context["risk_free"])
        outcomes.append({"seed": seed, **performance, "ending_value": float(wealth[-1])})
        paths.append(wealth)
        arrays.setdefault("GP_total_returns", np.empty((3, 24)))[offset] = total
    arrays["GP_wealth"] = np.array(paths)
    keys = [
        "conventional_sharpe",
        "compounded_total_return",
        "annualized_excess_vol",
        "max_drawdown_total_nav",
        "ending_value",
    ]
    gp_summary = {key: float(np.mean([r[key] for r in outcomes])) for key in keys}
    for key, value in selected["recorded_three_seed_mean_metrics"].items():
        if key != "decision_IC":
            np.testing.assert_allclose(gp_summary[key], value, rtol=0, atol=1e-12)
    results = {"Sharpe_leader": {"summary": gp_summary, "seeds": outcomes}, "benchmarks": {}}
    for name, profile in schedule["profiles"].items():
        selected_schedule = scheduled_weights(
            context["dates"], profile["initial_weights"], profile["verified_revisions"]
        )
        weights = np.zeros((24, 16))
        for index, proxy in enumerate(schedule["proxy_order"]):
            column = 15 if proxy == "CASH" else context["assets"].index(proxy)
            weights[:, column] = selected_schedule["weights"][:, index]
        total = portfolio_returns(context["actual"], context["risk_free"], weights)
        wealth = wealth_path(total)
        _audit_wealth(context["actual"], context["risk_free"], weights, wealth)
        performance = portfolio_performance(total, context["risk_free"])
        benchmark = {
            **performance,
            "ending_value": float(wealth[-1]),
            "applied_revision_count": selected_schedule["applied_revision_count"],
            "monthly_target_weights": selected_schedule["weights"].tolist(),
            "active_revision": selected_schedule["active_revision"],
            "GP_ending_value_advantage": gp_summary["ending_value"] - float(wealth[-1]),
        }
        benchmark["paired_exploratory"] = paired_block_bootstrap(
            (arrays["GP_total_returns"] - context["risk_free"]).T,
            np.broadcast_to((total - context["risk_free"])[:, None], (24, 3)),
        )
        results["benchmarks"][name] = benchmark
        arrays.update({f"{name}_weights": weights, f"{name}_total_returns": total, f"{name}_wealth": wealth})
    _write(
        OUTPUT / "contract.json",
        {
            "schema": "bayesfolio.schwab_benchmark.v1",
            "input_sha256": inputs,
            "initial_investment_dollars": 10000,
            "new_cash_flows": 0,
            "rebalance": "monthly",
            "cash_return": "same_frozen_risk_free_return_proxy; not_Schwab_actual_sweep_rate",
            "weight_schedule": schedule,
            "seeds": manifest["evaluation_reference"]["seeds"],
            "GP_curve": "mean_of_individual_seed_wealth_paths; not_new_ensemble_strategy",
            "Schwab_constraints": "published_target_weights_only; no_GP_effective_count_or_variance_budget_added",
            "data_refreshed": False,
            "GP_refits": 0,
            "new_optimizer_runs": 0,
            "costs": "gross_of_explicit_trading_costs_and_taxes",
            "future_weights_backfilled": False,
            "history_limitation": (
                "No verified target-weight revision found; not an exhaustive archived monthly record."
            ),
        },
    )
    base._snapshot(OUTPUT / "paths.npz", arrays)
    _write(OUTPUT / "results.json", results)
    _write(
        OUTPUT / "audit.json",
        {
            "passed": True,
            "months": 24,
            "GP_seed_paths": 3,
            "benchmark_paths": 2,
            "original_Sharpe_leader_performance_reproduced": True,
            "dollar_holdings_recursion_verified": True,
            "cash_RF_earned": True,
            "all_months_included": True,
            "baseline_and_prior_evidence_unchanged": True,
            "verified_target_revisions_applied": 0,
        },
    )
    _plot(arrays, results)
    base._verify(inputs)
    _write(
        OUTPUT / "EVALUATED.json",
        {
            "passed": True,
            "input_sha256": inputs,
            "artifact_sha256": {
                str(path): base._hash(path)
                for path in OUTPUT.iterdir()
                if path.is_file() and path.name not in {"EVALUATED.json", "COMPLETED.json"}
            },
        },
    )
    print(
        json.dumps(
            {
                "Sharpe_leader": gp_summary,
                "benchmarks": {
                    name: {
                        key: value for key, value in row.items() if key in keys or key == "GP_ending_value_advantage"
                    }
                    for name, row in results["benchmarks"].items()
                },
            }
        ),
        flush=True,
    )


def _seal() -> None:
    evidence = _read(OUTPUT / "EVALUATED.json")
    base._verify(evidence["input_sha256"])
    base._verify(evidence["artifact_sha256"])
    paths = [p for p in OUTPUT.iterdir() if p.is_file() and p.name != "COMPLETED.json"]
    paths.append(HERE / "FINDINGS.md")
    _write(
        OUTPUT / "COMPLETED.json",
        {
            "passed": True,
            "input_sha256": evidence["input_sha256"],
            "artifact_sha256": {str(path): base._hash(path) for path in paths},
        },
    )


def _verify_completed() -> None:
    completed = _read(OUTPUT / "COMPLETED.json")
    base._verify(completed["input_sha256"])
    base._verify(completed["artifact_sha256"])
    print("Verified Schwab-style comparison, unchanged Sharpe leader and all sealed prior evidence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true")
    group.add_argument("--seal", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.execute:
        _execute()
    elif args.seal:
        _seal()
    else:
        _verify_completed()
