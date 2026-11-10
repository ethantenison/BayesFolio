"""Complete the no-CVaR GP-scenario comparison using a sign-safe Sharpe solver.

Isolated research I/O orchestration. Reuse the identical frozen current GP means,
covariance, scenarios, ETF universe, dates, training history and two benchmarks.
Only the scenario allocator changes; no fitting or downloads. Returns and weights
are decimals; wealth is dollars. Writes only this new versioned run's outputs.

Usage from repository root:
    uv run --locked --no-sync python <this_file> --execute
    uv run --locked --no-sync python <this_file> --verify
Optional --workers defaults to min(15, CPU count minus two). Monthly allocation
jobs checkpoint independently; all portfolio metrics follow after those jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-scenario-riskfolio-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-scenario-riskfolio-cache")
HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parent
OUTPUT = HERE / "outputs"
PRIOR = FRAMEWORK / "scenario_riskfolio_v1/outputs"
sys.path.extend([str(FRAMEWORK), str(FRAMEWORK / "scenario_riskfolio_v1"), str(FRAMEWORK / "schwab_benchmark_v1")])

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from benchmark_math import portfolio_returns, wealth_path  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from metrics import portfolio_performance  # noqa: E402
from scenario_allocator import audit_weights, sample_joint  # noqa: E402
from signed_sharpe import allocate_scenarios  # noqa: E402

from artifacts import Reader  # noqa: E402

LABELS = {
    "gp_hard_rank": "Current GP rank + hard-rank allocation",
    "gp_scenarios": "GP scenarios + max Sharpe (Riskfolio + sign-safe fix)",
    "schwab_aggressive": "Schwab aggressive ETF proxy",
    "momentum_top7": "Simple momentum: top 7, equal weight",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_hashes(values: dict) -> None:
    for name, expected in values.items():
        if _hash(Path(name)) != expected:
            raise ValueError(f"Protected evidence changed: {name}")


def _write(path: Path, value: dict | bytes) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Only the new sign-safe experiment outputs are writable")
    if isinstance(value, dict):
        value = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"Refusing to alter existing evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _snapshot(path: Path, arrays: dict) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _write(path, buffer.getvalue())


def _prepare() -> dict:
    reader = Reader()
    old = reader.read(PRIOR / "contract.json")
    _verify_hashes(old["input_sha256"])
    panel = reader.read(PRIOR / "input_panel.npz")
    baseline = reader.read(FRAMEWORK / "CURRENT_BASELINES.json")
    reference = next(row for row in baseline["baselines"] if row["id"] == "sharpe_leader")
    current_path = FRAMEWORK / reference["evidence"]["npz"]
    current = reader.read(current_path)
    if reader.hashes[str(current_path.resolve())] != reference["evidence"]["npz_sha256"]:
        raise ValueError("Current GP source identity changed")
    benchmark_root = FRAMEWORK / "benchmark_suite_v1/outputs/current_sharpe_leader"
    benchmarks = reader.read(benchmark_root / "paths.npz")
    for key in ["dates", "assets", "actual", "risk_free"]:
        np.testing.assert_array_equal(panel[key], benchmarks[key])
    np.testing.assert_array_equal(panel["mean"], current["mean"])
    np.testing.assert_array_equal(panel["covariance"], current["covariance"])
    np.testing.assert_array_equal(panel["gp_hard_rank_weights"], current["weights"])
    if panel["assets"].tolist() != baseline["assets_in_order"] or panel["mean"].shape != (3, 24, 15):
        raise ValueError("The same 15 ETFs, 24 months and three GP seeds are required")
    for name in ["schwab_aggressive", "momentum_top7"]:
        for suffix in ["weights", "total_returns", "wealth"]:
            np.testing.assert_array_equal(panel[f"{name}_{suffix}"], benchmarks[f"{name}_{suffix}"])
    policy = reader.read(FRAMEWORK / "scenario_riskfolio_mv_v2/policy.json")
    if policy["CVaR"] or policy["risk_measure"] != "MV" or policy["objective"] != "Sharpe":
        raise ValueError("Preserve the user-approved no-CVaR maximum-Sharpe objective")
    for path in sorted(HERE.glob("*.py")):
        reader.read(path)
    scenario_paths = {}
    for seed in panel["seeds"]:
        for origin in panel["dates"]:
            name = f"s{seed}_{origin}"
            original = PRIOR / "cases" / f"{name}.npz"
            source = original if original.exists() else PRIOR / "failures" / f"{name}.npz"
            scenarios = reader.read(source)["scenarios"]
            if scenarios.shape != (5000, 15):
                raise ValueError("Original complete 5000-scenario sample required")
            scenario_paths[name] = str(source.resolve())
    reader.verify()
    protected = {**old["input_sha256"], **reader.hashes}
    contract = {
        "schema": "bayesfolio.sign_safe_scenario_sharpe.v3",
        "input_sha256": protected,
        "policy": policy,
        "source_GP": baseline,
        "scenario_paths": scenario_paths,
        "labels": LABELS,
        "fix": "Determine maximum feasible expected return ex ante. Use Riskfolio MV when positive; "
        "otherwise maximize the original signed mean/standard-deviation ratio directly.",
        "objective_changed_between_branches": False,
        "minimum_volatility_fallback": False,
        "GP_refits": 0,
        "new_scenario_draws": False,
        "constraint_relaxation": False,
        "negative_solver": "SLSQP with analytic gradients, fixed directional/eigenvector probes and 64 seeded starts",
        "limitations": [
            "Nonpositive-mean Sharpe optimization is nonconvex; multistart agreement is not a global certificate.",
            "Negative Sharpe can improve by increasing volatility; the objective is not silently replaced.",
            "Scenario arm uses effective count >=10 and cap 25%; hard rank uses >=7 and no ETF cap.",
            "Same reused exploratory 24-month window; three GP seeds are not independent market histories.",
            "Gross of explicit trading costs and taxes; Schwab is a simulated ETF proxy, not a managed account.",
        ],
    }
    _write(OUTPUT / "contract.json", contract)
    _snapshot(OUTPUT / "input_panel.npz", panel)
    return contract


def _job(task: tuple[int, int]) -> dict:
    seed_index, month_index = task
    contract = json.loads((OUTPUT / "contract.json").read_text())
    with np.load(OUTPUT / "input_panel.npz", allow_pickle=False) as panel:
        seed, origin = int(panel["seeds"][seed_index]), str(panel["dates"][month_index])
        mean, covariance = panel["mean"][seed_index, month_index], panel["covariance"][seed_index, month_index]
        assets = panel["assets"].tolist()
    name = f"s{seed}_{origin}"
    destination = OUTPUT / "cases" / f"{name}.json"
    if destination.exists():
        return json.loads(destination.read_text())
    source = Path(contract["scenario_paths"][name])
    if _hash(source) != contract["input_sha256"][str(source)]:
        raise ValueError("Saved scenarios changed")
    with np.load(source, allow_pickle=False) as arrays:
        scenarios = arrays["scenarios"].copy()
    np.testing.assert_array_equal(scenarios, sample_joint(mean, covariance, seed=seed, month=origin))
    started = time.monotonic()
    result = allocate_scenarios(scenarios, assets, seed=20261024 + seed + int(origin.replace("-", "")))
    record = {
        **result,
        "seed_index": seed_index,
        "month_index": month_index,
        "seed": seed,
        "origin": origin,
        "status": "allocated",
        "wall_seconds": time.monotonic() - started,
        "scenario_sha256": _hash(source),
        "scenario_path": str(source),
    }
    record["weights"], record["raw_weights"] = result["weights"].tolist(), result["raw_weights"].tolist()
    _write(destination, record)
    return record


def _plot(panel: dict) -> bytes:
    origins = panel["dates"].astype("datetime64[D]")
    endings = (origins.astype("datetime64[M]") + 2).astype("datetime64[D]") - 1
    timeline = np.r_[origins[:1], endings]
    figure, axis = plt.subplots(figsize=(14, 8.5), layout="constrained")
    for (name, label), color in zip(LABELS.items(), ["#2455a4", "#ba4536", "#b17c26", "#368275"], strict=True):
        paths = panel[f"{name}_wealth"]
        if paths.ndim == 2:
            axis.fill_between(timeline, paths.min(axis=0), paths.max(axis=0), color=color, alpha=0.1)
            paths = paths.mean(axis=0)
        axis.plot(timeline, paths, color=color, linewidth=2.7, label=f"{label} | ${paths[-1]:,.0f}")
    axis.set_title(
        "Same current GP forecasts: complete portfolio-allocation comparison\n"
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
        "Scenarios: 5,000 frozen joint draws/month; max signed Sharpe; effective assets ≥10; ETF cap 25%.\n"
        "No CVaR or minimum-volatility fallback. Hard rank retains effective assets ≥7 and its variance ceiling.\n"
        "Gross of explicit costs/taxes. Schwab is an ETF proxy simulation; momentum holds its top seven equally.",
        fontsize=9,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    return buffer.getvalue()


def _score(contract: dict) -> None:
    with np.load(OUTPUT / "input_panel.npz", allow_pickle=False) as source:
        panel = {key: source[key].copy() for key in source.files}
    jobs = [
        [json.loads((OUTPUT / "cases" / f"s{seed}_{origin}.json").read_text()) for origin in panel["dates"]]
        for seed in panel["seeds"]
    ]
    for seed_index, rows in enumerate(jobs):
        for month_index, row in enumerate(rows):
            if row["status"] != "allocated" or row["seed_index"] != seed_index or row["month_index"] != month_index:
                raise ValueError("Incomplete or misaligned allocation")
            audit_weights(np.asarray(row["weights"]))
            audit_weights(np.asarray(row["raw_weights"]))
    panel["gp_scenarios_weights"] = np.array([[row["weights"] for row in rows] for rows in jobs])
    panel["gp_scenarios_total_returns"] = np.array(
        [
            portfolio_returns(panel["actual"], panel["risk_free"], np.column_stack([weights, np.zeros(24)]))
            for weights in panel["gp_scenarios_weights"]
        ]
    )
    panel["gp_scenarios_wealth"] = wealth_path(panel["gp_scenarios_total_returns"])
    control_returns = np.array(
        [
            portfolio_returns(panel["actual"], panel["risk_free"], np.column_stack([weights, np.zeros(24)]))
            for weights in panel["gp_hard_rank_weights"]
        ]
    )
    np.testing.assert_allclose(control_returns, panel["gp_hard_rank_total_returns"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(wealth_path(control_returns), panel["gp_hard_rank_wealth"], rtol=0, atol=1e-9)
    results, table = {"strategies": {}, "limitations": contract["limitations"]}, []
    keys = [
        "ending_value",
        "compounded_total_return",
        "conventional_sharpe",
        "annualized_excess_vol",
        "max_drawdown_total_nav",
        "mean_effective_assets",
        "mean_equity_fraction",
    ]
    for name, label in LABELS.items():
        portfolios, totals = panel[f"{name}_weights"], panel[f"{name}_total_returns"]
        if totals.ndim == 1:
            portfolios, totals = portfolios[None], totals[None]
        runs = []
        for index, (weights, total) in enumerate(zip(portfolios, totals, strict=True)):
            runs.append(
                {
                    "seed": int(panel["seeds"][index]) if name.startswith("gp_") else None,
                    "ending_value": float(wealth_path(total)[-1]),
                    **portfolio_performance(total, panel["risk_free"]),
                    "mean_effective_assets": float(np.mean(1 / np.sum(weights**2, axis=1))),
                    "mean_equity_fraction": float(weights[:, :8].sum(axis=1).mean()),
                }
            )
        summary = {key: float(np.mean([row[key] for row in runs])) for key in keys}
        results["strategies"][name] = {"label": label, "summary": summary, "runs": runs}
        table.append({"strategy": name, **summary})
    _verify_hashes(contract["input_sha256"])
    _snapshot(OUTPUT / "paths.npz", panel)
    _write(OUTPUT / "results.json", results)
    _write(OUTPUT / "summary.csv", pd.DataFrame(table).to_csv(index=False).encode())
    _write(OUTPUT / "growth_comparison.png", _plot(panel))
    flattened = [row for rows in jobs for row in rows]
    signed = [row for row in flattened if row["solver_route"] == "direct_signed_sharpe_multistart"]
    audit = {
        "passed": True,
        "complete_month_seed_allocations": len(flattened),
        "GP_refits": 0,
        "CVaR": False,
        "constraint_relaxations": 0,
        "minimum_volatility_fallbacks": 0,
        "previous_weights_reused": 0,
        "dates_dropped": 0,
        "unchanged_current_GP_control": True,
        "unchanged_standing_benchmarks": True,
        "riskfolio_positive_solutions": len(flattened) - len(signed),
        "signed_negative_solutions": len(signed),
        "signed_case_records": [
            {
                key: row[key]
                for key in [
                    "seed",
                    "origin",
                    "maximum_feasible_mean",
                    "monthly_forecast_sharpe",
                    "best_objective_reproductions",
                    "successful_starts",
                ]
            }
            for row in signed
        ],
        "signed_branch_global_optimality_certified": False,
    }
    _write(OUTPUT / "audit.json", audit)


def _seal(contract: dict) -> None:
    _verify_hashes(contract["input_sha256"])
    outputs = [path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "COMPLETED.json"]
    _write(
        OUTPUT / "COMPLETED.json",
        {
            "passed": True,
            "input_sha256": contract["input_sha256"],
            "artifact_sha256": {str(path.resolve()): _hash(path) for path in outputs},
        },
    )


def _verify() -> None:
    seal = json.loads((OUTPUT / "COMPLETED.json").read_text())
    _verify_hashes(seal["input_sha256"])
    _verify_hashes(seal["artifact_sha256"])
    audit = json.loads((OUTPUT / "audit.json").read_text())
    if not seal["passed"] or not audit["passed"] or audit["complete_month_seed_allocations"] != 72:
        raise ValueError("Complete 72-allocation evidence required")
    with np.load(OUTPUT / "paths.npz", allow_pickle=False) as panel:
        totals = np.array(
            [
                portfolio_returns(panel["actual"], panel["risk_free"], np.column_stack([weights, np.zeros(24)]))
                for weights in panel["gp_scenarios_weights"]
            ]
        )
        np.testing.assert_allclose(totals, panel["gp_scenarios_total_returns"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(wealth_path(totals), panel["gp_scenarios_wealth"], rtol=0, atol=1e-9)
    print("Verified all 72 allocations, full return paths, unchanged GP/benchmark inputs and sealed evidence")


def _execute(workers: int) -> None:
    if (OUTPUT / "COMPLETED.json").exists():
        _verify()
        return
    started = time.monotonic()
    contract = _prepare()
    print(f"Running 72 no-CVaR maximum-Sharpe solves with {workers} workers; no GP fits", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, (seed, month)) for seed in range(3) for month in range(24)]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if index % 12 == 0 or index == 1:
                print(
                    f"{index}/72 accepted; {result['solver_route']}; elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    _score(contract)
    _seal(contract)
    _verify()
    print(f"Completed in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify", action="store_true")
    limit = min(15, max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--workers", type=int, default=limit)
    options = parser.parse_args()
    if not 1 <= options.workers <= limit:
        parser.error(f"Workers must be between one and {limit}")
    if options.execute:
        _execute(options.workers)
    else:
        _verify()
