"""Compare saved current GP hard ranks with the July scenario/Riskfolio policy.

Isolated research orchestration: no GP fits, data downloads, new model selection,
production edits or sealed-source changes. Reuses the Sharpe leader's frozen mean
and covariance for all 15 ETFs, plus the two approved frozen benchmarks. Outputs
only to this versioned folder. Returns/weights are decimals and wealth is dollars.

Usage from repository root:
    uv run --locked --no-sync python <this_file> --execute
    uv run --locked --no-sync python <this_file> --verify
Optional --workers defaults to min(15, CPU count minus two). Independent monthly
solves checkpoint; portfolio statistics are calculated together after all solves.
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
BENCHMARK = FRAMEWORK / "benchmark_suite_v1/outputs/current_sharpe_leader"
sys.path.extend([str(FRAMEWORK), str(FRAMEWORK / "schwab_benchmark_v1")])

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import riskfolio  # noqa: E402
from benchmark_math import portfolio_returns, wealth_path  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from metrics import portfolio_performance  # noqa: E402
from scenario_allocator import allocate_scenarios, audit_weights, sample_joint  # noqa: E402

from artifacts import Reader  # noqa: E402

LABELS = {
    "gp_hard_rank": "Current GP rank + hard-rank allocation",
    "gp_scenarios_riskfolio": "Current GP scenarios + old Riskfolio configuration",
    "schwab_aggressive": "Schwab aggressive ETF proxy",
    "momentum_top7": "Simple momentum: top 7, equal weight",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_hashes(hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        if _hash(Path(name)) != expected:
            raise ValueError(f"Protected evidence changed: {name}")


def _write(path: Path, payload: bytes | dict) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Only this experiment's new outputs may be written")
    if isinstance(payload, dict):
        payload = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite existing experiment evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _snapshot(path: Path, arrays: dict) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _write(path, buffer.getvalue())


def _prepare() -> dict:
    reader = Reader()
    seal = reader.read(BENCHMARK / "COMPLETED.json")
    if not seal["passed"]:
        raise ValueError("Frozen benchmark suite is incomplete")
    protected = {**seal["input_sha256"], **seal["artifact_sha256"]}
    _verify_hashes(protected)
    baseline = reader.read(FRAMEWORK / "CURRENT_BASELINES.json")
    benchmarks = reader.read(FRAMEWORK / "CURRENT_BENCHMARKS.json")
    reference = next(row for row in baseline["baselines"] if row["id"] == "sharpe_leader")
    current_path = FRAMEWORK / reference["evidence"]["npz"]
    current = reader.read(current_path)
    if reader.hashes[str(current_path.resolve())] != reference["evidence"]["npz_sha256"]:
        raise ValueError("Current GP control identity differs")
    frozen = reader.read(FRAMEWORK / "uncertainty_rank_v1/outputs/input_panel.npz")
    previous = reader.read(BENCHMARK / "paths.npz")
    old_path = Path("notebooks/outputs/20260702_july_portfolio_full3")
    old_manifest = reader.read(old_path / "manifest.json")
    reader.read(old_path / "JULY_PORTFOLIO_READOUT.md")
    reader.read(Path("experiments/2026-06-portfolio-optimization/run_monthly_optimization_walkforward.py"))
    for path in sorted(HERE.glob("*.py")):
        reader.read(path)
    if benchmarks["active_benchmark_ids"] != ["schwab_aggressive", "momentum_top7"]:
        raise ValueError("Approved benchmark scope changed")
    if riskfolio.__version__ != "7.2.1":
        raise ValueError("Raw solver audit requires locked Riskfolio 7.2.1")
    for key in ["dates", "assets", "actual", "risk_free"]:
        np.testing.assert_array_equal(frozen[key], previous[key])
    if frozen["assets"].tolist() != baseline["assets_in_order"]:
        raise ValueError("The current 15-ETF universe must remain unchanged")
    np.testing.assert_array_equal(current["mean"], frozen["mean_sharpe"])
    np.testing.assert_array_equal(
        current["covariance"], (frozen["broad_covariance"] + frozen["compact_covariance"]) / 2
    )
    np.testing.assert_array_equal(current["weights"], previous["comparison_weights"])
    if current["mean"].shape != (3, 24, 15):
        raise ValueError("All 24 forecast origins and three current GP seeds required")
    recommendation = old_manifest["riskfolio_recommendation"]
    if (
        recommendation["objective"],
        recommendation["risk_measure"],
        recommendation["recommended_nea"],
        recommendation["recommended_upperlng"],
    ) != ("Sharpe", "CVaR", 10, 0.25):
        raise ValueError("Historical optimizer recommendation changed")
    arrays = {key: frozen[key] for key in ["dates", "assets", "actual", "risk_free", "seeds"]}
    arrays.update(mean=current["mean"], covariance=current["covariance"], gp_hard_rank_weights=current["weights"])
    for name in ["schwab_aggressive", "momentum_top7"]:
        for suffix in ["weights", "total_returns", "wealth"]:
            arrays[f"{name}_{suffix}"] = previous[f"{name}_{suffix}"]
    arrays["gp_hard_rank_total_returns"] = previous["comparison_total_returns"]
    arrays["gp_hard_rank_wealth"] = previous["comparison_wealth"]
    reader.verify()
    for name, digest in reader.hashes.items():
        if name in protected and protected[name] != digest:
            raise ValueError(f"Source identity changed: {name}")
    protected.update(reader.hashes)
    contract = {
        "schema": "bayesfolio.current_gp_old_riskfolio_comparison.v1",
        "input_sha256": protected,
        "baseline": baseline,
        "labels": LABELS,
        "policy": {
            "posterior_scenarios": 5000,
            "distribution": "joint Gaussian from current frozen blended GP moments",
            "covariance": "same 50/50 full observed predictive covariance as current allocator, including noise",
            "model": "Classic",
            "risk_measure": "CVaR",
            "objective": "Sharpe",
            "alpha": 0.5,
            "method_mu": "hist",
            "method_cov": "hist",
            "hist": True,
            "rf": 0.0,
            "kelly": None,
            "minimum_effective_assets": 10,
            "ETF_position_cap": 0.25,
            "long_only": True,
            "budget": 1.0,
            "extra_variance_ceiling": None,
            "turnover_penalty": None,
            "fallback_weights": False,
            "constraint_relaxation": False,
            "GP_refits": 0,
            "new_data_downloads": 0,
            "scenario_rng": "numpy PCG64 SeedSequence([20261024, GP_seed, origin_YYYYMMDD])",
            "initial_dollars": 10000,
            "return_accounting": "same frozen discounted-excess to total conversion",
        },
        "limitations": [
            "Exploratory reused 24-month market window; three GP seeds are not independent histories.",
            "Old allocator includes different diversification/cap settings; not an objective-only ablation.",
            "5000 finite Monte Carlo draws introduce sampling variation; they are not a new GP fit.",
            "Frozen blended mean/covariance define the scenario Gaussian; not a mixture of component forecasts.",
            "Riskfolio Sharpe/CVaR is not the conventional realized Sharpe reported in evaluation.",
            "Gross returns; no explicit trading costs or taxes; Schwab is a proxy simulation.",
        ],
    }
    _write(OUTPUT / "contract.json", contract)
    _snapshot(OUTPUT / "input_panel.npz", arrays)
    return contract


def _job(task: tuple[int, int]) -> dict:
    seed_index, month_index = task
    with np.load(OUTPUT / "input_panel.npz", allow_pickle=False) as panel:
        mean, covariance = panel["mean"][seed_index, month_index], panel["covariance"][seed_index, month_index]
        seed, origin = int(panel["seeds"][seed_index]), str(panel["dates"][month_index])
        assets = panel["assets"].tolist()
    stem = OUTPUT / "cases" / f"s{seed}_{origin}"
    if stem.with_suffix(".json").exists():
        record = json.loads(stem.with_suffix(".json").read_text())
        if record["scenario_artifact_sha256"] != _hash(stem.with_suffix(".npz")):
            raise ValueError("Changed cached scenario/weight evidence")
        return record
    started = time.monotonic()
    scenarios = sample_joint(mean, covariance, seed=seed, month=origin)
    result = allocate_scenarios(scenarios, assets)
    sample_mean, sample_covariance = scenarios.mean(axis=0), np.cov(scenarios, rowvar=False)
    _snapshot(
        stem.with_suffix(".npz"),
        {
            "scenarios": scenarios,
            "weights": result["weights"],
            "raw_weights": result["raw_weights"],
            "sample_mean": sample_mean,
            "sample_covariance": sample_covariance,
        },
    )
    record = {
        "seed_index": seed_index,
        "month_index": month_index,
        "seed": seed,
        "origin": origin,
        "status": "allocated",
        "scenario_count": len(scenarios),
        "attempts": result["attempts"],
        "audit": result["audit"],
        "wall_seconds": time.monotonic() - started,
        "scenario_artifact_sha256": _hash(stem.with_suffix(".npz")),
        "sample_mean_max_abs_difference": float(np.max(np.abs(sample_mean - mean))),
        "sample_covariance_relative_frobenius_error": float(
            np.linalg.norm(sample_covariance - covariance) / np.linalg.norm(covariance)
        ),
    }
    _write(stem.with_suffix(".json"), record)
    return record


def _plot(arrays: dict) -> bytes:
    origins = arrays["dates"].astype("datetime64[D]")
    ends = (origins.astype("datetime64[M]") + 2).astype("datetime64[D]") - 1
    timeline = np.r_[origins[:1], ends]
    figure, axis = plt.subplots(figsize=(14, 8.5), layout="constrained")
    colors = ["#2455a4", "#ba4536", "#b17c26", "#368275"]
    for (name, label), color in zip(LABELS.items(), colors, strict=True):
        paths = arrays[f"{name}_wealth"]
        if paths.ndim == 2:
            axis.fill_between(timeline, paths.min(axis=0), paths.max(axis=0), color=color, alpha=0.10)
            paths = paths.mean(axis=0)
        axis.plot(timeline, paths, color=color, linewidth=2.7, label=f"{label}  |  ${paths[-1]:,.0f}")
    axis.set_title(
        "Same current GP forecasts, different portfolio allocation\n"
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
        "GP curves: mean of 3 saved model seeds; shading is seed range, not a confidence interval. No GP refits.\n"
        "Riskfolio: 5,000 joint scenarios/month; CVaR alpha 0.5; effective assets ≥10; ETF cap 25%.\n"
        "Hard rank: effective assets ≥7, no ETF cap, existing variance ceiling. Gross of explicit costs/taxes.\n"
        "Schwab is an ETF proxy simulation; momentum equally weights its seven highest-ranked ETFs.",
        fontsize=9,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    return buffer.getvalue()


def _score(contract: dict) -> None:
    with np.load(OUTPUT / "input_panel.npz", allow_pickle=False) as source:
        panel = {key: source[key].copy() for key in source.files}
    weights, jobs = [], []
    for seed_index, seed in enumerate(panel["seeds"]):
        rows = []
        for month_index, origin in enumerate(panel["dates"]):
            stem = OUTPUT / "cases" / f"s{seed}_{origin}"
            record = json.loads(stem.with_suffix(".json").read_text())
            if record["status"] != "allocated" or record["scenario_artifact_sha256"] != _hash(stem.with_suffix(".npz")):
                raise ValueError("Missing or changed monthly optimization evidence")
            if record["seed_index"] != seed_index or record["month_index"] != month_index:
                raise ValueError("Cached solver axes differ")
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as case:
                audit_weights(case["weights"])
                audit_weights(case["raw_weights"])
                rows.append(case["weights"].copy())
            jobs.append(record)
        weights.append(rows)
    panel["gp_scenarios_riskfolio_weights"] = np.asarray(weights)
    for name in ["gp_hard_rank", "gp_scenarios_riskfolio"]:
        totals = np.array(
            [
                portfolio_returns(panel["actual"], panel["risk_free"], np.column_stack([weight, np.zeros(24)]))
                for weight in panel[f"{name}_weights"]
            ]
        )
        wealth = wealth_path(totals)
        if name == "gp_hard_rank":
            np.testing.assert_allclose(totals, panel[f"{name}_total_returns"], rtol=0, atol=1e-14)
            np.testing.assert_allclose(wealth, panel[f"{name}_wealth"], rtol=0, atol=1e-9)
        panel[f"{name}_total_returns"], panel[f"{name}_wealth"] = totals, wealth
    results: dict = {"strategies": {}, "limitations": contract["limitations"]}
    summary_columns = [
        "ending_value",
        "compounded_total_return",
        "conventional_sharpe",
        "annualized_excess_vol",
        "max_drawdown_total_nav",
        "mean_effective_assets",
        "mean_equity_fraction",
        "mean_max_weight",
    ]
    equity = np.isin(panel["assets"], ["SPY", "MGK", "VTV", "IWM", "VEA", "VWO", "VSS", "EWX"])
    table_rows = []
    for name, label in LABELS.items():
        totals, portfolios = panel[f"{name}_total_returns"], panel[f"{name}_weights"]
        if totals.ndim == 1:
            totals, portfolios = totals[None], portfolios[None]
        runs = []
        for index, (total, portfolio) in enumerate(zip(totals, portfolios, strict=True)):
            full_equity = portfolio[:, :15][:, equity].sum(axis=1)
            runs.append(
                {
                    "seed": int(panel["seeds"][index]) if name.startswith("gp_") else None,
                    "ending_value": float(wealth_path(total)[-1]),
                    **portfolio_performance(total, panel["risk_free"]),
                    "mean_effective_assets": float(np.mean(1 / np.sum(portfolio**2, axis=1))),
                    "mean_equity_fraction": float(full_equity.mean()),
                    "mean_max_weight": float(portfolio.max(axis=1).mean()),
                }
            )
        summary = {key: float(np.mean([run[key] for run in runs])) for key in summary_columns}
        results["strategies"][name] = {"label": label, "summary": summary, "runs": runs}
        table_rows.append({"strategy": name, **summary})
    _verify_hashes(contract["input_sha256"])
    _snapshot(OUTPUT / "paths.npz", panel)
    _write(OUTPUT / "results.json", results)
    _write(OUTPUT / "summary.csv", pd.DataFrame(table_rows).to_csv(index=False).encode())
    _write(OUTPUT / "growth_comparison.png", _plot(panel))
    _write(
        OUTPUT / "audit.json",
        {
            "passed": True,
            "GP_refits": 0,
            "months_per_seed": 24,
            "GP_seeds": panel["seeds"].tolist(),
            "Riskfolio_solves": len(jobs),
            "fallback_weights": 0,
            "constraint_relaxations": 0,
            "raw_and_returned_weights_audited": True,
            "strict_optimal_months": len(jobs),
            "numerical_retries": sum(len(job["attempts"]) - 1 for job in jobs),
            "max_package_weight_alteration": max(job["attempts"][-1]["package_output_max_alteration"] for job in jobs),
            "sample_covariance_relative_error_max": max(
                job["sample_covariance_relative_frobenius_error"] for job in jobs
            ),
            "current_GP_control_reproduced": True,
            "standing_benchmark_paths_unchanged": True,
        },
    )


def _seal() -> None:
    contract = json.loads((OUTPUT / "contract.json").read_text())
    _verify_hashes(contract["input_sha256"])
    audit = json.loads((OUTPUT / "audit.json").read_text())
    if not audit["passed"] or audit["Riskfolio_solves"] != 72:
        raise ValueError("A complete 72-solve audit is required")
    paths = [path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "COMPLETED.json"]
    _write(
        OUTPUT / "COMPLETED.json",
        {
            "passed": True,
            "schema": "bayesfolio.current_gp_old_riskfolio_comparison.v1",
            "input_sha256": contract["input_sha256"],
            "artifact_sha256": {str(path.resolve()): _hash(path) for path in paths},
        },
    )


def _verify() -> None:
    seal = json.loads((OUTPUT / "COMPLETED.json").read_text())
    if not seal["passed"]:
        raise ValueError("Experiment seal failed")
    _verify_hashes(seal["input_sha256"])
    _verify_hashes(seal["artifact_sha256"])
    print("Verified 72 strict Riskfolio allocations, frozen GP/benchmark inputs, and all comparison outputs")


def _execute(workers: int) -> None:
    if (OUTPUT / "COMPLETED.json").exists():
        _verify()
        return
    started = time.monotonic()
    contract = _prepare()
    tasks = [(seed, month) for seed in range(3) for month in range(24)]
    print(f"Launching 72 independent monthly Riskfolio solves with {workers} workers; no GP fits", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            record = future.result()
            if completed % 6 == 0 or completed == 1:
                print(
                    f"{completed}/72 solved; last seed={record['seed']} origin={record['origin']}; "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    _score(contract)
    _seal()
    _verify()
    print(f"Completed comparison in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--workers", type=int, default=min(15, max(1, (os.cpu_count() or 1) - 2)))
    options = parser.parse_args()
    if options.workers < 1 or options.workers > min(15, max(1, (os.cpu_count() or 1) - 2)):
        parser.error("Workers must be between one and min(15, CPU count minus two)")
    if options.execute:
        _execute(options.workers)
    else:
        _verify()
