"""Replay numerical allocation failures without changing the sealed study.

Usage: .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/allocation_numerics_v2/run_repair.py
       Add --verify-only to validate saved repair evidence without solving.
Reads the original forecasts and scores; no GP fitting. Writes only this new
directory's outputs, retaining all seeds/months and unchanged decimal-return
metrics. Original study verification must pass before running this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[name] = "1"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
sys.path.insert(0, str(HERE.parent / "approved_scope_v1"))

import cvxpy as cp  # noqa: E402
import numpy as np  # noqa: E402
from stable_allocator import StableScoreAllocator  # noqa: E402
from study_common import FITS, ROOT, SEEDS, _context  # noqa: E402
from study_math import _fixed, _risk, _score, _scores  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: object) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Repair writes must stay in the new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError(f"Refusing to overwrite different repair evidence: {path}")
    path.write_text(payload)


def _aggregate(case: dict, seeds: list[dict]) -> dict:
    complete = all(seed["solved_months"] == 24 for seed in seeds)
    summary = {
        **case,
        "complete_seeds": sum(seed["solved_months"] == 24 for seed in seeds),
        "solved_months": sum(seed["solved_months"] for seed in seeds),
        "ETF_position_cap": None,
        "CVaR": False,
        "fallback_weights": False,
    }
    for metric in [
        "conventional_sharpe",
        "compounded_total_return",
        "annualized_excess_vol",
        "max_drawdown_total_nav",
        "mean_effective_count",
        "largest_weight",
    ]:
        numbers = [seed["performance"][metric] for seed in seeds] if complete else []
        summary[metric] = float(np.mean(numbers)) if complete else None
        summary[f"seed_sd_{metric}"] = float(np.std(numbers, ddof=1)) if complete else None
    for metric in ["error_rms_to_predicted_sd_rms", "coverage_95", "mean_nll", "mean_crps"]:
        summary[f"adaptive_{metric}"] = (
            float(np.mean([seed["adaptive_calibration"][metric] for seed in seeds])) if complete else None
        )
    return summary


def _job(original_path: Path) -> dict:
    original = json.loads(original_path.read_text())
    case, context = original["case"], _context()
    panel_path = ROOT / "panels" / f"{case['configuration']}.npz"
    source_hashes = {str(path): _hash(path) for path in [original_path, panel_path]}
    seeds, comparisons = [], []
    with np.load(panel_path, allow_pickle=False) as panel:
        for offset, seed in enumerate(SEEDS):
            mean = panel["mean"][offset]
            native = panel["covariance"][offset] if "covariance" in panel else None
            matrix, factors = _risk(
                case["risk_source"], context["actual"], mean, native, context["historical_covariance"]
            )
            scores, penalty = _scores(mean, context["baseline_mean"], case["objective"])
            allocator = StableScoreAllocator(
                matrix, context["risk_budget"] * case["budget_factor"], case["minimum"], penalty
            )
            records = allocator.allocate(scores)
            for old, new in zip(original["seeds"][offset]["months"], records, strict=True):
                comparison = {
                    "seed": seed,
                    "index": old["index"],
                    "old_status": old["status"],
                    "new_status": new["status"],
                }
                if old["weights"] is not None and new["weights"] is not None:
                    previous, current = np.asarray(old["weights"]), np.asarray(new["weights"])

                    def objective(weights: np.ndarray) -> float:
                        return float(scores[old["index"]] @ weights - penalty * (weights @ weights))

                    comparison.update(
                        max_abs_weight_change=float(np.max(np.abs(previous - current))),
                        raw_objective_change=objective(current) - objective(previous),
                    )
                comparisons.append(comparison)
            seeds.append(
                {
                    "seed": seed,
                    **_score(context, mean, matrix, records),
                    "prequential_variance_scale": factors.tolist(),
                    "fixed_reference_scores": _fixed(
                        context, case["configuration"], case["risk_source"], seed, mean, matrix
                    ),
                }
            )
    for path, expected in source_hashes.items():
        if _hash(Path(path)) != expected:
            raise ValueError("Protected forecast/evidence changed during replay")
    result = {
        "case": case,
        "summary": _aggregate(case, seeds),
        "seeds": seeds,
        "comparison_to_original": comparisons,
        "source_sha256": source_hashes,
    }
    destination = OUTPUT / "portfolio_cases" / original_path.name
    _save(destination, result)
    return {"path": str(destination), "summary": result["summary"], "comparisons": comparisons}


def _verify() -> None:
    seal = json.loads((OUTPUT / "COMPLETED.json").read_text())
    for group in ["input_sha256", "artifact_sha256"]:
        for path, expected in seal[group].items():
            if _hash(Path(path)) != expected:
                raise ValueError(f"Changed repair input/output: {path}")
    print("Verified unchanged sources, repaired portfolios and combined summary", flush=True)


def _run() -> None:
    original = {path.stem: json.loads(path.read_text()) for path in sorted((ROOT / "portfolio_cases").glob("*.json"))}
    failures = json.loads((ROOT / "failed_allocations.json").read_text())
    affected = {row["case_id"] for row in failures if row["status"] == "optimal_inaccurate"}
    selected = sorted(
        key for key, item in original.items() if key in affected or item["case"]["objective"] == "smooth_rank"
    )
    paths = [ROOT / "COMPLETED.json", ROOT / "contract.json", FITS / "context.json", FITS / "context_arrays.npz"]
    paths += list(HERE.glob("*.py"))
    paths += list((ROOT / "portfolio_cases").glob("*.json"))
    paths += list((ROOT / "panels").glob("*.npz"))
    inputs = {str(path.resolve()): _hash(path) for path in paths}
    workers = min(13, max(1, (os.cpu_count() or 4) - 2))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        repairs = list(pool.map(_job, [ROOT / "portfolio_cases" / f"{key}.json" for key in selected]))
    summaries = {key: item["summary"] for key, item in original.items()}
    summaries.update({row["summary"]["case_id"]: row["summary"] for row in repairs})
    comparisons = [item for row in repairs for item in row["comparisons"]]
    unresolved = [item for item in comparisons if item["new_status"] != "solved"]
    report = {
        "schema": "bayesfolio.allocation_numerics_v2",
        "gp_refits": 0,
        "workers": workers,
        "variance_budget_changed": False,
        "objective_changed": False,
        "constraints_relaxed": False,
        "replayed_cases": len(repairs),
        "replayed_monthly_optimizations": len(comparisons),
        "old_numerical_failures_recovered": sum(
            row["old_status"] == "optimal_inaccurate" and row["new_status"] == "solved" for row in comparisons
        ),
        "unresolved_replayed_months": unresolved,
        "all_cases": len(summaries),
        "all_complete_cases": sum(row["complete_seeds"] == 3 for row in summaries.values()),
        "all_solved_months": sum(row["solved_months"] for row in summaries.values()),
        "remaining_original_infeasible_months": sum(row["status"] == "infeasible" for row in failures),
        "maximum_previously_solved_weight_change": max(row.get("max_abs_weight_change", 0) for row in comparisons),
        "maximum_previously_solved_objective_change": max(
            abs(row.get("raw_objective_change", 0)) for row in comparisons
        ),
        "smooth_rank_complete_cases": sum(
            row["objective"] == "smooth_rank" and row["complete_seeds"] == 3 for row in summaries.values()
        ),
        "smooth_rank_total_cases": sum(row["objective"] == "smooth_rank" for row in summaries.values()),
        "original_failure_status_counts": dict(Counter(row["status"] for row in failures)),
        "solver_versions": {"cvxpy": cp.__version__, "numpy": np.__version__},
        "portfolio_summaries": list(summaries.values()),
    }
    _save(OUTPUT / "repair_summary.json", report)
    if unresolved:
        raise ValueError(f"Numerical repair still has {len(unresolved)} unresolved months; see repair_summary.json")
    for path, expected in inputs.items():
        if _hash(Path(path)) != expected:
            raise ValueError(f"Source changed during repair: {path}")
    artifacts = [OUTPUT / "repair_summary.json"] + [Path(row["path"]) for row in repairs]
    _save(
        OUTPUT / "COMPLETED.json",
        {"passed": True, "input_sha256": inputs, "artifact_sha256": {str(p): _hash(p) for p in artifacts}},
    )
    print(json.dumps({key: value for key, value in report.items() if key != "portfolio_summaries"}, indent=2))
    _verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    _verify() if arguments.verify_only else _run()
