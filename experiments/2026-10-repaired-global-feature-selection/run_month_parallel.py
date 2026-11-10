"""Replay frozen rank-five/36 ETF forecasts with a global monthly worker pool.

Experiment orchestration only: first verifies the single-window spawn/solver
parity, then cold-replays all 72 baseline fits, audits against the completed
serial study, and only then adds hy_spread_chg_1p for another 72 fits. Portfolio
solves are strict and independent; no fallback weights or relaxed constraints.
Decimal-return arithmetic and conventional Sharpe are independently audited.
Old source files, manifests, data and outputs remain untouched.

Usage:
    MPLCONFIGDIR=/private/tmp/bayesfolio-mpl .venv/bin/python \
        experiments/2026-10-repaired-global-feature-selection/run_month_parallel.py --execute --workers 14
    Matching completed job stamps support safe resume; cached jobs are excluded
    from any claim of a cold timing benchmark.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import month_parallel_worker as worker
import numpy as np
import pandas as pd
import repair_dataset as data
import run_study as study
from audit_joint_objectives import reconstruct
from followup_workflow import immutable_json

HERE = data.HERE / "month_parallel"
REFERENCE = data.HERE / "rank5_budget36"
BASE_ID = "global_e6c41f2020fd_rank5_iter36"
SEEDS = [27, 127, 227]
PARITY_TOLERANCES = {"predictions": 1e-8, "scenarios": 1e-8, "weights": 1e-6, "returns": 1e-8}


def worker_limit(requested: int, cores: int, peak_rss: int = 0) -> int:
    """Bound the global pool by CPU headroom and a conservative memory allocation.

    Args:
        requested: User-selected global worker ceiling, never a per-seed multiplier.
        cores: Physical/logical CPU count reported by this 16-core Mac Studio.
        peak_rss: Representative smoke-worker peak RSS in bytes, if available.

    Returns:
        At most cores minus two, fourteen, and 48 GiB divided by per-worker RSS.

    Raises:
        ValueError: An invalid requested ceiling would exceed the CPU budget.
    """
    cpu_limit = max(1, min(14, cores - 2))
    if not 1 <= requested <= cpu_limit:
        raise ValueError(f"Worker ceiling must be between one and {cpu_limit}")
    memory_limit = int(48 * 1024**3 // peak_rss) if peak_rss else requested
    return max(1, min(requested, memory_limit))


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(data.ROOT)): data.digest(path) for path in sorted(set(paths))}


def _fresh(contract: dict[str, Any]) -> None:
    if any(data.digest(data.ROOT / name) != digest for name, digest in contract["source_sha256"].items()):
        raise ValueError("A frozen source or baseline artifact changed")
    if data.digest(Path(contract["feature_path"])) != contract["feature_sha256"]:
        raise ValueError("Frozen feature panel changed")
    if data.digest(Path(contract["date_path"])) != contract["date_sha256"]:
        raise ValueError("Frozen date grid changed")


def _configuration(contract: dict[str, Any], phase: str) -> dict[str, Any]:
    value = copy.deepcopy(contract["variants"]["baseline" if phase == "smoke" else phase])
    value.update(feature_path=contract["feature_path"], dates=contract["dates"], tickers=contract["tickers"],
                 signature=data.digest(HERE / "contract.json"))
    return value


def _jobs(phase: str, dates: list[str]) -> list[dict[str, Any]]:
    seeds = [27] if phase == "smoke" else SEEDS
    selected = dates[:1] if phase == "smoke" else dates
    return [{"seed": seed, "date": date, "index": dates.index(date),
             "run": str(HERE / phase / "runs" / f"s{seed}")} for seed in seeds for date in selected]


def _stamp_path(phase: str, stage: str, job: dict[str, Any]) -> Path:
    return HERE / phase / "checkpoints" / f"{stage}_s{job['seed']}_{job['date']}.json"


def _stage(pool: ProcessPoolExecutor, phase: str, stage: str, jobs: list[dict[str, Any]],
           signature: str) -> tuple[list[dict[str, Any]], int, float]:
    started = time.perf_counter()
    completed, pending = [], []
    for job in jobs:
        stamp = _stamp_path(phase, stage, job)
        if stamp.exists():
            saved = json.loads(stamp.read_text())
            if saved["signature"] != signature or any(
                data.digest(data.ROOT / name) != digest for name, digest in saved["artifact_sha256"].items()
            ):
                raise ValueError(f"Stale or conflicting completed job: {stamp}")
            completed.append(saved["result"])
        else:
            pending.append(job)
    function = worker.fit_job if stage == "fit" else worker.portfolio_job
    futures = {pool.submit(function, job): job for job in pending}
    errors = []
    for future in as_completed(futures):
        job = futures[future]
        try:
            result = future.result()
            data.write_json(_stamp_path(phase, stage, job), {"signature": signature, "result": result,
                "artifact_sha256": _hashes([Path(path) for path in result["paths"].values()])})
            completed.append(result)
        except Exception as error:
            errors.append({"seed": job["seed"], "date": job["date"], "error": str(error)})
        data.write_json(HERE / "status.json", {"state": "running" if not errors else "failed",
            "phase": phase, "stage": stage, "completed_jobs": len(completed), "required_jobs": len(jobs),
            "errors": errors, "updated_at": datetime.now(UTC).isoformat()})
        print(f"{phase} {stage}: {len(completed)}/{len(jobs)} completed", flush=True)
    if errors:
        data.write_json(HERE / phase / f"{stage}_failures.json", {"errors": errors, "fallback_used": False})
        raise RuntimeError(f"{phase} {stage} failed; no replacement weights/results were supplied")
    return sorted(completed, key=lambda row: (row["seed"], row["date"])), len(pending), time.perf_counter() - started


def _parity(fits: list[dict[str, Any]], portfolios: list[dict[str, Any]]) -> dict[str, Any]:
    differences = {key: 0.0 for key in PARITY_TOLERANCES}
    for fit in fits:
        old = REFERENCE / f"runs/{BASE_ID}_s{fit['seed']}"
        new = HERE / ("smoke" if len(fits) == 1 else "baseline") / "runs" / f"s{fit['seed']}"
        for category, name, columns in [("predictions", "predictions", ["y_true", "y_pred", "y_std"]),
                                         ("scenarios", "scenarios", data.TICKERS)]:
            path = Path(fit["paths"][name])
            left, right = pd.read_csv(path), pd.read_csv(old / path.relative_to(new))
            if category == "predictions":
                np.testing.assert_array_equal(left[["date", "asset_id"]], right[["date", "asset_id"]])
            delta = float(np.abs(left[columns].to_numpy() - right[columns].to_numpy()).max())
            differences[category] = max(differences[category], delta)
    for row in portfolios:
        old = REFERENCE / f"runs/{BASE_ID}_s{row['seed']}"
        for category, filename, keys, column in [
            ("weights", "portfolio_weights.csv", ["date", "strategy", "asset_id"], "weight"),
            ("returns", "portfolio_returns.csv", ["date", "strategy"], "return")]:
            left = pd.read_csv(row["paths"][category]).sort_values(keys)
            right = pd.read_csv(old / filename)
            right = right[right.date.eq(row["date"])].sort_values(keys)
            np.testing.assert_array_equal(left[keys].to_numpy(), right[keys].to_numpy())
            differences[category] = max(differences[category], float(np.abs(
                left[column].to_numpy() - right[column].to_numpy()).max()))
    passed = all(differences[key] <= value for key, value in PARITY_TOLERANCES.items())
    return {"passed": passed, "max_absolute_differences": differences, "tolerances": PARITY_TOLERANCES,
            "windows_checked": len(fits), "strict_primary_portfolio_policy": True}


def _publish(phase: str, jobs: list[dict[str, Any]], fits: list[dict[str, Any]],
             portfolios: list[dict[str, Any]], contract: dict[str, Any]) -> list[dict[str, Any]]:
    objectives = []
    for seed in SEEDS:
        run = HERE / phase / "runs" / f"s{seed}"
        selected = [row for row in fits if row["seed"] == seed]
        predictions = pd.concat([pd.read_csv(row["paths"]["predictions"]) for row in selected], ignore_index=True)
        predictions.to_csv(run / "gp_window_predictions.csv", index=False)
        for key, filename in [("weights", "portfolio_weights.csv"), ("returns", "portfolio_returns.csv")]:
            pd.concat([pd.read_csv(row["paths"][key]) for row in portfolios if row["seed"] == seed],
                      ignore_index=True).to_csv(run / filename, index=False)
        pd.DataFrame([{"date": row["date"], "unique_train_dates": row["train_dates"],
                       "train_rows": row["train_rows"]} for row in selected]).to_csv(
            run / "window_training_history.csv", index=False)
        study.verify_training(run, data.TICKERS)
        metrics = reconstruct(run, data.TICKERS, contract["dates"])
        objectives.append({"phase": phase, "seed": seed, **metrics})
        data.write_json(run / "manifest.json", {"rank": 5, "maxiter": 36, "seed": seed,
            "input_columns": contract["variants"][phase]["inputs"], "dates": contract["dates"],
            "tickers": data.TICKERS, "source_signature": data.digest(HERE / "contract.json"),
            "previous_weights_fallback": False, "constraint_relaxations": False,
            "new_execution_scope": "month-parallel research; no production fallback policy was changed"})
    keys = ["phase", "seed", "mean_ic", "mean_gaussian_nll", "sharpe",
            "cumulative_return_diagnostic_only", "rmse_diagnostic_only"]
    pd.DataFrame(objectives)[keys].to_csv(HERE / phase / "objectives_by_seed.csv", index=False)
    data.write_json(HERE / phase / "independent_audit.json", {"passed": True, "seed_runs": 3,
        "checked": objectives, "fit_windows": len(fits), "portfolio_windows": len(portfolios),
        "source_signature": data.digest(HERE / "contract.json"), "no_fallback": True})
    return objectives


def _phase(phase: str, contract: dict[str, Any], workers: int) -> dict[str, Any]:
    _fresh(contract)
    folder = HERE / phase
    complete = folder / "completed.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        if previous["signature"] != data.digest(HERE / "contract.json") or any(
            data.digest(data.ROOT / name) != digest for name, digest in previous["artifact_sha256"].items()
        ):
            raise ValueError(f"Completed {phase} evidence no longer matches")
        return previous
    started = time.perf_counter()
    config = _configuration(contract, phase)
    jobs = _jobs(phase, contract["dates"])
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker.initialize_worker, initargs=(config,)) as pool:
        fits, fresh_fits, fit_wall = _stage(pool, phase, "fit", jobs, config["signature"])
        _fresh(contract)
        portfolios, fresh_portfolios, portfolio_wall = _stage(pool, phase, "portfolio", jobs, config["signature"])
    parity = _parity(fits, portfolios) if phase in ["smoke", "baseline"] else None
    if parity is not None:
        data.write_json(folder / "serial_parity.json", parity)
        if not parity["passed"]:
            raise ValueError(f"{phase} serial/parallel parity failed; HY addition will not launch")
    if phase != "smoke":
        _publish(phase, jobs, fits, portfolios, contract)
    _fresh(contract)
    timing = {"phase": phase, "workers": workers, "fit_wall_seconds": fit_wall,
              "portfolio_wall_seconds": portfolio_wall, "total_wall_seconds": time.perf_counter() - started,
              "fresh_gp_jobs": fresh_fits, "fresh_portfolio_jobs": fresh_portfolios,
              "is_cold_replay": fresh_fits == len(jobs), "sum_fit_job_seconds": sum(x["seconds"] for x in fits),
              "worker_peak_rss_bytes": max(x["peak_rss_bytes"] for x in fits),
              "fits_per_minute": fresh_fits / fit_wall * 60, "parity": parity}
    data.write_json(folder / "timing.json", timing)
    evidence = [folder / "timing.json", *folder.glob("*audit.json"), *folder.glob("*by_seed.csv")]
    for job in jobs:
        evidence.extend(_stamp_path(phase, stage, job) for stage in ["fit", "portfolio"])
    result = {"signature": config["signature"], "timing": timing, "artifact_sha256": _hashes(evidence)}
    data.write_json(complete, result)
    print(json.dumps(timing), flush=True)
    return result


def execute(workers: int) -> None:
    """Run isolated spawn parity, fresh monthly replay, and the single-variable addition.

    Args:
        workers: Global worker ceiling, constrained by core count and smoke-worker RSS.
    """
    cores = os.cpu_count() or 16
    worker_limit(workers, cores)
    reference_contract = json.loads((REFERENCE / "contract.json").read_text())
    seal = json.loads((REFERENCE / "RESULTS_VERIFICATION.json").read_text())
    sources = {**reference_contract["source_sha256"], **seal["artifact_sha256"]}
    sources.update(_hashes([Path(__file__).resolve(), Path(worker.__file__).resolve(),
                           data.HERE / "log_fit_budget36.py", REFERENCE / "contract.json",
                           REFERENCE / "RESULTS_VERIFICATION.json"]))
    command = reference_contract["commands"]["27"]
    inputs = reference_contract["expected_inputs"]
    baseline = {"command": command, "inputs": inputs}
    hy = copy.deepcopy(baseline)
    index = hy["command"].index("--custom-macro-cols") + 1
    macro = sorted([*hy["command"][index].split(","), "hy_spread_chg_1p"])
    hy["command"][index] = ",".join(macro)
    etf = command[command.index("--custom-etf-cols") + 1].split(",")
    hy["inputs"] = ["t_index", *etf, *macro]
    if set(hy["inputs"]) - set(inputs) != {"hy_spread_chg_1p"} or set(inputs) - set(hy["inputs"]):
        raise ValueError("The second mask must add exactly HY spread change")
    feature_path = command[command.index("--feature-path") + 1]
    date_path = command[command.index("--scored-date-file") + 1]
    contract = {"source_sha256": sources, "feature_path": feature_path, "date_path": date_path,
        "feature_sha256": reference_contract["feature_sha256"], "date_sha256": reference_contract["dates_sha256"],
        "dates": Path(date_path).read_text().splitlines(), "tickers": data.TICKERS, "seeds": SEEDS,
        "rank": 5, "maxiter": 36, "variants": {"baseline": baseline, "plus_hy_change": hy},
        "requested_worker_ceiling": workers, "cpu_count": cores, "memory_budget_gib": 48,
        "previous_weights_fallback": False, "equal_weight_failure_fallback": False, "constraint_relaxations": False,
        "parity_tolerances": PARITY_TOLERANCES,
        "serial_timing": {"seconds": (REFERENCE / "status.json").stat().st_mtime -
                          (REFERENCE / "contract.json").stat().st_mtime,
                          "method": "Approximate original end-to-end interval from immutable contract/status mtimes"}}
    immutable_json(HERE / "contract.json", contract)
    _fresh(contract)
    panel = pd.read_parquet(feature_path)
    if not np.isfinite(panel[hy["inputs"]].to_numpy()).all():
        raise ValueError("The added predictor is not fully covered on the frozen panel")
    smoke = _phase("smoke", contract, 1)
    actual_workers = worker_limit(workers, cores, smoke["timing"]["worker_peak_rss_bytes"])
    base = _phase("baseline", contract, actual_workers)
    addition = _phase("plus_hy_change", contract, actual_workers)
    comparison = {"completed": True, "rank": 5, "maxiter": 36, "actual_workers": actual_workers,
        "serial_timing": contract["serial_timing"], "baseline_timing": base["timing"],
        "hy_timing": addition["timing"], "estimated_historical_speedup":
            contract["serial_timing"]["seconds"] / base["timing"]["total_wall_seconds"],
        "new_fit_jobs": 144, "parity_smoke_extra_fit_jobs": 1, "new_seed_runs": 6,
        "all_primary_portfolios_solved": True, "previous_weights_reuse_disabled": True,
        "scope": "New experiment runner only; protected original/production code and results unchanged."}
    data.write_json(HERE / "comparison.json", comparison)
    data.write_json(HERE / "status.json", {"state": "completed", "comparison": "comparison.json"})
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "VECLIB_MAXIMUM_THREADS": "1", "MPLBACKEND": "Agg",
                       "MPLCONFIGDIR": "/private/tmp/bayesfolio-mpl"})
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "dispatcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args.workers)
