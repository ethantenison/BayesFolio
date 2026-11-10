"""Root-callable, locked queue for the registered missing v2 GP moments.

This isolated orchestration adapter never selects policies. It admits only the
frozen reference/compact-rank1 specifications, keeps one global compute lock,
counts all admitted/started attempts including retries, and exports all15-task
means (decimal excess returns) and observed covariance (squared decimals).
Usage: uv run --locked --no-sync python <this_file> --fit --workers 14
Importing this module or freezing its contract never starts a fit pool.
"""

from __future__ import annotations

import os

for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-v2-mpl")

import argparse  # noqa: E402
import fcntl  # noqa: E402
import multiprocessing  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from v2_common import (  # noqa: E402
    ASSETS,
    CHRONO,
    FRAMEWORK,
    HERE,
    REPO,
    SEEDS,
    TUNING,
    V1,
    arrays,
    digest,
    read,
    save,
    snapshot,
    status,
    verify_hashes,
)
from v2_expert_worker import fit_job, initialize  # noqa: E402
from v2_experts import (  # noqa: E402
    DATASET,
    REGISTERED,
    cache_inventory,
    checkpoint_path,
    load_cached_panel,
    missing_jobs,
    registered_configs,
    validate_fit,
)

ATTEMPT_CAP = 1200
FINAL_RESERVE = 96
MAX_IDENTICAL_ATTEMPTS = 2
FINAL_ORIGIN = "2026-10-30"


def prepare_contract() -> dict:
    """Freeze repeat-stable numerical/source/data evidence before pool creation.

    Returns:
        Immutable six-job registration and protected SHA256 map. Repeated calls
        verify the existing contract rather than recomputing its missing jobs.
    """
    path = HERE / "FIT_CONTRACT.json"
    if path.exists():
        contract = read(path)
        verify_hashes(contract["protected_sha256"])
        if contract["configs"] != read(HERE / "GP_VARIANTS.json")["specs"]:
            raise ValueError("GP registration changed after the fit contract froze")
        return contract
    variants = read(HERE / "GP_VARIANTS.json")
    configs = registered_configs()
    if (variants["specs"] != configs or variants["new_specs_count"] != 1
            or variants["kernel_or_prior_variants"] != [] or set(configs) != set(REGISTERED)):
        raise ValueError("Only initial registered compact-rank1 scope may enter this queue")
    jobs = missing_jobs(configs)
    if (len(jobs) != 6 or any(job["component"] != "compact_r1_i15_exp"
                             or job["date"] not in {"2026-09-30", FINAL_ORIGIN} for job in jobs)):
        raise ValueError("Initial fit registration must contain exactly six missing rank1 jobs")
    inventory = cache_inventory()
    files = [HERE / "GP_VARIANTS.json", HERE / "GOAL.md", HERE / "v2_common.py", HERE / "v2_experts.py",
             HERE / "v2_expert_worker.py", HERE / "v2_fit_queue.py", HERE / "expert_dates.txt",
             V1 / "data/COMPLETED.json", DATASET, CHRONO / "CONTRACT.json", TUNING / "tuning_worker.py",
             FRAMEWORK / "parallel_pipeline_v1/workers.py", REPO / "pyproject.toml", REPO / "uv.lock"]
    for folder in [TUNING.parent / "2026-07-international-equity-etf-features",
                   TUNING.parent / "2026-06-portfolio-optimization", TUNING.parent / "2026-06-task-covariance",
                   REPO / "bayesfolio/engine/forecast/gp"]:
        files.extend(sorted(folder.glob("*.py")))
    for value in inventory.values():
        files.extend(Path(name) for name in value["source_sha256"])
    protected = {str(item.resolve()): digest(item) for item in sorted(set(files))}
    contract = {"passed": True, "configs": configs, "jobs": jobs, "seeds": SEEDS, "tasks": ASSETS,
                "expected_new_fits": len(jobs), "total_fit_attempt_cap": ATTEMPT_CAP,
                "reserved_for_final_origin_or_repairs": FINAL_RESERVE, "worker_cap": 14,
                "maximum_identical_attempts": MAX_IDENTICAL_ATTEMPTS,
                "global_lock": str(HERE / "compute_queue.lock"), "cache_inventory": inventory,
                "no_portfolio_optimization_in_fit_worker": True, "protected_sha256": protected}
    save(path, contract)
    return contract


def attempt_inventory() -> dict:
    """Count global starts and conservative admissions, including abandoned retries.

    Returns:
        Unique attempt paths across every v2 started.json and fit-admission record.
        Admitted work counts even if a worker crashes before writing started.json.
    """
    starts = {str(path.parent.resolve()) for path in (HERE / "outputs").glob("**/started.json")}
    admissions = {str(Path(read(path)["run"]).resolve())
                  for path in (HERE / "outputs/fit_admissions").glob("*.json")}
    attempts = starts | admissions
    if any(not Path(name).is_relative_to(HERE) for name in attempts):
        raise ValueError("Out-of-scope attempt evidence")
    return {"started_attempts": len(starts), "admitted_or_started_attempts": len(attempts),
            "fit_attempts_used": len(attempts), "fit_attempts_remaining": max(0, ATTEMPT_CAP - len(attempts)),
            "attempt_paths": sorted(attempts), "attempts_remaining": max(0, ATTEMPT_CAP - len(attempts))}


def _check_budget(job: dict, inventory: dict) -> None:
    used = inventory["admitted_or_started_attempts"]
    limit = ATTEMPT_CAP if job["date"] == FINAL_ORIGIN else ATTEMPT_CAP - FINAL_RESERVE
    if used >= limit:
        raise RuntimeError(f"Fit budget prevents admission at {used}/{limit}; final reserve preserved")


def _checked_checkpoint(job: dict, contract: dict) -> dict:
    record = read(checkpoint_path(job))
    if record["job"] != job or record["contract_sha256"] != digest(HERE / "FIT_CONTRACT.json"):
        raise ValueError("Checkpoint job or frozen fit contract differs")
    validate_fit(record, job, contract["configs"][job["component"]])
    return record


def _export_complete_panels(contract: dict) -> list[Path]:
    artifacts = []
    for name in REGISTERED:
        values = load_cached_panel(name)
        if not values["available"].all():
            raise ValueError(f"Cannot export incomplete expert: {name}")
        path = HERE / "outputs/experts" / f"{name}.npz"
        snapshot(path, {key: values[key] for key in ["mean", "covariance", "dates", "assets", "seeds"]})
        artifacts.append(path)
    frame = pd.read_parquet(DATASET)
    for job in contract["jobs"]:
        record = _checked_checkpoint(job, contract)
        exported = arrays(Path(record["result"]["paths"]["analytic"]))
        expected = frame[frame.date.eq(pd.Timestamp(job["date"]))].set_index("asset_id").loc[
            ASSETS, "y_excess_lead"].to_numpy()
        np.testing.assert_allclose(exported["actual"], expected, rtol=0, atol=1e-15, equal_nan=True)
        artifacts.append(checkpoint_path(job))
        artifacts.extend(Path(name) for name in record["artifact_sha256"])
    return artifacts


def run_missing(workers: int = 14) -> dict:
    """Run six missing rank1 fits through the sole global compute lock and pool.

    Args:
        workers: Maximum single-thread workers, at most14 globally. The queue
            acquires the same nonblocking lock as v2_driver before any launch.

    Returns:
        Passed FITS_COMPLETED.json after all six records and four54-origin expert
        panels validate. Routine identical failures receive at most one retry.
    """
    if not 1 <= workers <= 14:
        raise ValueError("One pool of at most14 single-thread workers is required")
    with (HERE / "compute_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = prepare_contract()
        seal_path = HERE / "FITS_COMPLETED.json"
        if seal_path.exists():
            value = read(seal_path)
            verify_hashes(contract["protected_sha256"])
            verify_hashes(value["artifact_sha256"])
            return value
        pending = []
        for job in contract["jobs"]:
            if checkpoint_path(job).exists():
                _checked_checkpoint(job, contract)
            else:
                pending.append(job)
        started, attempt_numbers = time.monotonic(), {}
        if pending:
            with ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                    mp_context=multiprocessing.get_context("spawn"), initializer=initialize,
                    initargs=(contract["configs"], str(FRAMEWORK))) as queue:

                def submit(job: dict):
                    inventory = attempt_inventory()
                    _check_budget(job, inventory)
                    key = checkpoint_path(job).stem
                    used = [name for name in inventory["attempt_paths"] if Path(name).parent.name == key]
                    attempt = len(used) + 1
                    if attempt > MAX_IDENTICAL_ATTEMPTS:
                        raise RuntimeError(f"Identical fit failed repeatedly: {key}; inspect preserved evidence")
                    attempt_numbers[key] = attempt
                    run = HERE / "outputs/fits" / key / f"attempt_{attempt:02d}"
                    if str(run.resolve()) in inventory["attempt_paths"]:
                        raise ValueError("Attempt path already used")
                    save(HERE / "outputs/fit_admissions" / f"{key}_attempt_{attempt:02d}.json",
                         {"job": job, "run": str(run), "attempt": attempt,
                          "admitted_at_utc": datetime.now(UTC).isoformat(),
                          "contract_sha256": digest(HERE / "FIT_CONTRACT.json")})
                    return queue.submit(fit_job, {**job, "run": str(run)})

                futures = {submit(job): job for job in pending}
                while futures:
                    future = next(as_completed(futures))
                    job = futures.pop(future)
                    key = checkpoint_path(job).stem
                    try:
                        result = future.result()
                        record = {"job": job, "result": result,
                                  "contract_sha256": digest(HERE / "FIT_CONTRACT.json"),
                                  "artifact_sha256": {str(Path(name).resolve()): digest(Path(name))
                                                      for name in result["paths"].values()}}
                        validate_fit(record, job, contract["configs"][job["component"]])
                    except Exception:
                        attempt = attempt_numbers[key]
                        run = HERE / "outputs/fits" / key / f"attempt_{attempt:02d}"
                        failure = {"job": job, "attempt": attempt, "run": str(run),
                                   "traceback": traceback.format_exc(), "failed_at_utc": datetime.now(UTC).isoformat(),
                                   "partial_artifact_sha256": {str(path): digest(path)
                                                               for path in run.glob("**/*") if path.is_file()}}
                        save(HERE / "outputs/fit_failures" / f"{key}_attempt_{attempt:02d}.json", failure)
                        status("GP_fit_failure", "Inspect preserved evidence; retry only the identical intended job",
                               failure=failure, **attempt_inventory())
                        if attempt < MAX_IDENTICAL_ATTEMPTS:
                            futures[submit(job)] = job
                            continue
                        raise
                    save(checkpoint_path(job), record)
                    completed = sum(checkpoint_path(item).exists() for item in contract["jobs"])
                    status("registered_GP_fits", "Complete six rank1 exports, then assemble four54-origin panels",
                           completed_new_fits=completed, expected_new_fits=6, workers=workers,
                           elapsed_seconds=time.monotonic() - started, last_completed=job,
                           failures=[str(path) for path in (HERE / "outputs/fit_failures").glob("*.json")],
                           **attempt_inventory())
        artifacts = _export_complete_panels(contract)
        verify_hashes(contract["protected_sha256"])
        inventory = attempt_inventory()
        result = {"passed": True, "completed_new_fits": len(contract["jobs"]), "complete_experts": REGISTERED,
                  "origins_per_expert": 54, "total_month_seed_cells_per_expert": 162,
                  "contract_sha256": digest(HERE / "FIT_CONTRACT.json"), **inventory,
                  "artifact_sha256": {str(path.resolve()): digest(path) for path in sorted(set(artifacts))}}
        save(seal_path, result)
        status("GP_fits_complete", "Continue preregistered development selection; confirmation remains locked",
               completed_new_fits=6, **inventory)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true", required=True)
    parser.add_argument("--workers", type=int, default=14)
    arguments = parser.parse_args()
    run_missing(arguments.workers)
