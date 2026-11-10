"""One bounded month-parallel queue for recent and November GP forecasts.

Research orchestration only. New fits use fixed registered models and isolated
attempt directories; no trading or portfolio fallback. Outputs include exact
observed covariance in squared-decimal units and decimal predictive means.
Usage: uv run --locked --no-sync python <this_file> --fit --workers 14
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS"]:
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-gp-tuning-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-gp-tuning-cache")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import (  # noqa: E402
    CHRONO,
    COMPONENTS,
    FRAMEWORK,
    HERE,
    SEEDS,
    TUNING,
    arrays,
    digest,
    read,
    save,
    snapshot,
    status,
    verify_hashes,
)
from live_worker import fit_job, initialize  # noqa: E402
from prepare_decision import prepare  # noqa: E402


def prepare_fits() -> dict:
    """Freeze data/code/model registration after historical selection and data audit.

    Returns:
        Fit contract requiring eighteen new monthly fits with a hard128-attempt cap.
    """
    plan = prepare()
    if not read(HERE / "HISTORICAL_DECISION.json")["passed"]:
        raise ValueError("Freeze the historical decision before fresh outcome scoring")
    data = read(HERE / "data/COMPLETED.json")
    if not data["passed"]:
        raise ValueError("Complete validated October-close data required")
    path = HERE / "FIT_CONTRACT.json"
    if path.exists():
        value = read(path)
        verify_hashes(value["protected_sha256"])
        return value
    source = read(CHRONO / "CONTRACT.json")["configs"]
    configs = {}
    dates = ["2026-09-30", "2026-10-30"]
    save(HERE / "fit_dates.txt", ("\n".join(dates) + "\n").encode())
    for name in COMPONENTS:
        config = copy.deepcopy(source[name])
        config.update(dates=dates, reuse=None)
        command = config["command"]
        replacements = {"--feature-path": str(HERE / "data/dataset/global.parquet"),
                        "--output-root": str(HERE / "outputs/fits"), "--run-id": "november_" + name,
                        "--scored-date-file": str(HERE / "fit_dates.txt"),
                        "--gp-riskfolio-rm": "MV", "--gp-riskfolio-obj": "Sharpe", "--nea": "7"}
        for flag, value in replacements.items():
            command[command.index(flag) + 1] = value
        if config["rank"] != 2 or config["maxiter"] != 15:
            raise ValueError("November fitting controls changed")
        configs[name] = config
    files = [HERE / "PREREGISTRATION.json", HERE / "HISTORICAL_DECISION.json", HERE / "data/COMPLETED.json",
             HERE / "data/dataset/global.parquet", HERE / "live_worker.py", HERE / "live_fits.py",
             TUNING / "tuning_worker.py", FRAMEWORK / "parallel_pipeline_v1/workers.py"]
    for folder in [TUNING.parent / "2026-07-international-equity-etf-features",
                   TUNING.parent / "2026-06-portfolio-optimization", TUNING.parent / "2026-06-task-covariance",
                   HERE.parents[1] / "bayesfolio/engine/forecast/gp"]:
        files.extend(sorted(folder.glob("*.py")))
    value = {"passed": True, "configs": configs, "dates": dates, "seeds": SEEDS,
             "expected_new_fits": 18, "attempt_cap": plan["caps"]["monthly_fit_attempts"],
             "worker_cap": 14, "old_kernel_and_fit_seed_rules_preserved": True,
             "portfolio_optimizer_not_called_by_fit_worker": True,
             "legacy_riskfolio_flags_unused_for_forecast_extraction": True,
             "protected_sha256": {str(item.resolve()): digest(item) for item in files}}
    save(path, value)
    return value


def _jobs(contract: dict) -> list[dict]:
    return [{"component": name, "seed": seed, "index": index, "date": date, "audit": False}
            for name in COMPONENTS for seed in SEEDS for index, date in enumerate(contract["dates"], 28)]


def _checkpoint(job: dict):
    return HERE / "outputs/fit_checkpoints" / f"{job['component']}_s{job['seed']}_{job['date']}.json"


def _check(record: dict, job: dict, config: dict) -> None:
    verify_hashes(record["artifact_sha256"])
    result = record["result"]
    if (result["rank"], result["maxiter"], result["train_months"], result["global_window_index"]) != (
        2, 15, config["rolling_months"] or 84 + job["index"], job["index"]
    ):
        raise ValueError("Fit controls differ from registered November experiment")
    values = arrays(Path(result["paths"]["analytic"]))
    if not np.isfinite(values["mean"]).all() or not np.isfinite(values["covariance"]).all():
        raise ValueError("Invalid GP moments")
    np.linalg.cholesky(values["covariance"])
    if job["date"] == "2026-10-30" and not np.isnan(values["actual"]).all():
        raise ValueError("Live November targets must remain unknown")
    if job["date"] == "2026-09-30" and not np.isfinite(values["actual"]).all():
        raise ValueError("October historical outcomes must be available")


def fit(workers: int = 14) -> dict:
    """Run the sole bounded forecast queue, repairing only identical failed attempts.

    Args:
        workers: Concurrent single-thread workers, at most14 across all models.

    Returns:
        Passed fit completion seal after all18 jobs and analytic exports validate.
    """
    contract = prepare_fits()
    seal = HERE / "FITS_COMPLETED.json"
    if seal.exists():
        value = read(seal)
        verify_hashes(value["artifact_sha256"])
        return value
    if not 1 <= workers <= 14:
        raise ValueError("One shared pool of at most14 workers required")
    with (HERE / "fit_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs, pending = _jobs(contract), []
        for job in jobs:
            if _checkpoint(job).exists():
                _check(read(_checkpoint(job)), job, contract["configs"][job["component"]])
            else:
                pending.append(job)
        attempts_used = len(list((HERE / "outputs/fits").glob("**/started.json")))
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize, initargs=(contract["configs"], str(FRAMEWORK))) as pool:
            attempt_numbers = {}

            def submit(job: dict):
                nonlocal attempts_used
                if attempts_used >= contract["attempt_cap"]:
                    raise RuntimeError("Fit-attempt budget exhausted without a complete forecast")
                key = _checkpoint(job).stem
                attempt = len(list((HERE / "outputs/fits" / key).glob("attempt_*"))) + 1
                if attempt > 2:
                    raise RuntimeError(f"Repeated identical fit failure: {key}; inspect evidence")
                attempt_numbers[key] = attempt
                attempts_used += 1
                run = HERE / "outputs/fits" / key / f"attempt_{attempt:02d}"
                return pool.submit(fit_job, {**job, "run": str(run)})

            futures = {submit(job): job for job in pending}
            while futures:
                future = next(as_completed(futures))
                job = futures.pop(future)
                try:
                    result = future.result()
                except Exception:
                    key = _checkpoint(job).stem
                    save(HERE / "outputs/fit_failures" / f"{key}_attempt_{attempt_numbers[key]}.json",
                         {"job": job, "traceback": traceback.format_exc()})
                    if attempt_numbers[key] < 2:
                        futures[submit(job)] = job
                        continue
                    raise
                record = {"job": job, "result": result, "contract_sha256": digest(HERE / "FIT_CONTRACT.json"),
                          "artifact_sha256": {str(Path(item).resolve()): digest(Path(item))
                                              for item in result["paths"].values()}}
                _check(record, job, contract["configs"][job["component"]])
                save(_checkpoint(job), record)
                completed = len(list((HERE / "outputs/fit_checkpoints").glob("*.json")))
                status("recent_month_GP_fits", "Complete18 exports, then independently validate live weights",
                       completed_fits=completed, total_fits=18, fit_attempts_used=attempts_used,
                       fit_attempts_remaining=128 - attempts_used, workers=workers, last_completed=job,
                       elapsed_seconds=time.monotonic() - started,
                       failures=[str(path) for path in (HERE / "outputs/fit_failures").glob("*.json")])
        frame = pd.read_parquet(HERE / "data/dataset/global.parquet")
        context = arrays(CHRONO / "context52.npz")
        for name in COMPONENTS:
            mean, covariance = np.empty((3, 2, 15)), np.empty((3, 2, 15, 15))
            for job in [item for item in jobs if item["component"] == name]:
                record = read(_checkpoint(job))
                _check(record, job, contract["configs"][name])
                values = arrays(Path(record["result"]["paths"]["analytic"]))
                np.testing.assert_array_equal(values["assets"], context["assets"])
                seed, month = SEEDS.index(job["seed"]), job["index"] - 28
                mean[seed, month], covariance[seed, month] = values["mean"], values["covariance"]
                expected = frame[frame.date.eq(pd.Timestamp(job["date"]))].set_index("asset_id").loc[
                    context["assets"], "y_excess_lead"].to_numpy()
                np.testing.assert_allclose(values["actual"], expected, rtol=0, atol=1e-15, equal_nan=True)
            snapshot(HERE / "outputs/recent_components" / f"{name}.npz", {
                "mean": mean, "covariance": covariance, "assets": context["assets"],
                "seeds": np.array(SEEDS), "dates": np.array(contract["dates"])})
        verify_hashes(contract["protected_sha256"])
        artifacts = [*sorted((HERE / "outputs/fit_checkpoints").glob("*.json")),
                     *sorted((HERE / "outputs/recent_components").glob("*.npz"))]
        value = {"passed": True, "completed_fits": 18, "fit_attempts_used": attempts_used,
                 "fit_attempts_remaining": 128 - attempts_used,
                 "artifact_sha256": {str(item): digest(item) for item in artifacts}}
        save(seal, value)
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true", required=True)
    parser.add_argument("--workers", type=int, default=14)
    arguments = parser.parse_args()
    fit(arguments.workers)
