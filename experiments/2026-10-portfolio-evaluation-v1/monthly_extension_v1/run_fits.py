"""Run exactly 24 appended GP fits without changing any prior evidence.

Research I/O orchestration; original rank-two/maxiter-15 model construction and
deterministic global window seeds are unchanged. Financial predictions are
decimal discounted excess returns; covariance includes observation noise.
Usage: uv run --locked --no-sync python <this_file> --execute [--workers 14]
       uv run --locked --no-sync python <this_file> --verify
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS"]:
    os.environ[_key] = "1"

from extension_workers import fit_job, initialize  # noqa: E402

HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parent
REPO = FRAMEWORK.parents[1]
DATA = REPO / "experiments/2026-10-monthly-data-extension-v1/outputs"
OUTPUT = HERE / "outputs"
OLD = FRAMEWORK / "scenario_sharpe_v3/outputs"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_hashes(values: dict) -> None:
    for name, expected in values.items():
        if _hash(Path(name)) != expected:
            raise ValueError(f"Protected evidence changed: {name}")


def _write(path: Path, value: dict) -> None:
    if not path.resolve().is_relative_to(OUTPUT):
        raise ValueError("Only new extension outputs are writable")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Conflicting immutable checkpoint: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def _prepare() -> dict:
    if (OUTPUT / "fit_contract.json").exists():
        contract = json.loads((OUTPUT / "fit_contract.json").read_text())
        _verify_hashes(contract["input_sha256"])
        return contract
    protected = {}
    for root in [DATA, OLD]:
        path = root / "COMPLETED.json"
        seal = json.loads(path.read_text())
        if not seal["passed"]:
            raise ValueError("Successful sealed inputs required")
        for key in ["input_sha256", "artifact_sha256"]:
            protected.update(seal[key])
        protected[str(path)] = _hash(path)
    _verify_hashes(protected)
    sources = {
        "broad14": (FRAMEWORK / "full_rerun_v1/outputs/contract.json", "configs", "global_e6c41f2020fd"),
        "compact5": (FRAMEWORK / "approved_scope_v1/outputs/contract.json", "new_configs", "joint_existing_us_mask"),
    }
    dates = (DATA / "dataset/scored_dates_28.txt").read_text().splitlines()
    if len(dates) != 28 or dates[24:] != ["2026-05-29", "2026-06-30", "2026-07-31", "2026-08-31"]:
        raise ValueError("Only four complete additional outcomes may be appended")
    configs = {}
    for component, (source, group, name) in sources.items():
        protected[str(source)] = _hash(source)
        config = copy.deepcopy(json.loads(source.read_text())[group][name])
        if config["dates"] != dates[:24]:
            raise ValueError("Original date prefix changed")
        config.pop("old_run_root", None)
        config["dates"] = dates
        command = config["command"]
        for flag, value in [
            ("--feature-path", str(DATA / "dataset/global.parquet")),
            ("--scored-date-file", str(DATA / "dataset/scored_dates_28.txt")),
            ("--output-root", str(OUTPUT / "runs")),
            ("--run-id", f"{component}_s27"),
        ]:
            command[command.index(flag) + 1] = value
        configs[component] = config
    for path in [HERE / "run_fits.py", HERE / "extension_workers.py"]:
        protected[str(path)] = _hash(path)
    contract = {
        "schema": "bayesfolio.append_monthly_fits.v1", "configs": configs, "input_sha256": protected,
        "new_fit_count": 24, "old_refits": 0, "seeds": [27, 127, 227], "window_indices": [24, 25, 26, 27],
        "rank": 2, "maxiter": 15, "training_months": [108, 109, 110, 111],
        "legacy_allocation_invoked": False,
        "note": "Legacy CLI allocation flags are inert: only fit_gp_window_artifact and dense readout are called.",
    }
    _write(OUTPUT / "fit_contract.json", contract)
    return contract


def _checkpoint(job: dict) -> Path:
    return OUTPUT / "fit_checkpoints" / f"{job['component']}_s{job['seed']}_{job['date']}.json"


def _jobs(contract: dict) -> list[dict]:
    return [
        {"component": component, "seed": seed, "date": config["dates"][index], "index": index,
         "run": str(OUTPUT / "runs" / f"{component}_s{seed}")}
        for component, config in contract["configs"].items()
        for seed in contract["seeds"] for index in contract["window_indices"]
    ]


def _verify() -> None:
    contract = json.loads((OUTPUT / "fit_contract.json").read_text())
    _verify_hashes(contract["input_sha256"])
    for job in _jobs(contract):
        record = json.loads(_checkpoint(job).read_text())
        _verify_hashes(record["artifact_sha256"])
        result = record["result"]
        if (result["window_index"], result["train_dates"], result["rank"], result["maxiter"]) != (
            job["index"], 84 + job["index"], 2, 15
        ):
            raise ValueError("Actual fitting controls changed")
        log = [json.loads(line) for line in Path(result["paths"]["optimizer_log"]).read_text().splitlines()]
        if not log or any(row["iterations"] > 15 or row["window_index"] != job["index"] for row in log):
            raise ValueError("Optimizer telemetry does not meet registered budget")
    print("Verified 24 appended fits, dense readouts, global seed indices and protected prior inputs", flush=True)


def _execute(workers: int) -> None:
    contract = _prepare()
    pending = []
    for job in _jobs(contract):
        if _checkpoint(job).exists():
            _verify_hashes(json.loads(_checkpoint(job).read_text())["artifact_sha256"])
        else:
            pending.append(job)
    started = time.monotonic()
    print(f"Launching {len(pending)} remaining appended fits with {workers} single-thread workers", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1,
        initializer=initialize,
        initargs=(contract["configs"], str(FRAMEWORK),
                  str(REPO / "experiments/2026-10-repaired-global-feature-selection")),
    ) as pool:
        futures = {pool.submit(fit_job, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            hashes = {str(Path(name).resolve()): _hash(Path(name)) for name in result["paths"].values()}
            _write(_checkpoint(job), {"result": result, "artifact_sha256": hashes})
            print(f"Accepted {job['component']} s{job['seed']} {job['date']}; "
                  f"fit {result['seconds']:.1f}s; elapsed {time.monotonic() - started:.1f}s", flush=True)
    _verify()
    _write(OUTPUT / "FITS_COMPLETED.json", {
        "passed": True, "new_fits": 24, "old_refits": 0, "workers": workers,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoints": {str(_checkpoint(job)): _hash(_checkpoint(job)) for job in _jobs(contract)},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify", action="store_true")
    limit = min(15, max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--workers", type=int, default=limit)
    args = parser.parse_args()
    if not 1 <= args.workers <= limit:
        parser.error(f"Workers must be between one and {limit}")
    if args.execute:
        _execute(args.workers)
    else:
        _verify()
