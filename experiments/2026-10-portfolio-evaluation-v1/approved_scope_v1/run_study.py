"""Run the approved diversification, grouping and US-collapse follow-up study.

Usage: .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/approved_scope_v1/run_study.py --execute
Use --prepare for the immutable plan, --smoke-only for replay/allocation gates,
--status for progress, and --verify-only for completed evidence. One bounded
worker pool per stage; independent monthly fits precede final batch scoring.
Only approved scope 1-6 and 8 runs; no feature search, kernel redesign, training
window changes, new holdout/stability studies, CVaR, caps or failed-weight reuse.
All forecasts remain in discounted-excess decimals and risk in squared units.
"""

# Thread controls must be set before NumPy/BLAS, including under spawn.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import study_workers
from plan import _assemble, _prepare, _snapshot
from study_common import BASELINE, HERE, ROOT, SEEDS, _context, _hash, _portfolio_cases, _save, _verify
from study_math import _forecast_rows


def _status(stage: str, **values) -> None:
    _save(ROOT / "status.json", {"stage": stage, "updated_unix": time.time(), **values})
    print(json.dumps({"stage": stage, **values}), flush=True)


def _case_id(case: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:20]


def _run_jobs(
    pool: ProcessPoolExecutor, jobs: list[dict], worker, prefix: str, signature: str, workers: int
) -> list[dict]:
    completed, pending = [], []
    for job in jobs:
        stamp = ROOT / "checkpoints" / prefix / f"{job['job_id']}.json"
        if stamp.exists():
            value = json.loads(stamp.read_text())
            if value["signature"] != signature:
                raise ValueError("Checkpoint contract differs")
            _verify(value["result"]["artifact_sha256"])
            _verify(value["result"].get("source_sha256", {}))
            completed.append(value["result"])
        else:
            pending.append(job)
    iterator, futures = iter(pending), {}

    def submit() -> None:
        job = next(iterator, None)
        if job is not None:
            futures[pool.submit(worker, job)] = job

    for _ in range(workers * 3):
        submit()
    last_update = 0.0
    while futures:
        done, _ = wait(futures, timeout=20, return_when=FIRST_COMPLETED)
        for future in done:
            job = futures.pop(future)
            try:
                result = future.result()
            except BaseException:
                for queued in futures:
                    queued.cancel()
                raise
            _save(
                ROOT / "checkpoints" / prefix / f"{job['job_id']}.json",
                {"signature": signature, "result": result},
                immutable=True,
            )
            completed.append(result)
            submit()
        if time.monotonic() - last_update > 2 or not futures:
            _status(
                prefix,
                completed=len(completed),
                total=len(jobs),
                workers=workers,
                approved_items=[1, 2, 3, 4, 5, 6, 8],
                excluded_items=[7, 9],
            )
            last_update = time.monotonic()
    return completed


def _fit_worker(job: dict) -> dict:
    result = study_workers.fit_job(job)
    if (result["rank"], result["maxiter"], result["train_dates"]) != (2, 15, 84 + job["index"]):
        raise ValueError("Actual fitting controls differ")
    paths = [Path(value).resolve() for value in result["paths"].values()]
    if not all(path.is_relative_to(ROOT.resolve()) for path in paths):
        raise ValueError("Fit tried to write outside this study")
    return {"fit": result, "artifact_sha256": {str(path): _hash(path) for path in paths}}


def _fit_jobs(configs: dict) -> list[dict]:
    return [
        {
            "component": name,
            "seed": seed,
            "index": index,
            "date": date,
            "run": str(ROOT / f"runs/{name}_s{seed}"),
            "job_id": f"{name}_s{seed}_{date}",
        }
        for name in sorted(configs, key=lambda key: -len(configs[key]["tickers"]))
        for seed in SEEDS
        for index, date in enumerate(configs[name]["dates"])
    ]


def _evaluate(contract: dict, pilot: bool = False) -> list[dict]:
    models = _assemble(contract)
    jobs = []
    for model in models:
        for case in _portfolio_cases(model):
            case["case_id"] = _case_id(case)
            case["job_id"] = case["case_id"]
            jobs.append(case)
    if pilot:
        jobs = [
            job
            for job in jobs
            if job["configuration"] == BASELINE
            and job["minimum"] == 10
            and job["budget_factor"] == 1
            and job["objective"] == "rank"
            and job["risk_source"] == "GP"
        ]
        if len(jobs) != 1:
            raise ValueError("Allocation pilot not uniquely registered")
    signature = _hash(ROOT / "contract.json")
    workers = min(contract["worker_ceiling"], len(jobs))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=study_workers.initialize_portfolios,
        initargs=(_context(),),
    ) as pool:
        results = _run_jobs(pool, jobs, study_workers.portfolio_job, "portfolio_policies", signature, workers)
    if pilot:
        reference = json.loads(
            (HERE.parent / "portfolio_policy_comparison_v1/outputs/portfolios" / f"{BASELINE}.json").read_text()
        )
        pilot_result = json.loads(Path(next(iter(results[0]["artifact_sha256"]))).read_text())
        differences = []
        for expected, current in zip(reference["seeds"], pilot_result["seeds"], strict=True):
            old_weights = expected["portfolios"]["GP__risk_budget_plus_effective"]["performance"]["weights"]
            new_weights = current["performance"]["weights"]
            differences.append(float(np.max(np.abs(np.asarray(old_weights) - new_weights))))
        if max(differences) > 1e-6:
            raise ValueError("New rank allocator differs from protected baseline")
        _save(
            ROOT / "allocation_pilot.json",
            {"passed": True, "maximum_weight_difference": max(differences)},
            immutable=True,
        )
    else:
        frame = pd.DataFrame([value["summary"] for value in results]).sort_values(
            ["configuration", "risk_source", "objective", "budget_factor", "minimum"], na_position="first"
        )
        frame.to_csv(ROOT / "policy_metrics.csv", index=False)
        _save(
            ROOT / "portfolio_progress.json",
            {"models": len(models), "cases": len(jobs), "complete_cases": int(frame.complete_seeds.eq(3).sum())},
        )
    return results


def _smokes(contract: dict) -> int:
    configs = {**contract["new_configs"], **contract["audit_configs"]}
    audit_jobs = [job for job in _fit_jobs(contract["audit_configs"]) if job["seed"] == 27 and job["index"] in {0, 23}]
    jobs = _fit_jobs(contract["new_configs"])
    largest_triple = max(
        (name for name in configs if name.startswith("matched__")), key=lambda name: len(configs[name]["tickers"])
    )
    selected = [
        job
        for job in jobs
        if job["seed"] == 27
        and (
            (job["component"] == "joint_existing_us_mask" and job["index"] == 0)
            or (job["component"] == largest_triple and job["index"] == 23)
        )
    ]
    workers = min(contract["worker_ceiling"], len(audit_jobs) + len(selected))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=study_workers.initialize_fits,
        initargs=(configs,),
    ) as pool:
        audited = _run_jobs(pool, audit_jobs, _fit_worker, "audit_fits", _hash(ROOT / "contract.json"), workers)
        fitted = _run_jobs(pool, selected, _fit_worker, "matched_fits", _hash(ROOT / "contract.json"), workers)
    peak = max(row["fit"]["peak_rss_bytes"] for row in audited + fitted)
    allowed = max(1, min(contract["worker_ceiling"], int(48 * 1024**3 / (peak * 1.25))))
    _save(
        ROOT / "smoke_gate.json",
        {
            "passed": True,
            "audit_replays": len(audited),
            "new_component_smokes": len(fitted),
            "peak_worker_rss_bytes": peak,
            "workers_after_memory_gate": allowed,
        },
        immutable=True,
    )
    return allowed


def _consolidate(contract: dict) -> None:
    for name, config in contract["new_configs"].items():
        means, covariances = [], []
        for seed in SEEDS:
            per_mean, per_covariance = [], []
            for date in config["dates"]:
                path = ROOT / f"runs/{name}_s{seed}/analytic_months/covariance_{date}.npz"
                with np.load(path, allow_pickle=False) as values:
                    if values["assets"].tolist() != config["tickers"] or values["date"].item() != date:
                        raise ValueError("Fitted month/task axes differ")
                    per_mean.append(values["mean"].copy())
                    per_covariance.append(values["covariance"].copy())
            means.append(np.stack(per_mean))
            covariances.append(np.stack(per_covariance))
        _snapshot(
            ROOT / "components" / f"{name}.npz",
            mean=np.stack(means),
            covariance=np.stack(covariances),
            dates=np.array(config["dates"]),
            assets=np.array(config["tickers"]),
        )


def _diagnose(contract: dict) -> None:
    context = _context()
    etfs, ranks, predictions = [], [], []
    for name, component in contract["components"].items():
        path = ROOT / "components" / f"{name}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as values:
            for offset, seed in enumerate(SEEDS):
                rows, ranking = _forecast_rows(
                    context, name, seed, values["mean"][offset], values["covariance"][offset], component["assets"]
                )
                etfs.extend(rows)
                ranks.extend(ranking)
                for index, asset in enumerate(component["assets"]):
                    actual = context["actual"][:, context["assets"].index(asset)]
                    for month, date in enumerate(context["dates"]):
                        predictions.append(
                            {
                                "configuration": name,
                                "seed": seed,
                                "date": date,
                                "asset": asset,
                                "actual": float(actual[month]),
                                "predicted": float(values["mean"][offset, month, index]),
                                "predicted_sd": float(np.sqrt(values["covariance"][offset, month, index, index])),
                            }
                        )
    pd.DataFrame(etfs).to_csv(ROOT / "ETF_diagnostics.csv", index=False)
    pd.DataFrame(ranks).to_csv(ROOT / "ranking_diagnostics.csv", index=False)
    pd.DataFrame(predictions).to_csv(ROOT / "prediction_paths.csv", index=False)
    # All complete assemblies also get pooled and family IC, with no fabricated covariance.
    assemblies = []
    for model in _assemble(contract):
        with np.load(ROOT / "panels" / f"{model['name']}.npz", allow_pickle=False) as values:
            from metrics import rank_ic

            for offset, seed in enumerate(SEEDS):
                for scope, assets in {"pooled": context["assets"], **context["families"]}.items():
                    indices = [context["assets"].index(asset) for asset in assets]
                    monthly = rank_ic(context["actual"][:, indices], values["mean"][offset][:, indices])
                    if not np.isfinite(monthly).all():
                        raise ValueError("Undefined assembly IC; no silent omission")
                    assemblies.append(
                        {"configuration": model["name"], "seed": seed, "scope": scope, "mean_ic": float(monthly.mean())}
                    )
    pd.DataFrame(assemblies).to_csv(ROOT / "assembly_ranking.csv", index=False)


def _verify_complete() -> None:
    contract = json.loads((ROOT / "contract.json").read_text())
    seal = json.loads((ROOT / "COMPLETED.json").read_text())
    if not seal["passed"] or seal["signature"] != _hash(ROOT / "contract.json"):
        raise ValueError("Incomplete study seal")
    _verify(contract["input_sha256"])
    _verify(seal["artifact_sha256"])
    for prefix, expected in [("matched_fits", 648), ("audit_fits", 4), ("portfolio_policies", 372)]:
        stamps = list((ROOT / "checkpoints" / prefix).glob("*.json"))
        if len(stamps) != expected:
            raise ValueError(f"Wrong completed count for {prefix}: {len(stamps)}")
        for path in stamps:
            value = json.loads(path.read_text())
            if value["signature"] != seal["signature"]:
                raise ValueError("Checkpoint signature changed")
            _verify(value["result"]["artifact_sha256"])
            _verify(value["result"].get("source_sha256", {}))
    print("Verified scope 1-6 and 8: 652 new fits/replays, 30 models, 372 portfolio cases; 7 and 9 excluded")


def _execute(contract: dict, smoke_only: bool) -> None:
    _assemble(contract)
    _evaluate(contract, pilot=True)
    workers = _smokes(contract)
    if smoke_only:
        _status("smoke_complete", workers=workers, approved_items=contract["approved_items"], excluded_items=[7, 9])
        return
    _diagnose(contract)
    _evaluate(contract)
    configs = {**contract["new_configs"], **contract["audit_configs"]}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=study_workers.initialize_fits,
        initargs=(configs,),
    ) as pool:
        _run_jobs(
            pool,
            _fit_jobs(contract["new_configs"]),
            _fit_worker,
            "matched_fits",
            _hash(ROOT / "contract.json"),
            workers,
        )
    _consolidate(contract)
    _diagnose(contract)
    results = _evaluate(contract)
    if len(results) != 372:
        raise ValueError("Missing approved portfolio policies")
    from study_report import _report

    _report(contract)
    _verify(contract["input_sha256"])
    paths = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name
        not in {"status.json", "run.lock", "launch.lock", "COMPLETED.json", "background.log", "background_job.json"}
        and "runtime_cache" not in path.relative_to(ROOT).parts
    ]
    _save(
        ROOT / "COMPLETED.json",
        {
            "passed": True,
            "signature": _hash(ROOT / "contract.json"),
            "new_fits": 652,
            "models": 30,
            "portfolio_cases": 372,
            "excluded_items": [7, 9],
            "artifact_sha256": {str(p): _hash(p) for p in paths},
        },
        immutable=True,
    )
    _verify_complete()
    _status("completed", new_fits=652, models=30, portfolio_cases=372, excluded_items=[7, 9])


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["prepare", "smoke-only", "execute", "verify-only", "status"]:
        parser.add_argument("--" + flag, action="store_true")
    args = parser.parse_args()
    if args.status:
        print((ROOT / "status.json").read_text() if (ROOT / "status.json").exists() else "Not started")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.verify_only or (ROOT / "COMPLETED.json").exists():
            _verify_complete()
            return
        if not (args.prepare or args.smoke_only or args.execute):
            parser.error("Choose --prepare, --smoke-only, --execute, --verify-only or --status")
        contract = _prepare()
        if args.prepare:
            _status(
                "prepared",
                new_component_fits=648,
                audit_replay_fits=4,
                models=30,
                approved_items=contract["approved_items"],
                excluded_items=[7, 9],
            )
            return
        try:
            _execute(contract, args.smoke_only)
        except BaseException as error:
            _status("failed", error=f"{type(error).__name__}: {error}", fallback_weights=False)
            raise


if __name__ == "__main__":
    _main()
