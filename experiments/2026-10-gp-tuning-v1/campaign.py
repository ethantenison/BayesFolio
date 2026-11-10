"""Reproducible three-track GP tuning campaign with one bounded monthly fit queue.

Research orchestration only. Frozen data, baseline portfolios and original GP
builders remain read-only. Registered rank, iteration cap, feature mask and causal
training window vary; allocator economics do not. Returns are decimal discounted
excess, covariance decimal squared and NAV dollars. No trading or git mutations.
Usage: uv run --locked --no-sync python <this_file> --prepare|--pilot|--fit|--verify
Run run_study.py for the complete fit/evaluate/report workflow.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import io
import json
import multiprocessing
import os
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS"]:
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-gp-tuning-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-gp-tuning-cache")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tuning_worker import fit_job, initialize  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FRAMEWORK = REPO / "experiments/2026-10-portfolio-evaluation-v1"
PREVIOUS = FRAMEWORK / "monthly_extension_v1/outputs"
DATA = REPO / "experiments/2026-10-monthly-data-extension-v1/outputs"
OUTPUT = HERE / "outputs"
SEEDS = [27, 127, 227]
RANKS = [1, 2, 5]
ITERATIONS = [5, 15, 36, 75]
ALPHA = [0.5, 0.0, 0.375, 0.25]


def digest(path: Path) -> str:
    """Return a local artifact's SHA-256 fingerprint.

    Args:
        path: Exact file path, never a glob or directory.

    Returns:
        Lowercase hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hashes(values: dict) -> None:
    """Reject any changed protected evidence before reuse.

    Args:
        values: Absolute file names mapped to expected SHA-256 fingerprints.
    """
    for name, expected in values.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Protected evidence changed: {name}")


def save(path: Path, value: dict | bytes, *, immutable: bool = True) -> None:
    """Write one new campaign artifact atomically without altering prior research.

    Args:
        path: Destination strictly within this campaign's outputs directory.
        value: JSON object or already encoded deterministic artifact bytes.
        immutable: Reject changed existing evidence; false only for live status.
    """
    if not path.resolve().is_relative_to(OUTPUT):
        raise ValueError("Only new campaign outputs are writable")
    payload = value if isinstance(value, bytes) else (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Conflicting immutable evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def snapshot(path: Path, arrays: dict) -> None:
    """Save immutable named numerical arrays without pickle.

    Args:
        path: New campaign NPZ destination.
        arrays: Named finite numeric arrays and explicit string/date axes.
    """
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    save(path, buffer.getvalue())


def read_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load local named arrays without executable pickle content.

    Args:
        path: Existing NPZ artifact.

    Returns:
        Independent named array copies.
    """
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def _register(source: dict, name: str, rank: int, maxiter: int,
              rolling_months: int | None = None, features: dict | None = None) -> dict:
    config = copy.deepcopy(source)
    config.pop("old_run_root", None)
    config.update(rank=rank, maxiter=maxiter, rolling_months=rolling_months)
    command = config["command"]
    for flag, value in [("--mtgp-rank", str(rank)), ("--maxiter", str(maxiter)),
                        ("--output-root", str(OUTPUT / "runs")), ("--run-id", name)]:
        command[command.index(flag) + 1] = value
    if features is not None and features.get("custom_etf_cols") is not None:
        etf, macro = features["custom_etf_cols"], features["custom_macro_cols"]
        config["inputs"] = ["t_index", *etf, *macro]
        for flag, values in [("--custom-etf-cols", etf), ("--custom-macro-cols", macro)]:
            command[command.index(flag) + 1] = ",".join(values)
    return config


def prepare() -> dict:
    """Freeze the approved candidate registry, data and two cached component panels.

    Returns:
        Auditable campaign contract: 2,352 new fits plus four parity audits, all
        within the approved roughly 2,500-fit budget. No portfolios are promoted.
    """
    contract_path = OUTPUT / "contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        verify_hashes(contract["input_sha256"])
        verify_hashes(contract["prepared_sha256"])
        return contract
    seal_path = PREVIOUS / "COMPLETED.json"
    seal = json.loads(seal_path.read_text())
    if not seal["passed"]:
        raise ValueError("Completed protected comparison required")
    protected = {**seal["input_sha256"], **seal["artifact_sha256"], str(seal_path): digest(seal_path)}
    verify_hashes(protected)
    source = json.loads((PREVIOUS / "fit_contract.json").read_text())["configs"]
    specifications = json.loads((HERE / "track3_complements/components.json").read_text())["components"]
    configs = {}
    for family, config in [("broad", source["broad14"]), ("compact", source["compact5"])]:
        for rank in RANKS:
            for maxiter in ITERATIONS:
                name = f"{family}_r{rank}_i{maxiter}_exp"
                configs[name] = _register(config, name, rank, maxiter)
    for spec in specifications:
        name = spec["modelname"]
        configs[name] = _register(source["broad14" if spec["source_family"] == "broad" else "compact5"],
                                  name, spec["rank"], spec["maxiter"], spec["rolling_months"], spec)
    if len(configs) != 30 or len(specifications) != 6:
        raise ValueError("Exactly 24 grid plus six complementary components registered")
    reference = read_arrays(PREVIOUS / "paths.npz")
    features = pd.read_parquet(DATA / "dataset/global.parquet")
    for config in configs.values():
        if config["tickers"] != reference["assets"].tolist() or config["dates"] != reference["dates"].tolist():
            raise ValueError("Common universe/date axis changed")
        if not np.isfinite(features[config["inputs"]].to_numpy()).all():
            raise ValueError("Non-finite registered feature; no filling")
        if "hy_spread_chg_1p" in config["inputs"] or "hy_spread" in config["inputs"]:
            raise ValueError("Excluded raw/change credit features cannot be reintroduced")
    risk_old = read_arrays(FRAMEWORK / "uncertainty_rank_v1/outputs/input_panel.npz")["risk_budget"]
    risk_new = read_arrays(PREVIOUS / "new_forecasts.npz")["risk_budget"]
    context = {name: reference[name] for name in ["dates", "assets", "seeds", "actual", "risk_free"]}
    context.update(risk_budget=np.r_[risk_old, risk_new], baseline_mean=reference["mean"],
                   baseline_covariance=reference["covariance"], baseline_hard_weights=reference["gp_hard_rank_weights"],
                   baseline_scenario_weights=reference["gp_scenarios_weights"],
                   benchmark_schwab_total=reference["schwab_aggressive_total_returns"],
                   benchmark_momentum_total=reference["momentum_top7_total_returns"])
    snapshot(OUTPUT / "context.npz", context)
    tail = read_arrays(PREVIOUS / "new_forecasts.npz")
    cached = []
    for family in ["broad", "compact"]:
        if family == "broad":
            originals = [read_arrays(FRAMEWORK / f"full_rerun_v1/outputs/analytic/global_e6c41f2020fd/s{s}.npz")
                         for s in SEEDS]
            first_mean, first_cov = np.stack([row["mean"] for row in originals]), np.stack([
                row["covariance"] for row in originals])
        else:
            original = read_arrays(FRAMEWORK / "approved_scope_v1/outputs/components/joint_existing_us_mask.npz")
            first_mean, first_cov = original["mean"], original["covariance"]
        name = f"{family}_r2_i15_exp"
        snapshot(OUTPUT / "components" / f"{name}.npz", {
            "mean": np.concatenate([first_mean, tail[f"{family}_mean"]], axis=1),
            "covariance": np.concatenate([first_cov, tail[f"{family}_covariance"]], axis=1),
            **{key: reference[key] for key in ["dates", "assets", "seeds"]},
        })
        cached.append(name)
    for path in [HERE / "track3_complements/components.json", FRAMEWORK / "CURRENT_BASELINES.json",
                 FRAMEWORK / "CURRENT_BENCHMARKS.json"]:
        protected[str(path)] = digest(path)
    contract = {
        "schema": "bayesfolio.gp_tuning.campaign.v1", "created": "2026-10-25", "configs": configs,
        "input_sha256": protected, "cached_components": cached, "complements": [s["modelname"] for s in specifications],
        "new_fits": 2352, "parity_audit_refits": 4, "new_fit_budget": 2500, "all_component_month_fits": 2520,
        "seeds": SEEDS, "dates": reference["dates"].tolist(), "maximum_fit_workers": 14,
        "rank_grid": RANKS, "maxiter_grid": ITERATIONS,
        "objectives_maximize": ["conventional_sharpe", "compounded_total_return", "cross_sectional_rank_IC"],
        "blend_weights": "constant across all months and seeds for each preregistered recipe; no monthly adaptation",
        "selection": "first16 origins only; later12 temporal validation; all28 exploratory; no pristine holdout claim",
        "minimum_effective_hard_rank": 7, "hard_rank_cap": None, "scenario_minimum_effective": 10,
        "scenario_cap": .25, "CVaR": False, "fallback": False, "constraint_relaxation": False,
        "production_or_baseline_promotion": False,
        "prepared_sha256": {str(path): digest(path) for path in
                            [OUTPUT / "context.npz", *sorted((OUTPUT / "components").glob("*.npz"))]},
    }
    save(contract_path, contract)
    return contract


def _jobs(contract: dict, *, audit: bool = False) -> list[dict]:
    if audit:
        specs = [(name, 27, index) for name in contract["cached_components"] for index in [0, 27]]
    else:
        names = [name for name in contract["configs"] if name not in contract["cached_components"]]
        names.sort(key=lambda name: (name not in contract["complements"], contract["configs"][name]["maxiter"], name))
        specs = [(name, seed, index) for name in names for seed in SEEDS for index in range(28)]
    return [{"component": name, "seed": seed, "index": index, "date": contract["dates"][index],
             "audit": audit} for name, seed, index in specs]


def _checkpoint(job: dict) -> Path:
    folder = "audit_checkpoints" if job["audit"] else "fit_checkpoints"
    return OUTPUT / folder / f"{job['component']}_s{job['seed']}_{job['date']}.json"


def _check_record(record: dict, job: dict, config: dict) -> None:
    verify_hashes(record["artifact_sha256"])
    result = record["result"]
    count = config["rolling_months"] or 84 + job["index"]
    if (result["rank"], result["maxiter"], result["train_months"], result["global_window_index"]) != (
        config["rank"], config["maxiter"], count, job["index"]
    ):
        raise ValueError("Completed fit controls differ")
    if any(row["iterations"] > config["maxiter"] for row in result["optimizer"]):
        raise ValueError("Actual optimizer iterations exceeded cap")


def _assemble(contract: dict, component: str) -> None:
    destination = OUTPUT / "components" / f"{component}.npz"
    if destination.exists():
        return
    rows = []
    for seed in SEEDS:
        months = []
        for index, date in enumerate(contract["dates"]):
            job = {"component": component, "seed": seed, "index": index, "date": date, "audit": False}
            path = _checkpoint(job)
            if not path.exists():
                return
            record = json.loads(path.read_text())
            _check_record(record, job, contract["configs"][component])
            months.append(read_arrays(Path(record["result"]["paths"]["analytic"])))
        rows.append(months)
    context = read_arrays(OUTPUT / "context.npz")
    for months in rows:
        for index, value in enumerate(months):
            np.testing.assert_array_equal(value["assets"], context["assets"])
            np.testing.assert_allclose(value["actual"], context["actual"][index], rtol=0, atol=1e-15)
    snapshot(destination, {key: np.array([[row[key] for row in months] for months in rows])
                           for key in ["mean", "covariance"]} |
             {key: context[key] for key in ["dates", "assets", "seeds"]})


def fit_campaign(contract: dict, workers: int, *, pilot_only: bool = False) -> None:
    """Execute the single bounded fit queue with resumable independent checkpoints.

    Args:
        contract: Preregistered immutable components and protected input evidence.
        workers: One process per monthly fit, at most fourteen, one thread each.
        pilot_only: Stop after four baseline parity audits and one fit per new spec.
    """
    if not 1 <= workers <= 14:
        raise ValueError("Only one shared pool of at most fourteen fitting processes")
    verify_hashes(contract["input_sha256"])
    audit_jobs, jobs = _jobs(contract, audit=True), _jobs(contract)
    representatives = [job for job in jobs if job["seed"] == 27 and job["index"] == 27]
    phases = [audit_jobs, representatives] if pilot_only else [audit_jobs, representatives, jobs]
    started = time.monotonic()
    complete = len(list((OUTPUT / "fit_checkpoints").glob("*.json")))
    failures = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize,
                             initargs=(contract["configs"], str(FRAMEWORK))) as pool:
        for phase_index, phase in enumerate(phases):
            pending = []
            for job in phase:
                if _checkpoint(job).exists():
                    _check_record(json.loads(_checkpoint(job).read_text()), job, contract["configs"][job["component"]])
                else:
                    pending.append(job)
            print(f"Fit phase {phase_index + 1}: {len(pending)} pending, pool={workers}", flush=True)
            attempts = {}

            def submit(job: dict) -> Future:
                key = _checkpoint(job).stem
                existing = len(list((OUTPUT / "runs" / key).glob("attempt_*")))
                attempts[key] = existing + 1
                run = OUTPUT / "runs" / key / f"attempt_{existing + 1:02d}"
                return pool.submit(fit_job, {**job, "run": str(run)})

            futures = {submit(job): job for job in pending}
            while futures:
                future = next(as_completed(futures))
                job = futures.pop(future)
                try:
                    result = future.result()
                except Exception:
                    failures += 1
                    save(OUTPUT / "failures" / f"{_checkpoint(job).stem}_attempt{attempts[_checkpoint(job).stem]}.json",
                         {"job": job, "traceback": traceback.format_exc()})
                    # Retry the same model/seed/settings once, in a fresh output directory.
                    # No smaller window, changed features, or substitute portfolio is allowed.
                    reserved = len(list((OUTPUT / "failures").glob("*.json"))) + contract["new_fits"] + 4
                    if attempts[_checkpoint(job).stem] < 2 and reserved < contract["new_fit_budget"]:
                        futures[submit(job)] = job
                        continue
                    raise
                record = {"result": result,
                          "artifact_sha256": {str(Path(path).resolve()): digest(Path(path))
                                              for path in result["paths"].values()},
                          "implementation_sha256": {str(HERE / "tuning_worker.py"): digest(HERE / "tuning_worker.py")}}
                _check_record(record, job, contract["configs"][job["component"]])
                if job["audit"]:
                    expected = read_arrays(OUTPUT / "components" / f"{job['component']}.npz")
                    actual = read_arrays(Path(result["paths"]["analytic"]))
                    for key in ["mean", "covariance"]:
                        np.testing.assert_allclose(actual[key], expected[key][0, job["index"]], rtol=0, atol=1e-8)
                save(_checkpoint(job), record)
                if not job["audit"]:
                    complete += 1
                    _assemble(contract, job["component"])
                status = {"phase": "pilot_fits" if pilot_only else "fitting", "complete_new_fits": complete,
                          "target_new_fits": contract["new_fits"], "audit_fits": len(list(
                              (OUTPUT / "audit_checkpoints").glob("*.json"))),
                          "workers": workers, "failures_this_invocation": failures,
                          "elapsed_seconds_this_invocation": time.monotonic() - started,
                          "last_completed": {k: job[k] for k in ["component", "seed", "date"]},
                          "component_panels": len(list((OUTPUT / "components").glob("*.npz")))}
                save(OUTPUT / "status.json", status, immutable=False)
                if complete % 14 == 0 or job["audit"] or pilot_only:
                    print(json.dumps(status), flush=True)
    verify_hashes(contract["input_sha256"])
    if not pilot_only:
        for component in contract["configs"]:
            _assemble(contract, component)
        if len(list((OUTPUT / "components").glob("*.npz"))) != 30:
            raise ValueError("Missing complete component panels")
        save(OUTPUT / "FITS_COMPLETED.json", {"passed": True, "new_fits": complete,
                                               "audit_fits": 4, "total_new_monthly_fits": complete + 4})


def verify() -> None:
    """Verify protected inputs and all completed fit artifacts without launching jobs."""
    contract = json.loads((OUTPUT / "contract.json").read_text())
    verify_hashes(contract["input_sha256"])
    completed = 0
    for job in _jobs(contract) + _jobs(contract, audit=True):
        if _checkpoint(job).exists():
            _check_record(json.loads(_checkpoint(job).read_text()), job, contract["configs"][job["component"]])
            completed += 1
    print(f"Verified protected inputs and {completed} completed monthly fits", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ["prepare", "pilot", "fit", "verify"]:
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.verify:
        # Completed checkpoints are immutable/atomic, so read-only verification
        # does not need to interrupt the writer's campaign lock.
        verify()
    else:
        with (OUTPUT / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            contract = prepare()
            if args.pilot or args.fit:
                fit_campaign(contract, args.workers, pilot_only=args.pilot)
            else:
                print(f"Prepared {len(contract['configs'])} components, 2352 new fits +4 audits", flush=True)
