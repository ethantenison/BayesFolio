"""Resumable chronological feature/GP-setting/blend research orchestration.

Read frozen local inputs; write only this experiment's new outputs. Monthly
fits run independently before portfolio selection/evaluation. Return units are
decimal discounted excess, covariance squared decimals and NAV dollars.
Usage: uv run --locked --no-sync python <this_file> --prepare|--pilot|--execute|--status|--verify
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
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-gp-tuning-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-gp-tuning-cache")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from selection_math import catalog, recipe, select_static  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from worker import fit_job, initialize  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAMPAIGN = HERE.parent / "2026-10-gp-tuning-v1"
FRAMEWORK = HERE.parent / "2026-10-portfolio-evaluation-v1"
DATA = HERE.parent / "2026-10-monthly-data-extension-v1/outputs"
OUTPUT = HERE / "outputs"
SEEDS = [27, 127, 227]
sys.path[:0] = [str(CAMPAIGN / "three_way_joint_v1"), str(CAMPAIGN / "track1_scenarios")]
from joint_math import RankAllocator  # noqa: E402


def digest(path: Path) -> str:
    """Return an exact artifact fingerprint.

    Args:
        path: Existing local file.

    Returns:
        SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    """Read a local JSON evidence object.

    Args:
        path: Existing JSON file.

    Returns:
        Decoded evidence object.
    """
    return json.loads(path.read_text())


def arrays(path: Path) -> dict:
    """Read finite forecast arrays without permitting pickle execution.

    Args:
        path: Existing NPZ artifact.

    Returns:
        Independent copies of named arrays.
    """
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def save(path: Path, value: dict | list | bytes, *, mutable: bool = False) -> None:
    """Atomically save isolated outputs, rejecting changed sealed evidence.

    Args:
        path: Destination strictly within this experiment's outputs.
        value: JSON object/list or exact encoded artifact bytes.
        mutable: Only live status may be overwritten.
    """
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Only new experiment outputs are writable")
    content = value if isinstance(value, bytes) else (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists() and not mutable:
        if path.read_bytes() != content:
            raise ValueError(f"Changed immutable evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def snapshot(path: Path, values: dict) -> None:
    """Save immutable named numeric/string arrays.

    Args:
        path: New NPZ destination.
        values: Arrays with explicit date/asset/seed axes.
    """
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    save(path, buffer.getvalue())


def verify_hashes(hashes: dict) -> None:
    """Reject changed protected files before reuse or continuation.

    Args:
        hashes: Absolute paths mapped to SHA-256 fingerprints.
    """
    for name, expected in hashes.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Protected input/source changed: {name}")


def status(phase: str, **fields: object) -> None:
    """Write mutable progress, never used as evidence or a selection input.

    Args:
        phase: Current workflow phase.
        **fields: JSON-serializable counters/details.
    """
    payload = {"phase": phase, **fields}
    save(OUTPUT / "status.json", payload, mutable=True)
    print(json.dumps(payload), flush=True)


def _context(frame: pd.DataFrame, dates: list[str], assets: list[str]) -> dict:
    target = frame.pivot(index="date", columns="asset_id", values="y_excess_lead").reindex(columns=assets)
    irx = pd.read_parquet(DATA / "raw/combined/INDEX_IRX.parquet")["Adj Close"]
    daily = (irx / 100 / 252).reindex(pd.bdate_range(irx.index.min(), irx.index.max())).ffill()
    budget, rf = [], []
    for date in dates:
        point = pd.Timestamp(date)
        training = target.loc[target.index < point].to_numpy()
        if not np.isfinite(training).all():
            raise ValueError("No invalid historical targets may enter the risk ceiling")
        equal = np.full(15, 1 / 15)
        budget.append(float(equal @ LedoitWolf().fit(training).covariance_ @ equal))
        end = point + pd.offsets.BMonthEnd(1)
        interval = daily.loc[(daily.index > point) & (daily.index <= end)]
        if interval.empty or interval.isna().any() or end > daily.index.max():
            raise ValueError("Complete realized risk-free outcomes required")
        rf.append(float(np.expm1(interval.sum())))
    previous = arrays(CAMPAIGN / "outputs/context.npz")
    actual = target.loc[pd.to_datetime(dates)].to_numpy()
    np.testing.assert_array_equal(actual[24:], previous["actual"])
    np.testing.assert_allclose(rf[24:], previous["risk_free"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(budget[24:], previous["risk_budget"], rtol=1e-9, atol=1e-13)
    return {"dates": np.array(dates), "assets": np.array(assets), "seeds": np.array(SEEDS),
            "actual": actual, "risk_free": np.array(rf), "risk_budget": np.array(budget)}


def prepare() -> dict:
    """Freeze the data audit, bounded model catalog and temporal experiment design.

    Returns:
        Immutable contract with exact fit counts and protected source fingerprints.
        No GP fitting or test-outcome-based selection occurs in preparation.
    """
    destination = OUTPUT / "CONTRACT.json"
    if destination.exists():
        contract = read(destination)
        verify_hashes(contract["input_sha256"])
        verify_hashes(contract["prepared_sha256"])
        return contract
    old = read(CAMPAIGN / "outputs/contract.json")
    if not read(CAMPAIGN / "outputs/COMPLETED.json")["passed"]:
        raise ValueError("Original campaign must be sealed before reuse")
    frame = pd.read_parquet(DATA / "dataset/global.parquet")
    frame.date = pd.to_datetime(frame.date)
    dates = sorted(frame.date.unique())[60:]
    dates = [str(pd.Timestamp(date).date()) for date in dates]
    if len(dates) != 52 or dates[0] != "2022-05-31" or dates[-1] != "2026-08-31":
        raise ValueError("Unexpected complete frozen chronology")
    save(OUTPUT / "scored_dates_52.txt", ("\n".join(dates) + "\n").encode())
    configs, protected = {}, {}
    for spec in catalog():
        source = old["configs"][f"{spec['source']}_r2_i15_exp"]
        config = copy.deepcopy(source)
        config.update(dates=dates, rank=spec["rank"], maxiter=spec["maxiter"],
                      rolling_months=spec["rolling_months"], reuse=spec["reuse"], source=spec["source"])
        command = config["command"]
        for flag, value in [("--mtgp-rank", str(spec["rank"])), ("--maxiter", str(spec["maxiter"])),
                            ("--run-id", spec["id"]), ("--output-root", str(OUTPUT / "runs")),
                            ("--scored-date-file", str(OUTPUT / "scored_dates_52.txt"))]:
            command[command.index(flag) + 1] = value
        if "etf" in spec:
            config["inputs"] = ["t_index", *spec["etf"], *spec["macro"]]
            for flag, values in [("--custom-etf-cols", spec["etf"]), ("--custom-macro-cols", spec["macro"])]:
                command[command.index(flag) + 1] = ",".join(values)
        if (set(config["inputs"]) & {"hy_spread", "hy_spread_chg_1p", "hy_spread_chg_1m"}
                or "t_index" not in config["inputs"]):
            raise ValueError("Excluded credit feature or missing retained time input")
        if spec["reuse"]:
            path = CAMPAIGN / "outputs/components" / f"{spec['reuse']}.npz"
            cached = arrays(path)
            np.testing.assert_array_equal(cached["dates"], dates[24:])
            np.testing.assert_array_equal(cached["assets"], config["tickers"])
            np.testing.assert_array_equal(cached["seeds"], SEEDS)
            protected[str(path)] = digest(path)
        configs[spec["id"]] = config
    columns = sorted({column for config in configs.values() for column in config["inputs"]})
    if (len(frame) != 112 * 15 or frame.duplicated(["date", "asset_id"]).any()
            or not frame.groupby("date").size().eq(15).all()
            or not np.isfinite(frame[[*columns, "y_excess_lead"]].to_numpy()).all()):
        raise ValueError("Frozen eligible-feature coverage is incomplete; no imputation allowed")
    audit = {column: {"missing": int(frame[column].isna().sum()),
                      "zero_fraction": float(frame[column].eq(0).mean()),
                      "minimum": float(frame[column].min()), "maximum": float(frame[column].max()),
                      "unique_values": int(frame[column].nunique())} for column in columns}
    if any(row["unique_values"] < 2 for row in audit.values()):
        raise ValueError("Constant eligible predictor needs investigation")
    save(OUTPUT / "FEATURE_AUDIT.json", {
        "passed": True, "rows": len(frame), "monthly_dates": 112, "features": audit,
        "rich_economic_features": 25, "rich_with_t_index": 26,
        "excluded_july_features": ["hy_spread", "hy_spread_chg_1p"],
        "no_new_clipping_or_imputation": True, "prior_source_definitions_reused": True,
        "limitations": ["Latest-vintage CPI, not real-time macro vintages.",
                        "Existing extra one-period predictor lag preserved.",
                        "Original target convention and fixed15-task cross-sectional ranks preserved.",
                        "Catalog informed by previously inspected history; not untouched confirmation."]})
    snapshot(OUTPUT / "context52.npz", _context(frame, dates, next(iter(configs.values()))["tickers"]))
    dependencies = [DATA / "dataset/global.parquet", DATA / "raw/combined/INDEX_IRX.parquet",
                    DATA / "COMPLETED.json", CAMPAIGN / "outputs/context.npz",
                    CAMPAIGN / "outputs/contract.json", CAMPAIGN / "outputs/COMPLETED.json",
                    FRAMEWORK / "CURRENT_BASELINES.json", FRAMEWORK / "CURRENT_BENCHMARKS.json",
                    HERE / "README.md", *sorted(HERE.glob("*.py")), CAMPAIGN / "tuning_worker.py",
                    CAMPAIGN / "three_way_joint_v1/joint_math.py",
                    CAMPAIGN / "track1_scenarios/case_evaluator.py",
                    DATA.parent / "extend_data.py"]
    for folder in [REPO / "bayesfolio/engine/forecast/gp",
                   HERE.parent / "2026-07-international-equity-etf-features",
                   HERE.parent / "2026-06-task-covariance",
                   HERE.parent / "2026-06-portfolio-optimization",
                   FRAMEWORK / "parallel_pipeline_v1"]:
        dependencies.extend(sorted(folder.glob("*.py")))
    protected.update({str(path.resolve()): digest(path) for path in dependencies})
    new_fits = sum((24 if c["reuse"] else 52) * 3 for c in configs.values())
    if len(configs) != 19 or new_fits != 2208:
        raise ValueError("Bounded nineteen-expert, 2208-fit registration changed")
    contract = {
        "schema": "bayesfolio.chronological_gp_selection.v1", "configs": configs, "dates": dates,
        "seeds": SEEDS, "new_fits": new_fits, "audit_fits": 2, "attempt_budget": 2500,
        "maximum_workers": 14, "initial_meta_months": 24, "outer_blocks": [7, 7, 7, 7],
        "static_blend_within_each_test_block_and_all_seeds": True,
        "also_test_initial_recipe_frozen_all28": True, "maximum_experts_per_family": 3,
        "selection": "all expert/feature/setting/weight choices use earlier OOS ledger only",
        "selection_objectives": ["sharpe", "total_return", "rank_ic"],
        "covariance": "fixed50/50 broad-r2-i15/compact-r2-i15",
        "allocator": {"method": "unchanged_hard_rank", "minimum_effective_assets": 7,
                      "cap": None, "risk_budget": "unchanged causal historical-EW variance ceiling"},
        "CVaR": False, "fallback": False, "monthly_weight_tuning": False, "promotion": False,
        "historical_holdout_status": "retrospective only; candidate design inherits inspected history",
        "input_sha256": protected,
        "prepared_sha256": {str(path): digest(path) for path in
                            [OUTPUT / "context52.npz", OUTPUT / "FEATURE_AUDIT.json", OUTPUT / "scored_dates_52.txt"]}}
    save(destination, contract)
    status("prepared", experts=19, new_fits=2208, separate_audits=2)
    return contract


def _jobs(contract: dict, audit: bool = False) -> list[dict]:
    if audit:
        triples = [("broad_r2_i15_exp", 27, 0), ("compact_r2_i15_exp", 27, 27)]
    else:
        triples = [(name, seed, index) for name, config in contract["configs"].items()
                   for seed in SEEDS for index in range(-24, 28) if index < 0 or not config["reuse"]]
    return [{"component": name, "seed": seed, "index": index,
             "date": contract["dates"][index + 24], "audit": audit} for name, seed, index in triples]


def _checkpoint(job: dict) -> Path:
    folder = "audit_checkpoints" if job["audit"] else "fit_checkpoints"
    return OUTPUT / folder / f"{job['component']}_s{job['seed']}_{job['date']}.json"


def _check(record: dict, job: dict, config: dict) -> None:
    verify_hashes(record["artifact_sha256"])
    result = record["result"]
    if (result["rank"], result["maxiter"], result["train_months"], result["global_window_index"],
            result["seed"], result["component"]) != (
            config["rank"], config["maxiter"], config["rolling_months"] or 84 + job["index"],
            job["index"], job["seed"], job["component"]):
        raise ValueError("Fit controls/chronology mismatch")
    if record["contract_sha256"] != digest(OUTPUT / "CONTRACT.json"):
        raise ValueError("Fit checkpoint belongs to another experiment contract")


def fit_campaign(contract: dict, workers: int, *, pilot_only: bool = False) -> None:
    """Execute one bounded month-parallel queue with strict per-fit checkpoints.

    Args:
        contract: Frozen model/date/source registration.
        workers: At most14 processes, one Torch/BLAS thread per process.
        pilot_only: Stop after two parity audits and one early fit per expert.
    """
    if not 1 <= workers <= 14:
        raise ValueError("One shared pool of at most14 workers required")
    verify_hashes(contract["input_sha256"])
    jobs = _jobs(contract)
    pilots = [job for job in jobs if job["seed"] == 27 and job["index"] == -24]
    phases = [_jobs(contract, audit=True), pilots] + ([] if pilot_only else [jobs])
    started = time.monotonic()
    reserved_attempts = len(list((OUTPUT / "runs").glob("**/started.json")))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(contract["configs"], str(FRAMEWORK))) as pool:
        for phase in phases:
            pending = []
            for job in phase:
                if _checkpoint(job).exists():
                    _check(read(_checkpoint(job)), job, contract["configs"][job["component"]])
                else:
                    pending.append(job)
            attempts = {}

            def submit(job: dict):
                nonlocal reserved_attempts
                if reserved_attempts >= contract["attempt_budget"]:
                    raise RuntimeError("Registered attempt budget exhausted; no settings relaxed")
                key = _checkpoint(job).stem
                prior = len(list((OUTPUT / "runs" / key).glob("attempt_*")))
                if prior >= 2:
                    raise RuntimeError(f"Two identical attempts already failed: {key}; inspect preserved failures")
                reserved_attempts += 1
                attempts[key] = prior + 1
                return pool.submit(fit_job, {**job, "run": str(OUTPUT / "runs" / key / f"attempt_{prior + 1:02d}")})

            futures = {submit(job): job for job in pending}
            while futures:
                future = next(as_completed(futures))
                job = futures.pop(future)
                try:
                    result = future.result()
                except Exception:
                    key = _checkpoint(job).stem
                    save(OUTPUT / "failures" / f"{key}_attempt{attempts[key]}.json",
                         {"job": job, "traceback": traceback.format_exc()})
                    if attempts[key] < 2:
                        futures[submit(job)] = job
                        continue
                    raise
                record = {"contract_sha256": digest(OUTPUT / "CONTRACT.json"), "result": result,
                          "artifact_sha256": {str(Path(path).resolve()): digest(Path(path))
                                              for path in result["paths"].values()}}
                _check(record, job, contract["configs"][job["component"]])
                if job["audit"]:
                    cached = arrays(CAMPAIGN / "outputs/components" / f"{job['component']}.npz")
                    fitted = arrays(Path(result["paths"]["analytic"]))
                    for key in ["mean", "covariance"]:
                        np.testing.assert_allclose(fitted[key], cached[key][0, job["index"]], rtol=0, atol=1e-8)
                save(_checkpoint(job), record)
                completed = len(list((OUTPUT / "fit_checkpoints").glob("*.json")))
                status("pilot" if pilot_only else "fitting", completed_new_fits=completed,
                       total_new_fits=contract["new_fits"], fit_count_percent=100 * completed / contract["new_fits"],
                       separate_audit_fits=len(list((OUTPUT / "audit_checkpoints").glob("*.json"))),
                       total_audit_fits=2, workers=workers,
                       failures=len(list((OUTPUT / "failures").glob("*.json"))),
                       elapsed_seconds=time.monotonic() - started, last_completed=job,
                       note="fit-count percentage is not runtime percentage")
    verify_hashes(contract["input_sha256"])
    if not pilot_only:
        _assemble(contract)
        save(OUTPUT / "FITS_COMPLETED.json", {"passed": True, "new_fits": contract["new_fits"],
                                               "separate_audit_fits": 2, "complete_experts": 19})


def _assemble(contract: dict) -> None:
    context = arrays(OUTPUT / "context52.npz")
    for name, config in contract["configs"].items():
        destination = OUTPUT / "components" / f"{name}.npz"
        if destination.exists():
            continue
        mean = np.full((3, 52, 15), np.nan)
        covariance = np.full((3, 52, 15, 15), np.nan)
        if config["reuse"]:
            cached = arrays(CAMPAIGN / "outputs/components" / f"{config['reuse']}.npz")
            mean[:, 24:], covariance[:, 24:] = cached["mean"], cached["covariance"]
        for job in [row for row in _jobs(contract) if row["component"] == name]:
            record = read(_checkpoint(job))
            _check(record, job, config)
            values = arrays(Path(record["result"]["paths"]["analytic"]))
            s, t = SEEDS.index(job["seed"]), job["index"] + 24
            np.testing.assert_array_equal(values["assets"], context["assets"])
            np.testing.assert_allclose(values["actual"], context["actual"][t], rtol=0, atol=1e-15)
            mean[s, t], covariance[s, t] = values["mean"], values["covariance"]
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError(f"Missing expert forecasts: {name}")
        np.linalg.cholesky(covariance)
        snapshot(destination, {"mean": mean, "covariance": covariance,
                               **{key: context[key] for key in ["dates", "assets", "seeds"]}})


def evaluate(contract: dict) -> None:
    """Freeze earlier-only selections, then evaluate subsequent full portfolios.

    Args:
        contract: Completed forecast campaign with immutable source registration.
    """
    from report import create_report

    if not read(OUTPUT / "FITS_COMPLETED.json")["passed"]:
        raise ValueError("Complete fits required before any portfolio comparison")
    verify_hashes(contract["input_sha256"])
    context = arrays(OUTPUT / "context52.npz")
    names = list(contract["configs"])
    components = [arrays(OUTPUT / "components" / f"{name}.npz") for name in names]
    means = np.stack([item["mean"] for item in components])
    covariance = .5 * (components[names.index("broad_r2_i15_exp")]["covariance"]
                       + components[names.index("compact_r2_i15_exp")]["covariance"])
    counts = np.zeros((4, len(names)), dtype=int)
    compact = np.array([32, 0, 24, 16])
    counts[:, names.index("compact_r2_i15_exp")] = compact
    counts[:, names.index("broad_r2_i15_exp")] = 64 - compact
    incumbent = recipe(counts)
    selections = []
    for fold, stop in enumerate([24, 31, 38, 45]):
        destination = OUTPUT / "selections" / f"fold_{fold}.json"
        ledger = {"dates": context["dates"][:stop].tolist(), "expert_ids": names,
                  "last_origin": str(context["dates"][stop - 1]),
                  "outcomes_available_by_test_origin": str(context["dates"][stop]),
                  "earlier_input_sha256": hashlib.sha256(b"".join(x.tobytes() for x in
                      [means[:, :, :stop], context["actual"][:stop], context["risk_free"][:stop],
                       covariance[:, :stop], context["risk_budget"][:stop]])).hexdigest()}
        if destination.exists():
            frozen = read(destination)
            if frozen["ledger"] != ledger or frozen["contract_sha256"] != digest(OUTPUT / "CONTRACT.json"):
                raise ValueError("Frozen cutoff belongs to changed earlier evidence")
        else:
            status("earlier_only_selection", fold=fold, training_forecast_months=stop, test_months=7)
            # Copy only earlier observations into the selector. The selector's
            # API has no full context, test outcomes, file paths or report handle.
            selected, records = select_static(means[:, :, :stop].copy(), context["actual"][:stop].copy(),
                context["risk_free"][:stop].copy(), covariance[:, :stop].copy(),
                context["risk_budget"][:stop].copy(), incumbent, RankAllocator().allocate)
            save(OUTPUT / "selections" / f"fold_{fold}_search.json", records)
            frozen = {"fold": fold, "ledger": ledger, "contract_sha256": digest(OUTPUT / "CONTRACT.json"),
                      "test_origins": context["dates"][stop:stop + 7].tolist(), "selected": selected,
                      "recipes_evaluated": len(records),
                      "search_sha256": digest(OUTPUT / "selections" / f"fold_{fold}_search.json")}
            save(destination, frozen)
        selections.append(frozen)
    # All four recipes must be sealed before producing test-period scores.
    save(OUTPUT / "SELECTIONS_FROZEN.json", {"passed": True, "folds": 4,
         "selection_sha256": {str(path): digest(path) for path in sorted((OUTPUT / "selections").glob("*.json"))}})
    status("portfolio_evaluation", frozen_cutoffs=4, full_test_months=28)
    create_report(contract, context, means, covariance, selections, incumbent)


def verify() -> None:
    """Verify all completed evidence without launching or mutating computations."""
    contract = read(OUTPUT / "CONTRACT.json")
    verify_hashes(contract["input_sha256"])
    verify_hashes(contract["prepared_sha256"])
    count = 0
    for job in _jobs(contract) + _jobs(contract, audit=True):
        if _checkpoint(job).exists():
            _check(read(_checkpoint(job)), job, contract["configs"][job["component"]])
            count += 1
    if (OUTPUT / "COMPLETED.json").exists():
        seal = read(OUTPUT / "COMPLETED.json")
        if not seal["passed"] or count != 2210:
            raise ValueError("Incomplete final experiment")
        verify_hashes(seal["artifact_sha256"])
    print(json.dumps({"verified_completed_fits_including_audits": count,
                      "experiment_complete": (OUTPUT / "COMPLETED.json").exists()}), flush=True)


def main() -> None:
    """Run a read-only status/verification or the bounded resumable experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ["prepare", "pilot", "execute", "status", "verify"]:
        mode.add_argument("--" + name, action="store_true")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read(OUTPUT / "status.json"), indent=2))
        return
    if args.verify:
        verify()
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = prepare()
        if args.pilot or args.execute:
            try:
                if not (OUTPUT / "FITS_COMPLETED.json").exists():
                    fit_campaign(contract, args.workers, pilot_only=args.pilot)
                if args.execute and not (OUTPUT / "COMPLETED.json").exists():
                    evaluate(contract)
                if args.execute:
                    verify()
            except Exception:
                status("failed", traceback=traceback.format_exc(),
                       completed_new_fits=len(list((OUTPUT / "fit_checkpoints").glob("*.json"))),
                       total_new_fits=contract["new_fits"], preserved_all_evidence=True)
                raise


if __name__ == "__main__":
    main()
