"""Month-parallel baseline GP production followed by exact-covariance evaluation.

New research artifacts only; old data, sources and outputs remain read-only.
Jobs are seed/month independent, with a barrier before portfolio solves and a
second barrier before batched evaluation. All financial values use decimals.

Usage from repository root:
    PYTHONPATH=experiments/2026-10-portfolio-evaluation-v1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/parallel_pipeline_v1/run_pipeline.py --execute
    --smoke-only runs one fit/parity gate. --verify-only checks a completed run.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import workers
from evaluate import evaluate_model
from metrics import calibration_summary, paired_block_bootstrap, portfolio_performance, project_portfolios
from run_evaluation import _tables

from artifacts import Reader, aligned_matrix, load_context, load_covariance, verify_legacy_audit

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REPO = HERE.parents[2]
LEGACY = REPO / "experiments/2026-10-repaired-global-feature-selection"
SEEDS = [27, 127, 227]
MODEL = "global_e6c41f2020fd"
POLICIES = ["legacy_scenario_maxret", "risk_rank_cap35", "risk_rank_no_cap"]


def _bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _save(path: Path, value: object, *, immutable: bool = False) -> None:
    payload = _bytes(value)
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Conflicting immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _verify_hashes(hashes: dict[str, str]) -> None:
    for name, digest in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source/artifact changed: {name}")


def worker_limit(requested: int, cores: int, peak_rss: int = 0) -> int:
    """Bound the one global pool by cores minus two and observed memory demand.

    Args:
        requested: Requested process ceiling, not a separate budget per seed.
        cores: Available CPU count.
        peak_rss: Smoke process peak bytes; a 25% safety margin is applied.

    Returns:
        At most fourteen processes and a 48-GiB aggregate research memory budget.
    """
    ceiling = max(1, min(14, cores - 2))
    if not 1 <= requested <= ceiling:
        raise ValueError(f"Worker count must be between 1 and {ceiling}")
    memory = int(48 * 1024**3 / (peak_rss * 1.25)) if peak_rss else requested
    return max(1, min(requested, memory))


def _prepare(root: Path, requested: int) -> tuple[dict, dict]:
    reader = Reader()
    verified = verify_legacy_audit(reader, LEGACY)
    context = load_context(reader, LEGACY)
    old = LEGACY / "followup/runs" / f"{MODEL}_s27"
    manifest = reader.read(old / "manifest.json")
    inputs = reader.read(LEGACY / "followup/results/all_joint/exact_inputs.json")[MODEL]
    if len(inputs) != 15 or "t_index" not in inputs or "hy_spread_chg_1p" in inputs or "IWM" not in context["assets"]:
        raise ValueError("Accepted task/input baseline changed")
    for path in [*HERE.glob("*.py"), *PARENT.glob("*.py"), PARENT / "specification.json", REPO / "uv.lock",
                 LEGACY / "run_hy_rank2_iter15.py", LEGACY / "month_parallel_worker.py"]:
        reader.read(path)
    command = [sys.executable, *shlex.split(manifest["command"])]
    for flag, value in [("--output-root", str(root / "runs")), ("--run-id", "s27")]:
        command[command.index(flag) + 1] = value
    for flag, expected in [("--mtgp-rank", "2"), ("--maxiter", "15"), ("--gp-experiment", "signed_lkj_eta_2")]:
        if command[command.index(flag) + 1] != expected:
            raise ValueError(f"Baseline control differs: {flag}")
    config = {"command": command, "inputs": inputs, "tickers": context["assets"], "dates": context["dates"],
              "legacy_root": str(LEGACY), "historical_covariance": context["historical_covariance"].tolist(),
              "allocation_policies": POLICIES}
    contract = {"schema": "bayesfolio.research.parallel_exact_evaluation.v1", "model": MODEL,
        "config": config, "seeds": SEEDS, "rank": 2, "maxiter": 15, "requested_workers": requested,
        "source_sha256": reader.hashes, "verified_old_audit_files": verified,
        "required_independent_fits": 72, "initial_training_months": 84, "scored_months": 24,
        "phases": ["parallel_month_fits", "parallel_month_allocations", "batched_evaluation"],
        "analytic_export": {"observation_noise": True, "fast_pred_var": False,
                            "solves": "dense_cholesky_at_fitted_hyperparameters",
                            "hyperparameter_uncertainty_integrated": False},
        "risk_policy": {"common_covariance": "training_only_LedoitWolf", "research_variance_budget":
            "equal_weight_common_variance_each_origin", "effective_assets_minimum": 3,
            "caps": [0.35, 1.0], "live_investor_risk_target": False},
        "fallback_weights": False, "constraint_relaxation": False, "turnover_objective": False,
        "protected_old_studies_modified": False, "evaluation_status": "exploratory_reused_months",
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "torch", "botorch", "gpytorch", "cvxpy"]}}
    reader.verify()
    _save(root / "contract.json", contract, immutable=True)
    return contract, context


def _jobs(root: Path, dates: list[str]) -> list[dict]:
    return [{"seed": seed, "date": date, "index": index, "run": str(root / "runs" / f"s{seed}")}
            for seed in SEEDS for index, date in enumerate(dates)]


def _stage(root: Path, pool: ProcessPoolExecutor, jobs: list[dict], stage: str, signature: str) -> list[dict]:
    completed, pending = [], []
    for job in jobs:
        stamp = root / "checkpoints" / f"{stage}_s{job['seed']}_{job['date']}.json"
        if stamp.exists():
            cached = json.loads(stamp.read_text())
            if cached["signature"] != signature:
                raise ValueError(f"Checkpoint source signature differs: {stamp}")
            _verify_hashes(cached["artifact_sha256"])
            completed.append(cached["result"])
        else:
            pending.append(job)
    function = workers.fit_job if stage == "fit" else workers.portfolio_job
    futures = {pool.submit(function, job): job for job in pending}
    errors = []
    for future in as_completed(futures):
        job = futures[future]
        try:
            result = future.result()
            stamp = root / "checkpoints" / f"{stage}_s{job['seed']}_{job['date']}.json"
            _save(stamp, {"signature": signature, "result": result,
                "artifact_sha256": workers.hash_files([Path(value) for value in result["paths"].values()])},
                  immutable=True)
            completed.append(result)
        except Exception as error:
            errors.append({"seed": job["seed"], "date": job["date"], "error": str(error)})
        _save(root / "status.json", {"state": "running" if not errors else "failed", "stage": stage,
            "completed_jobs": len(completed), "required_jobs": len(jobs), "errors": errors})
        print(f"{stage}: {len(completed)}/{len(jobs)}", flush=True)
    if errors:
        raise RuntimeError(f"Monthly stage failed; valid checkpoints retained, no fallbacks: {errors}")
    return sorted(completed, key=lambda row: (row["seed"], row["date"]))


def _parity(root: Path, result: dict, assets: list[str]) -> dict:
    old = LEGACY / "followup/runs" / f"{MODEL}_s{result['seed']}"
    differences = {}
    for kind, columns in [("predictions", ["y_true", "y_pred", "y_std"]), ("scenarios", assets)]:
        path = Path(result["paths"][kind])
        fresh = pd.read_csv(path)
        baseline = pd.read_csv(old / path.relative_to(Path(root / "runs" / f"s{result['seed']}")))
        if kind == "predictions":
            np.testing.assert_array_equal(fresh[["date", "asset_id"]], baseline[["date", "asset_id"]])
        delta = float(np.max(np.abs(fresh[columns].to_numpy() - baseline[columns].to_numpy())))
        differences[kind] = delta
        if delta > 1e-8:
            raise ValueError(f"Unchanged legacy readout failed parallel-fit parity: {kind} {delta}")
    evidence = {"passed": True, "seed": result["seed"], "date": result["date"],
        "legacy_parallel_parity_atol": 1e-8, "max_abs_differences": differences,
        "analytic_vs_legacy_mean_max_abs": result["legacy_vs_analytic_mean_max_abs"],
        "analytic_vs_legacy_std_max_abs": result["legacy_vs_analytic_std_max_abs"]}
    _save(root / "parity_smoke.json", evidence, immutable=True)
    return evidence


def _assemble(root: Path, fits: list[dict], portfolios: list[dict], context: dict) -> None:
    for seed in SEEDS:
        run = root / "runs" / f"s{seed}"
        selected = [row for row in fits if row["seed"] == seed]
        predictions = pd.concat([pd.read_csv(row["paths"]["analytic_predictions"]) for row in selected],
                                ignore_index=True)
        prediction_path = run / "gp_window_predictions.csv"
        predictions.to_csv(prediction_path, index=False)
        covariances = []
        for row in selected:
            with np.load(row["paths"]["analytic_covariance"], allow_pickle=False) as arrays:
                if arrays["date"].item() != row["date"] or arrays["assets"].tolist() != context["assets"]:
                    raise ValueError("Monthly analytic artifact ordering differs")
                covariances.append(arrays["covariance"].copy())
        folder = root / "analytic"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / f"s{seed}.npz", dates=context["dates"], assets=context["assets"],
                 mean=aligned_matrix(predictions, "y_pred", context["dates"], context["assets"]),
                 covariance=np.stack(covariances))
        _save(folder / f"s{seed}.json", {"kind": "observed_predictive", "source": "analytic_gp",
            "units": "discounted_excess_decimal_squared",
            "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "computation": "dense_cholesky_fast_pred_var_false_at_fitted_hyperparameters"}, immutable=True)
        pd.concat([pd.read_csv(row["paths"]["weights"]) for row in portfolios if row["seed"] == seed],
                  ignore_index=True).to_csv(run / "allocation_weights.csv", index=False)
        pd.DataFrame([{key: row[key] for key in ["date", "train_dates", "train_rows", "fit_seed"]}
                      for row in selected]).to_csv(run / "window_training_history.csv", index=False)


def _evaluate(root: Path, context: dict) -> dict:
    reader = Reader()
    results = []
    contract = reader.read(root / "contract.json")
    for name, digest in contract["source_sha256"].items():
        reader.hashes[name] = digest
    for seed in SEEDS:
        run = {"run": root / "runs" / f"s{seed}"}
        predictions = reader.read(run["run"] / "gp_window_predictions.csv")
        for key, column in [("actual", "y_true"), ("mean", "y_pred"), ("std", "y_std")]:
            run[key] = aligned_matrix(predictions, column, context["dates"], context["assets"])
        np.testing.assert_allclose(run["actual"], context["actual"], rtol=0, atol=1e-12)
        allocations = reader.read(run["run"] / "allocation_weights.csv")
        run["legacy_weights"] = aligned_matrix(allocations[allocations.policy.eq("legacy_scenario_maxret")],
                                               "weight", context["dates"], context["assets"])
        covariance, meta = load_covariance(reader, run, context, "analytic", root / "analytic", seed)
        result = evaluate_model(run, context, covariance, meta)
        result.update(seed=seed, model=MODEL, input_count_excluding_time=14)
        for policy in ["risk_rank_cap35", "risk_rank_no_cap"]:
            weights = aligned_matrix(allocations[allocations.policy.eq(policy)], "weight",
                                     context["dates"], context["assets"])
            discounted = np.sum(weights * run["actual"], axis=1)
            total = (1 + discounted) * (1 + context["risk_free"]) - 1
            path = {"total": total.tolist(), "arithmetic_excess": (total - context["risk_free"]).tolist(),
                    "discounted_excess_legacy": discounted.tolist()}
            result["monthly_portfolio_paths"][policy] = path
            result["portfolios"].append({"portfolio": policy, **portfolio_performance(total, context["risk_free"]),
                "mean_effective_asset_count": float(np.mean(1 / np.sum(weights**2, axis=1))),
                "largest_weight": float(weights.max()), "role": "forecast_driven", "allocator": policy})
            for cov, source in [(covariance, meta["source"]),
                                (context["historical_covariance"], "common_training_ledoit_wolf_not_gp")]:
                projected = project_portfolios(run["actual"], run["mean"], cov, weights[:, None, :])
                row = {"portfolio": policy, "comparison_role": "adaptive_decision_diagnostic",
                    "covariance_source": source,
                    **calibration_summary({key: value[:, 0] for key, value in projected.items()})}
                result["portfolio_distributions"].append(row)
        results.append(result)
    comparisons = {}
    for policy in ["legacy_maxret", "linear_rank", "risk_rank_cap35", "risk_rank_no_cap"]:
        for benchmark in ["equal_weight", "fixed_60_40"]:
            left = np.array([item["monthly_portfolio_paths"][policy]["arithmetic_excess"] for item in results]).T
            right = np.array([item["monthly_portfolio_paths"][benchmark]["arithmetic_excess"] for item in results]).T
            comparison = paired_block_bootstrap(left, right)
            comparison["mean_total_return_difference"] = float(np.mean([
                np.prod(1 + np.array(item["monthly_portfolio_paths"][policy]["total"]))
                - np.prod(1 + np.array(item["monthly_portfolio_paths"][benchmark]["total"])) for item in results]))
            comparisons[f"{policy}_vs_{benchmark}"] = comparison
    reader.verify()
    output = root / "evaluation"
    _save(output / "results.json", results, immutable=True)
    _save(output / "paired_comparisons.json", comparisons, immutable=True)
    for name, table in _tables(results).items():
        path = output / f"{name}.csv"
        payload = table.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Conflicting evaluation table: {path}")
        path.write_bytes(payload)
    _save(output / "manifest.json", {"source_sha256": reader.hashes, "covariance": "analytic_observed_dense_cholesky",
        "cost_basis": "gross_costs_not_supplied", "evaluation_status": "exploratory_reused_months",
        "seeds_are_not_independent_market_months": True, "automatic_promotion": False}, immutable=True)
    reader.verify()
    return {"seed_runs": 3, "scored_months": 24,
            "joint_nll": [item["joint_distribution"]["joint_nll"] for item in results]}


def execute(root: Path, requested: int, smoke_only: bool) -> None:
    """Run parity gate, independent fits, independent allocations and batch scores.

    Args:
        root: New versioned pipeline output directory.
        requested: One global worker ceiling, up to cores minus two/fourteen.
        smoke_only: Stop after one verified seed/month instead of launching 72.
    """
    worker_limit(requested, os.cpu_count() or 16)
    contract, context = _prepare(root, requested)
    if (root / "COMPLETED.json").exists():
        verify(root)
        return
    signature = hashlib.sha256((root / "contract.json").read_bytes()).hexdigest()
    config = copy.deepcopy(contract["config"])
    config["signature"] = signature
    jobs = _jobs(root, context["dates"])
    started = time.perf_counter()
    mp = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=mp, initializer=workers.initialize, initargs=(config,)) as pool:
        smoke = _stage(root, pool, jobs[:1], "fit", signature)[0]
    _parity(root, smoke, context["assets"])
    if smoke_only:
        print(json.dumps({"smoke_passed": True, "seconds": time.perf_counter() - started}), flush=True)
        return
    count = worker_limit(requested, os.cpu_count() or 16, smoke["peak_rss_bytes"])
    fit_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=count, mp_context=mp, initializer=workers.initialize,
                             initargs=(config,)) as pool:
        fits = _stage(root, pool, jobs, "fit", signature)
        fit_seconds = time.perf_counter() - fit_start
        _verify_hashes(contract["source_sha256"])
        portfolio_start = time.perf_counter()
        portfolios = _stage(root, pool, jobs, "portfolio", signature)
        portfolio_seconds = time.perf_counter() - portfolio_start
    _assemble(root, fits, portfolios, context)
    evaluated = _evaluate(root, context)
    _verify_hashes(contract["source_sha256"])
    timing = {"workers": count, "fitting_stage_wall_seconds_excluding_smoke": fit_seconds,
        "portfolio_stage_wall_seconds": portfolio_seconds,
        "execution_wall_seconds_this_invocation": time.perf_counter() - started,
        "fit_checkpoint_count": len(fits), "portfolio_checkpoint_count": len(portfolios),
        "total_dense_readout_worker_seconds": sum(row["analytic_readout_seconds"] for row in fits),
        "max_dense_vs_legacy_mean_difference": max(row["legacy_vs_analytic_mean_max_abs"] for row in fits),
        "max_dense_vs_legacy_std_difference": max(row["legacy_vs_analytic_std_max_abs"] for row in fits)}
    _save(root / "timing.json", timing, immutable=True)
    artifacts = [*root.glob("checkpoints/*.json"), *root.glob("analytic/*"), *root.glob("evaluation/*"),
                 *root.glob("runs/s*/*.csv"), root / "parity_smoke.json", root / "timing.json"]
    seal = {"passed": True, "signature": signature, "artifact_sha256": workers.hash_files(artifacts), **evaluated}
    _save(root / "COMPLETED.json", seal, immutable=True)
    _save(root / "status.json", {"state": "completed", **evaluated})
    verify(root)
    print(json.dumps({"completed": True, "timing": timing, **evaluated}), flush=True)


def verify(root: Path) -> None:
    """Verify completed source, artifacts and every monthly checkpoint without fits.

    Args:
        root: Previously completed pipeline directory.
    """
    contract = json.loads((root / "contract.json").read_text())
    seal = json.loads((root / "COMPLETED.json").read_text())
    if hashlib.sha256((root / "contract.json").read_bytes()).hexdigest() != seal["signature"]:
        raise ValueError("Completed contract signature differs")
    _verify_hashes(contract["source_sha256"])
    _verify_hashes(seal["artifact_sha256"])
    for path in root.glob("checkpoints/*.json"):
        stamp = json.loads(path.read_text())
        if stamp["signature"] != seal["signature"]:
            raise ValueError("Monthly checkpoint signature differs")
        _verify_hashes(stamp["artifact_sha256"])
    print("Verified: sources, 72 fits, 72 allocations and analytic evaluation", flush=True)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--run-name", default="baseline_parallel_v1")
    args = parser.parse_args()
    if not (args.execute or args.verify_only or args.smoke_only):
        parser.error("Choose --execute, --smoke-only or --verify-only")
    root = HERE / "outputs" / args.run_name
    if root.resolve().parent != (HERE / "outputs").resolve():
        parser.error("Run name must be a single directory name")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.verify_only:
            verify(root)
        else:
            execute(root, args.workers, args.smoke_only)


if __name__ == "__main__":
    _main()
