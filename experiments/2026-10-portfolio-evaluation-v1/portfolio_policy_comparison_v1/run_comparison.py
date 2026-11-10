"""Automatically record both requested portfolio policies for all 499 GP fits.

Usage from repository root:
  .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/portfolio_policy_comparison_v1/run_comparison.py --follow
The one-shot dependency follower waits for the existing sealed rerun to finish
and release its writer lock before launching scoring workers. No GP fitting,
upstream writes, position caps, CVaR, or automatic promotion. Use --status for
read-only progress, --execute after upstream completion, --pilot-only for a
completed-baseline numerical check, or --verify-only after this study finishes.
"""

# Import paths and numerical thread controls precede NumPy/solver imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parent
FITS = FRAMEWORK / "full_rerun_v1/outputs"
ROOT = HERE / "outputs"
sys.path.insert(0, str(FRAMEWORK / "full_rerun_v1"))

import full_rerun
import numpy as np
import pandas as pd
import policy_workers
from policy_math import PolicyAllocator, _policies, _score_allocations

_save, _hashes, _verify = full_rerun._save, full_rerun._hashes, full_rerun._verify


def _status(stage: str, **values) -> None:
    value = {"stage": stage, "updated_unix": time.time(), **values}
    _save(ROOT / "status.json", value)
    print(json.dumps(value), flush=True)


def _prepare() -> dict:
    path = ROOT / "policy_contract.json"
    if path.exists():
        contract = json.loads(path.read_text())
        _verify(contract["source_sha256"])
        return contract
    upstream = json.loads((FITS / "contract.json").read_text())
    if len(upstream["candidates"]) != 499 or upstream["required_fresh_fits"] != 35928:
        raise ValueError("Wrong upstream study")
    _verify(upstream["source_sha256"])
    sources = [
        *HERE.glob("*.py"),
        HERE / "specification.json",
        FITS / "contract.json",
        FITS / "context.json",
        FITS / "context_arrays.npz",
        FRAMEWORK / "metrics.py",
        FRAMEWORK / "selection_refresh_v1/refresh_math.py",
        FRAMEWORK / "full_rerun_v1/full_rerun.py",
    ]
    contract = {
        "schema": "bayesfolio.paired_effective_count_policy.v1",
        "upstream_contract_sha256": hashlib.sha256((FITS / "contract.json").read_bytes()).hexdigest(),
        "candidates": upstream["candidates"],
        "configuration_count": 499,
        "risk_budget_only_minimum_effective": None,
        "joint_minimum_effective": 10,
        "family_conditional_portfolio_minimum_effective": 3,
        "ETF_position_cap": None,
        "CVaR": False,
        "long_only": True,
        "fully_invested": True,
        "risk_budget": upstream["risk_budget"],
        "fallback_weights": False,
        "constraint_relaxation": False,
        "historical_risk_policies_configurations": 499,
        "GP_risk_policies_joint_configurations": 35,
        "GP_cross_family_covariance_fabrication": False,
        "new_GP_fits": 0,
        "upstream_fits_restarted": False,
        "expected_configuration_policy_rows": 1068,
        "worker_ceiling": upstream["worker_ceiling"],
        "evaluation_status": upstream["evaluation_status"],
        "automatic_promotion": False,
        "source_sha256": _hashes(sources),
    }
    _save(path, contract, immutable=True)
    return contract


def _ready() -> bool:
    status = FITS / "status.json"
    if not (FITS / "COMPLETED.json").exists() or not status.exists():
        return False
    if json.loads(status.read_text())["stage"] != "completed":
        return False
    with (FITS / "run.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def _wait_upstream(contract: dict, follow: bool) -> dict:
    while not _ready():
        if not follow:
            raise RuntimeError("Full 499-fit study is still running; use --follow to queue scoring")
        progress = json.loads((FITS / "status.json").read_text())
        _status(
            "waiting_for_full_rerun",
            full_configuration_count=499,
            upstream_progress=progress,
            joint_minimum_effective=10,
            family_minimum_effective=3,
            other_variant_minimum_effective=None,
        )
        time.sleep(30)
    seal = json.loads((FITS / "COMPLETED.json").read_text())
    if not seal["passed"] or seal["signature"] != contract["upstream_contract_sha256"]:
        raise ValueError("Upstream completion seal/contract differs")
    _verify(seal["artifact_sha256"])
    return seal


def _job(candidate: dict, upstream_seal: dict) -> dict:
    path = FITS / "evaluation_checkpoints" / f"{candidate['configuration']}.json"
    stamp = json.loads(path.read_text())
    if stamp["signature"] != upstream_seal["signature"]:
        raise ValueError("Upstream seed-array checkpoint signature differs")
    expected = {name: digest for name, digest in stamp["result"]["artifact_sha256"].items() if name.endswith(".npz")}
    if len(expected) != 3:
        raise ValueError("Three complete exact-covariance seed arrays required")
    return {"candidate": candidate, "upstream": str(FITS), "output": str(ROOT), "expected_analytic_sha": expected}


def _execute(contract: dict, follow: bool) -> None:
    upstream_seal = _wait_upstream(contract, follow)
    _verify(contract["source_sha256"])
    signature = hashlib.sha256((ROOT / "policy_contract.json").read_bytes()).hexdigest()
    context, anchor = full_rerun._context_load()
    summaries, pending = [], []
    for candidate in contract["candidates"]:
        path = ROOT / "checkpoints" / f"{candidate['configuration']}.json"
        if path.exists():
            stamp = json.loads(path.read_text())
            if stamp["signature"] != signature:
                raise ValueError("Policy checkpoint signature mismatch")
            _verify(stamp["result"]["artifact_sha256"])
            _verify(stamp["result"]["upstream_sha256"])
            summaries.extend(stamp["result"]["summaries"])
        else:
            pending.append(candidate)
    with ProcessPoolExecutor(
        max_workers=contract["worker_ceiling"],
        mp_context=multiprocessing.get_context("spawn"),
        initializer=policy_workers.initialize,
        initargs=(context, anchor),
    ) as pool:
        futures = {
            pool.submit(policy_workers.evaluate, _job(candidate, upstream_seal)): candidate for candidate in pending
        }
        completed = 499 - len(pending)
        for future in as_completed(futures):
            value = future.result()
            _save(
                ROOT / "checkpoints" / f"{value['configuration']}.json",
                {"signature": signature, "result": value},
                immutable=True,
            )
            summaries.extend(value["summaries"])
            completed += 1
            _status(
                "recording_both_portfolios",
                completed_configurations=completed,
                total=499,
                joint_minimum_effective=10,
                family_minimum_effective=3,
                other_variant_minimum_effective=None,
            )
    frame = pd.DataFrame(summaries).sort_values(["family", "configuration", "covariance_policy", "variant"])
    if len(frame) != 1068 or frame.configuration.nunique() != 499:
        raise ValueError("All 499 configurations and both policies were not recorded")
    for _, group in frame.groupby(["configuration", "covariance_policy"]):
        if set(group.variant) != {"risk_budget_only", "risk_budget_plus_effective"}:
            raise ValueError("Missing policy comparison")
    frame.to_csv(ROOT / "all_portfolio_policies.csv", index=False)
    baseline = pd.read_csv(FITS / "all_499_configuration_metrics.csv")
    columns = [
        "minimum_effective",
        "conventional_sharpe",
        "compounded_total_return",
        "complete_seeds",
        "solved_months",
        "mean_effective_count",
        "largest_weight",
        "annualized_excess_vol",
        "max_drawdown_total_nav",
    ]
    # Drop old allocation columns from the source table to avoid treating its
    # superseded joint minimum-three policy as an approved joint-minimum-ten run.
    keep = [name for name in baseline.columns if not name.startswith(("historical_risk_no_cap_", "gp_risk_no_cap_"))]
    wide = baseline[keep].set_index("configuration")
    for (source, variant), group in frame.groupby(["covariance_policy", "variant"]):
        selected = group.set_index("configuration")[columns].add_prefix(f"{source}__{variant}__")
        wide = wide.join(selected, how="left")
    wide.reset_index().to_csv(ROOT / "all_499_policy_metrics.csv", index=False)
    _verify(contract["source_sha256"])
    outputs = [
        *ROOT.glob("checkpoints/*.json"),
        ROOT / "all_portfolio_policies.csv",
        ROOT / "all_499_policy_metrics.csv",
    ]
    _save(
        ROOT / "COMPLETED.json",
        {
            "passed": True,
            "signature": signature,
            "configurations": 499,
            "configuration_policy_rows": len(frame),
            "artifact_sha256": _hashes(outputs),
        },
        immutable=True,
    )
    _status(
        "completed",
        configurations=499,
        configuration_policy_rows=len(frame),
        joint_minimum_effective=10,
        family_minimum_effective=3,
        risk_budget_only_minimum_effective=None,
        CVaR=False,
        ETF_position_cap=None,
    )


def _pilot() -> None:
    context, _ = full_rerun._context_load()
    source = FRAMEWORK / "parallel_pipeline_v1/outputs/baseline_parallel_v1/analytic/s27.npz"
    with np.load(source, allow_pickle=False) as values:
        mean, covariance = values["mean"].copy(), values["covariance"].copy()
    scores = {}
    for risk_source, risk in [("historical", context["historical_covariance"]), ("GP", covariance)]:
        for variant, minimum in _policies("joint").items():
            records = PolicyAllocator(risk, context["risk_budget"], minimum).allocate(mean)
            scores[f"{risk_source}__{variant}"] = _score_allocations(
                context,
                mean,
                records,
                {"historical_control_not_GP": context["historical_covariance"], "exact_observed_GP": covariance},
            )
    _save(
        ROOT / "pilot.json",
        {
            "source_sha256": _hashes([source]),
            "scores": scores,
            "purpose": "numerical_check_on_completed_reference_not_full_499_results",
        },
        immutable=True,
    )
    _status(
        "pilot_complete",
        policies=4,
        results={
            key: {"status": value["status"], "solved_months": value["solved_months"]} for key, value in scores.items()
        },
    )


def _verify_complete() -> None:
    contract = json.loads((ROOT / "policy_contract.json").read_text())
    seal = json.loads((ROOT / "COMPLETED.json").read_text())
    if hashlib.sha256((ROOT / "policy_contract.json").read_bytes()).hexdigest() != seal["signature"]:
        raise ValueError("Completed policy contract differs")
    _verify(contract["source_sha256"])
    _verify(seal["artifact_sha256"])
    paths = list(ROOT.glob("checkpoints/*.json"))
    if len(paths) != 499:
        raise ValueError("Expected 499 policy checkpoints")
    for path in paths:
        stamp = json.loads(path.read_text())
        if stamp["signature"] != seal["signature"]:
            raise ValueError("Per-configuration policy signature differs")
        _verify(stamp["result"]["artifact_sha256"])
        _verify(stamp["result"]["upstream_sha256"])
    print(
        "Verified both portfolios for all 499 configurations: joint minimum 10, family minimum 3, no CVaR or ETF caps"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["follow", "execute", "pilot-only", "status", "verify-only"]:
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    if args.status:
        print((ROOT / "status.json").read_text() if (ROOT / "status.json").exists() else "Not queued")
        return
    if not (args.follow or args.execute or args.pilot_only or args.verify_only):
        parser.error("Choose --follow, --execute, --pilot-only, --status, or --verify-only")
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.verify_only or (ROOT / "COMPLETED.json").exists():
            _verify_complete()
            return
        contract = _prepare()
        try:
            if args.pilot_only:
                _pilot()
            else:
                _execute(contract, args.follow)
                _verify_complete()
        except BaseException as error:
            _status("failed", error=f"{type(error).__name__}: {error}", fallback_weights=False)
            raise


if __name__ == "__main__":
    _main()
