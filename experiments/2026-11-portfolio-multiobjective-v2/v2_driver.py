"""Root-owned finite policy queue and registered adaptive round construction.

Isolated research orchestration, no brokerage or trading. One single-threaded
compute pool at a time handles complete policy cases. Decimal returns/weights,
squared-decimal covariance. Confirmation cannot run before its frozen shortlist.
Usage: uv run --locked --no-sync python <this_file> --controls|--round1
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

import numpy as np  # noqa: E402
from v2_common import (  # noqa: E402
    ASSETS,
    HERE,
    SEEDS,
    V1,
    digest,
    read,
    save,
    snapshot,
    status,
    verify_hashes,
)
from v2_evaluate import evaluate, initialize, policy  # noqa: E402
from v2_experts import REGISTERED, cache_inventory, load_cached_panel, registered_configs  # noqa: E402
from v2_protocol import context, prepare  # noqa: E402


def inputs(stage: str = "development") -> str:
    """Freeze explicit source panels with no confirmation outcomes in dev inputs.

    Args:
        stage: Development41 or unlocked confirmation53.

    Returns:
        Exact isolated NPZ input path.
    """
    if stage == "confirmation" and not (HERE / "SHORTLIST_FROZEN.json").exists():
        raise ValueError("Confirmation is still locked")
    count = 41 if stage == "development" else 53
    source = context()
    panels = [load_cached_panel(name) for name in REGISTERED]
    for panel in panels:
        if not panel["available"][:, :count].all():
            raise ValueError("Necessary expert forecasts are unavailable")
    path = HERE / "outputs" / (stage + "_inputs.npz")
    snapshot(path, {"expert_mean": np.stack([p["mean"][:, :count] for p in panels]),
        "expert_covariance": np.stack([p["covariance"][:, :count] for p in panels]),
        "experts": np.array(REGISTERED), "assets": np.array(ASSETS), "seeds": np.array(SEEDS),
        "actual": source["actual"][:count], "rf": source["rf"][:count], "dates": source["dates"][:count],
        "historical_covariance": source["historical_covariance"][:count],
        "historical_EW_variance": source["historical_EW_variance"][:count]})
    return str(path)


def controls() -> list[dict]:
    """Return unchanged prediction/covariance recipes with matched MGK exclusion.

    Returns:
        Two content-addressed mandatory controls, NE7, causal14-EW ceiling.
    """
    reference = np.asarray(read(V1 / "CONFIG.json")["selected_recipe"]["fractions"])
    older = np.array([[.5, .5, 0], [1, 0, 0], [.625, .375, 0], [.75, .25, 0]])
    return [policy("v1_three_expert_reference_MGK0", reference, control=True),
            policy("older_two_expert_incumbent_MGK0", older, control=True)]


def distance(summary: dict, anchors: dict | None = None) -> float:
    """Apply the immutable fixed-anchor balanced shortfall distance.

    Args:
        summary: CAGR and signed maximum drawdown summary.
        anchors: Explicit original development-control anchor values.

    Returns:
        Lower-is-better dimensionless balanced distance.
    """
    anchors = read(HERE / "ANCHORS.json") if anchors is None else anchors
    return float(np.hypot(max(0., (anchors["aspiration_CAGR"] - summary["cagr"]) / .02),
                          max(0., (-summary["max_drawdown"] - anchors["aspiration_drawdown_loss"]) / .02)))


def run_cases(cases: list[dict], *, stage: str = "development", workers: int = 14) -> list[dict]:
    """Execute registered policy cases through the sole global compute queue.

    Args:
        cases: Exact complete policy records from a frozen batch or shortlist.
        stage: Locked development or explicitly unlocked final confirmation.
        workers: Globally at most14 single-thread workers.

    Returns:
        Complete or explicitly infeasible case records. Failures remain evidence.
    """
    prepare()
    if not 1 <= workers <= 14:
        raise ValueError("One at-most14-worker pool required")
    path = inputs(stage)
    results = {}
    started = time.monotonic()
    with (HERE / "compute_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ProcessPoolExecutor(max_workers=min(workers, len(cases)),
                mp_context=multiprocessing.get_context("spawn"), initializer=initialize,
                initargs=(path, stage)) as queue:
            pending = {queue.submit(evaluate, case): case for case in cases}
            for future in as_completed(pending):
                case = pending[future]
                try:
                    record = future.result()
                except Exception:
                    save(HERE / "outputs/failures" / f"{stage}_{case['id']}.json",
                         {"case": case, "stage": stage, "traceback": traceback.format_exc()})
                    raise
                results[case["id"]] = record
                status(stage + "_policy_evaluation", "Complete registered batch; inspect development evidence",
                       batch_completed=len(results), batch_total=len(cases),
                       complete=sum(r["complete"] for r in results.values()),
                       infeasible=sum(not r["complete"] for r in results.values()),
                       fit_attempts_used=len(list((HERE / "outputs/fits").glob("**/started.json"))),
                       elapsed_seconds=time.monotonic() - started, latest=case["id"],
                       confirmation_locked=not (HERE / "SHORTLIST_FROZEN.json").exists())
    return [results[case["id"]] for case in cases]


def initialize_controls(workers: int = 14) -> dict:
    """Audit cache compatibility, freeze GP scope, and instantiate fixed anchors.

    Args:
        workers: Shared queue worker maximum.

    Returns:
        Immutable control records and pre-search anchor values.
    """
    prepare()
    configs = registered_configs()
    save(HERE / "GP_VARIANTS.json", {"registered_at_utc": read(HERE / "PREREGISTRATION.json")["frozen_at_utc"],
        "specs": configs, "new_specs_count": 1,
        "hypothesis": "Compactrank1 may alter useful rank diversity at identicalfeatures/window/maxiter; "
                      "retunecompleteensembleandretainrank2control", "kernel_or_prior_variants": []})
    save(HERE / "CACHE_INVENTORY.json", cache_inventory())
    save(HERE / "MATH_CONTRACT.json", {"files_sha256": {str(path): digest(path) for path in
        [HERE / "v2_policy.py", HERE / "v2_metrics.py", HERE / "v2_evaluate.py", HERE / "v2_common.py",
         HERE / "PROTOCOL_CLARIFICATIONS.json", HERE / "GP_VARIANTS.json"]}})
    save(HERE / "CONTROL_REGISTRATION.json", {"cases": controls(),
        "registered_at_utc": read(HERE / "PREREGISTRATION.json")["frozen_at_utc"],
        "new_configurations": 0})
    rows = run_cases(controls(), workers=workers)
    if not all(row["complete"] for row in rows):
        raise ValueError("An unchanged matched control is infeasible; inspect before selecting replacements")
    baseline = rows[0]["summary"]
    anchors = {"source_control": rows[0]["id"], "source_panel": "development41 Jun2022-Oct2025",
        "reference_CAGR": baseline["cagr"], "reference_drawdown_loss": -baseline["max_drawdown"],
        "aspiration_CAGR": baseline["cagr"] + .02,
        "aspiration_drawdown_loss": max(0., -baseline["max_drawdown"] - .02),
        "CAGR_scale": .02, "drawdown_scale": .02,
        "hypervolume_reference_CAGR": baseline["cagr"] - .10,
        "hypervolume_reference_drawdown_loss": -baseline["max_drawdown"] + .10,
        "not_a_personal_loss_limit_or_return_forecast": True}
    save(HERE / "ANCHORS.json", anchors)
    save(HERE / "CONTROLS.json", {"rows": rows, "anchors": anchors,
                                 "best_development_control": min(rows, key=lambda r: (
                                     distance(r["summary"], anchors), r["id"]))["id"]})
    return {"controls": rows, "anchors": anchors}


def register_round(number: int, cases: list[dict], hypothesis: str) -> list[dict]:
    """Freeze an adaptive menu before any scoring in that round.

    Args:
        number: One through four.
        cases: Complete fixed policies, deduplicated against previous cases.
        hypothesis: Evidence-driven expected mechanism and matched controls.

    Returns:
        Exact newly admitted policies only.
    """
    prepare()
    verify_hashes(read(HERE / "MATH_CONTRACT.json")["files_sha256"])
    if number not in range(1, 5):
        raise ValueError("Round budget exceeded")
    seen = {case["id"] for case in controls()}
    for path in sorted((HERE / "rounds").glob("ROUND*.json")):
        if path.name == f"ROUND{number:02d}.json":
            return read(path)["cases"]
        seen.update(case["id"] for case in read(path)["cases"])
    unique = {}
    for case in cases:
        if case["id"] not in seen:
            unique.setdefault(case["id"], case)
    admitted = list(unique.values())
    if len(admitted) > 256 or len(seen) - 2 + len(admitted) > 1024:
        raise ValueError("Complete-policy budget exceeded")
    save(HERE / "rounds" / f"ROUND{number:02d}.json", {"round": number,
        "registered_at_utc": datetime.now(UTC).isoformat(), "hypothesis": hypothesis,
        "cases": admitted, "new_configurations": len(admitted),
        "prior_new_configurations": len(seen) - 2, "development_only": True,
        "input_sha256": digest(HERE / "outputs/development_inputs.npz"),
        "selection_rule_sha256": digest(HERE / "SELECTION_RULE.json"),
        "anchors_sha256": digest(HERE / "ANCHORS.json")})
    return admitted


def round1() -> list[dict]:
    """Register structural concentration/risk/covariance experiments before blends.

    Returns:
        At most256 complete policies spanning both unchanged prediction recipes.
    """
    cases = []
    for reference in controls():
        fractions = np.asarray(reference["prediction_fractions"])
        label = reference["label"]
        for nea in [3, 4, 5, 6, 7, 8, 10, 12, 14]:
            for risk in [None, .5, .75, 1., 1.25, 1.5, 2.]:
                cases.append(policy(f"{label}_NE{nea}_risk{risk}", fractions, nea=nea, risk=risk))
        for cov in [[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0], [0, 0, 0, 1.],
                    [.25, .25, 0, .5], [.25, .25, .5, 0]]:
            for nea in [5, 7, 10]:
                for risk in [.75, 1.]:
                    cases.append(policy(f"{label}_cov{cov}_NE{nea}_risk{risk}", fractions,
                                        nea=nea, risk=risk, covariance=cov))
        for cap in [.6, .8]:
            for nea in [5, 7, 10]:
                for risk in [None, 1.]:
                    cases.append(policy(f"{label}_equity{cap}_NE{nea}_risk{risk}", fractions,
                                        nea=nea, risk=risk, equity_cap=cap))
    return register_round(1, cases, "Use cached fixedprediction controls to isolate NE3..14, causal14-EW "
        "ceiling multipliers, observed/historicalPSD covariance combinations, and static equitycaps. "
        "No prediction retuning or GPfit confounds; no latest12 scoring.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--round1", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    if args.controls:
        initialize_controls(args.workers)
    if args.round1:
        if not (HERE / "CONTROLS.json").exists():
            initialize_controls(args.workers)
        result = run_cases(round1(), workers=args.workers)
        print(f"Round1 complete {sum(row['complete'] for row in result)}/{len(result)} feasible policies")
    if args.confirm:
        shortlist = read(HERE / "SHORTLIST_FROZEN.json")
        result = run_cases(shortlist["cases"], stage="confirmation", workers=args.workers)
        print(f"Final confirmation complete {sum(row['complete'] for row in result)}/{len(result)}")
