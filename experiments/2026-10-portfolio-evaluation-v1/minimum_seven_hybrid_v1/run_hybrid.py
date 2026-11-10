"""Approved cached U.S.-mean hybrid comparison at effective count seven.

Usage: .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/minimum_seven_hybrid_v1/run_hybrid.py --execute
       Use --verify-only to check completed evidence without allocation solves.
Research-only orchestration reads sealed predictions and fits no GPs. Only the
four U.S.-equity means are replaced; the full joint covariance remains intact.
No calibration, scaling, covariance blending, caps, CVaR or weight fallback.
Means/returns are decimals and covariances/budgets are decimal squared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
FRAMEWORK = HERE.parent
sys.path.extend(
    [
        str(FRAMEWORK),
        str(FRAMEWORK / "approved_scope_v1"),
        str(FRAMEWORK / "allocation_numerics_v2"),
        str(FRAMEWORK / "approved_scope_review_v2"),
    ]
)

import numpy as np  # noqa: E402
from feasibility import allocate_checked  # noqa: E402
from metrics import rank_ic  # noqa: E402
from review import _protected_inputs  # noqa: E402
from study_common import BASELINE, ROOT, SEEDS, US_MASK, _context  # noqa: E402
from study_math import _score, _scores  # noqa: E402


def replace_us_means(
    joint: np.ndarray, assets: list[str], local: np.ndarray, local_assets: list[str], targets: list[str]
) -> np.ndarray:
    """Replace only matched U.S.-equity means without touching other forecasts.

    Args:
        joint: Joint means, shape (3, 24, 15), discounted-excess decimal returns.
        assets: Unique ordered full-universe ticker labels.
        local: Family-only means, shape (3, 24, 4), same return units and dates.
        local_assets: Unique ordered labels for the family forecasts.
        targets: Exact four U.S.-equity tickers to replace.

    Returns:
        A copy retaining every non-U.S.-equity forecast bit for bit.
    """
    if (
        joint.shape != (3, 24, 15)
        or local.shape != (3, 24, 4)
        or len(assets) != 15
        or len(set(assets)) != 15
        or len(local_assets) != 4
        or len(set(local_assets)) != 4
        or set(local_assets) != set(targets)
        or len(set(targets)) != 4
        or not set(targets).issubset(assets)
        or not np.isfinite(joint).all()
        or not np.isfinite(local).all()
    ):
        raise ValueError("Exact finite aligned joint and U.S. forecast axes required")
    hybrid = joint.copy()
    for asset in targets:
        hybrid[:, :, assets.index(asset)] = local[:, :, local_assets.index(asset)]
    other = [index for index, asset in enumerate(assets) if asset not in targets]
    if not np.array_equal(hybrid[:, :, other], joint[:, :, other]):
        raise ValueError("Non-U.S.-equity forecasts changed")
    return hybrid


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(path: Path, value: object) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("New experiment outputs only")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError(f"Conflicting completed evidence: {path}")
    path.write_text(payload)


def _panel() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    context = _context()
    with np.load(ROOT / "components" / f"{BASELINE}.npz", allow_pickle=False) as joint:
        with np.load(ROOT / "components" / f"{US_MASK}.npz", allow_pickle=False) as local:
            if (
                joint["assets"].tolist() != context["assets"]
                or joint["dates"].tolist() != context["dates"]
                or local["dates"].tolist() != context["dates"]
            ):
                raise ValueError("Cached axes differ from the approved dates/universe")
            mean, covariance = joint["mean"].copy(), joint["covariance"].copy()
            hybrid = replace_us_means(
                mean, context["assets"], local["mean"], local["assets"].tolist(), context["families"]["us_equity"]
            )
    return context, mean, hybrid, covariance


def _job(arm: dict) -> dict:
    context, original, hybrid, covariance = _panel()
    mean = hybrid if arm["means"] == "hybrid_US_only" else original
    outcomes = []
    for offset, seed in enumerate(SEEDS):
        risk = covariance[offset] if arm["risk_source"] == "GP" else context["historical_covariance"]
        scores, penalty = _scores(mean[offset], context["baseline_mean"], "rank")
        allocation = allocate_checked(risk, context["risk_budget"], 7, penalty, scores)
        if allocation["status"] != "allocated":
            raise ValueError(f"Preflight blocked {arm['name']}: {allocation['preflight']}")
        scored = _score(context, mean[offset], risk, allocation["allocations"])
        if scored["solved_months"] != 24:
            raise ValueError(f"Incomplete allocation: {arm['name']}, seed {seed}")
        indices = [context["assets"].index(a) for a in context["families"]["us_equity"]]
        outcomes.append(
            {
                "seed": seed,
                **scored,
                "preflight": allocation["preflight"],
                "pooled_IC": float(rank_ic(context["actual"], mean[offset]).mean()),
                "US_equity_IC": float(rank_ic(context["actual"][:, indices], mean[offset][:, indices]).mean()),
            }
        )
    metrics = [
        "conventional_sharpe",
        "compounded_total_return",
        "annualized_excess_vol",
        "max_drawdown_total_nav",
        "mean_effective_count",
        "largest_weight",
    ]
    summary = {**arm, "complete_seeds": 3, "solved_months": 72, "minimum_effective_assets": 7}
    for metric in metrics:
        values = [row["performance"][metric] for row in outcomes]
        summary[metric] = float(np.mean(values))
        summary[f"seed_sd_{metric}"] = float(np.std(values, ddof=1))
    summary.update(
        pooled_IC=float(np.mean([row["pooled_IC"] for row in outcomes])),
        US_equity_IC=float(np.mean([row["US_equity_IC"] for row in outcomes])),
        risk_binding_months=sum(
            month["predicted_variance"] / month["variance_budget"] >= 1 - 1e-5
            for row in outcomes
            for month in row["months"]
        ),
    )
    result = {"policy": _read(HERE / "policy.json"), "arm": arm, "summary": summary, "seeds": outcomes}
    _save(OUTPUT / "arms" / f"{arm['name']}.json", result)
    return result


def _execute() -> None:
    inputs = _protected_inputs()
    for path in list(HERE.glob("*.py")) + [HERE / "policy.json"]:
        inputs[str(path)] = _hash(path)
    policy = _read(HERE / "policy.json")
    if policy["minimum_effective_assets"] != 7 or policy["uncertainty_calibration_experiments"] != "deferred":
        raise ValueError("Unapproved allocation or calibration policy")
    arms = [
        {"name": f"{means}__{source}", "means": means, "risk_source": source}
        for means in ["full_joint", "hybrid_US_only"]
        for source in ["GP", "historical"]
    ]
    _save(
        OUTPUT / "plan.json",
        {
            "policy": policy,
            "arms": arms,
            "input_sha256": inputs,
            "joint": BASELINE,
            "US_forecasts": US_MASK,
            "fixed_income_means": "unchanged_full_joint",
            "GP_covariance": "unchanged_full_joint_observed_return_covariance",
        },
    )
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_job, arms))
    comparisons = []
    for source in ["GP", "historical"]:
        original = next(row for row in results if row["arm"] == arms[0 if source == "GP" else 1])
        hybrid = next(row for row in results if row["arm"] == arms[2 if source == "GP" else 3])
        comparisons.append(
            {
                "risk_source": source,
                "hybrid_minus_joint_sharpe": hybrid["summary"]["conventional_sharpe"]
                - original["summary"]["conventional_sharpe"],
                "hybrid_minus_joint_total_return": hybrid["summary"]["compounded_total_return"]
                - original["summary"]["compounded_total_return"],
                "seed_sharpe_differences": [
                    b["performance"]["conventional_sharpe"] - a["performance"]["conventional_sharpe"]
                    for a, b in zip(original["seeds"], hybrid["seeds"], strict=True)
                ],
            }
        )
    _save(
        OUTPUT / "comparison.json",
        {
            "policy": policy,
            "new_GP_fits": 0,
            "monthly_allocations": 288,
            "summaries": [result["summary"] for result in results],
            "paired_comparisons": comparisons,
        },
    )
    for path, expected in inputs.items():
        if _hash(Path(path)) != expected:
            raise ValueError(f"Protected input changed: {path}")
    artifacts = list((OUTPUT / "arms").glob("*.json")) + [OUTPUT / "plan.json", OUTPUT / "comparison.json"]
    _save(
        OUTPUT / "COMPLETED.json",
        {"passed": True, "input_sha256": inputs, "artifact_sha256": {str(path): _hash(path) for path in artifacts}},
    )
    print(json.dumps(_read(OUTPUT / "comparison.json"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.execute == arguments.verify_only:
        parser.error("Choose --execute or --verify-only")
    if arguments.execute:
        _execute()
    else:
        seal = _read(OUTPUT / "COMPLETED.json")
        for group in ["input_sha256", "artifact_sha256"]:
            for name, expected in seal[group].items():
                if _hash(Path(name)) != expected:
                    raise ValueError(f"Changed hybrid evidence: {name}")
        print("Verified four matched cases, all seeds/months, unchanged sources and floor seven")
