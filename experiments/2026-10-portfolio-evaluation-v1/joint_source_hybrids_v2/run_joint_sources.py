"""Compare requested full-joint mean and covariance source combinations.

Usage from the repository root:
    .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/joint_source_hybrids_v2/run_joint_sources.py
Add --execute to allocate or --verify-only to check sealed results without solves.
Research-only orchestration reuses cached rank-two, maxiter-15 predictions and
retains full observed-return covariance. Means/returns are decimals; risk is
decimal squared. No GP fits, new feature search or calibration transformations.
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
PREVIOUS = FRAMEWORK / "minimum_seven_hybrid_v1/outputs"
sys.path.extend(
    [
        str(FRAMEWORK),
        str(FRAMEWORK / "approved_scope_v1"),
        str(FRAMEWORK / "allocation_numerics_v2"),
        str(FRAMEWORK / "approved_scope_review_v2"),
        str(FRAMEWORK / "minimum_seven_hybrid_v1"),
    ]
)

import numpy as np  # noqa: E402
from feasibility import allocate_checked  # noqa: E402
from metrics import joint_gaussian_scores  # noqa: E402
from review import _artifacts, _protected_inputs  # noqa: E402
from run_hybrid import replace_us_means  # noqa: E402
from study_common import ROOT, SEEDS, _context  # noqa: E402
from study_math import _fixed, _forecast_rows, _score, _scores  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(path: Path, value: object) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Only the new comparison outputs are writable")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != payload:
        raise ValueError(f"Conflicting immutable comparison artifact: {path}")
    path.write_text(payload)


def fuse_joint_sources(
    broad_mean: np.ndarray,
    compact_mean: np.ndarray,
    broad_covariance: np.ndarray,
    compact_covariance: np.ndarray,
    assets: list[str],
    us_assets: list[str],
) -> dict[str, np.ndarray]:
    """Use compact joint U.S. means and keep each covariance source intact.

    Args:
        broad_mean: Broad joint means with shape (3, 24, 15), decimal returns.
        compact_mean: Compact joint means on exactly the same axes and units.
        broad_covariance: Broad observed-return covariance, shape (3, 24, 15, 15).
        compact_covariance: Compact observed-return covariance on the same axes.
        assets: Unique ordered full-universe ticker labels.
        us_assets: Exact SPY, MGK, VTV and IWM labels to replace.

    Returns:
        One common hybrid mean and two unchanged complete covariance arrays.
    """
    if (
        compact_mean.shape != (3, 24, 15)
        or broad_covariance.shape != (3, 24, 15, 15)
        or compact_covariance.shape != broad_covariance.shape
        or not np.isfinite(compact_mean).all()
        or set(us_assets) != {"SPY", "MGK", "VTV", "IWM"}
    ):
        raise ValueError("Exact full-joint prediction axes and U.S. targets required")
    indices = [assets.index(asset) for asset in us_assets]
    mean = replace_us_means(broad_mean, assets, compact_mean[:, :, indices], us_assets, us_assets)
    for covariance in [broad_covariance, compact_covariance]:
        if not np.isfinite(covariance).all() or not np.allclose(
            covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-14
        ):
            raise ValueError("Finite symmetric complete covariance required; no repairs")
        np.linalg.cholesky(covariance)
    return {"mean": mean, "broad_joint": broad_covariance.copy(), "compact_joint": compact_covariance.copy()}


def _prepare(policy: dict) -> dict:
    context = _context()
    arrays = []
    for name in [policy["broad_joint"], policy["compact_joint"]]:
        with np.load(ROOT / "components" / f"{name}.npz", allow_pickle=False) as source:
            if source["dates"].tolist() != context["dates"] or source["assets"].tolist() != context["assets"]:
                raise ValueError("Both joint models must use exactly matched dates and ETF order")
            arrays.append((source["mean"].copy(), source["covariance"].copy()))
    panel = fuse_joint_sources(
        arrays[0][0], arrays[1][0], arrays[0][1], arrays[1][1], context["assets"], context["families"]["us_equity"]
    )
    panel.update(
        dates=np.array(context["dates"]),
        assets=np.array(context["assets"]),
        seeds=np.array(SEEDS),
        actual=context["actual"],
        risk_budget=context["risk_budget"],
        risk_free=context["risk_free"],
    )
    path = OUTPUT / "prediction_panel.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as prior:
            if set(prior.files) != set(panel) or any(
                not np.array_equal(prior[key], value) for key, value in panel.items()
            ):
                raise ValueError("Conflicting immutable prediction snapshot")
    else:
        np.savez(path, **panel)
    return context


def _job(source: str) -> dict:
    policy, context = _read(HERE / "policy.json"), _context()
    outcomes = []
    with np.load(OUTPUT / "prediction_panel.npz", allow_pickle=False) as panel:
        for offset, seed in enumerate(SEEDS):
            mean, covariance = panel["mean"][offset], panel[source][offset]
            scores, penalty = _scores(mean, context["baseline_mean"], policy["objective"])
            allocation = allocate_checked(covariance, context["risk_budget"], 7, penalty, scores)
            if allocation["status"] != "allocated":
                raise ValueError(f"Preflight failed for {source}, seed {seed}")
            outcome = _score(context, mean, covariance, allocation["allocations"])
            if outcome["solved_months"] != 24:
                raise ValueError(f"Incomplete solve for {source}, seed {seed}")
            asset_rows, ranking = _forecast_rows(context, source, seed, mean, covariance, context["assets"])
            density = joint_gaussian_scores(context["actual"], mean, covariance)
            outcomes.append(
                {
                    "seed": seed,
                    **outcome,
                    "preflight": allocation["preflight"],
                    "ranking": ranking,
                    "ETF_forecast_metrics": asset_rows,
                    "monthly_joint_predictive_scores": {key: value.tolist() for key, value in density.items()},
                    "fixed_reference_scores": _fixed(context, source, source, seed, mean, covariance),
                }
            )
    summary = {
        "name": f"compact_US_means__{source}_covariance",
        "covariance_source": source,
        "minimum_effective_assets": 7,
        "complete_seeds": 3,
        "solved_months": 72,
    }
    for metric in [
        "conventional_sharpe",
        "compounded_total_return",
        "annualized_excess_vol",
        "max_drawdown_total_nav",
        "mean_effective_count",
        "largest_weight",
    ]:
        values = [row["performance"][metric] for row in outcomes]
        summary[metric], summary[f"seed_sd_{metric}"] = float(np.mean(values)), float(np.std(values, ddof=1))
    for scope, field in [("pooled", "pooled_IC"), ("us_equity", "US_equity_IC")]:
        summary[field] = float(
            np.mean([next(item["mean_ic"] for item in row["ranking"] if item["scope"] == scope) for row in outcomes])
        )
    summary["risk_binding_months"] = sum(
        month["predicted_variance"] / month["variance_budget"] >= 1 - 1e-5
        for row in outcomes
        for month in row["months"]
    )
    result = {"policy": policy, "summary": summary, "seeds": outcomes}
    _save(OUTPUT / "arms" / f"{source}.json", result)
    return result


def _execute() -> None:
    inputs = _protected_inputs()
    prior = _read(PREVIOUS / "COMPLETED.json")
    inputs.update(prior["input_sha256"])
    inputs.update(prior["artifact_sha256"])
    for path in list(HERE.glob("*.py")) + [HERE / "policy.json", PREVIOUS / "COMPLETED.json"]:
        inputs[str(path)] = _hash(path)
    for name, digest in inputs.items():
        if _hash(Path(name)) != digest:
            raise ValueError(f"Changed protected evidence: {name}")
    policy = _read(HERE / "policy.json")
    if (
        policy["minimum_effective_assets"] != 7
        or policy["variance_budget_factor"] != 1
        or policy["uncertainty_calibration_experiments"] != "deferred"
    ):
        raise ValueError("Unapproved investment policy")
    _prepare(policy)
    _save(
        OUTPUT / "plan.json",
        {
            "policy": policy,
            "input_sha256": inputs,
            "new_monthly_allocations": 144,
            "GP_refits": 0,
            "prediction_panel_sha256": _hash(OUTPUT / "prediction_panel.npz"),
        },
    )
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_job, policy["covariance_sources"]))
    controls = [
        _read(PREVIOUS / "arms" / f"{name}.json")["summary"] for name in ["full_joint__GP", "hybrid_US_only__GP"]
    ]
    compact_control = next(
        a["summary"]
        for a in _artifacts().values()
        if a["case"]["configuration"] == policy["compact_joint"]
        and a["case"]["risk_source"] == "GP"
        and a["case"]["objective"] == "rank"
        and a["case"]["minimum"] == 7
        and a["case"]["budget_factor"] == 1
    )
    _save(
        OUTPUT / "comparison.json",
        {
            "policy": policy,
            "new_GP_fits": 0,
            "new_monthly_allocations": 144,
            "summaries": [row["summary"] for row in results],
            "previous_controls": controls,
            "pure_compact_joint_control": compact_control,
        },
    )
    for name, digest in inputs.items():
        if _hash(Path(name)) != digest:
            raise ValueError(f"Input changed during evaluation: {name}")
    paths = list((OUTPUT / "arms").glob("*.json")) + [
        OUTPUT / "plan.json",
        OUTPUT / "comparison.json",
        OUTPUT / "prediction_panel.npz",
    ]
    _save(
        OUTPUT / "COMPLETED.json",
        {"passed": True, "input_sha256": inputs, "artifact_sha256": {str(path): _hash(path) for path in paths}},
    )
    print(json.dumps({"new_results": [row["summary"] for row in results], "GP_refits": 0}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.execute == args.verify_only:
        parser.error("Choose --execute or --verify-only")
    if args.execute:
        _execute()
    else:
        seal = _read(OUTPUT / "COMPLETED.json")
        for group in ["input_sha256", "artifact_sha256"]:
            for name, digest in seal[group].items():
                if _hash(Path(name)) != digest:
                    raise ValueError(f"Changed joint-source evidence: {name}")
        print("Verified both requested hybrids, full prediction panel and all protected sources")
