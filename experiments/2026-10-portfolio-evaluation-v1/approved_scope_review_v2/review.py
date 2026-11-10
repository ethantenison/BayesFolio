"""Apply approved comparison policy and review corrected cached study results.

Usage: .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/approved_scope_review_v2/review.py
       Add --verify-only to check evidence without recomputing the review.
No GP fits or new return-optimizing allocations. Reads sealed original and repair
artifacts, checks training-only feasibility, and writes only outputs/ here.
Returns are decimal; risk is decimal squared; Sharpe is conventional and gross.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[variable] = "1"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
FRAMEWORK = HERE.parent
ORIGINAL = FRAMEWORK / "approved_scope_v1/outputs"
REPAIR = FRAMEWORK / "allocation_numerics_v2/outputs"
sys.path.extend([str(FRAMEWORK), str(FRAMEWORK / "approved_scope_v1"), str(FRAMEWORK / "allocation_numerics_v2")])

import numpy as np  # noqa: E402
from feasibility import preflight  # noqa: E402
from metrics import rank_ic  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from study_common import BASELINE, FITS, US_MASK, _context  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(path: Path, value: object) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Review outputs must stay in the new review directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _verify_hashes(hashes: dict) -> None:
    for name, expected in hashes.items():
        if _hash(Path(name)) != expected:
            raise ValueError(f"Changed sealed evidence: {name}")


def _protected_inputs() -> dict:
    hashes = {}
    original_contract = _read(ORIGINAL / "contract.json")
    hashes.update(original_contract["input_sha256"])
    hashes.update(_read(ORIGINAL / "COMPLETED.json")["artifact_sha256"])
    repair = _read(REPAIR / "COMPLETED.json")
    hashes.update(repair["input_sha256"])
    hashes.update(repair["artifact_sha256"])
    for path in [
        ORIGINAL / "contract.json",
        ORIGINAL / "COMPLETED.json",
        REPAIR / "COMPLETED.json",
        FITS / "context.json",
        FITS / "context_arrays.npz",
    ]:
        hashes[str(path)] = _hash(path)
    _verify_hashes(hashes)
    return hashes


def _artifacts() -> dict:
    result = {}
    for path in sorted((ORIGINAL / "portfolio_cases").glob("*.json")):
        replacement = REPAIR / "portfolio_cases" / path.name
        result[path.stem] = _read(replacement if replacement.exists() else path)
    if len(result) != 372:
        raise ValueError("Expected every original policy, including infeasible cases")
    return result


def _policy_rows(artifacts: dict, policy: dict) -> tuple[list[dict], list[dict]]:
    rows = []
    for artifact in artifacts.values():
        row = dict(artifact["summary"])
        row["analysis_role"] = (
            "primary"
            if row["budget_factor"] == policy["primary_variance_budget_factor"]
            else "sensitivity"
            if row["budget_factor"] == policy["sensitivity_variance_budget_factor"]
            else "stress"
        )
        row["evaluation_status"] = "complete" if row["complete_seeds"] == 3 else "infeasible"
        if row["complete_seeds"] != 3:
            statuses = {month["status"] for seed in artifact["seeds"] for month in seed["months"]}
            if statuses != {"infeasible"}:
                raise ValueError("An unresolved numerical failure must not be relabeled as infeasibility")
        row["matched_covariance_pair_eligible"] = False
        rows.append(row)
    index = {(r["configuration"], r["objective"], r["minimum"], r["budget_factor"], r["risk_source"]): r for r in rows}
    pairs = []
    for row in rows:
        if row["risk_source"] != "GP":
            continue
        key = (row["configuration"], row["objective"], row["minimum"], row["budget_factor"], "historical")
        historical = index.get(key)
        if historical is None:
            continue
        eligible = row["complete_seeds"] == historical["complete_seeds"] == 3
        row["matched_covariance_pair_eligible"] = historical["matched_covariance_pair_eligible"] = eligible
        pairs.append(
            {
                "configuration": row["configuration"],
                "objective": row["objective"],
                "minimum": row["minimum"],
                "budget_factor": row["budget_factor"],
                "analysis_role": row["analysis_role"],
                "eligible": eligible,
                "reason": "both_complete" if eligible else "historical_policy_infeasible",
                "GP_sharpe": row["conventional_sharpe"],
                "historical_sharpe": historical["conventional_sharpe"],
                "GP_minus_historical_sharpe": row["conventional_sharpe"] - historical["conventional_sharpe"]
                if eligible
                else None,
            }
        )
    return rows, pairs


def _rankings(context: dict, contract: dict) -> list[dict]:
    rows = []
    scopes = {
        "pooled": list(range(15)),
        **{family: [context["assets"].index(a) for a in tickers] for family, tickers in context["families"].items()},
    }
    for model in contract["models"]:
        with np.load(ORIGINAL / "panels" / f"{model['name']}.npz", allow_pickle=False) as panel:
            for scope, indices in scopes.items():
                scores = [rank_ic(context["actual"][:, indices], mean[:, indices]) for mean in panel["mean"]]
                rows.append(
                    {
                        "configuration": model["name"],
                        "model_kind": model["kind"],
                        "scope": scope,
                        "mean_ic": float(np.mean(scores)),
                        "seed_mean_ics": [float(values.mean()) for values in scores],
                    }
                )
    return rows


def _fixed_scores(artifacts: dict) -> list[dict]:
    unique = {}
    for artifact in artifacts.values():
        for seed in artifact["seeds"]:
            for row in seed["fixed_reference_scores"]:
                key = (row["configuration"], row["risk_source"], row["portfolio"], row["seed"])
                if key in unique and unique[key] != row:
                    raise ValueError("A supposedly fixed portfolio score changed across allocation policies")
                unique[key] = row
    groups = defaultdict(list)
    for (model, source, portfolio, _), row in unique.items():
        groups[(model, source, portfolio)].append(row)
    results = []
    for (model, source, portfolio), rows in sorted(groups.items()):
        if len(rows) != 3:
            raise ValueError("Fixed reference scores require all original seeds")
        averaged = {
            key: float(np.mean([r[key] for r in rows]))
            for key, value in rows[0].items()
            if isinstance(value, (int, float)) and key != "seed"
        }
        results.append({"configuration": model, "risk_source": source, "portfolio": portfolio, **averaged})
    return results


def _tracking(context: dict, component: str) -> list[dict]:
    with np.load(ORIGINAL / "components" / f"{component}.npz", allow_pickle=False) as values:
        assets = values["assets"].tolist()
        local = [assets.index(asset) for asset in context["families"]["us_equity"]]
        actual = context["actual"][:, [context["assets"].index(asset) for asset in context["families"]["us_equity"]]]
        means = values["mean"][:, :, local]
        local_ic = float(np.mean([rank_ic(actual, mean).mean() for mean in means]))
        result = []
        for index, asset in enumerate(context["families"]["us_equity"]):
            result.append(
                {
                    "component": component,
                    "asset": asset,
                    "US_cross_sectional_IC": local_ic,
                    "time_series_spearman": float(
                        np.mean([spearmanr(actual[:, index], m[:, index]).statistic for m in means])
                    ),
                    "prediction_sd_to_actual_sd": float(
                        np.mean([np.std(m[:, index], ddof=1) / np.std(actual[:, index], ddof=1) for m in means])
                    ),
                    "rmse": float(np.mean([np.sqrt(np.mean((actual[:, index] - m[:, index]) ** 2)) for m in means])),
                }
            )
        return result


def _task_groups(context: dict, contract: dict) -> list[dict]:
    rows = []
    for name, group in contract["groups"].items():
        component = BASELINE if len(group["families"]) == 4 else name
        with np.load(ORIGINAL / "components" / f"{component}.npz", allow_pickle=False) as values:
            for family in group["families"]:
                targets = context["families"][family]
                observed = context["actual"][:, [context["assets"].index(a) for a in targets]]
                indices = [values["assets"].tolist().index(a) for a in targets]
                rows.append(
                    {
                        "component": component,
                        "families_trained": group["families"],
                        "family_scored": family,
                        "mean_ic": float(
                            np.mean([rank_ic(observed, mean[:, indices]).mean() for mean in values["mean"]])
                        ),
                    }
                )
    return rows


def _binding(artifacts: dict, native: set[str]) -> list[dict]:
    groups = defaultdict(list)
    for artifact in artifacts.values():
        case = artifact["case"]
        if (
            case["configuration"] not in native
            or case["objective"] != "rank"
            or case["risk_source"] not in {"GP", "historical"}
            or case["budget_factor"] != 1
        ):
            continue
        for seed in artifact["seeds"]:
            for row in seed["months"]:
                if row["status"] == "solved":
                    groups[(case["risk_source"], case["minimum"])].append(
                        row["predicted_variance"] / row["variance_budget"]
                    )
    return [
        {
            "risk_source": key[0],
            "minimum": key[1],
            "months": len(values),
            "budget_binding_months": sum(value >= 1 - 1e-5 for value in values),
            "mean_predicted_variance_to_budget": float(np.mean(values)),
            "maximum_predicted_variance_to_budget": float(max(values)),
        }
        for key, values in groups.items()
    ]


def _analyze() -> None:
    inputs = _protected_inputs()
    policy, context = _read(HERE / "policy.json"), _context()
    contract, artifacts = _read(ORIGINAL / "contract.json"), _artifacts()
    rows, pairs = _policy_rows(artifacts, policy)
    native = {model["name"] for model in contract["models"] if model["native_covariance"]}
    checks = preflight(context["historical_covariance"], context["risk_budget"] * 0.5, 12)
    if any(row["status"] != "infeasible_dual_certificate" for row in checks):
        raise ValueError("Infeasible stress policy was not independently certified")
    primary_checks = {
        str(floor): preflight(context["historical_covariance"], context["risk_budget"], floor)
        for floor in policy["effective_count_floors"]
    }
    rank_pairs = [pair for pair in pairs if pair["objective"] == "rank"]
    pair_summary = []
    for factor in [1.0, 1.5, 0.5]:
        for floor in policy["effective_count_floors"]:
            eligible = [
                p for p in rank_pairs if p["budget_factor"] == factor and p["minimum"] == floor and p["eligible"]
            ]
            pair_summary.append(
                {
                    "budget_factor": factor,
                    "minimum": floor,
                    "pairs": len(eligible),
                    "GP_sharpe_higher": sum(p["GP_minus_historical_sharpe"] > 0 for p in eligible),
                    "mean_GP_minus_historical_sharpe": float(
                        np.mean([p["GP_minus_historical_sharpe"] for p in eligible])
                    )
                    if eligible
                    else None,
                }
            )
    result = {
        "policy": policy,
        "GP_refits": 0,
        "new_return_optimizing_allocations": 0,
        "configurations": len(rows),
        "complete_configurations": sum(r["complete_seeds"] == 3 for r in rows),
        "retained_infeasible_configurations": sum(r["complete_seeds"] != 3 for r in rows),
        "matched_rank_pairs_eligible": sum(p["eligible"] for p in rank_pairs),
        "matched_rank_pairs_excluded": sum(not p["eligible"] for p in rank_pairs),
        "models": contract["models"],
        "family_choices": contract["family_choices"],
        "policy_metrics": rows,
        "paired_covariance_comparisons": pairs,
        "paired_covariance_summary": pair_summary,
        "rankings": _rankings(context, contract),
        "fixed_reference_scores": _fixed_scores(artifacts),
        "US_equity_controlled_tracking": [
            row
            for name in [US_MASK, "joint_existing_us_mask", "matched__us_equity", BASELINE]
            for row in _tracking(context, name)
        ],
        "matched_task_group_ranking": _task_groups(context, contract),
        "risk_budget_binding": _binding(artifacts, native),
        "infeasible_stress_preflight": checks,
        "primary_historical_preflight": primary_checks,
    }
    _save(OUTPUT / "review.json", result)
    for path in list(HERE.glob("*.py")) + [HERE / "policy.json"] + list(HERE.glob("*.md")):
        inputs[str(path)] = _hash(path)
    _verify_hashes(inputs)
    _save(
        OUTPUT / "COMPLETED.json",
        {
            "passed": True,
            "input_sha256": inputs,
            "artifact_sha256": {str(OUTPUT / "review.json"): _hash(OUTPUT / "review.json")},
        },
    )
    print(
        f"Applied policy: {len(rows)} configurations, {sum(r['complete_seeds'] == 3 for r in rows)} complete; "
        f"{sum(p['eligible'] for p in rank_pairs)} eligible matched rank/covariance pairs; no GP refits"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        seal = _read(OUTPUT / "COMPLETED.json")
        _verify_hashes(seal["input_sha256"])
        _verify_hashes(seal["artifact_sha256"])
        print("Verified applied review policy, original evidence, numerical repair and corrected analysis")
    else:
        _analyze()
