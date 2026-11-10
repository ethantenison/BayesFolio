"""Run two adaptive cached blending rounds and matched 80/20 sleeve experiments.

Usage from repository root:
    .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/blend_sleeves_v1/run_blends.py --stage first
Stages: first, second, sleeves, compile, verify. Research orchestration writes only
this folder's outputs; sealed previous studies stay read-only. All financial
returns are decimals, risk is decimal squared; no new GP fits or calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_key] = "1"
HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parent
OUTPUT = HERE / "outputs"
FAMILIES = ["us_equity", "international_equity", "us_fixed_income", "international_fixed_income"]
sys.path.extend(
    [
        str(FRAMEWORK),
        str(FRAMEWORK / "approved_scope_v1"),
        str(FRAMEWORK / "allocation_numerics_v2"),
        str(FRAMEWORK / "approved_scope_review_v2"),
    ]
)

import numpy as np  # noqa: E402
from blend_math import allocation_scores, blend_panel, joint_allocate, sleeve_allocate  # noqa: E402
from feasibility import allocate_checked  # noqa: E402
from metrics import joint_gaussian_scores, rank_ic  # noqa: E402
from review import _protected_inputs  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from study_common import BASELINE, ROOT, SEEDS, _context  # noqa: E402
from study_math import _fixed, _forecast_rows, _score  # noqa: E402

POLICY = {
    "schema": "bayesfolio.blend_sleeves.v1",
    "broad": BASELINE,
    "compact": "joint_existing_us_mask",
    "families_in_alpha_order": FAMILIES,
    "rank": 2,
    "maxiter": 15,
    "seeds": SEEDS,
    "initial_training_months": 84,
    "forecast_months": 24,
    "minimum_effective": 7,
    "variance_budget": "unchanged_training_only_historical_equal_weight_15_ETF_variance",
    "selection": "highest_three_seed_mean_conventional_Sharpe; ties within 1e-8 use decision_IC then case_id",
    "IC_NLL": "recorded_and_Pareto_reported; not_NLL_guardrails",
    "ETF_position_cap": None,
    "CVaR": False,
    "fallback_weights": False,
    "constraint_relaxation": False,
    "calibration": "deferred",
    "production_promotion": False,
    "new_GP_fits": 0,
    "sleeves": {
        "capital": {"equity": 0.8, "fixed_income": 0.2},
        "minimum_effective": "7_combined_only_via_scalar_concentration_dual_price",
        "scores": "within_sleeve_percentile_rank_of_winning_decision_scores",
        "risk": "each_normalized_sleeve_variance_le_original_full_portfolio_budget",
        "risk_implication": "conservative_sufficient_condition_for_unchanged_combined_budget; not_equivalent",
        "combined_risk_evaluation": "complete_winning_covariance_including_cross_sleeve_terms",
    },
    "limitation": "two_adaptive_rounds_reuse_24_exploratory_months; seeds_are_not_independent_market_histories",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(path: Path, value: object) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Only new experiment outputs may be written")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Conflicting immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def _snapshot(path: Path, arrays: dict) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as prior:
            if set(prior.files) != set(arrays) or any(
                not np.array_equal(prior[key], value) for key, value in arrays.items()
            ):
                raise ValueError(f"Changed cached panel: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _verify(hashes: dict) -> None:
    for name, digest in hashes.items():
        if _hash(Path(name)) != digest:
            raise ValueError(f"Changed protected evidence: {name}")


def _prepare() -> dict:
    inputs = _protected_inputs()
    seal_path = FRAMEWORK / "joint_source_hybrids_v2/outputs/COMPLETED.json"
    seal = _read(seal_path)
    if seal["passed"] is not True:
        raise ValueError("Prior hybrid study did not complete")
    inputs.update(seal["input_sha256"])
    inputs.update(seal["artifact_sha256"])
    inputs[str(seal_path)] = _hash(seal_path)
    context, arrays = _context(), {}
    for label, name in [("broad", POLICY["broad"]), ("compact", POLICY["compact"])]:
        path = ROOT / "components" / f"{name}.npz"
        inputs[str(path)] = _hash(path)
        with np.load(path, allow_pickle=False) as source:
            if source["dates"].tolist() != context["dates"] or source["assets"].tolist() != context["assets"]:
                raise ValueError("Exact common month and ETF axes required")
            arrays[f"{label}_mean"], arrays[f"{label}_covariance"] = source["mean"].copy(), source["covariance"].copy()
    for key in ["actual", "risk_budget", "risk_free"]:
        arrays[key] = context[key]
    arrays.update(assets=np.array(context["assets"]), dates=np.array(context["dates"]), seeds=np.array(SEEDS))
    for path in sorted(HERE.glob("*.py")):
        inputs[str(path)] = _hash(path)
    _verify(inputs)
    _snapshot(OUTPUT / "source_panel.npz", arrays)
    _save(OUTPUT / "contract.json", {"policy": POLICY, "input_sha256": inputs})
    return context


def _case(alpha: list[float], beta: float = 0, covariance: str = "arithmetic", decision: str = "mean") -> dict:
    definition = {
        "alpha": [float(a) for a in alpha],
        "beta": float(beta),
        "covariance": covariance,
        "decision": decision,
    }
    identity = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()[:12]
    return {"id": identity, **definition}


def _round_one() -> list[dict]:
    candidates = []
    for alpha in [0, 0.25, 0.5, 0.75, 1]:
        for beta in [0, 0.5, 1]:
            candidates.append(_case([alpha, 0, 0, 0], beta))
    for family in [1, 2, 3]:
        for alpha in [0.25, 0.5, 1]:
            fractions = [1, 0, 0, 0]
            fractions[family] = alpha
            for beta in [0, 0.5, 1]:
                candidates.append(_case(fractions, beta))
    for alpha in [0.25, 0.5, 0.75, 1]:
        candidates.extend([_case([alpha] * 4, alpha), _case([alpha] * 4, alpha, "mixture")])
    for alpha in [0.5, 1]:
        for beta in [0, 0.5, 1]:
            candidates.append(_case([alpha, 0, 0, 0], beta, decision="borda"))
    for method in ["precision", "log", "correlation"]:
        for beta in [0.25, 0.5, 0.75]:
            candidates.append(_case([1, 0, 0, 0], beta, method))
    return list({case["id"]: case for case in candidates}.values())


def _alpha(case: dict, context: dict) -> np.ndarray:
    result = np.empty(15)
    for fraction, family in zip(case["alpha"], FAMILIES, strict=True):
        result[[context["assets"].index(asset) for asset in context["families"][family]]] = fraction
    return result


def _panel(case: dict, context: dict) -> dict:
    with np.load(OUTPUT / "source_panel.npz", allow_pickle=False) as source:
        broad = {key: source[f"broad_{key}"].copy() for key in ["mean", "covariance"]}
        compact = {key: source[f"compact_{key}"].copy() for key in ["mean", "covariance"]}
    alpha = _alpha(case, context)
    panel = blend_panel(broad, compact, alpha, case["covariance"], case["beta"])
    panel["scores"] = allocation_scores(broad["mean"], compact["mean"], panel["mean"], alpha, case["decision"])
    return panel


def _summarize(case: dict, outcomes: list[dict]) -> dict:
    result = {"case": case, "id": case["id"], "complete_seeds": len(outcomes), "solved_months": 72}
    for metric in [
        "conventional_sharpe",
        "compounded_total_return",
        "annualized_excess_vol",
        "max_drawdown_total_nav",
        "mean_effective_count",
        "largest_weight",
        "first_12_sharpe",
        "last_12_sharpe",
    ]:
        values = [row["performance"][metric] for row in outcomes]
        result[metric], result[f"seed_sd_{metric}"] = float(np.mean(values)), float(np.std(values, ddof=1))
        result[f"seed_values_{metric}"] = values
    result["decision_IC"] = float(np.mean([row["decision_IC"] for row in outcomes]))
    result["joint_nll"] = float(
        np.mean([np.mean(row["monthly_joint_predictive_scores"]["joint_nll"]) for row in outcomes])
    )
    result["rmse"] = float(np.mean([row["rmse"] for row in outcomes]))
    for scope in ["pooled", *FAMILIES]:
        result[f"{scope}_IC"] = float(
            np.mean([next(item["mean_ic"] for item in row["ranking"] if item["scope"] == scope) for row in outcomes])
        )
    result["risk_binding_months"] = sum(
        month["predicted_variance"] / month["variance_budget"] >= 1 - 1e-5
        for row in outcomes
        for month in row["months"]
    )
    return result


def _job(case: dict, mode: str = "blend") -> dict:
    path = OUTPUT / ("cases" if mode == "blend" else "sleeves") / f"{case['id']}.json"
    if path.exists():
        prior = _read(path)
        if prior["case"] != case or any(row["solved_months"] != 24 for row in prior["seeds"]):
            raise ValueError("Invalid cached result")
        return prior["summary"]
    context, outcomes = _context(), []
    panel = _panel(case, context)
    eq = [
        context["assets"].index(a)
        for family in ["us_equity", "international_equity"]
        for a in context["families"][family]
    ]
    fi = [i for i in range(15) if i not in eq]
    for offset, seed in enumerate(SEEDS):
        mean, covariance, scores = (panel[key][offset] for key in ["mean", "covariance", "scores"])
        if mode in {"fixed_local", "separate"}:
            scores = scores.copy()
            for indices in [eq, fi]:
                scores[:, indices] = rankdata(scores[:, indices], axis=1) / len(indices)
        if mode == "blend":
            allocated = allocate_checked(covariance, context["risk_budget"], 7, 1e-8, scores)
            if allocated["status"] != "allocated":
                raise ValueError(f"Blocked training-only preflight for {case}")
            records, checks = allocated["allocations"], allocated["preflight"]
        elif mode in {"fixed_global", "fixed_local"}:
            records = joint_allocate(covariance, context["risk_budget"], scores, 7, eq)
            checks = [{"status": "feasible_raw_solved_weights_witness"}]
        elif mode == "separate":
            records = sleeve_allocate(covariance, context["risk_budget"], scores, 7, eq)
            checks = [{"status": "feasible_raw_solved_weights_and_dual_certificate"}]
        else:
            raise ValueError("Unknown allocator mode")
        outcome = _score(context, mean, covariance, records)
        if outcome["solved_months"] != 24:
            raise ValueError(f"Incomplete {mode} seed {seed}: {case}")
        asset_rows, ranking = _forecast_rows(context, case["id"], seed, mean, covariance, context["assets"])
        density = joint_gaussian_scores(context["actual"], mean, covariance)
        weights = np.array([row["weights"] for row in records])
        panel.setdefault("weights", np.empty((3, 24, 15)))[offset] = weights
        outcome["performance"]["mean_equity_weight"] = float(weights[:, eq].sum(axis=1).mean())
        outcome["performance"]["mean_equity_sleeve_effective_count"] = float(
            np.mean(weights[:, eq].sum(axis=1) ** 2 / (weights[:, eq] ** 2).sum(axis=1))
        )
        outcomes.append(
            {
                "seed": seed,
                **outcome,
                "preflight": checks,
                "ranking": ranking,
                "ETF_forecast_metrics": asset_rows,
                "decision_IC": float(np.mean(rank_ic(context["actual"], scores))),
                "rmse": float(np.sqrt(np.mean((context["actual"] - mean) ** 2))),
                "monthly_joint_predictive_scores": {key: value.tolist() for key, value in density.items()},
                "fixed_reference_scores": _fixed(context, case["id"], case["covariance"], seed, mean, covariance),
            }
        )
    summary = _summarize(case, outcomes)
    summary["allocator"] = mode
    summary["mean_equity_weight"] = float(np.mean([o["performance"]["mean_equity_weight"] for o in outcomes]))
    summary["mean_equity_sleeve_effective_count"] = float(
        np.mean([o["performance"]["mean_equity_sleeve_effective_count"] for o in outcomes])
    )
    _snapshot(path.with_suffix(".npz"), panel)
    _save(path, {"case": case, "mode": mode, "summary": summary, "seeds": outcomes})
    return summary


def _ordered(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (-round(row["conventional_sharpe"], 8), -row["decision_IC"], row["id"]))


def _pareto(rows: list[dict]) -> list[str]:
    values = np.array([[row["conventional_sharpe"], row["decision_IC"], -row["joint_nll"]] for row in rows])
    return [
        row["id"]
        for i, row in enumerate(rows)
        if not np.any(np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1))
    ]


def _execute_round(number: int, candidates: list[dict], rationale: dict) -> None:
    started = time.monotonic()
    _save(OUTPUT / f"round_{number}_plan.json", {"cases": candidates, "rationale": rationale, "policy": POLICY})
    results = []
    workers = min(13, max(1, (os.cpu_count() or 3) - 2), len(candidates))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, case): case for case in candidates}
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 10 == 0 or len(results) == len(candidates):
                print(f"Round {number}: {len(results)}/{len(candidates)} complete", flush=True)
    rows = _ordered(results)
    _save(OUTPUT / f"round_{number}_results.json", {"summaries": rows, "pareto_ids": _pareto(rows)})
    print(
        json.dumps({"round": number, "workers": workers, "seconds": time.monotonic() - started, "leader": rows[0]}),
        flush=True,
    )


def _second_round() -> tuple[list[dict], dict]:
    rows = _read(OUTPUT / "round_1_results.json")["summaries"]
    leader = rows[0]["case"]
    original_ids = {row["id"] for row in rows}
    candidates = []
    # Refinement centers are observed round-one leaders, not a precomputed second grid.
    centers = []
    for row in rows:
        case = row["case"]
        if case["covariance"] != "mixture" and (case["alpha"], case["decision"]) not in centers:
            centers.append((case["alpha"], case["decision"]))
        if len(centers) == 2:
            break
    for center, decision in centers:
        for family in range(4):
            for change in [-0.125, 0.125]:
                fractions = center.copy()
                fractions[family] = float(np.clip(fractions[family] + change, 0, 1))
                for beta in [0, 0.5, 1]:
                    candidates.append(_case(fractions, beta, decision=decision))
    for method in ["arithmetic", "precision", "log", "correlation"]:
        for beta in [0.125, 0.375, 0.625, 0.875]:
            candidates.append(_case(leader["alpha"], beta, method, leader["decision"]))
    # Exploration: combine the best separately observed non-US family changes.
    interactions = leader["alpha"].copy()
    for family in [1, 2, 3]:
        alternatives = [row for row in rows if row["case"]["alpha"][family] > 0 and row["case"]["alpha"][0] == 1]
        if alternatives:
            interactions[family] = alternatives[0]["case"]["alpha"][family]
    for scale in [0.5, 1]:
        fractions = [leader["alpha"][0], *(value * scale for value in interactions[1:])]
        for beta in [0, 0.5, 1]:
            for decision in ["mean", "borda"]:
                candidates.append(_case(fractions, beta, decision=decision))
    unique = {case["id"]: case for case in candidates if case["id"] not in original_ids}
    rationale = {
        "observed_round_one_leader": rows[0],
        "refinement_centers": centers,
        "refinement": "one_eighth_family_mean_fraction_neighbors_plus_finer_covariance_grid",
        "exploration": "combine_best_separately_observed_non_US_family_fractions_and_compare_mean_vs_Borda",
        "interactions": interactions,
        "no_realized_outcomes_enter_individual_monthly_weight_solvers": True,
    }
    return list(unique.values()), rationale


def _all_rows() -> list[dict]:
    return _ordered([row for number in [1, 2] for row in _read(OUTPUT / f"round_{number}_results.json")["summaries"]])


def _sleeves() -> None:
    winner = _all_rows()[0]
    _save(
        OUTPUT / "sleeve_plan.json",
        {
            "selected_winner": winner,
            "policy": POLICY["sleeves"],
            "controls": ["winning_unconstrained_capital", "fixed_global", "fixed_local", "separate"],
        },
    )
    results = []
    # Three allocator modes execute concurrently; each includes all matched seeds/months.
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = []
        for mode in ["fixed_global", "fixed_local", "separate"]:
            case = {**winner["case"], "id": f"{winner['id']}__{mode}"}
            futures.append(pool.submit(_job, case, mode))
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps({"sleeve_completed": row}), flush=True)
    _save(
        OUTPUT / "sleeve_results.json",
        {"unrestricted_winner": winner, "summaries": sorted(results, key=lambda r: r["id"])},
    )


def _compile() -> None:
    rows = _all_rows()
    sleeves = _read(OUTPUT / "sleeve_results.json")
    _save(
        OUTPUT / "comparison.json",
        {
            "policy": POLICY,
            "summaries": rows,
            "winner": rows[0],
            "pareto_ids": _pareto(rows),
            "sleeves": sleeves,
            "blending_configuration_count": len(rows),
            "new_GP_fits": 0,
            "monthly_portfolio_allocations": (len(rows) + 3) * 72,
        },
    )
    contract = _read(OUTPUT / "contract.json")
    _verify(contract["input_sha256"])
    paths = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "COMPLETED.json")
    _save(
        OUTPUT / "COMPLETED.json",
        {
            "passed": True,
            "input_sha256": contract["input_sha256"],
            "artifact_sha256": {str(path): _hash(path) for path in paths},
        },
    )
    print(json.dumps({"configurations": len(rows), "winner": rows[0], "sleeves": sleeves["summaries"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["first", "second", "sleeves", "compile", "verify"])
    args = parser.parse_args()
    if args.stage == "verify":
        completed = _read(OUTPUT / "COMPLETED.json")
        _verify(completed["input_sha256"])
        _verify(completed["artifact_sha256"])
        print("Verified all experiment and protected source hashes")
    else:
        _prepare()
        if args.stage == "first":
            _execute_round(1, _round_one(), {"purpose": "broad_prediction_and_covariance_blending_screen"})
        elif args.stage == "second":
            proposed, reasons = _second_round()
            _execute_round(2, proposed, reasons)
        elif args.stage == "sleeves":
            _sleeves()
        else:
            _compile()
