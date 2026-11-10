"""Joint static three-way prediction/covariance retuning from frozen GP panels.

Isolated research I/O orchestration: no GP fits, data downloads or production
promotion. First16 months alone propose/select recipes. Only after a hashed
shortlist is frozen are later12/full28 outcomes scored. Returns are decimal,
covariance decimal squared, weights fractions and final NAV dollars.

Usage:
    uv run --locked --no-sync python <this_file> --execute --workers 14
    uv run --locked --no-sync python <this_file> --verify
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
SOURCE = CAMPAIGN / "scenario_numerics_v2/outputs"
PANELS = CAMPAIGN / "outputs/components"
OUTPUT = HERE / "outputs"
sys.path[:0] = [str(HERE), str(CAMPAIGN / "track1_scenarios"), str(CAMPAIGN / "track2_asymmetric"),
               str(CAMPAIGN / "scenario_ne7_uncapped_v1")]

import case_evaluator as evaluator  # noqa: E402
import run_uncapped  # noqa: E402
from joint_math import (  # noqa: E402
    BASE,
    COMPROMISE,
    FIXED_COV,
    REFINED,
    RankAllocator,
    blend,
    coarse_recipes,
    covariance_recipes,
    local_recipes,
    recipe,
    tuning_scores,
)
from recipe_math import select_candidates  # noqa: E402
from uncapped_math import allocate_scenarios  # noqa: E402

STATE: dict = {}
OBJECTIVES = ["sharpe", "total_return", "rank_ic"]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _save(path: Path, value: dict | list | bytes) -> None:
    if not path.resolve().is_relative_to(OUTPUT):
        raise ValueError("Only new joint-study outputs are writable")
    content = value if isinstance(value, bytes) else (
        json.dumps(evaluator._jsonable(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    evaluator._immutable_bytes(path, content)


def _status(phase: str, **fields: object) -> None:
    # Mutable status is never evidence or a selection input.
    path = OUTPUT / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"phase": phase, **fields}, indent=2) + "\n")
    print(json.dumps({"phase": phase, **fields}), flush=True)


def _models() -> list[str]:
    registry = _read(CAMPAIGN / "track3_complements/components.json")
    return sorted(c["modelname"] for c in registry["components"])


def _load_components(model: str, months: int) -> tuple[np.ndarray, np.ndarray]:
    components = [_arrays(PANELS / f"{name}.npz") for name in ["broad_r2_i15_exp", "compact_r2_i15_exp", model]]
    for item in components:
        np.testing.assert_array_equal(item["dates"], components[0]["dates"])
        np.testing.assert_array_equal(item["assets"], components[0]["assets"])
        np.testing.assert_array_equal(item["seeds"], [27, 127, 227])
        np.linalg.cholesky(item["covariance"])
    return (np.stack([c["mean"][:, :months] for c in components]),
            np.stack([c["covariance"][:, :months] for c in components]))


def _initialize() -> None:
    context = _arrays(SOURCE / "context.npz")
    STATE.clear()
    # This worker state deliberately excludes all later-month realized returns.
    STATE.update(actual=context["actual"][:16].copy(), rf=context["risk_free"][:16].copy(),
                 budget=context["risk_budget"][:16].copy(), allocator=RankAllocator(),
                 components={}, cache={}, covariance_cache={})


def _score_chunk(cases: list[dict]) -> list[dict]:
    payload = json.dumps(cases, sort_keys=True).encode()
    token = hashlib.sha256(payload).hexdigest()[:24]
    path = OUTPUT / "tuning_chunks" / f"{token}.json"
    if path.exists():
        cached = _read(path)
        if cached["contract_sha256"] != _hash(OUTPUT / "CONTRACT.json"):
            raise ValueError("Cached tuning chunk belongs to different inputs/code")
        return cached["records"]
    result = []
    for case in cases:
        model = case["third_id"]
        if model not in STATE["components"]:
            STATE["components"][model] = _load_components(model, 16)
        means, covariances = STATE["components"][model]
        mean, covariance = blend(means, covariances, case)
        ranks = rankdata(mean, axis=-1)
        cov_identity = case["covariance_counts"]
        risk_model = model if cov_identity[2] else "same_broad_compact"
        signature = hashlib.sha256(ranks.tobytes() + json.dumps([risk_model, cov_identity]).encode()).hexdigest()
        try:
            if signature in STATE["cache"]:
                score, constrained = STATE["cache"][signature]
                reused = True
            else:
                weights, constrained = STATE["allocator"].allocate(mean, covariance, STATE["budget"])
                score = tuning_scores(STATE["actual"], STATE["rf"], mean, weights)
                STATE["cache"][signature] = (score, constrained)
                reused = False
            result.append({"id": case["id"], "case": case, "status": "complete", **score,
                           "risk_constrained_pairs": constrained, "rank_covariance_signature": signature,
                           "exact_rank_state_cache_reused": reused})
        except (ValueError, RuntimeError, ArithmeticError) as error:
            result.append({"id": case["id"], "case": case, "status": "failed", "period": "tuning_first16",
                           "error": f"{type(error).__name__}: {error}", "partial_objective_used": False})
    _save(path, {"contract_sha256": _hash(OUTPUT / "CONTRACT.json"), "records": result})
    return result


def _simplicity(row: dict) -> tuple:
    counts = np.array(row["case"]["mean_counts"])
    return (int(counts[:, 2].sum()), int(np.count_nonzero(counts)), int(abs(counts - BASE).sum()), row["id"])


def _choose(records: list[dict], limit: int, *, extremes: bool = False) -> list[dict]:
    # Predictions are unidentified when different mixtures produce identical
    # ranks/portfolios. Prefer the simpler representative, not an arbitrary
    # apparently complex winner, for objective ties at 1e-10 resolution.
    unique = {}
    for row in records:
        if row["period"] != "tuning_first16":
            raise ValueError("Only first16 records may choose a recipe")
        if row.get("status") == "failed":
            continue
        key = tuple(round(row[k], 10) for k in OBJECTIVES)
        if key not in unique or _simplicity(row) < _simplicity(unique[key]):
            unique[key] = row
    rows = list(unique.values())
    by_id = {r["id"]: r for r in rows}
    ids = []
    if extremes:
        for objective in OBJECTIVES:
            ordered = sorted(rows, key=_simplicity)
            top = max(ordered, key=lambda r: tuple(round(r[k], 10) for k in
                                                   [objective, *[x for x in OBJECTIVES if x != objective]]))
            if top["id"] not in ids:
                ids.append(top["id"])
    others = select_candidates(rows, limit=max(limit, 6))
    ids.extend(x for x in others if x not in ids)
    return [by_id[x] for x in ids[:limit]]


def _score_stage(pool: ProcessPoolExecutor, cases: list[dict], stage: str, seen: dict) -> list[dict]:
    cases = sorted({c["id"]: c for c in cases}.values(), key=lambda c: c["id"])
    fresh = [c for c in cases if c["id"] not in seen]
    futures = [pool.submit(_score_chunk, fresh[i:i + 64]) for i in range(0, len(fresh), 64)]
    count = 0
    for future in as_completed(futures):
        for row in future.result():
            seen[row["id"]] = row
        count += 1
        if count % 20 == 0 or count == len(futures):
            _status(stage, chunks_completed=count, chunks_total=len(futures), distinct_tuning_recipes=len(seen))
    return [seen[c["id"]] for c in cases]


def _full_case(case: dict, allocator: str) -> dict:
    identity = f"{case['id']}_{allocator}"
    path = OUTPUT / "cases" / f"{identity}.json"
    if path.exists():
        row = _read(path)
        if row["recipe"] != case or row["artifact_sha256"] != _hash(path.with_suffix(".npz")):
            raise ValueError("Cached full case differs")
        return row
    if not (OUTPUT / "TUNING_FROZEN.json").exists():
        raise ValueError("Freeze every recipe before examining later outcomes")
    context = _arrays(SOURCE / "context.npz")
    means, covariances = _load_components(case["third_id"], 28)
    mean, covariance = blend(means, covariances, case)
    if allocator == "hard_rank":
        weights, count = RankAllocator().allocate(mean, covariance, context["risk_budget"])
        numerical = {"risk_constrained_pairs": count, "analytical_kkt_pairs": 84 - count}
    elif allocator == "scenarios":
        weights = np.full((3, 28, 15), np.nan)
        routes = []
        for s, seed in enumerate(context["seeds"]):
            for t, date in enumerate(context["dates"]):
                origin = str(date)
                scenarios = evaluator._dependency("sampling").sample_joint(mean[s, t], covariance[s, t],
                                                                          seed=int(seed), month=origin)
                solution = allocate_scenarios(scenarios, context["assets"].tolist(),
                    seed=20261024 + int(seed) + int(origin.replace("-", "")))
                weights[s, t] = solution["weights"]
                routes.append({"seed": int(seed), "origin": origin, **{
                    k: v for k, v in solution.items() if k not in {"weights", "raw_weights", "starts"}}})
        numerical = {"routes": routes, "position_cap": None, "minimum_effective_assets": 7}
    else:
        raise ValueError("Only unchanged hard rank and uncapped NE7 scenarios allowed")
    arrays = {key: context[key] for key in ["actual", "dates", "assets", "seeds", "risk_free"]}
    arrays.update(mean=mean, covariance=covariance, weights=weights, raw_weights=weights.copy(),
                  allocated=np.ones((3, 28), dtype=bool), effective_count=1 / np.sum(weights**2, axis=-1),
                  equity_weight=weights[:, :, :8].sum(-1),
                  predicted_variance=np.einsum("sta,stab,stb->st", weights, covariance, weights))
    periods, calibration = evaluator._scores(context, mean, covariance, arrays)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _save(path.with_suffix(".npz"), buffer.getvalue())
    row = {"id": identity, "recipe": case, "allocator": allocator, "status": "complete",
           "successful_allocations": 84, "periods": periods, "calibration": calibration, "numerical": numerical,
           "artifact_sha256": _hash(path.with_suffix(".npz")),
           "tuning_frozen_sha256": _hash(OUTPUT / "TUNING_FROZEN.json")}
    _save(path, row)
    return row


def _safe_full(case: dict, allocator: str) -> dict:
    try:
        return _full_case(case, allocator)
    except (ValueError, RuntimeError, ArithmeticError) as error:
        row = {"id": f"{case['id']}_{allocator}", "recipe": case, "allocator": allocator,
               "status": "failed", "periods": {}, "successful_allocations": 0,
               "error": f"{type(error).__name__}: {error}", "partial_objective_used": False,
               "tuning_frozen_sha256": _hash(OUTPUT / "TUNING_FROZEN.json")}
        _save(OUTPUT / "failures" / f"{row['id']}.json", row)
        return row


def _contract(workers: int) -> dict:
    inputs = [SOURCE / "COMPLETED.json", SOURCE / "context.npz", CAMPAIGN / "track3_complements/components.json",
              *[PANELS / f"{name}.npz" for name in ["broad_r2_i15_exp", "compact_r2_i15_exp", *_models()]],
              Path(evaluator.__file__), *evaluator._HELPERS.values(),
              CAMPAIGN / "scenario_ne7_uncapped_v1/uncapped_math.py", CAMPAIGN / "scenario_ne7_v1/ne7_math.py",
              CAMPAIGN / "track2_asymmetric/recipe_math.py", *HERE.glob("*.py")]
    return {"schema": "bayesfolio.joint_three_way.v1", "GP_fits": 0, "workers": workers,
            "primary_allocator": "hard_rank_NE7_uncapped_same_variance_ceiling",
            "secondary_allocator": "scenario_NE7_uncapped_checks_only_frozen_hard_rank_finalists",
            "broad_id": "broad_r2_i15_exp", "compact_id": "compact_r2_i15_exp", "third_models": _models(),
            "means": "four independent three-component simplexes; all families fully free",
            "covariance": "independent global three-component simplex; observed covariance arithmetic pool",
            "coarse": "1296 half-grid product-simplex +1024 Sobol+controls+old blends per model",
            "refinement": "six Pareto centers,steps8/64,4/64,2/64,1/64; transfers and simultaneous family probes",
            "covariance_retuning": "three first16 mean finalists/model; eighth-grid+64 Sobol covariances",
            "coupled_polish": "three first16 finalists/model; joint mean neighbors at1/64 with chosen covariances",
            "selection": "first16 only; up to5 including objective extremes per model, then global5",
            "secondary_selection": "No scenario-specific weight tuning or later-period selection",
            "month_adaptive_blending": False, "pristine_holdout": False, "automatic_promotion": False,
            "tie_parsimony": "Objectives tied at1e-10 prefer lower third mass, fewer active weights, smaller move",
            "input_sha256": {str(p): _hash(p) for p in sorted(set(inputs))}}


def _execute(workers: int) -> None:
    if (OUTPUT / "COMPLETED.json").exists():
        _verify()
        return
    run_uncapped._verify()
    contract = _contract(workers)
    _save(OUTPUT / "CONTRACT.json", contract)
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[key] = "1"
    seen = {}
    models = _models()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize) as pool:
        cases = [c for model in models for c in coarse_recipes(model)]
        _score_stage(pool, cases, "joint_coarse_search", seen)
        for round_index, step in enumerate([8, 4, 2, 1]):
            proposals = []
            for index, model in enumerate(models):
                rows = [r for r in seen.values() if r["case"]["third_id"] == model]
                centers = [r["case"] for r in _choose(rows, 6)]
                proposals.extend(local_recipes(centers, step, seed=20261027 + 100 * round_index + index))
            _score_stage(pool, proposals, f"joint_mean_refinement_{step}_of64", seen)
        covariance_cases = []
        for model in models:
            rows = [r for r in seen.values() if r["case"]["third_id"] == model]
            covariance_cases.extend(covariance_recipes([r["case"] for r in _choose(rows, 3, extremes=True)]))
        _score_stage(pool, covariance_cases, "independent_joint_covariance_retuning", seen)
        polish = []
        for index, model in enumerate(models):
            centers = _choose([r for r in seen.values() if r["case"]["third_id"] == model], 3, extremes=True)
            for center in centers:
                for r in local_recipes([center["case"]], 1, seed=20261030 + index):
                    polish.append(recipe(model, np.array(r["mean_counts"]), center["case"]["covariance_counts"]))
        _score_stage(pool, polish, "mean_covariance_coupled_polish", seen)
        paired_controls = []
        for model in models:
            for row in _choose([r for r in seen.values() if r["case"]["third_id"] == model], 5, extremes=True):
                paired_controls.append(recipe(model, np.array(row["case"]["mean_counts"]), FIXED_COV))
        _score_stage(pool, paired_controls, "paired_covariance_controls", seen)
        selected = {}
        for model in models:
            selected[model] = _choose([r for r in seen.values() if r["case"]["third_id"] == model], 5, extremes=True)
        global_selected = _choose(list(seen.values()), 5, extremes=True)
        controls = [recipe(models[0], c) for c in [BASE, REFINED, COMPROMISE]]
        final_cases = {r["id"]: r["case"] for group in selected.values() for r in group}
        final_cases.update({r["id"]: r["case"] for r in global_selected})
        for c in list(final_cases.values()):
            fixed = recipe(c["third_id"], np.array(c["mean_counts"]), FIXED_COV)
            final_cases[fixed["id"]] = fixed
        final_cases.update({c["id"]: c for c in controls})
        _save(OUTPUT / "TUNING_FROZEN.json", {
            "contract_sha256": _hash(OUTPUT / "CONTRACT.json"), "total_tuning_recipes": len(seen),
            "per_model_selected": selected, "global_selected": global_selected, "control_recipes": controls,
            "full_cases": sorted(final_cases.values(), key=lambda r: r["id"]),
            "scenario_check_ids": sorted(set([r["id"] for r in global_selected] + [controls[0]["id"]])),
            "selection_months": 16, "later_months_used_for_selection": 0})
        _status("shortlist_frozen_before_later12", tuning_recipes=len(seen), finalists=len(final_cases))
        full_futures = [pool.submit(_safe_full, c, "hard_rank") for c in final_cases.values()]
        full = [f.result() for f in as_completed(full_futures)]
        scenarios = [final_cases[i] for i in _read(OUTPUT / "TUNING_FROZEN.json")["scenario_check_ids"]]
        _status("uncapped_NE7_scenario_finalist_checks", cases=len(scenarios))
        scenario_futures = [pool.submit(_safe_full, c, "scenarios") for c in scenarios]
        full.extend(f.result() for f in as_completed(scenario_futures))
    from joint_report import report

    report(full, seen, selected, global_selected, controls)
    run_uncapped.previous._verify_hashes(contract["input_sha256"])
    run_uncapped._verify()
    _save(OUTPUT / "COMPLETED.json", {"passed": True, "GP_fits": 0, "tuning_recipes": len(seen),
          "full_cases": len(full), "failed_full_cases": sum(r["status"] != "complete" for r in full),
          "input_sha256": contract["input_sha256"],
          "artifact_sha256": {str(p): _hash(p) for p in sorted(OUTPUT.rglob("*"))
                              if p.is_file() and p.name not in {"status.json", "COMPLETED.json"}}})
    _status("complete", tuning_recipes=len(seen), full_cases=len(full))
    _verify()


def _verify() -> None:
    run_uncapped._verify()
    seal = _read(OUTPUT / "COMPLETED.json")
    if seal.get("passed") is not True:
        raise ValueError("Passed joint-study completion seal required")
    run_uncapped.previous._verify_hashes(seal["input_sha256"])
    run_uncapped.previous._verify_hashes(seal["artifact_sha256"])
    frozen = _read(OUTPUT / "TUNING_FROZEN.json")
    if frozen["later_months_used_for_selection"] != 0:
        raise ValueError("Later outcomes must never choose blends")
    for path in (OUTPUT / "cases").glob("*.json"):
        row = _read(path)
        if row["status"] != "complete" or row["successful_allocations"] != 84:
            raise ValueError("Partial final case")
        if row["artifact_sha256"] != _hash(path.with_suffix(".npz")):
            raise ValueError("Case arrays changed")
        if row["tuning_frozen_sha256"] != _hash(OUTPUT / "TUNING_FROZEN.json"):
            raise ValueError("Case evaluated with a different frozen shortlist")
    print("Verified joint three-way search, first16 freeze, "
          "complete final paths and untouched prior evidence", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify", action="store_true")
    parser.add_argument("--workers", type=int, default=14)
    options = parser.parse_args()
    if not 1 <= options.workers <= 14:
        parser.error("One through fourteen workers required")
    _execute(options.workers) if options.execute else _verify()
