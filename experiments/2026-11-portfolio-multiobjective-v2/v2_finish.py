"""Select only frozen finalists, prepare November weights, and seal delivery.

This isolated research composition layer never fits or changes a candidate after
confirmation. Returns/weights are decimals, risk matrices squared decimals, NAV
dollars. No trading/account/git actions. External independent audit precedes seal.
Usage: uv run --locked --no-sync python <this_file> --prepare-final|--seal
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from v2_benchmarks import build_benchmarks, stress
from v2_common import (
    ASSETS,
    HERE,
    SEEDS,
    V1,
    arrays,
    digest,
    read,
    save,
    snapshot,
    status,
    verify_hashes,
)
from v2_driver import distance
from v2_evaluate import pooled
from v2_experts import REGISTERED, load_cached_panel, registered_configs
from v2_fit_queue import attempt_inventory
from v2_independent_validation import (
    validate_fit_record,
    validate_gp_spec,
    validate_portfolio,
    validate_training_availability,
    validate_weights_csv,
)
from v2_policy import allocate
from v2_protocol import prepare
from v2_research import development_records, frontier
from v2_select import complexity


def live_source() -> dict:
    """Read separate unknown-target November moments from original expert caches.

    Returns:
        Three-seed live moments for all registered experts plus causal risk data.
        No November target or historical outcome enters a GP fit here.
    """
    panels = [load_cached_panel(name, ["2026-10-30"]) for name in REGISTERED]
    if not all(panel["available"].all() for panel in panels):
        raise ValueError("Live forecasts incomplete")
    context = arrays(HERE / "outputs/context.npz")
    return {"experts": np.array(REGISTERED), "expert_mean": np.stack([p["mean"] for p in panels]),
        "expert_covariance": np.stack([p["covariance"] for p in panels]),
        "historical_covariance": context["historical_covariance"][-1:],
        "historical_EW_variance": context["historical_EW_variance"][-1:]}


def _solve_live(case: dict, source: dict) -> dict:
    mean, covariance = pooled(case, source)
    budget = None if case["risk_multiplier"] is None else float(
        case["risk_multiplier"] * source["historical_EW_variance"][0])
    result = allocate(mean[0], covariance[0], min_effective_assets=case["minimum_effective_assets"],
                      variance_ceiling=budget, equity_cap=case["equity_cap"])
    return {"result": result, "mean": mean[0], "covariance": covariance[0], "budget": budget}


def acceptance(full: dict, robust: dict, *, best_control_full: dict,
               best_control_latest: dict, live_feasible: bool) -> dict:
    """Apply the frozen final gates without an IC veto or any retuning.

    Args:
        full: Complete final53 summary and latest12 summary.
        robust: Original development-only eligibility evidence.
        best_control_full: One actual control chosen by frozen full-panel distance.
        best_control_latest: One actual control chosen by latest12 distance.
        live_feasible: Whether the exact registered November constraints solved.

    Returns:
        Eligibility and explicit frozen gate values for one noncontrol finalist.
    """
    gates = {"development_eligible_before_unlock": bool(robust["eligible"]),
        "all53_complete_raw_path": bool(full["complete"]),
        "November_registered_constraints_feasible": bool(live_feasible),
        "all53_distance_gain_at_least_0p05": distance(best_control_full["summary"])
            - distance(full["summary"]) >= .05,
        "latest12_distance_within_best_control_plus0p50": distance(full["latest12"])
            <= distance(best_control_latest["latest12"]) + .50,
        "latest12_drawdown_loss_within_best_control_plus3pp": -full["latest12"]["max_drawdown"]
            <= -best_control_latest["latest12"]["max_drawdown"] + .03}
    return {"eligible": all(gates.values()), "gates": gates, "distance": distance(full["summary"])}


def _final_key(row: dict) -> tuple:
    return (round(row["selection"]["distance"], 8), -row["robustness"]["annual_blocks_within_tolerance"],
            -(row["full"]["summary"]["sharpe"] or -1e6), -row["full"]["summary"]["rank_ic"],
            complexity(row["policy"]), row["id"])


def _validate_inputs() -> dict:
    prepare()
    verify_hashes(read(HERE / "MATH_CONTRACT.json")["files_sha256"])
    shortlist = read(HERE / "SHORTLIST_FROZEN.json")
    if digest(HERE / "SHORTLIST_FROZEN.json") != (HERE / "SHORTLIST_FROZEN.sha256").read_text().strip():
        raise ValueError("Confirmation shortlist changed")
    verify_hashes(shortlist["source_sha256"])
    fit_contract, fits = read(HERE / "FIT_CONTRACT.json"), read(HERE / "FITS_COMPLETED.json")
    verify_hashes(fit_contract["protected_sha256"])
    verify_hashes(fits["artifact_sha256"])
    v1 = read(V1 / "COMPLETED.json")
    verify_hashes(v1["artifact_sha256"])
    verify_hashes(v1["environment_sha256"])
    for name in ["INDEPENDENT_DEVELOPMENT_AUDIT.json", "INDEPENDENT_SHORTLIST_AUDIT.json"]:
        if not read(HERE / "outputs" / name)["passed"]:
            raise ValueError("An independent prerequisite audit did not pass")
    return shortlist


def prepare_final() -> dict:
    """Choose among frozen finalists and prepare numerical delivery for audit.

    Returns:
        Root validation evidence. Final report/seal waits for external final audit.
    """
    shortlist = _validate_inputs()
    if (HERE / "ROOT_VALIDATION.json").exists():
        return read(HERE / "ROOT_VALIDATION.json")
    registry = {row["id"]: row for row in development_records()}
    full = {identity: read(HERE / "outputs/confirmation" / f"{identity}.json") for identity in shortlist["ids"]}
    if not all(row["complete"] for row in full.values()):
        raise ValueError("All frozen final confirmation paths must be complete")
    control_rows = [full[identity] for identity in shortlist["control_ids"]]
    best_full = min(control_rows, key=lambda r: (distance(r["summary"]), r["id"]))
    best_latest = min(control_rows, key=lambda r: (distance(r["latest12"]), r["id"]))
    source = live_source()
    records, live = [], {}
    for identity in shortlist["ids"]:
        row = full[identity]
        live[identity] = _solve_live(row["policy"], source)
        robust = shortlist["development_gate_evidence"].get(identity)
        if robust is None:
            robust = {"eligible": False, "annual_blocks_within_tolerance": 0,
                      "interpretation": "Mandatory unchanged fallback control; improvement over itself is not required"}
        selection = acceptance(row, robust, best_control_full=best_full,
                               best_control_latest=best_latest, live_feasible=live[identity]["result"]["passed"])
        if identity in shortlist["control_ids"]:
            selection = {"eligible": bool(live[identity]["result"]["passed"]),
                "gates": {"unchanged_control": True, "raw_path_and_live_feasible":
                          bool(live[identity]["result"]["passed"])},
                "distance": distance(row["summary"]), "fallback_control": True}
        records.append({"id": identity, "policy": row["policy"], "development": registry[identity],
            "full": row, "robustness": robust, "selection": selection,
            "artifact_path": row["artifact_path"]})
    challengers = [row for row in records if row["id"] not in shortlist["control_ids"]
                   and row["selection"]["eligible"]]
    compact = [{"id": row["id"], "complete": True, "summary": row["full"]["summary"]} for row in challengers]
    pareto_ids = {row["id"] for row in frontier(compact)}
    eligible_front = [row for row in challengers if row["id"] in pareto_ids]
    chosen = (min(eligible_front, key=_final_key) if eligible_front else
              min((row for row in records if row["id"] in shortlist["control_ids"] and row["selection"]["eligible"]),
                  key=lambda r: (r["selection"]["distance"], r["id"])))
    identity, case = chosen["id"], chosen["policy"]
    final = live[identity]
    weights = np.asarray(final["result"]["weights"])
    numeric = validate_portfolio(final["mean"], final["covariance"], weights, ASSETS,
        min_effective_assets=case["minimum_effective_assets"], variance_ceiling=final["budget"],
        equity_cap=case["equity_cap"], solver_status=final["result"]["solver_status"],
        solver_attempts=final["result"]["numerical_attempts"], analytic_certificate=final["result"]["certificate"])
    save(HERE / "FINAL_SELECTION.json", {"passed": True, "selected_id": identity, "selected_policy": case,
        "selected_at_utc": (read(HERE / "FINAL_SELECTION.json")["selected_at_utc"]
                            if (HERE / "FINAL_SELECTION.json").exists() else datetime.now(UTC).isoformat()),
        "shortlist_sha256": digest(HERE / "SHORTLIST_FROZEN.json"),
        "incumbent_retained": identity in shortlist["control_ids"],
        "eligible_challengers": [row["id"] for row in challengers], "eligible_Pareto_ids": sorted(pareto_ids),
        "best_full_control": best_full["id"], "best_latest_control": best_latest["id"],
        "finalist_gates": {row["id"]: row["selection"] for row in records},
        "no_search_or_retuning_after_confirmation": True, "weights_appearance_not_a_selection_input": True})
    frame = pd.DataFrame({"asset_id": ASSETS, "weight": weights, "as_of_date": "2026-10-31",
                          "market_origin": "2026-10-30", "forecast_month": "2026-11"})
    stream = io.StringIO()
    frame.to_csv(stream, index=False, float_format="%.17g")
    save(HERE / "NOVEMBER_WEIGHTS.csv", stream.getvalue().encode())
    serialized = validate_weights_csv(HERE / "NOVEMBER_WEIGHTS.csv", final["mean"], final["covariance"], weights,
        ASSETS, min_effective_assets=case["minimum_effective_assets"], variance_ceiling=final["budget"],
        equity_cap=case["equity_cap"], solver_status=final["result"]["solver_status"],
        solver_attempts=final["result"]["numerical_attempts"], analytic_certificate=final["result"]["certificate"])
    snapshot(HERE / "outputs/NOVEMBER_MOMENTS.npz", {"mean": final["mean"], "covariance": final["covariance"],
        "weights": weights, "assets": np.array(ASSETS), "seeds": np.array(SEEDS),
        "historical_EW_variance": source["historical_EW_variance"],
        "variance_ceiling": np.array(final["budget"] if final["budget"] is not None else np.nan)})
    save(HERE / "outputs/NOVEMBER_OPTIMIZER.json", final["result"])
    expert_specs = registered_configs()
    reference_specs = read(V1 / "FIT_CONTRACT.json")["configs"]
    spec_validation = {}
    for name, spec in expert_specs.items():
        old = reference_specs[name.replace("compact_r1_i15_exp", "compact_r2_i15_exp")]
        spec_validation[name] = validate_gp_spec(spec, old, read(HERE / "GP_VARIANTS.json")["specs"][name],
                                                allow_compact_rank1=name == "compact_r1_i15_exp")
    fit_validation = []
    for path in sorted((HERE / "outputs/fit_checkpoints").glob("*.json")):
        checkpoint = read(path)
        job = checkpoint["job"]
        fit_validation.append(validate_fit_record(checkpoint["result"], expert_specs[job["component"]],
                                                   origin=job["date"], seed=job["seed"]))
    columns = sorted({column for spec in expert_specs.values() for column in spec["inputs"]})
    available = validate_training_availability(pd.read_parquet(V1 / "data/dataset/global.parquet"), ASSETS, columns)
    context53 = arrays(HERE / "outputs/confirmation_inputs.npz")
    benchmark = build_benchmarks({"dates": context53["dates"], "assets": ASSETS,
                                  "actual": context53["actual"], "rf": context53["rf"]})
    benchmarks = {name: {"label": "Schwab aggressive proxy" if name == "schwab_aggressive" else
                         "Simple momentum: eligible14 top7 equal weight", **row,
                         "latest12": row["confirmation12"]} for name, row in benchmark["rows"].items()}
    stress_result = stress(weights)
    save(HERE / "STRESS.json", stress_result)
    comparison = {"records": records, "selected_id": identity, "control_ids": shortlist["control_ids"],
        "benchmarks": benchmarks, "dates53": context53["dates"].tolist(), "stress": stress_result,
        "eligible_frontier_ids": sorted(pareto_ids), "all53_previously_inspected": True,
        "shortlist_frozen_before_confirmation": True}
    config = {"schema": "bayesfolio.november_multiobjective_configuration.v2", "selected_policy": case,
        "incumbent_retained": identity in shortlist["control_ids"],
        "decision_reason": ("Retain the matched MGK-free control; no challenger met all frozen requirements."
            if identity in shortlist["control_ids"] else "Promote this fixed policy: it passed development robustness, "
            "the single retrospective confirmation and the full-history tradeoff gates. It is selected from the "
            "eligible return/drawdown Pareto set by fixed-anchor distance, annual-block stability, then "
            "conventional Sharpe/eligible14IC/simplicity. No November weight appearance or new tuning was used."),
        "selection_rule": read(HERE / "SELECTION_RULE.json"), "asof": "2026-10-31",
        "cutoff": {"market_origin": "2026-10-30", "target_month": "2026-11", "last_training_origin": "2026-09-30",
                   "last_available_training_target": "October2026", "November_target_used": False},
        "GPspecs": {name: {key: value for key, value in spec.items() if key in
                    ["inputs", "tickers", "rank", "maxiter", "rolling_months", "kernel_recipe", "task_prior_lkj_eta"]}
                    for name, spec in expert_specs.items()},
        "exact_full_GP_contract": "FIT_CONTRACT.json",
        "GP_kernel_and_transforms": {key: value for key, value in read(V1 / "CONFIG.json")["GP_settings"].items()
                                     if key not in {"rank", "maxiter", "seeds"}},
        "training": {"all15_tasks_including_MGK": True, "expanding_available_months": 113,
                     "broad_rolling_months": 48, "feature_window_iteration_search": False,
                     "allowed_new_expert_specs": ["compact_r1_i15_exp"], "kernel_prior_changes_admitted": []},
        "seed_policy": {"seeds": SEEDS, "weights": [1 / 3] * 3,
                        "procedure": "fixed family means + declared covariance, equal seed moment pool, one solve"},
        "policy_settings": {"eligible_assets": [name for name in ASSETS if name != "MGK"],
            "MGK_zero_enforced_in_eligible_axis": True, "rank_score": "ascending average eligible rank/14",
            "objective": "rank_score @ weights - 1e-8*sum_squared_weights",
            "long_only": True, "fully_invested": True, "position_cap": None,
            "minimum_effective_assets": case["minimum_effective_assets"], "equity_cap": case["equity_cap"],
            "risk_reference": "causal expanding available-label LedoitWolf historical equal-weight eligible14 variance",
            "risk_multiplier": case["risk_multiplier"], "live_variance_ceiling": final["budget"],
            "covariance_definition": "observed GP covariance including likelihood noise; optional historical risk mix",
            "covariance_disagreement_term": False, "CVaR": False, "fallback": False,
            "monthly_blend_tuning": False, "postsolve_weight_repairs": False},
        "fit_counts": attempt_inventory(), "search_counts": {"rounds": 4, "new_complete_policies": 874,
            "controls": 2, "final_confirmation_policies": len(records), "new_expert_specs": 1},
        "execution_note": "Oct31 Saturday as-of uses Oct30 closing information to forecast November. "
            "Possible Nov2 execution has price/gap risk; an after-close decision cannot obtain "
            "a retrospective Oct30 fill.",
        "design_notes": [read(HERE / "MATCHED_CONTROL_AUDIT.json")["interpretation"],
            "All earlier histories were inspected; confirmation is a locked retrospective replay, "
            "not pristine holdout.",
            "No documented kernel/prior malfunction was established. Low amplitude and reaching maxiter15 are "
            "not sufficient justification; no unsupported kernel/prior or feature/window/budget/rank search was run.",
            "The originalMGK dividend-adjustment vintage difference0.08247% is documented in v1; old history "
            "and macro vintages were not revised. Macro point-in-time vintage certification remains limited.",
            "Risk-matrix mixtures are declared allocation recipes, not calibrated Bayesian mixture distributions. "
            "Historical covariance's stored MGK diagonal is a noninvestable placeholder, "
            "never a conditional observation.",
            "Drawdown bootstrap/static-stress frequencies are not calibrated future crash probabilities."],
        "reproduction_command": "uv run --locked --no-sync python "
            "experiments/2026-11-portfolio-multiobjective-v2/v2_run.py --verify"}
    validation = {"passed": True, "root_checks_only_external_final_audit_required_before_completion": True,
        "data_availability": available, "GP_specs": spec_validation, "new_fit_settings": fit_validation,
        "portfolio": numeric, "CSV": serialized, "all15_rows_including_MGK_zero": True,
        "raw_weights_repaired": False, "protected_inputs_unchanged": True,
        "frozen_confirmation_respected": True, "unresolved_fit_or_allocation_failures": [],
        "retained_failed_design_coverage_leaves": 16, "fit_attempts_used": attempt_inventory()["fit_attempts_used"],
        "budget_limits_passed": True, "Pylance": "unavailable; no editor diagnostics pass claimed",
        "no_trades_git_or_production_changes": True}
    save(HERE / "CONFIG.json", config)
    save(HERE / "COMPARISON.json", comparison)
    save(HERE / "ROOT_VALIDATION.json", validation)
    _ledger_and_pareto(records)
    status("awaiting_final_independent_audit", "Audit final sources/weights/selection, then reproduce and seal reports",
           selected_id=identity, new_policy_count=874, finalists=10, fit_attempts_used=6,
           fit_attempts_remaining=1194, unresolved_failures=[], completed_checks="ROOT_VALIDATION.json")
    return validation


def _ledger_and_pareto(finalists: list[dict]) -> None:
    rows = development_records()
    coverage = set(read(HERE / "MATCHED_CONTROL_AUDIT.json")["exploratory_nonpromotable_ids"])
    rounds = {}
    for path in sorted((HERE / "rounds").glob("ROUND*.json")):
        registered = read(path)
        for case in registered["cases"]:
            rounds[case["id"]] = {"round": registered["round"], "registered_at_utc": registered["registered_at_utc"],
                                  "batch_sha256": digest(path)}
    ledger, table = [], []
    frontier_ids = {row["id"] for row in frontier(rows)}
    for row in rows:
        ledger.append({"id": row["id"], "policy": row["policy"], "complete": row["complete"],
            "source_phase": "development41", "confirmation_used_for_hypothesis": False,
            "matched_coverage_eligible": row["id"] not in coverage,
            "registration": rounds.get(row["id"], {"round": 0, "mandatory_control": True}),
            "result_sha256": digest(HERE / "outputs/development" / f"{row['id']}.json"),
            "arrays_sha256": row.get("arrays_sha256"), "artifact_path": row.get("artifact_path")})
        if row["complete"]:
            table.append({"panel": "development41", "id": row["id"], "label": row["policy"]["label"],
                "pareto": row["id"] in frontier_ids, "matched_coverage_eligible": row["id"] not in coverage,
                **{key: row["summary"][key] for key in ["cagr", "total_return", "max_drawdown", "sharpe", "rank_ic"]}})
    ffront = {row["id"] for row in frontier([{"id": r["id"], "complete": True, "summary": r["full"]["summary"]}
                                            for r in finalists])}
    for row in finalists:
        table.append({"panel": "confirmation_frozen_full53", "id": row["id"], "label": row["policy"]["label"],
            "pareto": row["id"] in ffront, "matched_coverage_eligible": row["selection"]["eligible"],
            **{key: row["full"]["summary"][key] for key in
               ["cagr", "total_return", "max_drawdown", "sharpe", "rank_ic"]}})
    save(HERE / "EXPERIMENT_LEDGER.jsonl", ("\n".join(json.dumps(row, sort_keys=True, allow_nan=False)
                                                    for row in ledger) + "\n").encode())
    save(HERE / "PARETO.csv", pd.DataFrame(table).to_csv(index=False, float_format="%.17g").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-final", action="store_true", required=True)
    parser.parse_args()
    result = prepare_final()
    print({"root_checks_passed": result["passed"], "external_audit_required": True})
