"""Correct November's hard minimum using existing frozen GP forecasts only.

Experiment composition/I/O, no fitting/trading/production changes. All fifteen
tasks remain in the GP; fourteen eligible ETFs are ranked BEFORE optimization.
Returns/weights are decimals, covariance squared decimals, NAV dollars.
Usage: uv run --locked --no-sync python <this_file> --execute|--verify
"""

from __future__ import annotations

import os

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS"]:
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-v3-mpl")

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "2026-11-portfolio-multiobjective-v2"
sys.path.insert(0, str(V2))

from v2_common import ASSETS, FAMILIES, _jsonable, arrays, digest, read, verify_hashes  # noqa: E402
from v2_driver import distance  # noqa: E402
from v2_evaluate import policy, pooled  # noqa: E402
from v2_finish import acceptance, live_source  # noqa: E402
from v2_independent_validation import validate_portfolio, validate_weights_csv  # noqa: E402
from v2_metrics import score_portfolio  # noqa: E402
from v2_policy import allocate  # noqa: E402
from v2_robustness import assess  # noqa: E402


def write(path: Path, value) -> None:
    """Write new-version evidence without replacing sealed previous experiments.

    Args:
        path: New v3 artifact only.
        value: Exact bytes or JSON-compatible numerical evidence.
    """
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError("Only the new correction folder is writable")
    content = value if isinstance(value, bytes) else (
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Immutable correction conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def snapshot(path: Path, values: dict) -> None:
    """Save explicit numerical correction evidence.

    Args:
        path: New NPZ artifact.
        values: Explicit named arrays; financial values in decimal units.
    """
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    write(path, stream.getvalue())


def checked_allocation(mean: np.ndarray, covariance: np.ndarray, case: dict, budget: float | None) -> dict:
    """Enforce both NE>=7 and at least seven meaningful eligible holdings.

    Args:
        mean: Original fifteen-task pooled decimal predictions.
        covariance: Original observed fifteen-task covariance.
        case: Exact complete policy, which must state NE floor at least seven.
        budget: Registered causal variance ceiling, squared decimals.

    Returns:
        Accepted unmodified eligible14 allocation, with exact MGK zero.
    """
    if case["minimum_effective_assets"] < 7:
        raise ValueError("November's minimum effective asset floor is seven")
    result = allocate(mean, covariance, min_effective_assets=case["minimum_effective_assets"],
                      variance_ceiling=budget, equity_cap=case["equity_cap"])
    if not result["passed"]:
        raise ValueError("The registered corrected portfolio is infeasible; no fallback or relaxation")
    weights = np.asarray(result["weights"])
    count = int(np.count_nonzero(weights >= .0001))
    if weights[ASSETS.index("MGK")] != 0 or count < 7 or 1 / (weights @ weights) < 7 - 1e-6:
        raise ValueError("MGK exclusion or corrected seven-asset requirement failed")
    return {**result, "meaningful_holdings": count, "count_threshold": .0001}


def register() -> dict:
    """Freeze the narrow correction and unchanged old decision rule before scoring.

    Returns:
        Three fixed cached policies: corrected blend and two min7 controls.
    """
    path = ROOT / "CORRECTION_SPEC.json"
    if path.exists():
        value = read(path)
        verify_hashes(value["protected_sha256"])
        return value
    seal = read(V2 / "COMPLETED.json")
    verify_hashes(seal["artifact_sha256"])
    old = read(V2 / "CONFIG.json")["selected_policy"]
    corrected = policy("corrected_v2_blend_NE7_MGK_excluded_before_allocation",
        np.asarray(old["prediction_fractions"]), nea=7, risk=old["risk_multiplier"],
        covariance=old["covariance_fractions"], equity_cap=old["equity_cap"],
        prediction_experts=old["prediction_experts"], covariance_experts=old["covariance_experts"])
    previous = read(V2 / "COMPARISON.json")
    controls = [row["policy"] for row in previous["records"] if row["id"] in previous["control_ids"]]
    value = {"registered_at_utc": datetime.now(UTC).isoformat(), "minimum_effective_assets": 7,
        "minimum_meaningful_holdings": 7, "meaningful_weight_threshold": .0001,
        "GP_tasks": ASSETS, "eligible_assets": [asset for asset in ASSETS if asset != "MGK"],
        "MGK_exclusion": "before ranking and optimization, not after a top7 allocation",
        "GP_fits": 0, "new_blending_search": False, "cases": [corrected, *controls],
        "control_ids": [case["id"] for case in controls], "corrected_candidate_id": corrected["id"],
        "selection": "Reuse v2 frozen anchors, development robustness and final acceptance rules for the "
                     "one corrected fixed-blend challenger; retain best unchanged min7 control if it fails.",
        "evaluation": "Retrospective correction on the same53months; no new untouched holdout claim.",
        "old_recommendation": "v2 NE3 weights do not meet the now-hard November minimum and are superseded.",
        "protected_sha256": {str(p): digest(p) for p in [V2 / "COMPLETED.json", V2 / "CONFIG.json",
            V2 / "ANCHORS.json", V2 / "SELECTION_RULE.json", V2 / "outputs/confirmation_inputs.npz",
            V2 / "outputs/context.npz", V2 / "v2_policy.py", V2 / "v2_evaluate.py",
            V2 / "v2_metrics.py", V2 / "v2_finish.py", V2 / "v2_robustness.py",
            V2 / "v2_independent_validation.py"]}}
    write(path, value)
    return value


def evaluate(case: dict, source: dict, live: dict) -> dict:
    """Replay one fixed policy with the corrected constraint on every month.

    Args:
        case: Registered min7 portfolio policy.
        source: Frozen53-month expert and outcome arrays.
        live: Separate unknown-November-target forecast moments.

    Returns:
        Complete history, development/latest metrics, live raw allocation.
    """
    mean, covariance = pooled(case, source)
    solved = []
    for index in range(53):
        budget = None if case["risk_multiplier"] is None else float(
            case["risk_multiplier"] * source["historical_EW_variance"][index])
        solved.append(checked_allocation(mean[index], covariance[index], case, budget))
    weights = np.asarray([row["weights"] for row in solved])
    full, paths = score_portfolio(source["actual"], source["rf"], mean, weights, assets=ASSETS, families=FAMILIES)
    dev, _ = score_portfolio(source["actual"][:41], source["rf"][:41], mean[:41], weights[:41],
                              assets=ASSETS, families=FAMILIES)
    latest, _ = score_portfolio(source["actual"][41:], source["rf"][41:], mean[41:], weights[41:],
                                 assets=ASSETS, families=FAMILIES)
    new_mean, new_covariance = pooled(case, live)
    budget = None if case["risk_multiplier"] is None else float(
        case["risk_multiplier"] * live["historical_EW_variance"][0])
    november = checked_allocation(new_mean[0], new_covariance[0], case, budget)
    path = ROOT / "outputs" / f"{case['id']}.npz"
    numeric = {"mean": mean, "covariance": covariance, "weights": weights, "dates": source["dates"],
        "rf": source["rf"], "actual": source["actual"], "wealth": paths["wealth"],
        "drawdown": paths["drawdown"], "total_returns": paths["total_returns"], "rank_ic": paths["rank_ic"],
        "cost_10_total_returns": paths["cost_cases"]["10"]["total_returns"],
        "live_mean": new_mean[0], "live_covariance": new_covariance[0],
        "live_weights": np.asarray(november["weights"]), "live_budget": np.array(budget)}
    snapshot(path, numeric)
    development_path = path.with_name(path.stem + "_development.npz")
    snapshot(development_path, {key: value[:41] for key, value in numeric.items()
                               if key in ["dates", "rf", "total_returns", "rank_ic", "cost_10_total_returns"]})
    result = {"id": case["id"], "policy": case, "complete": True, "full": full, "development": dev,
        "latest12": latest, "november": november, "november_budget": budget, "numerical": solved,
        "minimum_historical_holdings": min(row["meaningful_holdings"] for row in solved),
        "artifact_path": str(path), "development_artifact_path": str(development_path)}
    write(path.with_suffix(".json"), result)
    return result


def execute() -> dict:
    """Reallocate and deliver a min7 November recommendation without any GP fits.

    Returns:
        Passed correction validation after independent conic and reproduction.
    """
    spec = register()
    if (ROOT / "COMPLETED.json").exists():
        return verify()
    source = arrays(V2 / "outputs/confirmation_inputs.npz")
    live = live_source()
    rows = [evaluate(case, source, live) for case in spec["cases"]]
    controls = [row for row in rows if row["id"] in spec["control_ids"]]
    reference = min(controls, key=lambda r: (distance(r["development"]), r["id"]))
    candidate = rows[0]
    candidate_dev = {**candidate, "artifact_path": candidate["development_artifact_path"],
                     "numerical": candidate["numerical"][:41], "feasibility_complete": True}
    reference_dev = {**reference, "artifact_path": reference["development_artifact_path"],
                     "numerical": reference["numerical"][:41], "feasibility_complete": True}
    robustness = assess(candidate_dev, reference_dev, read(V2 / "ANCHORS.json"))
    best = min(controls, key=lambda r: (distance(r["full"]), r["id"]))
    best_latest = min(controls, key=lambda r: (distance(r["latest12"]), r["id"]))
    gates = acceptance({"complete": True, "summary": candidate["full"], "latest12": candidate["latest12"]},
        robustness, best_control_full={"summary": best["full"]},
        best_control_latest={"latest12": best_latest["latest12"]}, live_feasible=True)
    chosen = candidate if gates["eligible"] else best
    values = arrays(Path(chosen["artifact_path"]))
    case, raw = chosen["policy"], chosen["november"]
    weights = values["live_weights"]
    independent = validate_portfolio(values["live_mean"], values["live_covariance"], weights, ASSETS,
        min_effective_assets=7, variance_ceiling=chosen["november_budget"], equity_cap=case["equity_cap"],
        solver_status=raw["solver_status"], solver_attempts=raw["numerical_attempts"],
        analytic_certificate=raw["certificate"])
    frame = pd.DataFrame({"asset_id": ASSETS, "weight": weights, "as_of_date": "2026-10-31",
                          "market_origin": "2026-10-30", "forecast_month": "2026-11"})
    stream = io.StringIO()
    frame.to_csv(stream, index=False, float_format="%.17g")
    write(ROOT / "NOVEMBER_WEIGHTS.csv", stream.getvalue().encode())
    csv = validate_weights_csv(ROOT / "NOVEMBER_WEIGHTS.csv", values["live_mean"], values["live_covariance"],
        weights, ASSETS, min_effective_assets=7, variance_ceiling=chosen["november_budget"],
        equity_cap=case["equity_cap"], solver_status=raw["solver_status"],
        solver_attempts=raw["numerical_attempts"], analytic_certificate=raw["certificate"])
    old_config = read(V2 / "CONFIG.json")
    write(ROOT / "CONFIG.json", {"selected_policy": case, "GPspecs": old_config["GPspecs"],
        "GP_kernel_and_transforms": old_config["GP_kernel_and_transforms"], "training": old_config["training"],
        "seed_policy": old_config["seed_policy"], "cutoff": old_config["cutoff"],
        "minimum_effective_assets": 7, "minimum_meaningful_holdings": 7, "meaningful_weight_threshold": .0001,
        "MGK_training_task": True, "MGK_allocation": 0,
        "MGK_excluded_before_ranking_and_optimization": True, "posterior_GP_refits": 0,
        "selected_corrected_challenger": chosen["id"] == candidate["id"],
        "old_NE3_recommendation_superseded": True})
    write(ROOT / "COMPARISON.json", {"records": rows, "selected_id": chosen["id"],
        "corrected_candidate_gates": gates, "development_robustness": robustness,
        "retrospective_correction_not_new_holdout": True})
    repeated = checked_allocation(values["live_mean"], values["live_covariance"], case, chosen["november_budget"])
    np.testing.assert_allclose(repeated["weights"], weights, rtol=0, atol=1e-12)
    verify_hashes(read(V2 / "COMPLETED.json")["artifact_sha256"])
    validation = {"passed": True, "minimum_effective_assets": 7, "minimum_meaningful_holdings": 7,
        "November_holdings": raw["meaningful_holdings"],
        "November_effective_assets": raw["raw_checks"]["effective_assets"],
        "all53_months_meet_minimum": chosen["minimum_historical_holdings"] >= 7,
        "minimum_historical_holdings": chosen["minimum_historical_holdings"], "MGK_exactly_zero": weights[1] == 0,
        "MGK_preoptimization_exclusion": True, "GP_training_population_unchanged": ASSETS,
        "new_GP_fits": 0, "independent_optimizer": independent, "CSV": csv,
        "deterministic_reproduction_maximum_difference": float(np.max(abs(np.array(repeated["weights"]) - weights))),
        "old_experiments_unchanged": True, "trades_or_git_mutations": False}
    write(ROOT / "VALIDATION.json", validation)
    _report(rows, chosen, gates)
    files = [path for path in ROOT.rglob("*") if path.is_file()
             and path.name != "COMPLETED.json" and path.suffix not in {".log", ".pyc"}
             and "__pycache__" not in path.parts]
    write(ROOT / "COMPLETED.json", {"passed": True, "artifact_sha256": {str(path): digest(path) for path in files}})
    return verify()


def _report(rows: list[dict], chosen: dict, gates: dict) -> None:
    raw = chosen["november"]
    weight = np.asarray(raw["weights"])
    lines = ["# Corrected November recommendation: minimum seven", "",
        "This supersedes the prior NE3 allocation for November. The allocator now enforces NE>=7 and at least "
        "seven holdings of0.01% or more, AFTER excluding MGK from the eligible set. It ranks all14eligible "
        "ETFs; it never makes a top7 portfolio and deletes MGK afterward.", "",
        f"Chosen policy: `{chosen['policy']['label']}`. Holdings: {raw['meaningful_holdings']}; "
        f"effective assets: {raw['raw_checks']['effective_assets']:.8f}; MGK exactly0.", "",
        "All15 GP training tasks, model features, kernels, rank/maxiter settings and training windows are unchanged. "
        "No GP fits or new blend search occurred. The corrected fixed blend was compared with unchanged min7 "
        "controls using the prior frozen decision rules; old NE3 performance is not attributed to these weights.", "",
        "All data were inspected previously. This is a retrospective constraint correction, not a fresh holdout.", "",
        "| Policy | Full53 return | Full53 max drawdown | Sharpe | Latest12 return |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        full = row["full"]
        lines.append(f"| {row['policy']['label']} | {full['total_return']:.2%} | {full['max_drawdown']:.2%} | "
                     f"{full['sharpe']:.3f} | {row['latest12']['total_return']:.2%} |")
    lines += ["", "| ETF | Target (%) |", "|---|---:|"]
    lines += [f"| {asset} | {value:.4%} |" for asset, value in zip(ASSETS, weight, strict=True)]
    lines += ["", "Displayed percentages are rounded. The CSV retains raw decimal weights.", "",
        f"Corrected challenger passed the inherited promotion rule: {gates['eligible']}.", "",
        "Data origin remains October30,2026 predicting November. A decision made afterward cannot assume an "
        "October30closing fill. No trades, production changes, commits or pushes occurred.", "",
        "Verify from the repository root:", "", "```sh",
        "uv run --locked --no-sync python experiments/2026-11-portfolio-min7-correction-v3/run.py --verify", "```", ""]
    write(ROOT / "CORRECTION.md", "\n".join(lines).encode())


def verify() -> dict:
    """Independently verify the corrected raw CSV and minimum before handoff.

    Returns:
        Compact passed verification evidence. No files or models are modified.
    """
    completed = read(ROOT / "COMPLETED.json")
    verify_hashes(completed["artifact_sha256"])
    verify_hashes(read(ROOT / "CORRECTION_SPEC.json")["protected_sha256"])
    verify_hashes(read(V2 / "COMPLETED.json")["artifact_sha256"])
    configuration = read(ROOT / "CONFIG.json")
    case = configuration["selected_policy"]
    comparison = read(ROOT / "COMPARISON.json")
    row = next(item for item in comparison["records"] if item["id"] == case["id"])
    mean, covariance = pooled(case, live_source())
    repeated = checked_allocation(mean[0], covariance[0], case, row["november_budget"])
    weights = np.asarray(repeated["weights"])
    validate_weights_csv(ROOT / "NOVEMBER_WEIGHTS.csv", mean[0], covariance[0], weights, ASSETS,
        min_effective_assets=7, variance_ceiling=row["november_budget"], equity_cap=case["equity_cap"],
        solver_status=repeated["solver_status"], solver_attempts=repeated["numerical_attempts"],
        analytic_certificate=repeated["certificate"])
    return {"passed": True, "meaningful_holdings": repeated["meaningful_holdings"],
        "effective_assets": repeated["raw_checks"]["effective_assets"], "MGK_weight": weights[1],
        "minimum_effective_assets": 7, "new_GP_fits": 0, "old_experiments_unchanged": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(_jsonable(execute() if args.execute else verify()), indent=2))
