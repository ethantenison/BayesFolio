"""Versioned repair/report adapter for twelve frozen scenario allocation failures.

Reads the sealed campaign without modifying it. Writes only this version's
outputs directory. Decimal returns/covariance and fraction weights retain the
original units; deterministic scores, selection rules and static blend recipes
are reused unchanged. No GP fitting, downloads, production promotion or fallback.

Usage:
    uv run --locked --no-sync python <this_file> --execute
    uv run --locked --no-sync python <this_file> --verify
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
SOURCE = CAMPAIGN / "outputs"
OUTPUT = HERE / "outputs"
sys.path[:0] = [str(HERE), str(CAMPAIGN), str(CAMPAIGN / "track1_scenarios")]

import campaign  # noqa: E402
import case_evaluator as evaluator  # noqa: E402
import run_study  # noqa: E402
import run_track as track1  # noqa: E402
from allocator import allocate_scenarios  # noqa: E402
from positive_sharpe import solve_positive  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict | bytes) -> None:
    if not path.resolve().is_relative_to(OUTPUT):
        raise ValueError("Only new versioned outputs may be written")
    content = value if isinstance(value, bytes) else (
        json.dumps(evaluator._jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    evaluator._immutable_bytes(path, content)


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def _snapshot(path: Path, arrays: dict) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _write(path, buffer.getvalue())


def _verify_original() -> dict:
    seal = _read(SOURCE / "COMPLETED.json")
    if seal.get("passed") is not True:
        raise ValueError("Original campaign completion seal required")
    for key in ["input_sha256", "artifact_sha256", "implementation_sha256"]:
        campaign.verify_hashes(seal[key])
    return seal


def _sample(arrays: dict, identity: dict) -> np.ndarray:
    s, m = identity["seed_index"], identity["month_index"]
    return evaluator._dependency("sampling").sample_joint(
        arrays["mean"][s, m], arrays["covariance"][s, m],
        seed=identity["seed"], month=identity["origin"], count=5000)


def _repair_case(source_path: Path, context: dict) -> tuple[dict, list[dict]]:
    old = _read(source_path)
    record = copy.deepcopy(old)
    arrays = _arrays(source_path.with_suffix(".npz"))
    original_arrays = {name: value.copy() for name, value in arrays.items()}
    checks = []
    repair_keys = set()
    for failure in old["failures"]:
        if failure.get("stage") != "allocation" or "optimal_inaccurate" not in failure["error"]:
            raise ValueError("Repair is scoped exclusively to the known inaccurate allocation failures")
        s, m = failure["seed_index"], failure["month_index"]
        if arrays["allocated"][s, m]:
            raise ValueError("Cannot replace an already successful allocation")
        scenarios = _sample(arrays, failure)
        mean, covariance = scenarios.mean(0), np.cov(scenarios, rowvar=False)
        _, max_mean = evaluator._dependency("scenarios")._maximum_mean(mean)
        if max_mean <= 1e-9:
            raise ValueError("The nonpositive signed-Sharpe branch is outside this repair")
        result = allocate_scenarios(scenarios, arrays["assets"].tolist(),
                                    seed=20261024 + failure["seed"] + int(failure["origin"].replace("-", "")))
        independent = evaluator._dependency("scenarios").solve_signed(
            mean, covariance, seed=20261024 + failure["seed"] + int(failure["origin"].replace("-", "")),
            random_starts=4)
        difference = abs(result["monthly_forecast_sharpe"] - independent["monthly_forecast_sharpe"])
        if difference > 1e-8:
            raise ValueError("Conic and independent direct-Sharpe objectives disagree")
        audit = evaluator._audit(result["weights"], arrays["covariance"][s, m],
                                 float(context["risk_budget"][m]), "scenarios")
        evaluator._audit(result["raw_weights"], arrays["covariance"][s, m],
                         float(context["risk_budget"][m]), "scenarios")
        arrays["weights"][s, m] = result["weights"]
        arrays["raw_weights"][s, m] = result["raw_weights"]
        arrays["allocated"][s, m] = True
        for key in ["effective_count", "equity_weight", "predicted_variance"]:
            arrays[key][s, m] = audit[key]
        identity = {key: failure[key] for key in ["seed", "origin", "seed_index", "month_index"]}
        numerical = {**identity, "status": "allocated", "weight_audit": audit, **{
            key: value for key, value in result.items() if key not in {"weights", "raw_weights"}}}
        record["numerical_records"] = [numerical if (r["seed_index"], r["month_index"]) == (s, m) else r
                                       for r in record["numerical_records"]]
        check = {"case_id": old["id"], **identity, "original_failure": failure,
                 "sample_sha256": hashlib.sha256(scenarios.tobytes()).hexdigest(),
                 "maximum_feasible_mean": max_mean, "numerical": numerical,
                 "independent_direct_sharpe": independent["monthly_forecast_sharpe"],
                 "independent_successful_starts": independent["successful_starts"],
                 "independent_objective_reproductions": independent["best_objective_reproductions"],
                 "independent_objective_difference": difference}
        checks.append(check)
        repair_keys.add((s, m))
    expected = set(map(tuple, np.argwhere(~original_arrays["allocated"])))
    if repair_keys != expected or not arrays["allocated"].all():
        raise ValueError("Every and only originally failed month must be repaired")
    mask = original_arrays["allocated"]
    for key in ["weights", "raw_weights", "effective_count", "equity_weight", "predicted_variance"]:
        np.testing.assert_array_equal(arrays[key][mask], original_arrays[key][mask],
                                      err_msg="Previously successful allocation changed")
    for key in ["mean", "covariance", "dates", "assets", "seeds", "actual", "risk_free"]:
        np.testing.assert_array_equal(arrays[key], original_arrays[key])
    periods, calibration = evaluator._scores(context, arrays["mean"], arrays["covariance"], arrays)
    arrays["status"] = np.array("complete")
    record.update(status="complete", successful_allocations=84, failed_allocation_pairs=0, failures=[],
                  failure_kind_counts={}, periods=periods, calibration=calibration,
                  numerical_route_counts=dict(Counter(r["solver_route"] for r in record["numerical_records"])))
    record["repair_provenance"] = {"original_record": str(source_path), "original_sha256": _hash(source_path),
                                    "original_failures": old["failures"], "new_monthly_allocations": len(checks),
                                    "successful_original_weights_bitwise_preserved": True,
                                    "forecast_arrays_bitwise_preserved": True, "GP_refits": 0}
    record["input_sha256"].update({str(source_path): _hash(source_path),
                                     str(source_path.with_suffix(".npz")): _hash(source_path.with_suffix(".npz")),
                                     str(HERE / "positive_sharpe.py"): _hash(HERE / "positive_sharpe.py"),
                                     str(HERE / "allocator.py"): _hash(HERE / "allocator.py"),
                                     str(Path(__file__)): _hash(Path(__file__))})
    artifact = OUTPUT / "cases" / f"{old['id']}.npz"
    _snapshot(artifact, arrays)
    record["artifact_sha256"] = _hash(artifact)
    _write(artifact.with_suffix(".json"), record)
    return record, checks


def _positive_controls() -> list[dict]:
    # Replay every previously successful positive-mean solve: this is a full
    # numerical regression, not a favorable subset or a new portfolio selection.
    controls = []
    for path in sorted((SOURCE / "cases").glob("t1_r*_*.json")):
        record = _read(path)
        if record["case"]["allocator"] != "scenarios":
            continue
        candidates = [r for r in record["numerical_records"]
                      if r.get("solver_route") == "riskfolio_MV_positive_mean" and r["status"] == "allocated"]
        if not candidates:  # Exact baseline cache never invoked this numerical solver.
            continue
        arrays = _arrays(path.with_suffix(".npz"))
        for identity in candidates:
            scenarios = _sample(arrays, identity)
            mean, covariance = scenarios.mean(0), np.cov(scenarios, rowvar=False)
            result = solve_positive(mean, covariance)
            old_score = float(identity["monthly_forecast_sharpe"])
            difference = result["monthly_forecast_sharpe"] - old_score
            if abs(difference) > 1e-7:
                raise ValueError(f"Equivalent formulations disagree on successful control: {path.stem}")
            controls.append({"case_id": path.stem, "seed": identity["seed"], "origin": identity["origin"],
                             "original_riskfolio_sharpe": old_score,
                             "reformulated_sharpe": result["monthly_forecast_sharpe"],
                             "difference": difference, "certificate": result["certificate"]})
        print(f"Numerical regression {path.stem}: {len(candidates)} passed", flush=True)
    return controls


def _report_track1(records: list[dict], manifest: dict) -> dict:
    selector = track1._recipes()
    selected = selector.select_candidates(track1._candidate_rows(records, "tuning_first16"), limit=3)
    periods = ["tuning_first16", "validation_last12", "all28"]
    frontiers = {p: selector.pareto_ids(track1._candidate_rows(records, p)) for p in periods}
    by_id = {r["id"]: r for r in records}
    summary = {**_read(SOURCE / "track1/summary.json"),
               "schema": "bayesfolio.gp_tuning.track1_results.numerics_v2", "selected_case_ids": selected,
               "selected_best_case": by_id[selected[0]]["case"],
               "selected_cases": [by_id[name]["case"] for name in selected],
               "complete_scenario_cases": 36, "failed_scenario_cases": 0,
               "pareto_case_ids_by_period": frontiers, "paired_contrasts": track1._contrasts(records),
               "failures": [], "case_manifest_sha256": _hash(OUTPUT / "track1/case_manifest.json"),
               "case_sha256": {str(OUTPUT / "cases" / f"{row['id']}.{extension}"):
                               _hash(OUTPUT / "cases" / f"{row['id']}.{extension}")
                               for row in records for extension in ["json", "npz"]},
               "repair_manifest": manifest}
    _write(OUTPUT / "track1/table.csv", track1._table(records, selected, frontiers["all28"]))
    _write(OUTPUT / "track1/RESULTS.md", track1._report(summary, records))
    _write(OUTPUT / "track1/summary.json", summary)
    return summary


def _execute() -> None:
    if (OUTPUT / "COMPLETED.json").exists():
        _verify()
        return
    started = time.monotonic()
    _verify_original()
    print("Original sealed campaign verified; no fitting or input changes", flush=True)
    source_paths = sorted((SOURCE / "cases").glob("*.json"))
    failed_paths = [p for p in source_paths if _read(p)["status"] != "complete"]
    if len(failed_paths) != 10 or sum(len(_read(p)["failures"]) for p in failed_paths) != 12:
        raise ValueError("Expected exactly ten failed cases and twelve missing allocations")
    manifest = {"schema": "bayesfolio.gp_tuning.scenario_numerics_v2", "original_seal": str(SOURCE / "COMPLETED.json"),
                "original_seal_sha256": _hash(SOURCE / "COMPLETED.json"),
                "failed_cases": [p.stem for p in failed_paths], "repair_allocations": 12,
                "unchanged_objective": "signed scenario mean divided by sample standard deviation, rf=0",
                "constraints": {"long_only": True, "fully_invested": True, "effective_assets_min": 10,
                                "max_position": .25}, "GP_fits": 0, "fallback": False,
                "monthly_blend_adaptation": False, "only_missing_allocations_replaced": True,
                "input_sha256": {str(SOURCE / "COMPLETED.json"): _hash(SOURCE / "COMPLETED.json")},
                "implementation_sha256": {str(p): _hash(p) for p in sorted(HERE.glob("*.py"))}}
    _write(OUTPUT / "REPAIR_MANIFEST.json", manifest)
    controls = _positive_controls()
    _write(OUTPUT / "POSITIVE_CONTROL_CHECKS.json", {"passed": True, "controls": controls})
    print(f"{len(controls)} successful Riskfolio controls match the equivalent formulation", flush=True)
    context = _arrays(SOURCE / "context.npz")
    checks = []
    failed_ids = {p.stem for p in failed_paths}
    for path in source_paths:
        if path.stem in failed_ids:
            record, local = _repair_case(path, context)
            checks.extend(local)
            print(f"Repaired {record['id']}: {record['successful_allocations']}/84", flush=True)
        else:
            # Exact byte copies, not hard links; originals cannot be overwritten
            # through the new view. Frozen internal provenance remains original.
            for extension in ["json", "npz"]:
                old = path.with_suffix(f".{extension}")
                _write(OUTPUT / "cases" / old.name, old.read_bytes())
    _write(OUTPUT / "ALLOCATION_REPAIRS.json", {"passed": True, "repaired": checks})
    for name in ["context.npz", "FITS_COMPLETED.json"]:
        _write(OUTPUT / name, (SOURCE / name).read_bytes())
    _write(OUTPUT / "track1/case_manifest.json", {
        **_read(SOURCE / "track1/case_manifest.json"), "numerical_repair": manifest})
    records = [_read(OUTPUT / "cases" / p.name) for p in source_paths if p.stem.startswith("t1_")]
    summary1 = _report_track1(records, manifest)
    # Track 2 and Track 3 have no affected allocations or scenario-based
    # selection dependency. Their reports and scores remain exact byte copies.
    for folder in ["track2", "track3"]:
        for path in sorted((SOURCE / folder).rglob("*")):
            if path.is_file():
                _write(OUTPUT / folder / path.relative_to(SOURCE / folder), path.read_bytes())
    summaries = {"track1": summary1, "track2": _read(SOURCE / "track2/summary.json"),
                 "track3": _read(SOURCE / "track3/summary.json")}
    with patch.object(campaign, "OUTPUT", OUTPUT), patch.object(run_study, "OUTPUT", OUTPUT):
        run_study._combine(summaries)
    _write_diagnostic_report(checks, controls)
    _verify_original()
    seal = {"passed": True, "new_GP_fits": 0, "repaired_monthly_allocations": len(checks),
            "repaired_cases": len(failed_paths), "complete_cases": 589, "failed_cases": 0,
            "complete_scenario_cases": 36, "successful_original_weights_bitwise_preserved": True,
            "original_seal_sha256": _hash(SOURCE / "COMPLETED.json"),
            "seconds": time.monotonic() - started,
            "implementation_sha256": manifest["implementation_sha256"],
            "artifact_sha256": {str(p): _hash(p) for p in sorted(OUTPUT.rglob("*"))
                                if p.is_file() and p.name != "COMPLETED.json"}}
    _write(OUTPUT / "COMPLETED.json", seal)
    _verify()


def _write_diagnostic_report(checks: list[dict], controls: list[dict]) -> None:
    before = _read(SOURCE / "GLOBAL_SUMMARY.json")
    after = _read(OUTPUT / "GLOBAL_SUMMARY.json")
    old_track = _read(SOURCE / "track1/summary.json")
    new_track = _read(OUTPUT / "track1/summary.json")
    max_residual = max(max(c["numerical"]["certificate"]["primal_residual"],
                           c["numerical"]["certificate"]["dual_residual"]) for c in checks)
    shortlist_changed = before["first16_selected_case_ids"] != after["first16_selected_case_ids"]
    leaders_changed = (before["all28_descriptive_leaders_not_selection_inputs"]
                       != after["all28_descriptive_leaders_not_selection_inputs"])
    lines = ["# Scenario numerical repair: complete comparison", "",
             "## Result", "",
             "All 12 formerly failed allocations now have strict optimal status and pass raw-weight, "
             "primal/dual residual, duality-gap and independent direct-Sharpe checks. "
             "All 36 scenario configurations and 589 total portfolio cases are now complete; zero remain excluded.",
             "No GP refits. All previously valid weights, GP forecasts, actuals and scenarios are unchanged. "
             "Tracks 2/3 and baseline portfolios are unchanged. The original sealed campaign remains untouched.", "",
             "## Cause and fix", "",
             "Riskfolio MV/Sharpe's positive-mean branch imposes mu @ x = 1 and minimizes scaled variance. "
             "With nearly cancelling means and diversification constraints, the largest feasible portfolio mean "
             "can be tiny. Its homogeneous scale must then be enormous: k >= 1/max_feasible_mean. "
             "The observed failed cases imply lower bounds on k of roughly 866 to 164,126. "
             "This is a numerical scaling issue, not an infeasible portfolio or non-SPD covariance.", "",
             "The replacement solves the mathematically equivalent positive-Sharpe SOCP: "
             "maximize (mu/s) @ x subject to ||L.T @ x/s|| <= 1, x >= 0, "
             "x <= .25 sum(x), and ||x|| <= sum(x)/sqrt(10), then w=x/sum(x). "
             "Here L L.T is the unchanged sample covariance and s is equal-weight volatility. "
             "The fixed unit-risk constraint normalizes auxiliary holdings, not portfolio risk: "
             "there is no new variance budget on the resulting fully invested w.", "",
             "This is not minimum variance, an alternative allocator, acceptance of optimal_inaccurate, "
             "or a fallback portfolio. It maximizes the same signed mean/standard-deviation objective. "
             "Nonpositive-mean allocation results retain their existing signed-Sharpe route unchanged. "
             "Full-accuracy solver gaps remain 1e-10; scaled feasibility tolerance is 1e-9; "
             "original economic constraint tolerances are unchanged.", "",
             "The reusable `allocator.allocate_scenarios` supports mixed and all-negative returns; "
             "there is no positive-return floor, clipping, shifting or dropping of observations. "
             "`allocator.evaluate_case` wires it into NEW evaluation batches by default and fingerprints "
             "the new numerical code. Sealed historical runners remain untouched.", "",
             "## Validation", "",
             f"- {len(checks)} strict-optimal repairs, each reproduced by direct signed-Sharpe multistart.",
             f"- {len(controls)} previously successful Riskfolio controls agree within 1e-7 monthly forecast Sharpe.",
             f"- Largest independent repaired-objective difference: "
             f"{max(c['independent_objective_difference'] for c in checks):.3g}.",
             f"- Largest repaired primal/dual residual: {max_residual:.3g}.",
             "- Original full evidence and implementation hashes reverified before and after the repair.", "",
             "## Selection impact", "",
             f"- Previous Track 1 first16 shortlist: {', '.join(old_track['selected_case_ids'])}.",
             f"- Repaired Track 1 first16 shortlist: {', '.join(new_track['selected_case_ids'])}.",
             f"- Global first16 shortlist changed: {shortlist_changed}.",
             f"- All28 descriptive leaders changed: {leaders_changed}.",
             "Selection still uses first16 only, with later12 and all28 separate descriptions. "
             "No baseline promotion; history was previously viewed and is not a pristine holdout.", "",
             "## Updated evidence", "",
             "- [Integrated results](RESULTS.md)", "- [Scenario comparison](track1/RESULTS.md)",
             "- [Global comparison plot](GLOBAL_COMPARISON.png)",
             "- `ALLOCATION_REPAIRS.json`: original failures, sample hashes, raw audits and solver certificates.",
             "- `POSITIVE_CONTROL_CHECKS.json`: objective parity on previously successful solves.", "",
             "Primary numerical references: [CVXPY SOCP formulation and duals]"
             "(https://www.cvxpy.org/examples/basic/socp.html), [Clarabel accuracy settings]"
             "(https://clarabel.org/stable/api_settings/).", ""]
    _write(OUTPUT / "REPAIR_REPORT.md", "\n".join(lines).encode())


def _verify() -> None:
    _verify_original()
    seal = _read(OUTPUT / "COMPLETED.json")
    if seal.get("passed") is not True or seal["original_seal_sha256"] != _hash(SOURCE / "COMPLETED.json"):
        raise ValueError("Original and repaired seals must both pass")
    campaign.verify_hashes(seal["implementation_sha256"])
    campaign.verify_hashes(seal["artifact_sha256"])
    records = [_read(p) for p in sorted((OUTPUT / "cases").glob("*.json"))]
    relevant = [r for r in records if r["case"]["track"] != "audit"]
    if len(relevant) != 589 or any(r["status"] != "complete" or r["successful_allocations"] != 84 for r in relevant):
        raise ValueError("Every one of 589 cases must be complete")
    for row in records:
        path = OUTPUT / "cases" / f"{row['id']}.npz"
        if row["artifact_sha256"] != _hash(path):
            raise ValueError("Case artifact seal mismatch")
        old = _read(SOURCE / "cases" / f"{row['id']}.json")
        if "repair_provenance" not in row:
            if row != old or _hash(path) != _hash(SOURCE / "cases" / path.name):
                raise ValueError("An originally complete case changed")
        else:
            previous = _arrays(SOURCE / "cases" / path.name)
            current = _arrays(path)
            mask = previous["allocated"]
            for key in ["weights", "raw_weights", "effective_count", "equity_weight", "predicted_variance"]:
                np.testing.assert_array_equal(previous[key][mask], current[key][mask])
            for key in ["mean", "covariance", "dates", "assets", "seeds", "actual", "risk_free"]:
                np.testing.assert_array_equal(previous[key], current[key])
    repairs = _read(OUTPUT / "ALLOCATION_REPAIRS.json")["repaired"]
    if len(repairs) != 12 or any(r["numerical"]["solver_status"] != "optimal" for r in repairs):
        raise ValueError("Twelve strict-optimal repairs required")
    global_summary = _read(OUTPUT / "GLOBAL_SUMMARY.json")
    if global_summary["complete_cases"] != 589 or global_summary["failed_cases"] != 0:
        raise ValueError("Global report incomplete")
    print("Verified: 12/12 repairs; 36/36 scenario and 589/589 total cases; original evidence unchanged", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    _execute() if arguments.execute else _verify()
