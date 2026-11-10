"""Package audited research fronts and explicit hosted decisions without new fits.

This experiment-only reporting command joins the existing deterministic tables
to a human-reviewed decision file. It never selects a weighted winner, launches
experiments, changes frozen evidence, or certifies economic generalization.
Returns, RMSE and forecast spreads are decimals; turnover is not consumed.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/package_final_results.py

Requires planning/final_decisions.json and successful complete-scope audits.
Writes results/final_recommendations.json and results/FINAL_PARETO_TABLES.md.
The separate final scientific report and completion review remain required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import repair_dataset as data


def validate_decisions(decisions: dict[str, Any], fronts: dict[str, set[str]]) -> None:
    """Reject unaudited recommendations or an implicit production promotion.

    Args:
        decisions: Hosted interpretation with explicit selected frontier IDs.
        fronts: Current complete Pareto identifiers by model universe.

    Raises:
        ValueError: A selected mask is not on its current front or scope changed.
    """
    if decisions.get("production_promotion") is not False:
        raise ValueError("This study does not authorize production promotion")
    if set(decisions["selected_configurations"]) != set(fronts):
        raise ValueError("Recommendations must cover the four families and actual joint model")
    for family, choices in decisions["selected_configurations"].items():
        if not choices or any(identifier not in fronts[family] for identifier in choices.values()):
            raise ValueError(f"Recommendation outside the verified {family} Pareto front")
    if decisions["shared_research_lead"] not in decisions["selected_configurations"]["global"].values():
        raise ValueError("The shared research lead must be an explicitly selected joint finalist")


def validate_sources(report: dict[str, Any], root: Path) -> None:
    """Check every recorded source digest before consuming an audit or synthesis.

    Args:
        report: Machine-readable evidence with repository-relative source hashes.
        root: Repository root owning those paths.

    Raises:
        ValueError: A source no longer matches its recorded digest.
    """
    hashes = report.get("source_sha256")
    if not hashes:
        raise ValueError("Final packaging requires source-hashed evidence")
    for relative, expected in hashes.items():
        if data.digest(root / relative) != expected:
            raise ValueError(f"Stale evidence source: {relative}")


def package() -> None:
    """Write exact final fronts and hosted recommendations from complete audits.

    Raises:
        ValueError: Registered work, metrics, source fingerprints or decisions are unresolved.
    """
    results = data.HERE / "results"
    cumulative = results / "cumulative"
    decision_path = data.HERE / "planning/final_decisions.json"
    decisions = json.loads(decision_path.read_text())
    audit_paths = [results / name for name in ["execution_coverage_audit.json", "joint_objective_audit.json",
                                              "model_contract_audit.json", "component_synthesis/manifest.json"]]
    audits = [json.loads(path.read_text()) for path in audit_paths]
    execution, joint, model, synthesis = audits
    if not execution["execution_scope_complete"] or execution["unresolved"]:
        raise ValueError("Execution scope is not complete")
    if not joint["passed"] or joint["unresolved"] or not model["passed"]:
        raise ValueError("Final independent arithmetic/model audit has not passed")
    for report in audits:
        validate_sources(report, data.ROOT)
    if json.loads((cumulative / "incomplete_runs.json").read_text()):
        raise ValueError("Incomplete registered runs remain")
    status = json.loads((cumulative / "status.json").read_text())
    expected: dict[str, set[str]] = {}
    sources = [decision_path, *audit_paths, cumulative / "status.json", cumulative / "incomplete_runs.json"]
    sources.extend(data.HERE / relative for relative in decisions["evidence"])
    for path in sorted((data.HERE / "plans").glob("*.json")):
        plan = json.loads(path.read_text())
        expected.setdefault(plan["family"], set()).update(item["id"] for item in plan["configurations"])
        sources.append(path)
    expected_count = sum(len(value) for value in expected.values())
    if execution["verified_configurations_in_complete_batches"] != expected_count:
        raise ValueError("Execution audit does not cover the current registered masks")
    if joint["checked_seed_runs"] != len(expected["global"]) * len(data.STUDY["seeds"]):
        raise ValueError("Joint arithmetic audit does not cover every registered seed")
    fronts: dict[str, set[str]] = {}
    records: dict[str, dict[str, Any]] = {}
    baselines: dict[str, dict[str, Any]] = {}
    control = json.loads((data.HERE / "plans/00_global_all_input_control.json").read_text())
    all_inputs = set(control["configurations"][0]["expected_input_columns"])
    tables = ["# Complete final Pareto tables", "",
              "All points use three complete seeds (27, 127, 227). Mean objectives and sample seed SD are shown.",
              "Counts exclude t_index, which is retained in every exact list. Return and RMSE are diagnostics only.",
              "Return means compounded monthly excess return, not total nominal investment wealth; RMSE is decimal.",
              "Seed dispersion is not economic confidence. Adaptive research fronts are not deployment approval.",
              ""]
    for family in [*data.FAMILIES, "global"]:
        folder = cumulative / family
        names = ["configuration_aggregates.csv", "objectives_by_seed.csv", "family_diagnostics_by_seed.csv",
                 "pareto.csv", "exact_inputs.json", "hypervolume.json"]
        sources.extend(folder / name for name in names)
        aggregate = pd.read_csv(folder / "configuration_aggregates.csv")
        seeds = pd.read_csv(folder / "objectives_by_seed.csv")
        diagnostics = pd.read_csv(folder / "family_diagnostics_by_seed.csv")
        frontier = pd.read_csv(folder / "pareto.csv").sort_values("mean_ic", ascending=False)
        exact = json.loads((folder / "exact_inputs.json").read_text())
        if set(aggregate.configuration) != expected[family] or len(aggregate) != len(expected[family]):
            raise ValueError(f"Cumulative masks do not match registration for {family}")
        if status[family]["complete_configurations"] != len(aggregate) or (
            synthesis["registered_configurations"][family] != len(aggregate)
        ):
            raise ValueError(f"Stale cumulative/synthesis counts for {family}")
        fronts[family] = set(frontier.configuration)
        family_records = {}
        baseline_ids = [identifier for identifier, columns in exact.items() if set(columns) == all_inputs]
        if len(baseline_ids) != 1:
            raise ValueError(f"The current all-input comparator is not uniquely identified for {family}")
        baseline_id = baseline_ids[0]
        baseline = aggregate[aggregate.configuration.eq(baseline_id)].iloc[0].to_dict()
        baselines[family] = {"aggregate": baseline, "input_columns": exact[baseline_id],
            "by_seed": seeds[seeds.configuration.eq(baseline_id)].to_dict("records"),
            "family_diagnostics_by_seed": diagnostics[diagnostics.configuration.eq(baseline_id)].to_dict("records")}
        tables.extend([f"## {family}", "", f"Complete configurations: {len(aggregate)}; frontier: {len(frontier)}.",
                       "", f"Compute-matched all-input control: {baseline_id}; "
                       f"IC {baseline['mean_ic']:.6f}, NLL {baseline['mean_gaussian_nll']:.6f}, "
                       f"Sharpe {baseline['sharpe']:.6f} (27 non-time inputs).",
                       "", "| Configuration | Inputs | IC ± SD | NLL ± SD | Sharpe ± SD | Excess return | RMSE |",
                       "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in frontier.to_dict("records"):
            identifier = row["configuration"]
            selected_seed = seeds[seeds.configuration.eq(identifier)]
            if sorted(selected_seed.seed.tolist()) != sorted(data.STUDY["seeds"]):
                raise ValueError(f"Incomplete seed list for {identifier}")
            if "t_index" not in exact[identifier] or len(exact[identifier]) - 1 != row["input_count_excluding_time"]:
                raise ValueError(f"Incorrect input list for {identifier}")
            family_records[identifier] = {"aggregate": row, "input_columns": exact[identifier],
                "by_seed": selected_seed.to_dict("records"),
                "family_diagnostics_by_seed": diagnostics[diagnostics.configuration.eq(identifier)].to_dict("records")}
            tables.append(f"| {identifier} | {int(row['input_count_excluding_time'])} | "
                          f"{row['mean_ic']:.6f} ± {row['std_ic']:.6f} | "
                          f"{row['mean_gaussian_nll']:.6f} ± {row['std_gaussian_nll']:.6f} | "
                          f"{row['sharpe']:.6f} ± {row['std_sharpe']:.6f} | "
                          f"{row['cumulative_return_diagnostic_only']:.6f} | {row['rmse_diagnostic_only']:.6f} |")
        tables.extend(["", "### Exact input lists", ""])
        for identifier, record in family_records.items():
            tables.extend([f"- {identifier}: " + ", ".join(record["input_columns"]), ""])
        records[family] = family_records
    validate_decisions(decisions, fronts)
    sources.append(Path(__file__).resolve())
    fingerprints = {str(path.relative_to(data.ROOT)): data.digest(path) for path in sources}
    output = {"scope": "Completed adaptive research, not production promotion or untouched confirmation",
              "counts": {"configurations": expected_count, "seed_runs": expected_count * len(data.STUDY["seeds"]),
                         "scored_gp_windows": expected_count * len(data.STUDY["seeds"]) * data.STUDY["scored_months"],
                         "by_universe": {family: len(value) for family, value in expected.items()}},
              "decisions": decisions, "all_pareto_configurations": records, "all_input_baselines": baselines,
              "exact_shared_lead_inputs": records["global"][decisions["shared_research_lead"]]["input_columns"],
              "source_sha256": fingerprints}
    lead = set(output["exact_shared_lead_inputs"])
    output["lead_changes_vs_all_input"] = {"removed": sorted(all_inputs - lead),
                                           "added": sorted(lead - all_inputs),
                                           "retained_excluding_time": sorted((lead & all_inputs) - {"t_index"})}
    # Nullable allocation diagnostics are intentionally null, never fabricated zero.
    sanitized = json.loads(json.dumps(output), parse_constant=lambda _: None)
    data.write_json(results / "final_recommendations.json", sanitized)
    (results / "FINAL_PARETO_TABLES.md").write_text("\n".join(tables) + "\n")
    print(json.dumps({"configurations": expected_count, "frontiers": {k: len(v) for k, v in fronts.items()},
                      "shared_research_lead": decisions["shared_research_lead"]}))


if __name__ == "__main__":
    package()
