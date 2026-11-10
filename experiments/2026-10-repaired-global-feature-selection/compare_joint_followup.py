"""Recompare every original and follow-up joint mask after both batches complete.

This experiment-only reporter revalidates all run signatures, independently
reconstructs IC/NLL/conventional Sharpe and family calibration, and preserves the
three-objective Pareto front. No fits, tuning, source repair or deployment occurs.
Return, RMSE and predictive spread use decimal units; turnover is not consumed.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/compare_joint_followup.py

Writes followup/results/all_joint/. Incomplete or stale evidence fails closed.
Original completed-study publications are read-only controls, never overwritten.
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any

import followup_workflow as followup
import numpy as np
import pandas as pd
import repair_dataset as data
import run_study as study
import workflow
from analyze_results import paired_evidence
from audit_joint_objectives import reconstruct
from synthesize_components import window_diagnostics

METRICS = ["mean_ic", "mean_gaussian_nll", "sharpe", "cumulative_return_diagnostic_only", "rmse_diagnostic_only"]


def validate_seed_coverage(rows: pd.DataFrame, identifiers: set[str]) -> None:
    """Require exact unique three-seed objective coverage without accepting partial points.

    Args:
        rows: Published seed rows with decimal-return diagnostics.
        identifiers: All and only registered original/follow-up mask identifiers.

    Raises:
        ValueError: Seeds, configurations, finite objectives or validation flags differ.
    """
    if set(rows.configuration) != identifiers or rows.duplicated(["configuration", "seed"]).any():
        raise ValueError("Missing, extra or duplicated registered configuration/seed rows")
    if not rows.groupby("configuration").seed.apply(lambda seeds: sorted(seeds) == data.STUDY["seeds"]).all():
        raise ValueError("A registered mask does not have exactly the three required seeds")
    if not np.isfinite(rows[METRICS].to_numpy()).all() or not rows.validation_passed.all():
        raise ValueError("Nonfinite or unvalidated seed objectives cannot enter the comparison")


def compare() -> dict[str, Any]:
    """Audit all registered joint runs and write a fresh combined comparison package.

    Returns:
        Complete configuration/seed counts, final front identifiers and hypervolume.

    Raises:
        ValueError: Either follow-up batch or any original control is incomplete/stale.
    """
    followup.validate_base()
    registry: dict[str, tuple[dict[str, Any], dict[str, Any], Path, dict[str, str], str]] = {}
    sources = [Path(__file__).resolve(), followup.CONTRACT, data.HERE / "synthesize_components.py"]
    original_hashes = study.code_hashes()
    for plan in sorted((data.HERE / "plans").glob("*.json")):
        config = json.loads(plan.read_text())
        if config["family"] != "global":
            continue
        workflow.validate_plan(config, plan, workflow.validate_frozen_data())
        if not workflow.completed_batch_is_valid(plan, config):
            raise ValueError(f"Original joint batch is no longer valid: {plan.name}")
        sources.append(plan)
        for item in config["configurations"]:
            registry[item["id"]] = config, item, data.HERE / "runs", original_hashes, "original"
    original_count = len(registry)
    plans = sorted((followup.HERE / "plans").glob("*.json"))
    if len(plans) != 2:
        raise ValueError("Both authorized follow-up batches must be registered and complete")
    with followup.execution_scope():
        followup_hashes = study.code_hashes()
        for index, plan in enumerate(plans, 1):
            config = followup.validate_plan(plan)
            if config["round_index"] != index or not followup.completed(plan, config):
                raise ValueError("Follow-up batch sequence is incomplete or stale")
            sources.append(plan)
            sources.extend(data.ROOT / path for path in config["evidence_reports"])
            for item in config["configurations"]:
                if item["id"] in registry:
                    raise ValueError("A follow-up mask duplicates another registered joint configuration")
                registry[item["id"]] = config, item, followup.HERE / "runs", followup_hashes, f"followup_batch{index}"
    folder = followup.HERE / "results/batch2"
    publications = [folder / name for name in ["objectives_by_seed.csv", "configuration_aggregates.csv",
                   "exact_inputs.json", "family_diagnostics_by_seed.csv", "hypervolume.json"]]
    sources.extend(publications)
    published = pd.read_csv(folder / "objectives_by_seed.csv")
    validate_seed_coverage(published, set(registry))
    exact = json.loads((folder / "exact_inputs.json").read_text())
    if set(exact) != set(registry) or any(
        exact[identifier] != item["expected_input_columns"] for identifier, (_, item, _, _, _) in registry.items()
    ):
        raise ValueError("Published exact inputs differ from registered masks")
    families = pd.read_csv(folder / "family_diagnostics_by_seed.csv")
    if families.duplicated(["configuration", "seed", "family"]).any() or len(families) != len(published) * 4:
        raise ValueError("Family evidence has duplicated or missing entries")
    targets = published.set_index(["configuration", "seed"])
    family_targets = families.set_index(["configuration", "seed", "family"])
    dates = study.previous.read_dates(data.ARTIFACTS / "scored_dates_24.txt")
    checked = []
    for identifier, (config, item, run_root, hashes, stage) in registry.items():
        for seed in data.STUDY["seeds"]:
            run = run_root / f"{identifier}_s{seed}"
            stamp = json.loads((run / "repaired_execution_fingerprint.json").read_text())
            if stamp.get("state") != "completed" or stamp["signature"] != study.signature(config, item, seed, hashes):
                raise ValueError(f"Wrong source/data/config/date/seed signature: {run.name}")
            study.verify_training(run, config["tickers"])
            independent = reconstruct(run, config["tickers"], dates)
            for metric in METRICS:
                np.testing.assert_allclose(independent[metric], targets.loc[(identifier, seed), metric],
                                           atol=1e-9, rtol=0)
            for family, values in independent["families"].items():
                for metric, value in values.items():
                    np.testing.assert_allclose(value, family_targets.loc[(identifier, seed, family), metric],
                                               atol=1e-9, rtol=0, equal_nan=True)
            checked.append({"configuration": identifier, "seed": seed, "stage": stage, **independent})
            sources.extend(run / name for name in ["gp_window_predictions.csv", "portfolio_returns.csv",
                                                   "portfolio_weights.csv", "repaired_execution_fingerprint.json"])
    aggregates = study.aggregate_configurations(published)
    aggregates["input_count_excluding_time"] = aggregates.configuration.map({k: len(v) - 1 for k, v in exact.items()})
    aggregates["stage"] = aggregates.configuration.map({k: v[4] for k, v in registry.items()})
    aggregates["hypothesis"] = aggregates.configuration.map({k: v[1].get("hypothesis_name", k)
                                                              for k, v in registry.items()})
    expected = pd.read_csv(folder / "configuration_aggregates.csv").set_index("configuration")
    for row in aggregates.to_dict("records"):
        for metric in ["mean_ic", "mean_gaussian_nll", "sharpe"]:
            np.testing.assert_allclose(row[metric], expected.loc[row["configuration"], metric], atol=1e-10, rtol=0)
        if bool(row["pareto_nondominated"]) != bool(expected.loc[row["configuration"], "pareto_nondominated"]):
            raise ValueError("Published cumulative front membership is incorrect")
    frontier = aggregates[aggregates.pareto_nondominated].sort_values("mean_ic", ascending=False)
    family_means = families.drop(columns="seed").groupby(
        ["configuration", "family"], as_index=False).mean(numeric_only=True)
    parent_id = json.loads(followup.CONTRACT.read_text())["base_configuration"]
    controls = ["global_c55e49933d10", parent_id]
    for control in controls:
        reference = family_means[family_means.configuration.eq(control)].set_index("family")
        for metric in ["within_family_ic", "mean_gaussian_nll"]:
            family_means[f"delta_{metric}_vs_{control}"] = (
                family_means[metric] - family_means.family.map(reference[metric]))
    finalists = {"highest_sharpe": str(frontier.loc[frontier.sharpe.idxmax(), "configuration"]),
                 "highest_ic": str(frontier.loc[frontier.mean_ic.idxmax(), "configuration"]),
                 "best_nll": str(frontier.loc[frontier.mean_gaussian_nll.idxmin(), "configuration"])}
    hv = study.calculate_hypervolume(aggregates, next(iter(registry.values()))[0])
    out = followup.HERE / "results/all_joint"
    out.mkdir(parents=True, exist_ok=True)
    aggregates.to_csv(out / "configuration_aggregates.csv", index=False)
    frontier.to_csv(out / "pareto.csv", index=False)
    published.to_csv(out / "objectives_by_seed.csv", index=False)
    families.to_csv(out / "family_diagnostics_by_seed.csv", index=False)
    family_means.to_csv(out / "family_mean_diagnostics.csv", index=False)
    paired_evidence({key: {"expected_input_columns": value} for key, value in exact.items()}, published).to_csv(
        out / "conditional_feature_pairs.csv", index=False)
    monthly = []
    for identifier in sorted(set(frontier.configuration) | set(controls)):
        for seed in data.STUDY["seeds"]:
            run = registry[identifier][2] / f"{identifier}_s{seed}"
            frame = window_diagnostics(pd.read_csv(run / "gp_window_predictions.csv"))
            frame.insert(0, "seed", seed)
            frame.insert(0, "configuration", identifier)
            monthly.append(frame)
    pd.concat(monthly, ignore_index=True).to_csv(out / "monthly_diagnostics.csv", index=False)
    data.write_json(out / "exact_inputs.json", exact)
    data.write_json(out / "hypervolume.json", hv)
    sources.extend(data.ROOT / relative for relative in {**original_hashes, **followup_hashes})
    hashes = {str(path.relative_to(data.ROOT)): data.digest(path) for path in sorted(set(sources))}
    audit = {"passed": True, "checked_seed_runs": len(checked), "checked": checked,
             "source_sha256": hashes, "unresolved": [],
             "scope": "All original and both follow-up batches; independent arithmetic and exact fingerprints."}
    data.write_json(out / "independent_audit.json", audit)
    result = {"configurations": len(registry), "original_configurations": original_count,
              "followup_configurations": len(registry) - original_count, "seed_runs": len(checked),
              "pareto_configurations": len(frontier), "hypervolume": hv["value"], "axis_finalists": finalists,
              "all_pareto_ids": frontier.configuration.tolist(), "source_sha256": hashes,
              "boundary": "Axes are reporting alternatives, not a weighted winner. No untouched holdout or deployment."}
    data.write_json(out / "comparison.json", result)
    columns = ["configuration", "hypothesis", "input_count_excluding_time", "mean_ic", "std_ic",
               "mean_gaussian_nll", "std_gaussian_nll", "sharpe", "std_sharpe", "pareto_nondominated"]
    lines = ["# All joint model results", "", f"Validated configurations: {len(registry)}; seeds: {len(checked)}.",
             "", f"Fixed-reference hypervolume: {hv['value']:.9f}.", "", "## Full Pareto front", "",
             frontier[columns].to_markdown(index=False, floatfmt=".6f"), "", "## Every configuration", "",
             aggregates[columns].to_markdown(index=False, floatfmt=".6f"), "", "## Exact frontier inputs", ""]
    for identifier in frontier.configuration:
        lines.extend([f"- {identifier}: " + ", ".join(exact[identifier]), ""])
    family_columns = ["configuration", "family", "within_family_ic", "mean_gaussian_nll",
                      "coverage_95", "mean_predictive_std", "mean_family_capital_weight"]
    lines.extend(["## Family outcomes of frontier and controls", "",
        family_means.loc[family_means.configuration.isin(set(frontier.configuration) | set(controls)),
                         family_columns].to_markdown(
            index=False, floatfmt=".6f"), "", "Monthly uncertainty and exact conditional effects are adjacent CSVs.",
        "Three objectives remain active; return/RMSE/width/count are not objectives. Reused-date adaptive research."])
    (out / "TABLES.md").write_text("\n".join(lines) + "\n")
    if study.code_hashes() != original_hashes or any(
        data.digest(data.ROOT / relative) != digest for relative, digest in hashes.items()
    ):
        raise ValueError("Sources or evidence changed during the all-joint comparison")
    return {key: value for key, value in result.items() if key != "source_sha256"}


if __name__ == "__main__":
    with (followup.HERE / "dispatcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(compare()))
