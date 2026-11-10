"""Run a separately authorized bounded joint-feature follow-up using frozen inputs.

This research-only adapter reuses the completed study's candidate factory,
validator, dispatcher, fit signatures, summarizer and independent arithmetic.
A reversible in-process execution scope redirects only runner output paths and
adds this adapter/contract to fingerprints; no old file or GP rule is changed.
Returns, RMSE and predictive spreads are decimal units. No trading or promotion.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/followup_workflow.py --prepare
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/followup_workflow.py --execute --workers 15
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/followup_workflow.py --check

Batch two requires --register-second after hosted review writes
followup/INTERPRETATION_BATCH1.md. Execution never invents a reasoning turn.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import repair_dataset as data
import run_study as study
import workflow
from analyze_results import family_diagnostics, paired_evidence
from audit_joint_objectives import reconstruct
from compare_round import frontier_progress
from extend_feature_pool import EXTRA_ETF
from register_pool_history import pool_candidate, pool_config
from register_round import apply_operations

HERE = Path(__file__).resolve().parent / "followup"
CONTRACT = HERE / "contract.json"
BASE = data.HERE


def immutable_json(path: Path, value: object) -> None:
    """Create a registration once, rejecting any attempted overwrite with changed content."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing to change frozen registration: {path}")
    else:
        data.write_json(path, value)


def snapshot(original: Path, destination: Path) -> None:
    """Freeze existing evidence without replacing an earlier snapshot."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if data.digest(destination) != data.digest(original):
            raise ValueError(f"Evidence changed since registration: {original}")
    else:
        shutil.copy2(original, destination)


@contextmanager
def execution_scope() -> Iterator[None]:
    """Redirect legacy output globals reversibly and extend exact execution signatures.

    Yields:
        A scope in which the unmodified dispatcher writes only follow-up outputs.
    """
    old_here, old_hashes = study.HERE, study.code_hashes
    paths = [data.ROOT / relative for relative in old_hashes()]
    paths.extend([Path(__file__).resolve(), CONTRACT])
    paths.extend(BASE / name for name in ["workflow.py", "candidate_masks.py", "register_pool_history.py",
        "register_round.py", "extend_feature_pool.py", "repair_dataset.py", "analyze_results.py",
        "audit_joint_objectives.py", "compare_round.py"])

    def fingerprints() -> dict[str, str]:
        return {str(path.relative_to(data.ROOT)): data.digest(path) for path in sorted(set(paths))}

    try:
        study.HERE, study.code_hashes = HERE, fingerprints
        yield
    finally:
        study.HERE, study.code_hashes = old_here, old_hashes


def validate_base() -> set[str]:
    """Validate frozen data and the completed parent evidence, without refitting controls."""
    available = workflow.validate_frozen_data()
    final = json.loads((BASE / "results/final_recommendations.json").read_text())
    for relative, expected in final["source_sha256"].items():
        if data.digest(data.ROOT / relative) != expected:
            raise ValueError(f"Completed parent evidence changed: {relative}")
    smoke = json.loads((BASE / "executions/00_global_all_input_control/smoke_status.json").read_text())
    if smoke["state"] != "completed":
        raise ValueError("Existing same-runner full-window smoke did not pass")
    control = BASE / "plans/00_global_all_input_control.json"
    if not workflow.completed_batch_is_valid(control, json.loads(control.read_text())):
        raise ValueError("The completed full-window runner control is no longer valid")
    contract = json.loads(CONTRACT.read_text())
    if contract["seeds"] != data.STUDY["seeds"] or contract["max_batches"] != 2 or (
        contract["max_new_configurations_per_batch"] != 6 or contract["max_workers_total"] != 15
    ):
        raise ValueError("Follow-up authorization or inherited controls changed")
    return available


def candidate_for(hypothesis: dict[str, Any], inputs: list[str], evidence: list[str]) -> dict[str, Any]:
    """Resolve one approved operation list through the existing frozen candidate factory."""
    local = {*data.ETF_COLUMNS, *EXTRA_ETF, "lag_y_excess_lead", "lag2_y_excess_lead"}
    etf = [column for column in inputs if column in local]
    macro = [column for column in inputs if column not in local and column != "t_index"]
    etf = apply_operations(etf, hypothesis.get("etf_add", []), hypothesis.get("etf_drop", []), None)
    macro = apply_operations(macro, hypothesis.get("macro_add", []), hypothesis.get("macro_drop", []), None)
    item = pool_candidate("global", etf, macro, evidence)
    item.update({"hypothesis_name": hypothesis["name"], "hypothesis": hypothesis["hypothesis"],
                 "category": "authorized_joint_followup", "base_configuration": json.loads(
                     CONTRACT.read_text())["base_configuration"],
                 "expected_tradeoff": "Report IC/NLL/conventional Sharpe, all seeds and family diagnostics.",
                 "falsification": "A dominated result does not support the hypothesis; effects remain conditional."})
    return item


def register(index: int) -> Path:
    """Register an authorized six-mask batch with immutable before-result evidence."""
    available = validate_base()
    if index not in [1, 2]:
        raise ValueError("Only two follow-up batches are authorized")
    if index == 2:
        interpretation = HERE / "INTERPRETATION_BATCH1.md"
        if not interpretation.is_file() or len(interpretation.read_text()) < 200:
            raise ValueError("Hosted interpretation of completed batch one is required")
        report = json.loads((HERE / "results/batch1/audit.json").read_text())
        if not report["passed"] or report["checked_seed_runs"] != 18:
            raise ValueError("First-batch independent arithmetic audit is incomplete")
    folder = HERE / "evidence" / f"batch{index}"
    source = BASE / "results/cumulative/global" if index == 1 else HERE / "results/batch1"
    names = ["objectives_by_seed.csv", "configuration_aggregates.csv", "exact_inputs.json",
             "family_diagnostics_by_seed.csv", "conditional_feature_pairs.csv", "hypervolume.json"]
    for name in names:
        snapshot(source / name, folder / name)
    snapshot(CONTRACT, folder / "contract.json")
    if index == 2:
        snapshot(HERE / "INTERPRETATION_BATCH1.md", folder / "INTERPRETATION_BATCH1.md")
        snapshot(HERE / "results/batch1/audit.json", folder / "audit.json")
    frozen = sorted(folder.iterdir())
    evidence = [str(path.relative_to(data.ROOT)) for path in frozen]
    before = json.loads((folder / "exact_inputs.json").read_text())
    contract = json.loads(CONTRACT.read_text())
    lead = before[contract["base_configuration"]]
    items = [candidate_for(hypothesis, lead, evidence) for hypothesis in contract["batches"][index - 1]]
    if len(items) != 6 or len({item["id"] for item in items}) != 6 or any(item["id"] in before for item in items):
        raise ValueError("The batch must have six distinct new authorized masks")
    name = f"10_global_round{index}"
    config = pool_config("global", items, name)
    config.update({"output_root": str((HERE / "runs").relative_to(data.ROOT)),
                   "snapshot_root": str((HERE / "executions").relative_to(data.ROOT)),
                   "study_stage": "global_refinement", "round_index": index,
                   "evidence_reports": evidence,
                   "evidence_sha256": {str(path.relative_to(data.ROOT)): data.digest(path) for path in frozen},
                   "round_interpretation": "Separate authorized follow-up; completed campaign cap stays closed."
                   if index == 1 else (HERE / "INTERPRETATION_BATCH1.md").read_text()})
    path = HERE / "plans" / f"{name}.json"
    pending = HERE / "planning" / f"{name}.pending.json"
    data.write_json(pending, config)
    workflow.validate_plan(config, pending, available)
    immutable_json(path, config)
    return path


def validate_plan(path: Path) -> dict[str, Any]:
    """Check inherited controls, authorized operations, output isolation and frozen evidence."""
    config = json.loads(path.read_text())
    workflow.validate_plan(config, path, workflow.validate_frozen_data())
    index = config["round_index"]
    if index not in [1, 2] or config["family"] != "global":
        raise ValueError("Out-of-scope follow-up batch")
    if config["output_root"] != str((HERE / "runs").relative_to(data.ROOT)) or (
        config["snapshot_root"] != str((HERE / "executions").relative_to(data.ROOT))
    ):
        raise ValueError("Follow-up must not write original run or execution directories")
    contract = json.loads(CONTRACT.read_text())
    folder = HERE / "evidence" / f"batch{index}"
    if data.digest(CONTRACT) != data.digest(folder / "contract.json"):
        raise ValueError("Authorized campaign contract changed after registration")
    inputs = json.loads((folder / "exact_inputs.json").read_text())
    expected = [candidate_for(hypothesis, inputs[contract["base_configuration"]], config["evidence_reports"])
                for hypothesis in contract["batches"][index - 1]]
    if config["configurations"] != expected:
        raise ValueError("Registered operations differ from the approved six-mask batch")
    return config


def completed(path: Path, config: dict[str, Any]) -> bool:
    """Check every seed's full output, training and extended follow-up signature."""
    state = HERE / "executions" / path.stem / "status.json"
    if not state.is_file() or json.loads(state.read_text())["state"] != "completed":
        return False
    paths = study.previous.validate_config(config, path)
    hashes = study.code_hashes()
    for item in config["configurations"]:
        for seed in data.STUDY["seeds"]:
            run = HERE / "runs" / f"{item['id']}_s{seed}"
            stamp = run / "repaired_execution_fingerprint.json"
            if not stamp.is_file():
                return False
            saved = json.loads(stamp.read_text())
            if saved.get("state") != "completed" or saved["signature"] != study.signature(config, item, seed, hashes):
                return False
            if not study.previous.existing_run_is_valid(config, paths, item, seed):
                return False
            study.verify_training(run, config["tickers"])
    return True


def report_batch(path: Path, config: dict[str, Any]) -> None:
    """Write cumulative fronts, family evidence and independent arithmetic after a full batch."""
    if not completed(path, config):
        raise ValueError("Cannot report partial or stale fits as a complete batch")
    index = config["round_index"]
    out = HERE / "results" / f"batch{index}"
    out.mkdir(parents=True, exist_ok=True)
    evidence = HERE / "evidence" / f"batch{index}"
    before = pd.read_csv(evidence / "objectives_by_seed.csv")
    exact = json.loads((evidence / "exact_inputs.json").read_text())
    old_families = pd.read_csv(evidence / "family_diagnostics_by_seed.csv")
    dates = study.previous.read_dates(data.ARTIFACTS / "scored_dates_24.txt")
    rows, families, checked = [], [], []
    for item in config["configurations"]:
        exact[item["id"]] = item["expected_input_columns"]
        for seed in data.STUDY["seeds"]:
            run = HERE / "runs" / f"{item['id']}_s{seed}"
            row, failures = study.previous.summarize_run(
                item["id"], item, run, dates, config["tickers"], expected_seed=seed)
            if failures:
                raise ValueError(f"Invalid output: {failures}")
            independent = reconstruct(run, config["tickers"], dates)
            for metric in ["mean_ic", "mean_gaussian_nll", "sharpe", "cumulative_return_diagnostic_only",
                           "rmse_diagnostic_only"]:
                np.testing.assert_allclose(row[metric], independent[metric], rtol=0, atol=1e-9)
            rows.append(row)
            families.extend(family_diagnostics(run, item["id"], seed))
            checked.append({"configuration": item["id"], "seed": seed, **independent})
    combined = pd.concat([before, pd.DataFrame(rows)], ignore_index=True)
    if combined.duplicated(["configuration", "seed"]).any():
        raise ValueError("Duplicate controls or candidate seeds cannot count twice")
    aggregates = study.aggregate_configurations(combined)
    aggregates["input_count_excluding_time"] = aggregates.configuration.map({k: len(v) - 1 for k, v in exact.items()})
    old_aggregates = pd.read_csv(evidence / "configuration_aggregates.csv")
    old_hv = study.calculate_hypervolume(old_aggregates, config)["value"]
    hv = study.calculate_hypervolume(aggregates, config)
    changes = []
    for item in config["configurations"]:
        new = combined[combined.configuration.eq(item["id"])].set_index("seed")
        parent = combined[combined.configuration.eq(item["base_configuration"])].set_index("seed")
        delta = new[["mean_ic", "mean_gaussian_nll", "sharpe"]] - parent[
            ["mean_ic", "mean_gaussian_nll", "sharpe"]]
        changes.append({"configuration": item["id"], "hypothesis": item["hypothesis_name"],
            "delta_ic": float(delta.mean_ic.mean()), "delta_nll": float(delta.mean_gaussian_nll.mean()),
            "delta_sharpe": float(delta.sharpe.mean()), "seed_ic_wins": int(delta.mean_ic.gt(0).sum()),
            "seed_nll_wins": int(delta.mean_gaussian_nll.lt(0).sum()),
            "seed_sharpe_wins": int(delta.sharpe.gt(0).sum())})
    combined.to_csv(out / "objectives_by_seed.csv", index=False)
    aggregates.to_csv(out / "configuration_aggregates.csv", index=False)
    aggregates[aggregates.pareto_nondominated].to_csv(out / "pareto.csv", index=False)
    pd.concat([old_families, pd.DataFrame(families)], ignore_index=True).to_csv(
        out / "family_diagnostics_by_seed.csv", index=False)
    paired_evidence({key: {"expected_input_columns": value} for key, value in exact.items()}, combined).to_csv(
        out / "conditional_feature_pairs.csv", index=False)
    data.write_json(out / "exact_inputs.json", exact)
    data.write_json(out / "hypervolume.json", hv)
    data.write_json(out / "audit.json", {"passed": True, "checked_seed_runs": len(checked), "checked": checked,
        "source_sha256": study.code_hashes(), "plan_sha256": data.digest(path)})
    data.write_json(out / "comparison.json", {"hypervolume_before": old_hv, "hypervolume_after": hv["value"],
        **frontier_progress(old_aggregates, aggregates, {item["id"] for item in config["configurations"]}),
        "hypothesis_comparisons": changes})
    (out / "RESULTS.md").write_text("# Joint follow-up batch " + str(index) + "\n\n" +
        f"Hypervolume: {old_hv:.9f} -> {hv['value']:.9f}.\n\n" + pd.DataFrame(changes).to_markdown(index=False) +
        "\n\nAll three objectives remain active. Family diagnostics are separate. Reused-date research results.\n")
    print(json.dumps({"batch": index, "independently_checked_seeds": len(checked),
                      "hypervolume_before": old_hv, "hypervolume_after": hv["value"]}), flush=True)


def main() -> None:
    """Prepare/check or execute registered work under a single follow-up process lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--register-second", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.workers <= 15:
        raise ValueError("Use one to fifteen fit workers across the whole campaign")
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "dispatcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_base()
        if args.prepare:
            register(1)
        if args.register_second:
            register(2)
        plans = sorted((HERE / "plans").glob("*.json"))
        configs = [(path, validate_plan(path)) for path in plans]
        if not configs or len(configs) > 2:
            raise ValueError("Prepare one or two authorized batches before execution")
        print(json.dumps({"validated_batches": len(configs), "registered_masks": sum(
            len(config["configurations"]) for _, config in configs), "max_fit_workers": args.workers}), flush=True)
        if args.execute:
            with execution_scope():
                for path, config in configs:
                    if not completed(path, config):
                        study.execute_batch(path, args.workers, False)
                    report_batch(path, config)


if __name__ == "__main__":
    main()
