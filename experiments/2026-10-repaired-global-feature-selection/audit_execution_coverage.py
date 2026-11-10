"""Verify research execution scope without treating it as scientific goal completion.

This experiment-only audit reuses the frozen-data, plan, fingerprint, walk-forward
output and round-comparison validators. It detects missing audited historical
masks, incomplete three-seed batches, duplicate rounds and invalid stopping.
It never launches fits. Objective/recommendation quality still needs final review.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_execution_coverage.py

The command exits nonzero while execution scope is incomplete. Completed round
reports are deterministically refreshed from their frozen pre-round evidence.
"""

from __future__ import annotations

import json
from typing import Any

import compare_round
import repair_dataset as data
import run_study as study
import workflow


def adaptive_closure(reports: dict[int, dict[str, Any]], maximum: int) -> tuple[bool, str]:
    """Check a consecutive verified adaptive sequence against its stopping contract.

    Args:
        reports: Completed/recomputed adaptive reports, excluding initial round zero.
        maximum: Frozen maximum permitted adaptive rounds.

    Returns:
        Whether the sequence is closed and a concrete reason; missing work stays open.
    """
    if any(index < 1 or index > maximum for index in reports):
        return False, "Adaptive sequence contains an out-of-budget round"
    previous_stalled = False
    for index in range(1, maximum + 1):
        if index not in reports:
            return False, f"Adaptive round {index} is missing or not fully verified"
        stalled = bool(reports[index]["stalled"])
        if stalled and previous_stalled:
            if any(other > index for other in reports):
                return False, "Experiments continued after two consecutive stalled adaptive rounds"
            return True, f"Stopped after consecutive stalled adaptive rounds {index - 1}/{index}"
        previous_stalled = stalled
    return True, f"Reached the {maximum}-round cap; no convergence claim"


def audit() -> dict[str, Any]:
    """Revalidate all registered research batches and write an execution-only scope audit.

    Returns:
        Explicit checks, unresolved requirements, verified counts and source hashes.
    """
    available = workflow.validate_frozen_data()
    hashes = study.code_hashes()
    plans = sorted((data.HERE / "plans").glob("*.json"))
    plan_hashes = {str(path.relative_to(data.ROOT)): data.digest(path) for path in plans}
    policy_sources = [data.HERE / name for name in [
        "audit_execution_coverage.py", "workflow.py", "candidate_masks.py", "compare_round.py", "GOAL_CONTRACT.md",
        "historical_inventory.json", "results/historical_scope_audit.json",
        "results/historical_pool_scope_audit.json", "results/historical_macro_only_audit.json"]]
    policy_hashes = {str(path.relative_to(data.ROOT)): data.digest(path) for path in policy_sources}
    ids: dict[str, set[str]] = {}
    baselines: dict[str, set[str]] = {}
    rounds: dict[str, dict[int, dict[str, Any]]] = {}
    seen_rounds = set()
    batches, unresolved, initial = [], [], []
    for path in plans:
        config = json.loads(path.read_text())
        workflow.validate_plan(config, path, available)
        family, stage = config["family"], workflow.plan_stage(config, path)
        names = {item["id"] for item in config["configurations"]}
        if len(names) != len(config["configurations"]) or names & ids.setdefault(family, set()):
            raise ValueError("Duplicate masks cannot count twice toward coverage")
        ids[family].update(names)
        if stage == "historical_rebaseline":
            baselines.setdefault(family, set()).update(names)
        else:
            key = (family, config["round_index"])
            if key in seen_rounds:
                raise ValueError("Multiple files cannot expand one round's candidate budget")
            seen_rounds.add(key)
        complete = workflow.completed_batch_is_valid(path, config)
        state_path = data.HERE / "executions" / path.stem / "status.json"
        if state_path.is_file():
            workers = json.loads(state_path.read_text())["workers"]
            if not 1 <= workers <= data.STUDY["max_workers_total"]:
                raise ValueError("An execution pool exceeds the global worker ceiling")
        if not complete:
            unresolved.append({"requirement": "complete_source_matched_three_seed_batch", "plan": path.name})
        elif stage != "historical_rebaseline":
            report = compare_round.compare(path)
            if stage == "global_initial":
                initial.append(report)
            else:
                rounds.setdefault(family, {})[config["round_index"]] = report
        batches.append({"family": family, "stage": stage, "plan": str(path.relative_to(data.ROOT)),
                        "plan_sha256": data.digest(path), "configurations": len(names),
                        "verified_complete": complete})
    historical = {}
    for family in [*data.FAMILIES, "global"]:
        required = workflow.audited_baseline_ids(family)
        registered = baselines.get(family, set())
        missing, extra = required - registered, registered - required
        historical[family] = {"required": len(required), "registered": len(registered),
                              "missing_ids": sorted(missing), "unaudited_ids": sorted(extra)}
        if missing or extra:
            unresolved.append({"requirement": "audited_historical_mask_coverage", "family": family,
                               "missing": sorted(missing), "unaudited": sorted(extra)})
    closure = {}
    for family in [*data.FAMILIES, "global"]:
        maximum = data.STUDY["global_rounds" if family == "global" else "family_rounds"]
        closed, reason = adaptive_closure(rounds.get(family, {}), maximum)
        closure[family] = {"closed": closed, "reason": reason,
                           "verified_rounds": sorted(rounds.get(family, {}))}
        if not closed:
            unresolved.append({"requirement": "adaptive_sequence_closed", "family": family, "reason": reason})
    if len(initial) != 1:
        unresolved.append({"requirement": "one_complete_actual_joint_initial_comparison"})
    verified_masks = sum(row["configurations"] for row in batches if row["verified_complete"])
    report = {"execution_scope_complete": not unresolved, "unresolved": unresolved,
              "registered_configurations": sum(len(value) for value in ids.values()),
              "verified_configurations_in_complete_batches": verified_masks,
              "verified_seed_runs_in_complete_batches": verified_masks * len(data.STUDY["seeds"]),
              "verified_scored_gp_windows_in_complete_batches": verified_masks * len(
                  data.STUDY["seeds"]) * data.STUDY["scored_months"],
              "historical_coverage": historical, "adaptive_closure": closure, "batches": batches,
              "source_sha256": {**hashes, **policy_hashes, **plan_hashes},
              "limits": ["This proves execution scope only, not scientific recommendations or goal completion.",
                         "Partial batches are deliberately not counted as complete scope.",
                         "Global concurrent-worker compliance also needs the recorded scheduler/native evidence; "
                         "per-pool ceilings alone are insufficient."]}
    if hashes != study.code_hashes() or any(
        data.digest(data.ROOT / relative) != digest for relative, digest in {**policy_hashes, **plan_hashes}.items()
    ) or sorted((data.HERE / "plans").glob("*.json")) != plans:
        raise ValueError("Sources or the registered plan set changed during coverage verification")
    data.write_json(data.HERE / "results/execution_coverage_audit.json", report)
    return report


if __name__ == "__main__":
    result = audit()
    print(json.dumps({key: result[key] for key in ["execution_scope_complete", "registered_configurations",
        "verified_seed_runs_in_complete_batches", "unresolved"]}))
    raise SystemExit(0 if result["execution_scope_complete"] else 1)
