"""Resolve a hosted-Codex hypothesis proposal into an immutable executable batch.

Codex supplies interpretations and explicit feature operations; this registrar
does not choose candidates or optimize scores. Deterministic code snapshots the
evidence, resolves masks, enforces novelty/caps, and validates the existing runner
contract. It never launches fits. All new adaptive masks use the frozen v2 pool.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/register_round.py \
      --proposal experiments/2026-10-repaired-global-feature-selection/planning/international_equity_round1.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import repair_dataset as data
import workflow
from extend_feature_pool import EXTRA_ETF
from register_pool_history import pool_candidate, pool_config


def apply_operations(
    base: list[str], additions: list[str], removals: list[str], replacement: list[str] | None
) -> list[str]:
    """Apply explicit list operations without silently ignoring invalid removals."""
    if replacement is not None:
        if additions or removals:
            raise ValueError("A replacement block cannot also specify additions/removals")
        return replacement
    missing = set(removals) - set(base)
    if missing:
        raise ValueError(f"Cannot remove absent features: {sorted(missing)}")
    return sorted((set(base) - set(removals)) | set(additions))


def resolve_proposal(proposal_path: Path) -> Path:
    """Freeze evidence and register one valid, novel round without changing old plans."""
    proposal = json.loads(proposal_path.read_text())
    family, index = proposal["family"], proposal["round_index"]
    stage = proposal.get("study_stage", "global_refinement" if family == "global" else "family_refinement")
    if family not in {*data.FAMILIES, "global"}:
        raise ValueError("Unsupported family")
    cap = data.STUDY["max_new_global_configurations_per_round" if family == "global"
                     else "max_new_family_configurations_per_round"]
    if not 1 <= len(proposal["candidates"]) <= cap:
        raise ValueError("Candidate count violates the accepted round cap")
    evidence_dir = data.HERE / "planning/evidence" / f"{family}_round{index}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = []
    evidence_hashes = {}
    for raw in proposal["evidence_reports"]:
        original = (data.ROOT / raw).resolve()
        if not original.is_relative_to(data.HERE) or not original.is_file():
            raise ValueError(f"Missing or out-of-scope planning evidence: {original}")
        destination = evidence_dir / original.name
        if destination.exists() and data.digest(destination) != data.digest(original):
            raise ValueError("Evidence snapshot already exists with different contents")
        if not destination.exists():
            shutil.copy2(original, destination)
        relative = str(destination.relative_to(data.ROOT))
        evidence.append(relative)
        evidence_hashes[relative] = data.digest(destination)
    source = next((data.ROOT / path for path in evidence if Path(path).name == "exact_inputs.json"), None)
    if source is None:
        raise ValueError("The evidence bundle must include exact_inputs.json")
    bases = json.loads(source.read_text())
    already = set()
    for path in (data.HERE / "plans").glob("*.json"):
        config = json.loads(path.read_text())
        if config["family"] == family and config.get("round_index") != index:
            already.update(item["id"] for item in config["configurations"])
    candidates = []
    allowed_etf = {*data.ETF_COLUMNS, *EXTRA_ETF, "lag_y_excess_lead", "lag2_y_excess_lead"}
    for hypothesis in proposal["candidates"]:
        identifier = hypothesis["base_configuration"]
        if identifier not in bases:
            raise ValueError(f"Hypothesis base is absent from frozen evidence: {identifier}")
        columns = bases[identifier]
        etf = [column for column in columns if column in allowed_etf]
        macro = [column for column in columns if column not in allowed_etf and column != "t_index"]
        etf = apply_operations(etf, hypothesis.get("etf_add", []), hypothesis.get("etf_drop", []),
                               hypothesis.get("etf_replace"))
        macro = apply_operations(macro, hypothesis.get("macro_add", []), hypothesis.get("macro_drop", []),
                                 hypothesis.get("macro_replace"))
        item = pool_candidate(family, etf, macro, [*evidence, f"base_configuration={identifier}"])
        if item["id"] in already or item["id"] in {other["id"] for other in candidates}:
            raise ValueError(f"Hypothesis is not a genuinely new mask: {hypothesis['name']}")
        item.update({"hypothesis_name": hypothesis["name"], "category": hypothesis["category"],
                     "hypothesis": hypothesis["hypothesis"], "falsification": hypothesis.get("falsification",
                         proposal.get("default_falsification", "Reject a dominated, noncompetitive result.")),
                     "base_configuration": identifier, "expected_tradeoff": hypothesis.get("expected_tradeoff",
                         proposal.get("default_expected_tradeoff", "Report the full three-objective tradeoff."))})
        candidates.append(item)
    prefix = "05" if stage == "global_initial" else "10"
    name = f"{prefix}_{family}_round{index}"
    config = pool_config(family, candidates, name)
    config.update({"study_stage": stage, "round_index": index, "evidence_reports": evidence,
                   "evidence_sha256": evidence_hashes, "round_interpretation": proposal["round_interpretation"],
                   "proposal_path": str(proposal_path.relative_to(data.ROOT)),
                   "proposal_sha256": data.digest(proposal_path),
                   "planning_method": "Hosted Codex hypothesis generation"})
    pending = data.HERE / "planning" / (name + ".pending.json")
    data.write_json(pending, config)
    workflow.validate_plan(config, pending, workflow.validate_frozen_data())
    target = data.HERE / "plans" / (name + ".json")
    if target.exists() and json.loads(target.read_text()) != config:
        raise ValueError("Refusing to alter an already registered round")
    if not target.exists():
        data.write_json(target, config)
    return target


def main() -> None:
    """Register a concrete reasoning proposal; execution remains a separate deterministic step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    args = parser.parse_args()
    path = resolve_proposal(args.proposal.resolve())
    print(json.dumps({"registered_plan": str(path), "new_configurations": len(
        json.loads(path.read_text())["configurations"])}))


if __name__ == "__main__":
    main()
