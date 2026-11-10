"""Restore valid historical macro-only masks omitted by an unintended local-block gate.

Recheck only the five already-discovered source manifests, preserve the old scope
audit as historical evidence, and register two deduplicated supplemental baselines.
Use the existing fixed kernel rule and frozen v2 data. This does not launch fits
or enlarge any adaptive budget.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/register_macro_only_history.py
"""

from __future__ import annotations

import json

import inventory_history as history
import repair_dataset as data
import workflow
from register_pool_history import pool_candidate, pool_config


def register() -> dict:
    """Revalidate existing exclusion evidence and register legitimate historical masks.

    Returns:
        Source hashes, qualification records and deduplicated supplemental counts.
    """
    available = workflow.validate_frozen_data()
    source = data.HERE / "results/historical_scope_audit.json"
    old = json.loads(source.read_text())
    records, candidates = [], {}
    for record in old["records"]:
        if record.get("reason") != "no ETF-local block; outside the fixed block-kernel comparison":
            continue
        path = data.ROOT / record["manifest"]
        if data.digest(path) != record["manifest_sha256"]:
            raise ValueError("Historical source changed after the original inventory")
        manifest = json.loads(path.read_text())
        family, reason = history.historical_mask(manifest, path, available)
        if reason or family is None:
            raise ValueError(f"Supplemental mask is not valid: {path}: {reason}")
        wrapper = manifest["us_equity_first_round_wrapper"]
        assert not wrapper["etf_columns"]
        item = pool_candidate(family, [], wrapper["macro_columns"], [record["manifest"]])
        group = candidates.setdefault(family, {})
        if item["id"] in group:
            group[item["id"]]["source_evidence"].append(record["manifest"])
        else:
            group[item["id"]] = item
        records.append({"family": family, "manifest": record["manifest"],
                        "manifest_sha256": record["manifest_sha256"], "repaired_mask_id": item["id"],
                        "disposition": "valid_macro_only_historical_mask",
                        "previous_exclusion_not_authorized_by_goal": True})
    if len(records) != 5 or {family: len(items) for family, items in candidates.items()} != {
        "us_equity": 1, "international_equity": 1
    }:
        raise ValueError("Supplemental inventory differs from the reviewed five-source/two-mask scope")
    report = {"source_audit_path": str(source.relative_to(data.ROOT)), "source_audit_sha256": data.digest(source),
              "records": records, "new_historical_masks": {family: len(items) for family, items in candidates.items()},
              "reason": "The goal does not mandate a nonempty ETF-local block. The existing wrapper can omit it "
                        "under the same default covariance construction rule. This repairs inventory scope, "
                        "not the data or model code; original results and adaptive caps remain unchanged."}
    audit_path = data.HERE / "results/historical_macro_only_audit.json"
    if audit_path.exists() and json.loads(audit_path.read_text()) != report:
        raise ValueError("Refusing to replace a different supplemental inventory")
    data.write_json(audit_path, report)
    for family, items in candidates.items():
        name = f"00_{family}_macro_only_historical_extension"
        config = pool_config(family, list(items.values()), name)
        config["historical_scope"] = (
            "Previously omitted valid macro-only masks, revalidated with explicit source hashes.")
        path = data.HERE / "plans" / f"{name}.json"
        pending = data.HERE / "planning" / f"{name}.pending.json"
        data.write_json(pending, config)
        workflow.validate_plan(config, pending, available)
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Refusing to replace a different registered supplemental baseline")
        if not path.exists():
            data.write_json(path, config)
    return report


if __name__ == "__main__":
    result = register()
    print(json.dumps({"registered_historical_masks": result["new_historical_masks"],
                      "source_manifests": len(result["records"])}))
