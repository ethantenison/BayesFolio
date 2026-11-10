"""Register previously quarantined July masks supported by the frozen v2 pool.

Existing v1 plans and fitted results remain unchanged. New masks retain the same
kernel rule, dates, seeds, labels, and common feature values, but record their
actual v2 dataset hashes. Unsupported liquidity/turnover blocks remain explicit.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/register_pool_history.py
"""

from __future__ import annotations

import json
from typing import Any

import inventory_history as history
import repair_dataset as data
import run_study as study
from candidate_masks import research_candidate
from extend_feature_pool import OUTPUT


def pool_candidate(family: str, etf: list[str], macro: list[str], evidence: list[str]) -> dict[str, Any]:
    """Create a fixed-kernel candidate that explicitly records the v2 artifact hash."""
    result = research_candidate(family, etf, macro, evidence)
    result["expected_feature_sha256"] = data.digest(OUTPUT / f"{family}.parquet")
    return result


def pool_config(family: str, candidates: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Return a v2 input-pool batch without changing comparison or portfolio controls."""
    result = study.phase_config(family, candidates, name)
    result["feature_path"] = str((OUTPUT / f"{family}.parquet").relative_to(data.ROOT))
    result["feature_manifest"] = str((OUTPUT / f"{family}.manifest.json").relative_to(data.ROOT))
    result["dataset_version"] = "v2"
    result["cross_version_equality_audit"] = str((data.HERE / "results/feature_pool_extension_audit.json").relative_to(
        data.ROOT))
    return result


def main() -> None:
    """Audit July sources again and register only new, supported v2 historical masks."""
    meta = json.loads((OUTPUT / "global.manifest.json").read_text())
    available = set(meta["required_columns"])
    seen = {family: set() for family in data.FAMILIES}
    for path in (data.HERE / "plans").glob("*.json"):
        if path.name.endswith("_july_pool_extension.json"):
            continue
        config = json.loads(path.read_text())
        if config["family"] in seen:
            seen[config["family"]].update(item["id"] for item in config["configurations"])
    extra: dict[str, dict[str, Any]] = {family: {} for family in data.FAMILIES}
    audit = []
    for root in history.HISTORICAL_ROOTS:
        for path in sorted(root.rglob("manifest.json")):
            manifest = json.loads(path.read_text())
            family, reason = history.historical_mask(manifest, path, available)
            if reason:
                if reason.startswith("quarantined"):
                    audit.append({"manifest": str(path.relative_to(data.ROOT)), "disposition": "still_quarantined",
                                  "reason": reason})
                continue
            assert family is not None
            wrapper = manifest["us_equity_first_round_wrapper"]
            item = pool_candidate(family, wrapper["etf_columns"], wrapper["macro_columns"],
                                  [str(path.relative_to(data.ROOT))])
            if item["id"] in seen[family]:
                continue
            if item["id"] in extra[family]:
                extra[family][item["id"]]["source_evidence"].extend(item["source_evidence"])
            else:
                extra[family][item["id"]] = item
            audit.append({"manifest": str(path.relative_to(data.ROOT)), "disposition": "new_v2_mask",
                          "family": family, "repaired_mask_id": item["id"], "manifest_sha256": data.digest(path)})
    for family, items in extra.items():
        if not items:
            continue
        name = f"00_{family}_july_pool_extension"
        path = data.HERE / "plans" / (name + ".json")
        config = pool_config(family, list(items.values()), name)
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError(f"Refusing to change a registered batch: {path}")
        data.write_json(path, config)
    counts = {family: len(items) for family, items in extra.items()}
    data.write_json(data.HERE / "results/historical_pool_scope_audit.json", {
        "new_v2_masks": counts, "records": audit,
        "shared_value_equality_required": True, "original_v1_plans_unchanged": True})
    print(json.dumps({"new_v2_masks": counts, "still_quarantined_runs": sum(
        item["disposition"] == "still_quarantined" for item in audit)}))


if __name__ == "__main__":
    main()
