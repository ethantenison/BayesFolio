"""Reconcile July discovery masks with the frozen October rebaseline inventory.

This read-only source audit preserves all historical artifacts and registers only
previously tested, completed, correctly scoped masks supported by the repaired v1
feature pool. Unsupported predictors are explicitly quarantined, not silently
substituted. Old metrics are never reused as new results.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/inventory_history.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import repair_dataset as data
import run_study as study
from candidate_masks import research_candidate

HISTORICAL_ROOTS = [data.ROOT / "experiments/2026-07-us-equity-etf-features",
                    data.ROOT / "experiments/2026-07-international-equity-etf-features"]


def historical_mask(manifest: dict[str, Any], path: Path, available: set[str]) -> tuple[str | None, str | None]:
    """Classify a historical run without misclassifying shorter scored panels as invalid masks."""
    wrapper = manifest.get("us_equity_first_round_wrapper", {})
    universe = manifest.get("data", {}).get("training_universe", [])
    family = next((key for key, tickers in data.FAMILIES.items() if universe == tickers), None)
    if not wrapper:
        return family, "not a GP walk-forward wrapper manifest; source/provenance or non-GP study"
    if family is None:
        return family, "not an exact approved ETF universe"
    if not wrapper.get("time_index_included"):
        return family, "time index omitted"
    if len(manifest.get("rebalance_dates", [])) < 12:
        return family, "smoke/scout only; fewer than 12 scored months"
    if not (path.parent / "gp_window_predictions.csv").is_file():
        return family, "no completed prediction artifact"
    predictions = pd.read_csv(path.parent / "gp_window_predictions.csv")
    scored_dates = manifest["rebalance_dates"]
    required = ["date", "asset_id", "y_true", "y_pred", "y_std"]
    if not set(required).issubset(predictions.columns):
        return family, "incomplete historical prediction schema"
    if len(predictions) != len(scored_dates) * len(universe):
        return family, "incomplete historical prediction row count"
    if sorted(pd.to_datetime(predictions.date).dt.date.astype(str).unique()) != sorted(scored_dates):
        return family, "historical prediction dates do not match the manifest"
    finite = np.isfinite(predictions[["y_true", "y_pred", "y_std"]]).all().all()
    if set(predictions.asset_id) != set(universe) or not finite:
        return family, "invalid historical task coverage or nonfinite predictions"
    if family == "international_equity" and wrapper.get("covar_grouping_proposal") == "split_groups":
        return family, "invalid historical international hard-coded split_groups screen"
    columns = study.canonical([*wrapper.get("etf_columns", []), *wrapper.get("macro_columns", [])])
    missing = sorted(set(columns) - available)
    if missing:
        return family, "quarantined: predictors not validated in frozen v1 pool: " + ", ".join(missing)
    return family, None


def main() -> None:
    """Write the complete scope audit and immutable supplemental historical batches."""
    available = set(json.loads((data.ARTIFACTS / "global.manifest.json").read_text())["required_columns"])
    already: dict[str, set[str]] = {family: set() for family in data.FAMILIES}
    plans = list((data.HERE / "plans").glob("*.json"))
    for path in plans:
        config = json.loads(path.read_text())
        if config["family"] in already and not path.name.endswith("_july_extension.json"):
            already[config["family"]].update(item["id"] for item in config["configurations"])
    supplemental: dict[str, dict[str, Any]] = {family: {} for family in data.FAMILIES}
    records = []
    for root in HISTORICAL_ROOTS:
        for path in sorted(root.rglob("manifest.json")):
            manifest = json.loads(path.read_text())
            family, reason = historical_mask(manifest, path, available)
            record: dict[str, Any] = {"manifest": str(path.relative_to(data.ROOT)), "family": family,
                                      "manifest_sha256": data.digest(path)}
            if reason:
                record.update({"disposition": "quarantined" if reason.startswith("quarantined") else "excluded",
                               "reason": reason})
            else:
                assert family is not None
                wrapper = manifest["us_equity_first_round_wrapper"]
                item = research_candidate(family, wrapper["etf_columns"], wrapper["macro_columns"],
                                          [str(path.relative_to(data.ROOT))])
                record["repaired_mask_id"] = item["id"]
                if item["id"] in already[family]:
                    record["disposition"] = "already_registered_october_mask"
                elif item["id"] in supplemental[family]:
                    supplemental[family][item["id"]]["source_evidence"].extend(item["source_evidence"])
                    record["disposition"] = "deduplicated_july_mask"
                else:
                    supplemental[family][item["id"]] = item
                    record["disposition"] = "registered_july_extension"
            records.append(record)
    counts = {family: len(items) for family, items in supplemental.items()}
    for family, items in supplemental.items():
        if not items:
            continue
        name = f"00_{family}_july_extension"
        config = study.phase_config(family, list(items.values()), name)
        config["historical_scope"] = "Valid July feature masks supported by frozen repaired v1; metrics not pooled."
        path = data.HERE / "plans" / (name + ".json")
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError(f"Refusing to overwrite a different historical extension: {path}")
        data.write_json(path, config)
    out = data.HERE / "results/historical_scope_audit.json"
    dispositions = dict(Counter(record["disposition"] for record in records))
    data.write_json(out, {"scanned_roots": [str(root.relative_to(data.ROOT)) for root in HISTORICAL_ROOTS],
                         "source_manifest_count": len(records), "dispositions": dispositions,
                         "additional_masks": counts, "records": records,
                         "policy": "12-month full runs may generate masks; old scored-panel length is not reused.",
                         "quarantine_policy": "Unsupported columns are not invented or silently substituted/dropped."})
    lines = ["# Historical scope reconciliation", "",
             "The first inventory covered October's 24-window studies. This additional audit explicitly",
             "covers July, including valid full 12-month runs. Old metrics remain separate and invalid",
             "hard-coded international split-group screens remain excluded.", "",
             f"Historical manifests audited: {len(records)}.", "",
             "Additional unique masks supported by the repaired v1 feature pool:", "",
             *[f"- {family}: {count}" for family, count in counts.items()], "",
             "Disposition counts:", "", *[f"- {key}: {value}" for key, value in dispositions.items()], "",
             "Detailed per-run reasons, manifest hashes, and repaired mask identifiers are recorded in",
             "`historical_scope_audit.json`. Quarantined predictors were not validated in frozen v1;",
             "their absence is not evidence that they are unimportant. Adding them requires a separately",
             "versioned feature-pool extension, not modification of frozen datasets or silent substitution.", ""]
    (data.HERE / "results/HISTORICAL_SCOPE_AUDIT.md").write_text("\n".join(lines))
    print(json.dumps({"manifests": len(records), "additional_masks": counts, "dispositions": dispositions}))


if __name__ == "__main__":
    main()
