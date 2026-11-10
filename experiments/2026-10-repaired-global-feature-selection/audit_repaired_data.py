"""Audit frozen repaired values against source caches without changing datasets.

This experiment-only reporting entrypoint writes coverage, repair-cell counts,
source-definition notes, and a reproducibility fingerprint. All financial returns
and credit spreads are decimal units. No network access or GP fits occur.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_repaired_data.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import repair_dataset as data


def compare_columns(new: pd.DataFrame, old: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    """Compare corresponding aligned cells, including missing and placeholder repairs."""
    keys = ["date", "asset_id"]
    joined = new.merge(old[[*keys, *columns]], on=keys, how="left", suffixes=("_new", "_old"),
                       validate="one_to_one")
    rows = []
    for column in columns:
        before, after = joined[column + "_old"], joined[column + "_new"]
        changed = ~np.isclose(before, after, rtol=1e-7, atol=1e-10, equal_nan=True)
        rows.append({"column": column, "compared_cells": len(joined), "changed_cells": int(changed.sum()),
                     "old_missing": int(before.isna().sum()), "old_zero": int(before.eq(0).sum()),
                     "missing_to_valid": int((before.isna() & after.notna()).sum()),
                     "zero_to_nonzero": int((before.eq(0) & after.ne(0)).sum()),
                     "new_missing": int(after.isna().sum()), "new_zero": int(after.eq(0).sum())})
    return rows


def main() -> None:
    """Validate immutable outputs and write readable plus machine-readable repair evidence."""
    manifest = json.loads((data.ARTIFACTS / "dataset.manifest.json").read_text())
    for output in manifest["outputs"]:
        if data.digest(data.ROOT / output["path"]) != output["sha256"]:
            raise ValueError(f"Frozen output changed: {output['path']}")
    new = pd.read_parquet(data.ARTIFACTS / "global.parquet")
    coverage = data.validate_panel(new)
    old_etf = pd.read_parquet(data.RAW / "old_etf_features.parquet")
    old_etf = old_etf[old_etf.asset_id.isin(data.TICKERS)].sort_values(["asset_id", "date"])
    shared_etf = [column for column in data.ETF_COLUMNS if column in old_etf]
    old_etf[shared_etf] = old_etf.groupby("asset_id")[shared_etf].shift(1)
    comparison = compare_columns(new, old_etf, shared_etf)
    old_macro = pd.read_parquet(data.RAW / "old_macro.parquet").sort_values("date")
    old_macro = old_macro.rename(columns=data.ALIASES)
    macro_columns = [column for column in old_macro if column in new and column != "date"]
    old_macro[macro_columns] = old_macro[macro_columns].shift(1)
    old_macro = pd.concat([old_macro.assign(asset_id=ticker) for ticker in data.TICKERS], ignore_index=True)
    comparison.extend(compare_columns(new, old_macro, macro_columns))
    labels = pd.read_parquet(data.RAW / "labels.parquet")
    checked = new[["date", "asset_id", "y_excess_lead"]].merge(
        labels, on=["date", "asset_id"], suffixes=("_new", "_old"), validate="one_to_one")
    np.testing.assert_array_equal(checked.y_excess_lead_new, checked.y_excess_lead_old)
    report = {"schema": "bayesfolio.repaired_dataset_audit.v1", "dataset_sha256": data.digest(
        data.ARTIFACTS / "global.parquet"), "coverage": coverage, "targets_bitwise_preserved": True,
        "old_vs_repaired_aligned_cells": comparison,
        "interpretation_limits": ["New dataset uses freshly frozen adjusted market sources.",
                                  "Source revisions, repair, ranks, and history changes are not separable effects.",
                                  "All original experiment data and results remain untouched.",
                                  "Latest-vintage CPI; not a vintage-controlled forecasting claim."]}
    data.write_json(data.HERE / "results/dataset_repair_audit.json", report)
    repairs = pd.DataFrame(comparison)
    repairs.to_csv(data.HERE / "results/dataset_repair_cells.csv", index=False)
    source_audit = json.loads((data.ARTIFACTS / "coverage_audit.json").read_text())
    lines = ["# Repaired dataset validation", "", "Dataset: `artifacts/v1/global.parquet`.", "",
             f"SHA-256: `{report['dataset_sha256']}`.", "",
             "All 15 tasks have exactly 108 complete monthly rows: 84 initial training dates and 24 scored dates.",
             "The original decimal excess-return targets are bitwise unchanged; predictors are lagged exactly once.",
             "", "## Aligned cache repairs", "", repairs.to_markdown(index=False), "",
             "## Inflation availability", "",
             json.dumps(source_audit["inflation_missingness_policy"], indent=2), "",
             "## Exact-index HY source checks", "", json.dumps(source_audit["hy_source_overlap"], indent=2), "",
             "## Interpretation limits", "", *[f"- {note}" for note in report["interpretation_limits"]], "",
             "No fit is authorised by this audit alone; the active study separately authorises bounded execution.", ""]
    (data.HERE / "results/DATASET_REPAIR_AUDIT.md").write_text("\n".join(lines))
    print(json.dumps({"passed": True, "rows": len(new), "targets_preserved": True,
                      "columns_compared": len(comparison)}))


if __name__ == "__main__":
    main()
