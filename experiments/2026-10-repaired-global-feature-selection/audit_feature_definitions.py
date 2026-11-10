"""Identify exact feature-definition redundancies without inferring performance importance.

The audit reads the frozen v2 pool, records exact identities and naming aliases,
and proposes paired removal tests. It neither transforms model inputs nor treats
correlation/identity as proof that a feature can be dropped from a finite-budget
GP without changing its kernel geometry, prior, or optimization behavior.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_feature_definitions.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import repair_dataset as data
from extend_feature_pool import OUTPUT


def main() -> None:
    """Write algebraic source-definition evidence for later conditional ablations."""
    frame = pd.read_parquet(OUTPUT / "global.parquet")
    identities = {
        "tnote10y = y10_nominal": frame.tnote10y - frame.y10_nominal,
        "chmom = mom12m - mom6m": frame.chmom - (frame.mom12m - frame.mom6m),
        "term_spread = tnote10y - tbill3m": frame.term_spread - (frame.tnote10y - frame.tbill3m),
        "erp = spy_ret - tbill3m": frame.erp - (frame.spy_ret - frame.tbill3m),
        "vix_slope = log(vix3m) - log(vix)": frame.vix_slope - (np.log(frame.vix3m) - np.log(frame.vix)),
        "gold_crude_ratio = gold / oil": frame.gold_crude_ratio - frame.gold / frame.oil,
    }
    checks = {name: {"rows_checked": len(errors), "max_absolute_residual": float(errors.abs().max()),
                     "passed": bool(np.isfinite(errors).all() and errors.abs().max() < 1e-12)}
              for name, errors in identities.items()}
    if not all(check["passed"] for check in checks.values()):
        raise ValueError("A recorded feature identity does not hold in the frozen data")
    aliases = {}
    for alias, original in data.ALIASES.items():
        if not frame[alias].equals(frame[original]):
            raise ValueError(f"Verified monthly naming alias changed: {alias}")
        aliases[alias] = original
    payload = {"schema": "bayesfolio.feature_definition_redundancy.v1",
               "dataset_sha256": data.digest(OUTPUT / "global.parquet"), "identities": checks,
               "canonical_monthly_aliases": aliases,
               "interpretation": "Exact definitions are not ablation outcomes. Fit paired removals before claiming "
               "unchanged IC/NLL/Sharpe; duplicate axes alter GP geometry and optimization at the fixed budget.",
               "priority_hypotheses": [
                   "Keep only one of the identical Treasury-level columns in broad rate-aware candidates.",
                   "Test chmom versus mom6m/mom12m bases without assuming a nonlinear GP is invariant.",
                   "Test ERP/bill/market-return redundancy where multiple components are present.",
                   "Treat ratio-versus-components tests as representation changes, not independent economic data."]}
    data.write_json(data.HERE / "results/feature_definition_redundancy_audit.json", payload)
    rows = pd.DataFrame([{"identity": name, **check} for name, check in checks.items()])
    lines = ["# Feature-definition redundancy audit", "", rows.to_markdown(index=False), "",
             "Monthly `_1m`/`_1p` and `_12m`/`_12p` aliases are canonicalized before mask deduplication.",
             "The two Treasury level names were not silently canonicalized out of historical masks; their",
             "removal is a useful explicit paired experiment under the fixed kernel/fitting budget.", "",
             payload["interpretation"], "", "Priority hypotheses:", "",
             *[f"- {hypothesis}" for hypothesis in payload["priority_hypotheses"]], ""]
    (data.HERE / "results/FEATURE_DEFINITION_REDUNDANCY_AUDIT.md").write_text("\n".join(lines))
    print(json.dumps({"passed": True, "verified_identities": len(checks), "naming_aliases": len(aliases)}))


if __name__ == "__main__":
    main()
