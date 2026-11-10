"""Build feature masks including legitimate macro-only historical baselines.

This experiment-only compatibility adapter preserves the frozen fit-critical
runner and normal nonempty-local candidate bytes. Empty local blocks use the
existing wrapper's natural block omission, not zero masking or a new kernel.
Financial input units and frozen artifacts are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import repair_dataset as data
import run_study as study


def research_candidate(family: str, etf: list[str], macro: list[str], evidence: list[str]) -> dict[str, Any]:
    """Create a canonical mask without inventing a mandatory ETF-local feature.

    Args:
        family: Approved family or joint universe.
        etf: ETF-local feature names, possibly empty for a legitimate baseline.
        macro: Shared macro names; t_index remains implicit and mandatory.
        evidence: Exact source references used to motivate/register the mask.

    Returns:
        The unchanged normal candidate contract, allowing an omitted local block.
    """
    if etf:
        return study.candidate(family, etf, macro, evidence)
    macro = study.canonical(macro)
    key = hashlib.sha256(json.dumps([family, [], macro, study.KERNEL_POLICY], sort_keys=True).encode()).hexdigest()[:12]
    identifier = f"{family}_{key}"
    return {"id": identifier, "label": identifier, "expected_input_columns": ["t_index", *macro],
            "expected_feature_sha256": data.digest(data.ARTIFACTS / f"{family}.parquet"),
            "expected_training_start": data.STUDY["training_start"],
            "args": ["--feature-set", "momentum_plus_macro_market", "--custom-etf-cols", "",
                     "--custom-macro-cols", ",".join(macro), "--covar-grouping-proposal", "default",
                     "--macro-component-proposal", "default", "--time-modulation-mode", "neither"],
            "source_evidence": evidence, "etf_columns": [], "macro_columns": macro,
            "kernel_policy": study.KERNEL_POLICY}
