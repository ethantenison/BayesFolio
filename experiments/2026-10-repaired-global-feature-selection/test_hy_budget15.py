"""Guard the one-factor HY-change fitting-budget comparison; no financial I/O."""

from __future__ import annotations

import copy

import pytest
from run_hy_budget15 import budget15_variant


def test_only_budget_changes_without_mutating_reference():
    reference = {"variants": {"plus_hy_change": {
        "command": ["python", "wrapper.py", "--maxiter", "36", "--seed", "27"],
        "inputs": ["t_index", "hy_spread_chg_1p", "hy_spread_z_12p"]}}}
    before = copy.deepcopy(reference)
    changed = budget15_variant(reference)
    assert reference == before
    expected = copy.deepcopy(before["variants"]["plus_hy_change"])
    expected["command"][3] = "15"
    assert changed == expected


def test_rejects_missing_hy_change():
    with pytest.raises(ValueError, match="HY-change/36"):
        budget15_variant({"variants": {"plus_hy_change": {
            "command": ["python", "--maxiter", "36"], "inputs": ["t_index"]}}})


def test_rejects_wrong_reference_cap():
    with pytest.raises(ValueError, match="HY-change/36"):
        budget15_variant({"variants": {"plus_hy_change": {
            "command": ["python", "--maxiter", "150"], "inputs": ["hy_spread_chg_1p"]}}})
