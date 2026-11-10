"""Guard the rank-only HY-change comparison without fitting or financial I/O."""

from __future__ import annotations

import copy

import pytest
from run_hy_rank2_iter15 import rank2_variant


def _reference():
    return {"variants": {"hy_iter15": {
        "command": ["python", "wrapper.py", "--mtgp-rank", "5", "--maxiter", "15", "--seed", "27"],
        "inputs": ["t_index", "hy_spread_chg_1p", "hy_spread_z_12p"]}}}


def test_only_rank_changes_without_mutating_completed_reference():
    reference = _reference()
    before = copy.deepcopy(reference)
    changed = rank2_variant(reference)
    expected = copy.deepcopy(before["variants"]["hy_iter15"])
    expected["command"][3] = "2"
    assert changed == expected
    assert reference == before


@pytest.mark.parametrize("index,value", [(3, "2"), (5, "36")])
def test_rejects_mismatched_reference_rank_or_budget(index, value):
    reference = _reference()
    reference["variants"]["hy_iter15"]["command"][index] = value
    with pytest.raises(ValueError, match="rank-five HY-change/15"):
        rank2_variant(reference)


def test_rejects_reference_without_hy_change():
    reference = _reference()
    reference["variants"]["hy_iter15"]["inputs"] = ["t_index", "hy_spread_z_12p"]
    with pytest.raises(ValueError, match="rank-five HY-change/15"):
        rank2_variant(reference)
