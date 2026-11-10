"""Regression checks for the corrected portfolio floor and pre-solve MGK exclusion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("minimum_correction", Path(__file__).with_name("run.py"))
assert _spec is not None and _spec.loader is not None
correction = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(correction)


def test_deployment_rejects_the_previous_floor_three() -> None:
    """An exploratory low-diversification setting cannot produce November weights."""
    with pytest.raises(ValueError, match="minimum effective asset floor is seven"):
        correction.checked_allocation(np.arange(15, dtype=float), np.eye(15) * .001,
            {"minimum_effective_assets": 3, "equity_cap": None}, .001)


def test_MGK_is_excluded_before_allocation_and_seven_survive() -> None:
    """Even the highest MGK forecast cannot consume a required investment slot."""
    mean = np.arange(15, dtype=float)
    mean[1] = 1000
    case = {"minimum_effective_assets": 7, "equity_cap": None}
    first = correction.checked_allocation(mean, np.eye(15) * .001, case, .001)
    mean[1] = -1000
    second = correction.checked_allocation(mean, np.eye(15) * .001, case, .001)
    np.testing.assert_array_equal(first["weights"], second["weights"])
    assert first["weights"][1] == 0
    assert first["meaningful_holdings"] >= 7
    assert first["raw_checks"]["effective_assets"] == pytest.approx(7)
    assert len(correction.ASSETS) == 15
    assert "MGK" in correction.ASSETS
