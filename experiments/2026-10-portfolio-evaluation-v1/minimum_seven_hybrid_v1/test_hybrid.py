"""Validate mean-only fusion and explicit floor-seven controls.

Usage: .venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/minimum_seven_hybrid_v1
Synthetic decimal forecasts only; no GP fits or allocation solves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from run_hybrid import replace_us_means


def test_only_US_equity_means_change_with_labels_not_column_positions() -> None:
    """The hybrid preserves all fixed-income and international-equity means exactly."""
    assets = ["SPY", "MGK", "VTV", "IWM"] + [f"OTHER{i}" for i in range(11)]
    means = np.arange(3 * 24 * 15, dtype=float).reshape(3, 24, 15) / 100000
    local_assets = ["IWM", "SPY", "MGK", "VTV"]
    local = np.full((3, 24, 4), 0.01)
    local[:, :, 0] = 0.02
    result = replace_us_means(means, assets, local, local_assets, assets[:4])
    np.testing.assert_array_equal(result[:, :, 4:], means[:, :, 4:])
    np.testing.assert_array_equal(result[:, :, 3], local[:, :, 0])
    assert not np.shares_memory(result, means)


def test_duplicate_or_missing_labels_are_rejected() -> None:
    """Forecast fusion must never guess an ETF alignment."""
    assets = [f"ETF{i}" for i in range(15)]
    with pytest.raises(ValueError, match="aligned"):
        replace_us_means(
            np.zeros((3, 24, 15)), assets, np.zeros((3, 24, 4)), ["ETF0", "ETF0", "ETF2", "ETF3"], assets[:4]
        )


def test_policy_keeps_seven_and_defers_calibration() -> None:
    """No stale floor-ten default, calibration experiment, cap or CVaR enters this run."""
    policy = json.loads((Path(__file__).parent / "policy.json").read_text())
    assert policy["minimum_effective_assets"] == 7
    assert policy["uncertainty_calibration_experiments"] == "deferred"
    assert not policy["GP_covariance_scaling"] and not policy["GP_historical_blends"]
    assert policy["ETF_position_cap"] is None and not policy["CVaR"]
    assert not policy["fallback_weights"] and not policy["constraint_relaxation"]
