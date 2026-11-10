"""Validate identical means and unspliced covariance in the two requested arms.

Usage: .venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/joint_source_hybrids_v2
Synthetic decimal arrays only; no fitting, portfolio optimization or network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from run_joint_sources import fuse_joint_sources


def _arrays() -> tuple:
    assets = ["SPY", "MGK", "VTV", "IWM"] + [f"ETF{i}" for i in range(11)]
    broad = np.full((3, 24, 15), 0.01)
    compact = np.full((3, 24, 15), 0.02)
    covariance = np.broadcast_to(np.eye(15) * 0.001, (3, 24, 15, 15)).copy()
    return broad, compact, covariance, covariance * 2, assets, assets[:4]


def test_both_covariances_are_retained_whole() -> None:
    """Mean fusion does not combine covariance rows, blocks or variance scales."""
    args = _arrays()
    result = fuse_joint_sources(*args)
    np.testing.assert_array_equal(result["broad_joint"], args[2])
    np.testing.assert_array_equal(result["compact_joint"], args[3])
    assert not np.shares_memory(result["broad_joint"], args[2])
    assert not np.shares_memory(result["compact_joint"], args[3])


def test_only_the_four_US_equity_means_are_replaced() -> None:
    """The same hybrid mean is used for both covariance-source alternatives."""
    args = _arrays()
    result = fuse_joint_sources(*args)
    np.testing.assert_array_equal(result["mean"][:, :, :4], args[1][:, :, :4])
    np.testing.assert_array_equal(result["mean"][:, :, 4:], args[0][:, :, 4:])
    np.testing.assert_array_equal(args[0], np.full((3, 24, 15), 0.01))


def test_nonpositive_covariance_is_not_repaired() -> None:
    """Invalid matrices fail rather than receiving diagonal jitter or clipping."""
    args = list(_arrays())
    args[2][0, 0, 0, 0] = -1
    with pytest.raises(np.linalg.LinAlgError):
        fuse_joint_sources(*args)


def test_compact_means_full_joint_not_US_only_or_NLL_leader() -> None:
    """The compact source is the five-input full-joint SPY comparison model."""
    policy = json.loads((Path(__file__).parent / "policy.json").read_text())
    assert policy["compact_joint"] == "joint_existing_us_mask"
    assert len(policy["compact_non_time_inputs"]) == 5
    assert policy["minimum_effective_assets"] == 7
    assert policy["uncertainty_calibration_experiments"] == "deferred"
    assert not policy["covariance_scaling"] and not policy["covariance_blending"]
