"""Synthetic tests for append-only indexing and unchanged allocation economics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from extension_math import append_paths, blend, hard_rank
from extension_workers import validate_job


def test_global_indices_never_restart_at_zero() -> None:
    """New fits retain the original month-keyed initialization seeds."""
    config = {"dates": [str(i) for i in range(28)]}
    for index in range(24, 28):
        validate_job({"index": index, "date": str(index), "seed": 27}, config)
    for index in [0, 23, 28]:
        with pytest.raises(ValueError):
            validate_job({"index": index, "date": str(index), "seed": 27}, config)


def test_mismatched_origin_rejected() -> None:
    """A global index cannot silently attach to a different month's data."""
    with pytest.raises(ValueError):
        validate_job({"index": 24, "date": "25", "seed": 127}, {"dates": [str(i) for i in range(28)]})


def test_frozen_family_mean_blend_and_covariance() -> None:
    """Family fractions affect means, while covariance stays globally 50/50."""
    broad, compact = np.zeros((3, 4, 15)), np.ones((3, 4, 15))
    covariance = np.broadcast_to(np.eye(15) * 0.001, (3, 4, 15, 15))
    mean, risk = blend(broad, compact, covariance, covariance * 3)
    np.testing.assert_array_equal(mean[0, 0], np.r_[np.full(4, .5), np.zeros(4), np.full(4, .375), np.full(3, .25)])
    np.testing.assert_array_equal(risk, covariance * 2)


def test_rank_ignores_mean_magnitude_and_preserves_raw_constraints() -> None:
    """The new per-month adapter retains rank economics, NE7 and no position cap."""
    mean = np.linspace(-.03, .03, 15)
    risk, budget = np.eye(15) * 0.001, 0.001
    record = hard_rank(mean, risk, budget)
    altered = hard_rank(mean * 20 + 4, risk, budget)
    weights = np.array(record["weights"])
    np.testing.assert_allclose(weights, altered["weights"], rtol=0, atol=1e-12)
    assert weights.sum() == pytest.approx(1, abs=1e-7)
    assert weights.min() >= -1e-8
    assert weights @ weights <= 1/7 + 1e-7
    assert weights @ risk @ weights <= budget * (1 + 1e-6)
    assert record["fallback_used"] is False


def test_append_preserves_the_exact_prefix() -> None:
    """Appending wealth never recomputes the original NAV observations."""
    old = {"assets": np.array(["a"]), "seeds": np.array([27]),
           "dates": np.array(["old"] * 24), "gp_test_wealth": np.arange(25)[None, :] * 1.01}
    tail = {"dates": np.array(["new"] * 4), "gp_test_wealth": np.arange(4)[None, :] + 30}
    combined = append_paths(old, tail)
    np.testing.assert_array_equal(combined["gp_test_wealth"][:, :25], old["gp_test_wealth"])
    assert combined["gp_test_wealth"].shape == (1, 29)


def test_month_adapter_replays_frozen_hard_rank_weights() -> None:
    """The dynamic-date adapter reproduces existing allocator controls, no GP refit."""
    root = Path(__file__).resolve().parents[1]
    with (np.load(root / "scenario_sharpe_v3/outputs/paths.npz") as old,
          np.load(root / "uncertainty_rank_v1/outputs/input_panel.npz") as inputs):
        for month in [0, 11, 23]:
            for seed in range(3):
                replay = hard_rank(old["mean"][seed, month], old["covariance"][seed, month],
                                   float(inputs["risk_budget"][month]))
                np.testing.assert_allclose(replay["weights"], old["gp_hard_rank_weights"][seed, month],
                                           rtol=0, atol=1e-8)
