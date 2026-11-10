"""Validate scenario dependence, determinism, constraints and historical parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scenario_allocator import allocate_scenarios, audit_weights, sample_joint


def test_joint_sampling_is_reproducible_and_keeps_cross_asset_dependence() -> None:
    mean = np.array([0.01, -0.005])
    covariance = np.array([[0.001, 0.0005], [0.0005, 0.0008]])
    first = sample_joint(mean, covariance, seed=27, month="2024-05-31", count=100000)
    np.testing.assert_array_equal(first, sample_joint(mean, covariance, seed=27, month="2024-05-31", count=100000))
    np.testing.assert_allclose(first.mean(axis=0), mean, rtol=0, atol=4e-4)
    np.testing.assert_allclose(np.cov(first, rowvar=False), covariance, rtol=0.025, atol=0)
    assert not np.array_equal(first[:100], sample_joint(mean, covariance, seed=127, month="2024-05-31", count=100))


@pytest.mark.parametrize(
    "weights",
    [
        np.r_[0.3, np.repeat(0.05, 14)],
        np.r_[np.repeat(1 / 9, 9), np.zeros(6)],
        np.repeat(0.05, 15),
        np.r_[np.nan, np.zeros(14)],
    ],
)
def test_constraint_violations_are_rejected_without_repair(weights: np.ndarray) -> None:
    with pytest.raises(ValueError):
        audit_weights(weights)


def test_equal_weight_is_feasible_and_unchanged() -> None:
    weights = np.ones(15) / 15
    result = audit_weights(weights)
    assert result["effective_assets"] == pytest.approx(15)
    np.testing.assert_array_equal(weights, np.ones(15) / 15)


def test_nonpositive_definite_covariance_is_not_repaired() -> None:
    with pytest.raises(np.linalg.LinAlgError):
        sample_joint(np.zeros(2), np.ones((2, 2)), seed=27, month="2024-05-31")


def test_actual_riskfolio_matches_the_saved_july_optimizer_fixture() -> None:
    fixture = Path(__file__).resolve().parents[3] / "tests/fixtures/regression/july_2026"
    scenarios = pd.read_csv(fixture / "scenarios.csv")
    expected = pd.read_csv(fixture / "expected_weights.csv")
    assert scenarios.columns.tolist() == expected["asset"].tolist()
    result = allocate_scenarios(scenarios.to_numpy(), scenarios.columns.tolist())
    np.testing.assert_allclose(result["weights"], expected["weight"].to_numpy(), rtol=0, atol=1e-10)
    assert result["attempts"][-1]["status"] == "optimal"
    assert result["attempts"][-1]["package_output_max_alteration"] < 1e-8
