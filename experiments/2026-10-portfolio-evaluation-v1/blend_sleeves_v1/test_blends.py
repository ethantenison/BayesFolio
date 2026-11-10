"""Deterministic synthetic tests for blending and combined sleeve constraints.

Research-only tests use decimal returns and decimal-squared covariance. No
network, GP fitting or old experiment writes. Usage: pytest this directory.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest
from blend_math import _solve, allocation_scores, blend_panel, joint_allocate, sleeve_allocate


def _sources() -> tuple[dict, dict]:
    a = np.array([[[0.04, 0.006], [0.006, 0.02]]])
    b = np.array([[[0.02, -0.002], [-0.002, 0.03]]])
    return {"mean": np.array([[0.01, -0.02]]), "covariance": a}, {
        "mean": np.array([[-0.01, 0.03]]),
        "covariance": b,
    }


@pytest.mark.parametrize("method", ["arithmetic", "precision", "log", "correlation", "mixture"])
def test_spd_and_endpoints(method: str) -> None:
    """Every covariance method retains SPD and exact source endpoint moments."""
    a, b = _sources()
    for fraction, source in [(0, a), (1, b)]:
        panel = blend_panel(a, b, np.full(2, fraction), method, fraction)
        np.testing.assert_allclose(panel["mean"], source["mean"], atol=1e-14, rtol=1e-10)
        np.testing.assert_allclose(panel["covariance"], source["covariance"], atol=1e-14, rtol=1e-10)
    middle = blend_panel(a, b, np.full(2, 0.5), method, 0.5)
    np.linalg.cholesky(middle["covariance"])


def test_mixture_between_model_variance() -> None:
    """Mixture covariance includes disagreement and rejects heterogeneous weights."""
    a, b = _sources()
    difference = b["mean"][0] - a["mean"][0]
    expected = 0.5 * (a["covariance"] + b["covariance"]) + 0.25 * np.outer(difference, difference)
    np.testing.assert_allclose(blend_panel(a, b, np.full(2, 0.5), "mixture", 0.5)["covariance"], expected)
    with pytest.raises(ValueError, match="common mixture"):
        blend_panel(a, b, np.array([0, 1]), "mixture", 0.5)


def test_family_blend_leaves_other_means_untouched() -> None:
    """Asset-specific fractions do not change unselected model means."""
    a, b = _sources()
    panel = blend_panel(a, b, np.array([1.0, 0]), "arithmetic", 0)
    np.testing.assert_allclose(panel["mean"], [[-0.01, -0.02]], atol=1e-14)
    np.testing.assert_allclose(a["mean"], [[0.01, -0.02]], atol=1e-14)


def test_borda_is_not_labeled_a_return() -> None:
    """Decision scores differ from ranks of raw-return averages when scale differs."""
    a, b = np.array([[0.001, 0.002, 0.003]]), np.array([[0.03, 0.02, 0.01]])
    alpha, mean = np.full(3, 0.5), (a + b) / 2
    decision = allocation_scores(a, b, mean, alpha, "borda")
    np.testing.assert_allclose(decision, np.full((1, 3), 2 / 3), atol=1e-14)
    assert not np.allclose(decision, allocation_scores(a, b, mean, alpha, "mean"))


def test_invalid_covariance_not_repaired() -> None:
    """Indefinite covariance is rejected rather than eigenvalue clipped."""
    a, b = _sources()
    a["covariance"][0, 0, 0] = -1
    with pytest.raises(np.linalg.LinAlgError):
        blend_panel(a, b, np.zeros(2), "arithmetic", 0)


def _sleeve_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariance = (np.eye(15) * 0.0002 + np.ones((15, 15)) * 0.00005)[None]
    scores = np.r_[np.arange(1, 9) / 8, np.arange(1, 8) / 7][None]
    return covariance, np.array([0.002]), scores


def test_separate_sleeves_match_convex_control_and_include_cross_risk() -> None:
    """Dual-separated solves enforce combined seven, not seven per sleeve."""
    covariance, budget, scores = _sleeve_data()
    record = sleeve_allocate(covariance, budget, scores, 7, list(range(8)))[0]
    actual = np.array(record["weights"])
    assert actual[:8].sum() == pytest.approx(0.8, abs=1e-7)
    assert actual[8:].sum() == pytest.approx(0.2, abs=1e-7)
    assert record["effective_count"] >= 7 - 1e-6
    assert 1 / np.sum((actual[8:] / 0.2) ** 2) < 7
    assert record["cross_sleeve_variance_contribution"] > 0
    risk_parts = sum(
        record[key]
        for key in [
            "equity_variance_contribution",
            "fixed_income_variance_contribution",
            "cross_sleeve_variance_contribution",
        ]
    )
    assert risk_parts == pytest.approx(record["predicted_variance"], abs=1e-12)
    # Independent control with identical local-risk and combined-concentration constraints.
    weights = cp.Variable(15)
    constraints = [
        weights >= 0,
        cp.sum(weights[:8]) == 0.8,
        cp.sum(weights[8:]) == 0.2,
        cp.norm(weights) <= 1 / np.sqrt(7),
    ]
    for indices, share in [(list(range(8)), 0.8), (list(range(8, 15)), 0.2)]:
        factor = np.linalg.cholesky(covariance[0][np.ix_(indices, indices)] / budget[0]).T
        constraints.append(cp.norm(factor @ weights[indices] / share) <= 1)
    problem = cp.Problem(cp.Maximize(scores[0] @ weights - 1e-8 * cp.sum_squares(weights)), constraints)
    _solve(problem)
    assert scores[0] @ actual == pytest.approx(float(scores[0] @ weights.value), abs=2e-7)
    assert record["concentration_dual_gap"] < 5e-8


def test_joint_fixed_capital_control() -> None:
    """Full-vector control retains all cross-covariance and hard constraints."""
    covariance, budget, scores = _sleeve_data()
    row = joint_allocate(covariance, budget, scores, 7, list(range(8)))[0]
    assert sum(row["weights"][:8]) == pytest.approx(0.8, abs=1e-7)
    assert row["effective_count"] >= 7 - 1e-6
    assert row["fallback_used"] is False
