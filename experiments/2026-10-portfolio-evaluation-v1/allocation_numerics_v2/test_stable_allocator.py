"""Deterministic equivalence and strict failure-policy gates for numerical repair.

Usage: .venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/allocation_numerics_v2
Synthetic arrays only; no GP fitting, networking or study artifact writes.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest
from stable_allocator import StableScoreAllocator


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariance = np.broadcast_to(np.eye(15) * 0.0004, (24, 15, 15)).copy()
    return covariance, np.full(24, 0.01), np.tile(np.arange(1, 16) / 15, (24, 1))


def test_smooth_solution_matches_analytic_simplex_projection() -> None:
    """With inactive risk, smooth rank is an exact Euclidean projection."""
    covariance, budget, scores = _inputs()
    rows = StableScoreAllocator(covariance, budget, None, 0.25).allocate(scores)
    target = scores[0] / 0.5
    ordered = np.sort(target)[::-1]
    threshold = (np.cumsum(ordered) - 1) / np.arange(1, 16)
    last = np.flatnonzero(ordered > threshold)[-1]
    expected = np.maximum(target - threshold[last], 0)
    assert all(row["status"] == "solved" for row in rows)
    np.testing.assert_allclose(rows[0]["weights"], expected, rtol=0, atol=2e-5)


@pytest.mark.parametrize("minimum", [None, 3, 10, 12])
def test_original_covariance_and_effective_count_constraints(minimum: int | None) -> None:
    """Check raw weights against original quadratic constraints, not cone surrogates."""
    covariance, _, scores = _inputs()
    rng = np.random.default_rng(27)
    factor = rng.normal(size=(15, 15))
    covariance[:] = factor @ factor.T * 0.0001 + np.eye(15) * 0.0003
    equal = np.ones(15) / 15
    budget = np.einsum("a,tab,b->t", equal, covariance, equal)
    rows = StableScoreAllocator(covariance, budget, minimum, 0.25).allocate(scores)
    assert all(row["status"] == "solved" for row in rows)
    for row in rows:
        weights = np.asarray(row["weights"])
        assert weights.min() >= -1e-8
        assert weights.sum() == pytest.approx(1, abs=1e-7)
        assert weights @ covariance[row["index"]] @ weights <= budget[row["index"]] * (1 + 1e-6)
        if minimum:
            assert weights @ weights <= 1 / minimum + 1e-7


def test_infeasible_month_stays_null_and_later_months_continue() -> None:
    """No solver retry may loosen the budget or reuse earlier weights."""
    covariance, budget, scores = _inputs()
    covariance[3] = np.eye(15)
    rows = StableScoreAllocator(covariance, budget, 10, 0.25).allocate(scores)
    assert rows[3]["status"] == "infeasible" and rows[3]["weights"] is None
    assert len(rows[3]["numerical_attempts"]) == 1
    assert rows[4]["status"] == "solved"
    assert all(not row["fallback_used"] and not row["constraint_relaxation"] for row in rows)


def test_inaccurate_status_never_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even feasible approximate weights remain null without certified optimality."""
    original = cp.Problem.solve

    def inaccurate(problem, *args, **kwargs):
        value = original(problem, *args, **kwargs)
        problem._status = cp.OPTIMAL_INACCURATE
        return value

    monkeypatch.setattr(cp.Problem, "solve", inaccurate)
    covariance, budget, scores = _inputs()
    rows = StableScoreAllocator(covariance, budget, None, 0.25).allocate(scores)
    assert all(row["status"] == "optimal_inaccurate" and row["weights"] is None for row in rows)


def test_second_algorithm_solves_same_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    """A numerical retry is not a substitute allocation or changed investment policy."""
    original = cp.Problem.solve

    def reject_clarabel(problem, *args, **kwargs):
        if kwargs.get("solver") == "CLARABEL":
            raise cp.error.SolverError("Synthetic first-algorithm failure")
        return original(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", reject_clarabel)
    covariance, budget, scores = _inputs()
    rows = StableScoreAllocator(covariance, budget, 10, 0.25).allocate(scores)
    assert all(row["status"] == "solved" for row in rows)
    assert all(row["numerical_attempts"][-1]["solver"] == "SCS" for row in rows)
    assert all(not row["fallback_used"] and not row["constraint_relaxation"] for row in rows)


def test_objective_centering_and_scaling_preserve_raw_objective() -> None:
    """The transformed objective differs only by an explicit positive scale and constant."""
    covariance, budget, scores = _inputs()
    row = StableScoreAllocator(covariance, budget, 10, 0.25).allocate(scores)[0]
    weights = np.asarray(row["weights"])
    original = scores[0] @ weights - 0.25 * (weights @ weights)
    assert row["scaled_objective"] / 4 + scores[0].mean() == pytest.approx(original, abs=1e-9)
