"""Synthetic mathematical, strict failure and runner-isolation regression tests.

Usage: .venv/bin/pytest -q <this_directory>. No study writes, GP fits or network.
"""

from __future__ import annotations

import ast

import cvxpy as cp
import numpy as np
import pytest
import run_rank_risk as study
from rank_risk_math import RankRiskAllocator, allocate_rank_risk_checked
from stable_allocator import StableScoreAllocator


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariance = np.broadcast_to(np.diag(np.linspace(0.0001, 0.004, 15)), (24, 15, 15)).copy()
    return covariance, np.full(24, 0.01), np.tile(np.arange(1, 16) / 15, (24, 1))


def test_zero_penalty_exact_previous_raw_weights() -> None:
    """Zero lambda leaves the prior problem and numerical path unchanged."""
    covariance, budget, scores = _inputs()
    old = StableScoreAllocator(covariance, budget, 7, 1e-8).allocate(scores)
    new = RankRiskAllocator(covariance, budget, 7, 1e-8, 0).allocate(scores)
    np.testing.assert_array_equal([r["weights"] for r in old], [r["weights"] for r in new])


def test_risk_penalty_changes_weights_without_binding_ceiling() -> None:
    """Risk affects holdings even when every solution is well below the ceiling."""
    covariance, budget, scores = _inputs()
    rows = [RankRiskAllocator(covariance, budget, 7, 1e-8, value).allocate(scores)[0] for value in [0, 3, 10]]
    assert all(row["status"] == "solved" and row["normalized_predicted_variance"] < 0.2 for row in rows)
    assert rows[-1]["normalized_predicted_variance"] < rows[0]["normalized_predicted_variance"] - 0.005
    assert rows[-1]["weighted_rank_score"] < rows[0]["weighted_rank_score"] - 0.01
    assert np.max(np.abs(np.array(rows[-1]["weights"]) - rows[0]["weights"])) > 0.01


def test_utility_equals_explicit_quadratic_objective() -> None:
    """Factored variance and objective centering do not change economic utility."""
    covariance, budget, scores = _inputs()
    row = RankRiskAllocator(covariance, budget, 7, 1e-8, 0.5).allocate(scores)[0]
    weights = np.asarray(row["weights"])
    expected = scores[0] @ weights - 0.5 * (weights @ covariance[0] @ weights) / budget[0] - 1e-8 * (weights @ weights)
    assert row["rank_risk_utility"] == pytest.approx(expected, abs=1e-12)
    assert row["scaled_objective"] + scores[0].mean() == pytest.approx(expected, abs=1e-9)


def test_common_variance_scale_is_unit_invariant() -> None:
    """Scaling both variance units and the common reference preserves weights."""
    covariance, budget, scores = _inputs()
    left = RankRiskAllocator(covariance, budget, 7, 1e-8, 0.5).allocate(scores)
    right = RankRiskAllocator(covariance * 10000, budget * 10000, 7, 1e-8, 0.5).allocate(scores)
    np.testing.assert_allclose([r["weights"] for r in left], [r["weights"] for r in right], rtol=0, atol=1e-6)


def test_equal_scores_recover_minimum_variance_solution() -> None:
    """With no ranking preference, positive lambda minimizes variance."""
    covariance, budget, _ = _inputs()
    rows = RankRiskAllocator(covariance, budget, None, 1e-8, 1).allocate(np.full((24, 15), 0.5))
    inverse = 1 / (np.diag(covariance[0]) / budget[0] + 1e-8)
    expected = inverse / inverse.sum()
    np.testing.assert_allclose(rows[0]["weights"], expected, rtol=0, atol=2e-5)


@pytest.mark.parametrize("risk_aversion", [-1, np.inf, np.nan])
def test_invalid_risk_aversion_rejected(risk_aversion: float) -> None:
    """The declared tradeoff cannot be negative or nonfinite."""
    covariance, budget, _ = _inputs()
    with pytest.raises(ValueError, match="Risk aversion"):
        RankRiskAllocator(covariance, budget, 7, 1e-8, risk_aversion)


def test_infeasible_policy_has_no_relaxation_or_partial_portfolio() -> None:
    """Preflight blocks an impossible month without using realized returns."""
    covariance, budget, scores = _inputs()
    covariance[0] = np.eye(15)
    result = allocate_rank_risk_checked(covariance, budget, 7, 1e-8, scores, risk_aversion=0.5)
    assert result["status"] == "blocked_preflight" and result["allocations"] is None
    assert result["preflight"][0]["status"] == "infeasible_dual_certificate"
    assert all(not row["budget_relaxed"] for row in result["preflight"])


def test_inaccurate_solver_status_never_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The extension inherits the same rejection of inaccurate allocations."""
    original = cp.Problem.solve

    def inaccurate(problem, *args, **kwargs):
        """Inject an inaccurate status after the real synthetic solve."""
        result = original(problem, *args, **kwargs)
        problem._status = cp.OPTIMAL_INACCURATE
        return result

    monkeypatch.setattr(cp.Problem, "solve", inaccurate)
    covariance, budget, scores = _inputs()
    rows = RankRiskAllocator(covariance, budget, 7, 1e-8, 0.5).allocate(scores)
    assert all(row["weights"] is None and row["status"] == "optimal_inaccurate" for row in rows)
    assert all(not row["fallback_used"] and not row["constraint_relaxation"] for row in rows)


def test_registered_grid_and_covariance_ablations() -> None:
    """Only the two frozen recipes and four approved sources enter the grid."""
    cases = study._cases()
    assert len(cases) == len({case["id"] for case in cases}) == 56
    covariance, _, _ = _inputs()
    source = {
        "broad_covariance": np.broadcast_to(covariance, (3, 24, 15, 15)),
        "compact_covariance": np.broadcast_to(2 * covariance, (3, 24, 15, 15)),
        "historical_covariance": 3 * covariance,
    }
    half, diagonal = (study._covariance(source, method) for method in ["half", "diagonal_half"])
    np.testing.assert_array_equal(np.diagonal(diagonal, axis1=-2, axis2=-1), np.diagonal(half, axis1=-2, axis2=-1))
    assert study._covariance(source, "historical").shape == (3, 24, 15, 15)
    with pytest.raises(ValueError, match="Unknown covariance"):
        study._covariance(source, "unregistered")


def test_worker_adapters_are_process_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frozen scorer is reused with only panel, destination and allocator adapters."""
    for name in ["OUTPUT", "_panel", "allocate_checked"]:
        monkeypatch.setattr(study.base, name, getattr(study.base, name))
    monkeypatch.setattr(
        study.base,
        "_job",
        lambda case: {
            "case": case,
            "output": study.base.OUTPUT,
            "panel": study.base._panel,
            "allocator": study.base.allocate_checked,
        },
    )
    case = study._case("sharpe", "half", 0.25)
    result = study._worker(case)
    assert result["output"] == study.OUTPUT and result["panel"] is study._panel
    assert result["allocator"].func is allocate_rank_risk_checked
    assert result["allocator"].keywords == {"risk_aversion": 0.25}


def test_off_diagonal_covariance_affects_allocation() -> None:
    """Dependence changes holdings with marginal variances and ranks held fixed."""
    covariance, budget, scores = _inputs()
    correlated = covariance.copy()
    correlated[:, -1, -2] = correlated[:, -2, -1] = 0.9 * np.sqrt(covariance[:, -1, -1] * covariance[:, -2, -2])
    left = RankRiskAllocator(covariance, budget, 7, 1e-8, 1).allocate(scores)[0]
    right = RankRiskAllocator(correlated, budget, 7, 1e-8, 1).allocate(scores)[0]
    np.testing.assert_array_equal(
        np.diagonal(covariance, axis1=-2, axis2=-1), np.diagonal(correlated, axis1=-2, axis2=-1)
    )
    assert np.max(np.abs(np.asarray(left["weights"]) - right["weights"])) > 0.001
    assert left["normalized_predicted_variance"] < 1 and right["normalized_predicted_variance"] < 1


def test_common_score_offset_does_not_change_weights() -> None:
    """Objective centering preserves invariance to a shared rank-score offset."""
    covariance, budget, scores = _inputs()
    left = RankRiskAllocator(covariance, budget, 7, 1e-8, 0.5).allocate(scores)
    right = RankRiskAllocator(covariance, budget, 7, 1e-8, 0.5).allocate(scores + 2)
    np.testing.assert_allclose([r["weights"] for r in left], [r["weights"] for r in right], rtol=0, atol=1e-6)


def test_all_new_modules_and_public_symbols_are_documented() -> None:
    """Cover experiment files excluded from the repository's normal doc gate."""
    for path in study.HERE.glob("*.py"):
        tree = ast.parse(path.read_text())
        assert ast.get_docstring(tree), path
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith(
                "_"
            ):
                assert ast.get_docstring(node), (path, node.name)
