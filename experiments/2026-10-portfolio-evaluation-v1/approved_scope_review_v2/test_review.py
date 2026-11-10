"""Synthetic gates for unchanged risk budgets and certified feasibility.

Usage: .venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/approved_scope_review_v2
No network, GP fitting or reads/writes of completed study artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest
from feasibility import allocate_checked, preflight


@pytest.mark.parametrize("minimum", [None, 3, 5, 7, 10, 12, 15])
def test_original_historical_budget_has_equal_weight_witness(minimum: int | None) -> None:
    """Equal weighting guarantees feasibility at the original historical budget."""
    rng = np.random.default_rng(27)
    factor = rng.normal(size=(15, 15))
    matrix = factor @ factor.T * 0.0001 + np.eye(15) * 0.0002
    covariance = np.broadcast_to(matrix, (24, 15, 15)).copy()
    equal = np.ones(15) / 15
    budget = np.einsum("a,tab,b->t", equal, covariance, equal)
    checks = preflight(covariance, budget, minimum)
    assert all(row["status"] == "feasible_equal_weight_witness" for row in checks)
    assert all(not row["budget_relaxed"] for row in checks)


def test_infeasibility_has_a_valid_lower_bound() -> None:
    """A covariance with known minimum variance supplies a numerical dual certificate."""
    covariance = np.eye(15)[None, :, :]
    checks = preflight(covariance, np.array([0.01]), 12)
    assert checks[0]["status"] == "infeasible_dual_certificate"
    lower = checks[0]["minimum_variance_lower_to_budget"]
    upper = checks[0]["minimum_variance_upper_to_budget"]
    assert lower <= 1 / (15 * 0.01) + 1e-7
    assert lower == pytest.approx(upper, abs=1e-7)
    assert not checks[0]["budget_relaxed"]


def test_checked_allocator_blocks_before_score_optimization(monkeypatch: pytest.MonkeyPatch) -> None:
    """An impossible policy cannot become an automatic higher-budget allocation."""
    from stable_allocator import StableScoreAllocator

    def forbidden(*args, **kwargs):
        raise AssertionError("Allocation must not run after infeasible preflight")

    monkeypatch.setattr(StableScoreAllocator, "allocate", forbidden)
    covariance = np.broadcast_to(np.eye(15), (24, 15, 15)).copy()
    result = allocate_checked(covariance, np.full(24, 0.01), 12, 0.25, np.ones((24, 15)))
    assert result["status"] == "blocked_preflight" and result["allocations"] is None
    assert len(result["preflight"]) == 24


def test_checked_allocator_uses_repaired_backend_when_feasible() -> None:
    """The approved entrypoint preserves all seeds' monthly axes and raw constraints."""
    covariance = np.broadcast_to(np.eye(15) * 0.0004, (24, 15, 15)).copy()
    result = allocate_checked(covariance, np.full(24, 0.01), 10, 0.25, np.tile(np.arange(15) / 15, (24, 1)))
    assert result["status"] == "allocated"
    assert len(result["allocations"]) == 24
    assert all(row["status"] == "solved" for row in result["allocations"])
    assert all(not row["fallback_used"] and not row["constraint_relaxation"] for row in result["allocations"])


def test_infeasible_comparator_cannot_be_counted_as_a_GP_win() -> None:
    """A complete GP policy remains recorded but excluded from an incomplete pair."""
    from review import _policy_rows

    def artifact(source: str, complete: bool) -> dict:
        return {
            "summary": {
                "configuration": "joint",
                "risk_source": source,
                "objective": "rank",
                "minimum": 12,
                "budget_factor": 0.5,
                "complete_seeds": 3 if complete else 0,
                "conventional_sharpe": 1.5 if complete else None,
            },
            "seeds": [{"months": [{"status": "solved" if complete else "infeasible"}]}],
        }

    rows, pairs = _policy_rows(
        {"GP": artifact("GP", True), "historical": artifact("historical", False)},
        {"primary_variance_budget_factor": 1.0, "sensitivity_variance_budget_factor": 1.5},
    )
    assert len(rows) == 2 and all(row["analysis_role"] == "stress" for row in rows)
    assert not pairs[0]["eligible"] and pairs[0]["GP_minus_historical_sharpe"] is None
    assert pairs[0]["reason"] == "historical_policy_infeasible"


def test_numerical_failure_not_mislabeled_as_infeasible() -> None:
    """Reporting must not merge uncertified optimization with conflicting constraints."""
    from review import _policy_rows

    artifact = {
        "summary": {
            "configuration": "joint",
            "risk_source": "historical",
            "objective": "rank",
            "minimum": 12,
            "budget_factor": 0.5,
            "complete_seeds": 0,
        },
        "seeds": [{"months": [{"status": "optimal_inaccurate"}]}],
    }
    with pytest.raises(ValueError, match="unresolved numerical failure"):
        _policy_rows(
            {"case": artifact}, {"primary_variance_budget_factor": 1.0, "sensitivity_variance_budget_factor": 1.5}
        )
