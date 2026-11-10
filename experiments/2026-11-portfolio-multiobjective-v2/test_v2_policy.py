"""Independent synthetic checks of arbitrary eligible-rank concentration floors."""

from __future__ import annotations

import numpy as np
import pytest
from v2_common import ASSETS
from v2_independent_validation import validate_portfolio
from v2_policy import allocate


@pytest.mark.parametrize("effective", range(3, 15))
def test_all_effective_counts_match_independent_conic(effective: int) -> None:
    """Eligible-rank KKT weights match independently expressed concentration cones."""
    mean = np.random.default_rng(271).normal(size=15) / 100
    covariance = np.eye(15) / 1000
    result = allocate(mean, covariance, min_effective_assets=effective)
    weights = np.asarray(result["weights"])
    assert weights[ASSETS.index("MGK")] == 0
    checked = validate_portfolio(mean, covariance, weights, ASSETS, min_effective_assets=effective,
        solver_status=result["solver_status"], solver_attempts=result["numerical_attempts"],
        analytic_certificate=result["certificate"])
    assert checked["passed"]
    assert weights.min() >= 0
    assert weights.sum() == pytest.approx(1, abs=1e-12)
    assert 1 / (weights @ weights) == pytest.approx(effective, abs=1e-10)


def test_ineligible_MGK_forecast_does_not_change_eligible_rank_scores() -> None:
    """MGK trains jointly but cannot alter gaps in the fourteen allocator scores."""
    mean, covariance = np.arange(15, dtype=float), np.eye(15) / 1000
    first = allocate(mean, covariance)
    mean[1] = -10000
    second = allocate(mean, covariance)
    np.testing.assert_array_equal(first["weights"], second["weights"])
    np.testing.assert_array_equal(first["eligible_ranks"], second["eligible_ranks"])


def test_equity_cap_and_risk_ceiling_are_enforced_without_weight_repairs() -> None:
    """A binding registered equity cap triggers the strict conic path."""
    result = allocate(np.r_[np.arange(8) + 20, np.arange(7)], np.eye(15) * .001,
                      min_effective_assets=7, variance_ceiling=.001, equity_cap=.4)
    assert result["solver_status"] == "optimal"
    weights = np.array(result["weights"])
    assert weights[:8].sum() <= .4 + 1e-7
    assert not result["raw_weights_changed"]
    assert result["raw_checks"]["passed"]


def test_NE14_incompatible_risk_is_an_explicit_infeasible_candidate() -> None:
    """An infeasible risk floor is reported, never replaced by equal weights."""
    result = allocate(np.arange(15, dtype=float), np.eye(15) * .001,
                      min_effective_assets=14, variance_ceiling=.00001)
    assert result["infeasible"]
    assert not result["passed"]
    assert "weights" not in result


def test_NE14_tied_means_still_has_unique_equal_weight_solution() -> None:
    """Equal-rank predictions do not change the mathematical NE14 feasibility."""
    result = allocate(np.zeros(15), np.eye(15) * .001, min_effective_assets=14)
    assert result["passed"]
    assert result["weights"][1] == 0
    assert result["raw_checks"]["effective_assets"] == pytest.approx(14)
