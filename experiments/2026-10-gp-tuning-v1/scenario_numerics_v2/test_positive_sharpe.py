"""Numerical unit and frozen-failure regressions; no GP fits or altered returns.

Usage: uv run --locked --no-sync pytest -q <this_file>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "track1_scenarios"))
from allocator import allocate_scenarios  # noqa: E402
from case_evaluator import _dependency  # noqa: E402
from positive_sharpe import solve_positive  # noqa: E402

SOURCE = HERE.parent / "outputs"
FAILED = [(p.stem, f) for p in sorted((SOURCE / "cases").glob("t1_*.json"))
          for f in json.loads(p.read_text())["failures"]]


def test_equal_moments() -> None:
    """Equal independent assets uniquely maximize Sharpe at equal weight."""
    result = solve_positive(np.full(15, .01), np.eye(15) * .0025)
    np.testing.assert_allclose(result["weights"], np.full(15, 1 / 15), atol=1e-7)
    assert result["monthly_forecast_sharpe"] == pytest.approx(.2 * np.sqrt(15), abs=1e-9)


@pytest.mark.parametrize("case_id,failure", FAILED, ids=[f"{c}_{f['seed']}_{f['origin']}" for c, f in FAILED])
def test_frozen_failure(case_id: str, failure: dict) -> None:
    """Every formerly inaccurate allocation solves and matches direct Sharpe."""
    with np.load(SOURCE / "cases" / f"{case_id}.npz") as arrays:
        s, m = failure["seed_index"], failure["month_index"]
        scenarios = _dependency("sampling").sample_joint(
            arrays["mean"][s, m], arrays["covariance"][s, m], seed=failure["seed"], month=failure["origin"])
    mean, covariance = scenarios.mean(0), np.cov(scenarios, rowvar=False)
    result = solve_positive(mean, covariance)
    independent = _dependency("scenarios").solve_signed(mean, covariance, seed=20261025, random_starts=4)
    assert result["solver_status"] == "optimal"
    assert result["certificate"]["duality_gap"] < 1e-8
    assert result["monthly_forecast_sharpe"] == pytest.approx(independent["monthly_forecast_sharpe"], abs=1e-8)
    # Scaling the returns does not change the underlying investment problem.
    other = solve_positive(mean * 100, covariance * 10000)
    np.testing.assert_allclose(result["weights"], other["weights"], atol=2e-7, rtol=0)


def test_reject_indefinite_covariance() -> None:
    """Never silently repair a non-SPD covariance."""
    with pytest.raises(np.linalg.LinAlgError):
        solve_positive(np.full(15, .01), -np.eye(15))


def test_reject_nonpositive_branch() -> None:
    """Nonpositive means must retain the preexisting signed-Sharpe route."""
    with pytest.raises(ValueError, match="Positive-Sharpe|positive feasible|certificate"):
        solve_positive(np.full(15, -.01), np.eye(15))


@pytest.mark.parametrize("kind", ["all_negative", "mixed_negative_optimum", "mixed_positive_optimum"])
def test_negative_returns_allowed(kind: str) -> None:
    """Negative observations and even an optimal negative mean are valid."""
    scenarios = np.random.default_rng(20261025).uniform(-.004, .004, size=(5000, 15)) - .02
    if kind == "mixed_negative_optimum":
        scenarios[:, 0] += .022
    elif kind == "mixed_positive_optimum":
        scenarios[:, :8] += .04
    original = scenarios.copy()
    result = allocate_scenarios(scenarios, [f"ETF{i}" for i in range(15)], seed=27)
    np.testing.assert_array_equal(scenarios, original)
    assert result["negative_scenario_observations"] > 0
    assert result["minimum_portfolio_return_constraint"] is False
    assert np.isfinite(result["monthly_forecast_sharpe"])
    if kind != "mixed_positive_optimum":
        assert result["maximum_feasible_mean"] < 0
        assert result["monthly_forecast_sharpe"] < 0
        assert result["solver_route"] == "direct_signed_sharpe_multistart"
    else:
        assert result["solver_status"] == "optimal"


def test_sealed_campaign_cannot_be_changed() -> None:
    """The reusable evaluation entry point refuses completed historical data."""
    from allocator import evaluate_case

    with pytest.raises(ValueError, match="sealed"):
        evaluate_case({}, str(SOURCE))
