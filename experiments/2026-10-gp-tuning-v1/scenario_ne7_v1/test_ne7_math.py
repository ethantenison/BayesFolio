"""Explicit-policy and negative-return regression tests; no fits or data refresh."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ne7_math import Policy, _maximum_mean, _positive, allocate_scenarios, audit_weights, solve_direct  # noqa: E402


@pytest.mark.parametrize("minimum", [7., 10.])
def test_equal_independent_assets(minimum: float) -> None:
    """An unchanged Sharpe objective gives equal weights for identical assets."""
    result = _positive(np.full(15, .01), np.eye(15) * .0025, Policy(minimum))
    np.testing.assert_allclose(result["weights"], np.full(15, 1 / 15), atol=1e-7)
    assert result["monthly_forecast_sharpe"] == pytest.approx(.2 * np.sqrt(15), abs=1e-8)


def test_cap_cannot_be_ignored_after_changing_effective_count() -> None:
    """The 25% cap is not redundant at NE7 and must remain enforced."""
    mean = np.r_[.1, np.zeros(14)]
    maximum, feasibility = _maximum_mean(mean, Policy())
    assert maximum[0] == pytest.approx(.25, abs=1e-7)
    assert feasibility["maximum_feasible_mean"] == pytest.approx(.025, abs=1e-8)
    bad = np.r_[1 / 3, np.full(14, (2 / 3) / 14)]
    assert 1 / (bad @ bad) == pytest.approx(7)
    with pytest.raises(ValueError):
        audit_weights(bad, Policy())


@pytest.mark.parametrize("kind", ["negative", "mixed_negative", "positive"])
def test_negative_returns_are_valid(kind: str) -> None:
    """Neither mixed observations nor a negative optimum imposes a return floor."""
    scenarios = np.random.default_rng(20261025).uniform(-.004, .004, size=(5000, 15)) - .02
    if kind == "mixed_negative":
        scenarios[:, 0] += .022
    elif kind == "positive":
        scenarios[:, :8] += .04
    original = scenarios.copy()
    result = allocate_scenarios(scenarios, [f"ETF{i}" for i in range(15)], seed=27)
    audit_weights(result["weights"], Policy())
    np.testing.assert_array_equal(scenarios, original)
    assert result["negative_scenario_observations"] > 0
    assert result["minimum_portfolio_return_constraint"] is False
    if kind != "positive":
        assert result["monthly_forecast_sharpe"] < 0


def test_ne10_parity_with_sealed_positive_repair() -> None:
    """Parameterization preserves the prior repaired NE10 numerical objective."""
    path = HERE.parent / "scenario_numerics_v2/positive_sharpe.py"
    spec = importlib.util.spec_from_file_location("_sealed_ne10_math", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(127)
    factor = rng.standard_normal((15, 15)) * .01
    covariance = factor @ factor.T + np.eye(15) * .001
    mean = rng.uniform(-.004, .02, 15)
    old = module.solve_positive(mean, covariance)
    new = _positive(mean, covariance, Policy(10))
    assert new["monthly_forecast_sharpe"] == pytest.approx(old["monthly_forecast_sharpe"], abs=1e-8)


def test_conic_and_direct_sharpe_agree_under_ne7() -> None:
    """The conic scaling does not change the ratio or economic constraints."""
    rng = np.random.default_rng(227)
    factor = rng.standard_normal((15, 15)) * .01
    covariance = factor @ factor.T + np.eye(15) * .001
    mean = rng.uniform(-.004, .02, 15)
    conic = _positive(mean, covariance, Policy())
    direct = solve_direct(mean, covariance, Policy(), seed=27, random_starts=4)
    assert conic["monthly_forecast_sharpe"] == pytest.approx(direct["monthly_forecast_sharpe"], abs=1e-8)
    stricter = _positive(mean, covariance, Policy(10))
    assert conic["monthly_forecast_sharpe"] >= stricter["monthly_forecast_sharpe"] - 1e-8
