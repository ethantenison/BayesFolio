"""Analytic and actual-failure regressions for sign-safe Sharpe maximization."""

from __future__ import annotations

from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest
from signed_sharpe import _loss_gradient, _maximum_mean, allocate_scenarios, solve_signed


def test_closed_form_feasible_mean_matches_independent_conic_solutions() -> None:
    generator = np.random.default_rng(20261024)
    for _ in range(8):
        mean = generator.normal(0, 0.02, 15)
        result, value = _maximum_mean(mean)
        weights = cp.Variable(15)
        problem = cp.Problem(
            cp.Maximize(mean @ weights),
            [weights >= 0, cp.sum(weights) == 1, weights <= 0.25, cp.sum_squares(weights) <= 0.1],
        )
        problem.solve(solver="CLARABEL")
        assert value == pytest.approx(problem.value, abs=1e-8)
        assert result.sum() == pytest.approx(1, abs=1e-12)


def test_negative_sharpe_has_known_optimum_not_minimum_volatility() -> None:
    mean, covariance = np.repeat(-0.01, 15), np.eye(15) * 0.001
    result = solve_signed(mean, covariance, seed=27, random_starts=8)
    expected = -0.01 / np.sqrt(0.001 / 10)
    assert result["monthly_forecast_sharpe"] == pytest.approx(expected, abs=1e-8)
    assert result["weights"] @ result["weights"] == pytest.approx(0.1, abs=1e-9)
    assert result["best_objective_reproductions"] >= 2
    assert result["global_optimality_certified"] is False


def test_zero_means_are_valid_and_do_not_require_positive_normalization() -> None:
    result = solve_signed(np.zeros(15), np.eye(15) * 0.001, seed=27, random_starts=2)
    assert result["monthly_forecast_sharpe"] == pytest.approx(0, abs=1e-14)
    assert result["weights"].sum() == pytest.approx(1, abs=1e-7)


def test_positive_sharpe_matches_the_known_equal_weight_optimum() -> None:
    result = solve_signed(np.repeat(0.01, 15), np.eye(15) * 0.001, seed=27, random_starts=4)
    np.testing.assert_allclose(result["weights"], np.ones(15) / 15, rtol=0, atol=1e-6)
    assert result["monthly_forecast_sharpe"] == pytest.approx(0.01 / np.sqrt(0.001 / 15), abs=1e-9)


def test_riskfolio_positive_branch_matches_the_known_sharpe_optimum() -> None:
    scenarios = 0.01 + np.r_[np.eye(15), -np.eye(15)] * np.sqrt(0.001 * 29 / 2)
    result = allocate_scenarios(scenarios, [f"ETF{i}" for i in range(15)], seed=27)
    assert result["solver_route"] == "riskfolio_MV_positive_mean"
    np.testing.assert_allclose(result["weights"], np.ones(15) / 15, rtol=0, atol=1e-5)
    assert result["monthly_forecast_sharpe"] == pytest.approx(0.01 / np.sqrt(0.001 / 15), abs=1e-8)


def test_signed_gradient_matches_finite_differences() -> None:
    mean = np.linspace(-0.02, 0.01, 15)
    covariance = np.diag(np.linspace(0.0005, 0.002, 15))
    weights = np.ones(15) / 15
    _, gradient = _loss_gradient(weights, mean, covariance)
    epsilon = 1e-6
    finite = np.array(
        [
            (
                _loss_gradient(weights + direction * epsilon, mean, covariance)[0]
                - _loss_gradient(weights - direction * epsilon, mean, covariance)[0]
            )
            / (2 * epsilon)
            for direction in np.eye(15)
        ]
    )
    np.testing.assert_allclose(gradient, finite, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("seed", [27, 127, 227])
def test_previously_infeasible_august_portfolios_keep_the_same_objective_and_constraints(seed: int) -> None:
    root = Path(__file__).resolve().parents[1]
    with np.load(root / "scenario_riskfolio_v1/outputs/failures" / f"s{seed}_2025-07-31.npz") as source:
        scenarios = source["scenarios"]
    with np.load(root / "scenario_riskfolio_v1/outputs/input_panel.npz") as panel:
        assets = panel["assets"].tolist()
    result = allocate_scenarios(scenarios, assets, seed=seed)
    assert result["maximum_feasible_mean"] < 0
    assert result["monthly_forecast_sharpe"] < 0
    assert result["solver_route"] == "direct_signed_sharpe_multistart"
    assert result["weights"].sum() == pytest.approx(1, abs=1e-7)
    assert result["weights"].min() >= -1e-8
    assert result["weights"].max() <= 0.25 + 1e-8
    assert result["audit"]["effective_assets"] >= 10 - 1e-6
