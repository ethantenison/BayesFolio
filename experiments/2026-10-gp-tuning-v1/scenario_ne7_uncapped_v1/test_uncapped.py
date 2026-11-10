"""Regression coverage for cap removal and the capped diagnostic's mean-sign edge."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "track1_scenarios")]
from case_evaluator import _dependency  # noqa: E402
from uncapped_math import Policy, allocate_scenarios, ne7_math, strict_solve  # noqa: E402


def test_no_nonredundant_position_cap() -> None:
    """An NE7-optimal single-leading-asset portfolio may exceed 25 percent."""
    mean = np.r_[.1, np.full(14, .001)]
    covariance = np.eye(15) * .0025
    with patch.object(ne7_math, "_strict_solve", strict_solve):
        result = ne7_math._positive(mean, covariance, Policy(7, 1))
    assert result["weights"].max() > .25
    assert result["audit"]["effective_count"] >= 7 - 1e-6


@pytest.mark.parametrize("cap", [.25, 1.])
def test_mean_sign_numerical_edge(cap: float) -> None:
    """The one inaccurate capped mean-sign test now has bounded strict retries."""
    path = HERE.parent / "scenario_numerics_v2/outputs/cases/t1_r5_i36_compact.npz"
    with np.load(path) as values:
        scenarios = _dependency("sampling").sample_joint(values["mean"][0, 22], values["covariance"][0, 22],
                                                          seed=27, month="2026-03-31")
        assets = values["assets"].tolist()
    result = allocate_scenarios(scenarios, assets, seed=27, policy=Policy(7, cap))
    assert result["solver_status"] == "optimal"
    assert result["feasibility"]["certificate"]["status"] == "optimal"


def test_all_negative_returns_uncapped() -> None:
    """Removing a cap does not add a positive-return requirement."""
    scenarios = np.random.default_rng(20261025).uniform(-.025, -.015, size=(5000, 15))
    result = allocate_scenarios(scenarios, [f"ETF{i}" for i in range(15)], seed=27)
    assert result["monthly_forecast_sharpe"] < 0
    assert result["minimum_portfolio_return_constraint"] is False
    assert result["audit"]["effective_count"] >= 7 - 1e-6
