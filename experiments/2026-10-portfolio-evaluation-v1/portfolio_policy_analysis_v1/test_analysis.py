"""Fast numerical tests for saved-policy ranking analysis.

Usage: .venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/portfolio_policy_analysis_v1
Synthetic values only; no fits, network, or completed-study writes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analyze import _bootstrap, _pair, _pareto


def test_pareto_directions_and_ties() -> None:
    """Retain distinct tradeoffs and equal ties, excluding strictly dominated rows."""
    points = np.array([[1, 2, 3], [0, 1, 2], [3, 1, 2], [1, 2, 3]])
    np.testing.assert_array_equal(_pareto(points), [True, False, True, True])


def test_pareto_rejects_missing_objective() -> None:
    """Incomplete objectives cannot accidentally enter the frontier."""
    with pytest.raises(ValueError):
        _pareto(np.array([[np.nan, 1]]))


def test_paired_bootstrap_identical_returns() -> None:
    """Identical models have exactly zero paired differences in every draw."""
    panel = np.random.default_rng(27).normal(0.01, 0.04, (3, 24))
    score = _bootstrap(panel, panel, draws=100)
    assert score["mean_seed_sharpe_difference"] == pytest.approx(0)
    np.testing.assert_allclose(score["conditional_95pct_interval"], [0, 0], atol=1e-14)


def test_pair_excludes_incomplete_policy_without_crossing_covariance() -> None:
    """Pair within the same model and risk source, never across policies/sources."""
    rows = [
        {
            "family": "joint",
            "configuration": "a",
            "covariance_policy": source,
            "variant": variant,
            "conventional_sharpe": value,
        }
        for source, variant, value in [
            ("GP", "risk_budget_only", 1),
            ("GP", "risk_budget_plus_effective", 2),
            ("historical", "risk_budget_only", 4),
        ]
    ]
    paired = _pair(pd.DataFrame(rows))
    assert len(paired) == 1 and paired.iloc[0].covariance_policy == "GP"
    assert paired.iloc[0].diversified_minus_risk_only == pytest.approx(1)
