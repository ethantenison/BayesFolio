"""Synthetic tests of objective direction and honest three-seed interpretation.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_analysis.py
"""

from __future__ import annotations

import math

import analyze_results as analysis
import numpy as np
import pandas as pd
import pytest


def test_conventional_sharpe_not_cagr_ratio() -> None:
    """Research objectives use arithmetic excess returns and sample deviation."""
    returns = pd.Series([0.02, -0.01] * 12)
    expected = float(returns.mean() / returns.std(ddof=1) * math.sqrt(12))
    assert analysis.sharpe(returns) == pytest.approx(expected)
    assert analysis.sharpe(returns.iloc[:-1]) is None
    assert analysis.sharpe(pd.Series(np.zeros(24))) is None


def test_lower_nll_is_favorable_in_conditional_feature_evidence() -> None:
    """A higher NLL cannot accidentally be treated as an improvement."""
    items = {"without": {"expected_input_columns": ["t_index", "mom12m"]},
             "with": {"expected_input_columns": ["t_index", "mom12m", "oil_ret"]}}
    rows = [{"configuration": name, "seed": seed, "mean_ic": 0.2, "sharpe": 0.7,
             "mean_gaussian_nll": -2.5 if name == "with" else -2.4}
            for name in items for seed in [27, 127, 227]]
    result = analysis.paired_evidence(items, pd.DataFrame(rows))
    assert len(result) == 1
    assert result.iloc[0].feature == "oil_ret"
    assert result.iloc[0].conditional_finding == "helpful_in_this_context"
    assert result.iloc[0].seed_nll_wins == 3
    assert result.iloc[0].delta_nll_with_minus_without == pytest.approx(-0.1)


def test_mixed_objective_effects_remain_tradeoffs() -> None:
    """IC improvement cannot conceal the opposite Sharpe effect in a weighted score."""
    items = {"without": {"expected_input_columns": ["t_index", "mom12m"]},
             "with": {"expected_input_columns": ["t_index", "mom12m", "oil_ret"]}}
    rows = [{"configuration": name, "seed": seed, "mean_ic": 0.3 if name == "with" else 0.2,
             "sharpe": 0.5 if name == "with" else 0.7, "mean_gaussian_nll": -2.4}
            for name in items for seed in [27, 127, 227]]
    result = analysis.paired_evidence(items, pd.DataFrame(rows))
    assert result.iloc[0].conditional_finding == "objective_tradeoff_or_interaction_dependent"
