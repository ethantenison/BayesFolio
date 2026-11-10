"""Synthetic total-return accounting, monthly rebalancing and dated-revision tests.

Usage: .venv/bin/pytest -q <this_directory>. No fitting, network or study writes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from benchmark_math import portfolio_returns, scheduled_weights, wealth_path


def test_cash_earns_RF_and_is_not_double_counted() -> None:
    """A 100 percent cash portfolio earns RF, not zero or twice RF."""
    actual, rf = np.zeros((3, 1)), np.array([0.01, 0.02, 0.03])
    weights = np.tile([0, 1], (3, 1))
    np.testing.assert_allclose(portfolio_returns(actual, rf, weights), rf, rtol=0, atol=1e-14)


def test_mixed_portfolio_agrees_with_total_return_weighting() -> None:
    """Discounted-excess reconstruction equals a direct ETF and cash sum."""
    total = np.array([[0.1, -0.02], [-0.03, 0.01], [0.02, 0.04]])
    rf = np.array([0.005, 0.004, 0.003])
    actual = (1 + total) / (1 + rf[:, None]) - 1
    weights = np.tile([0.5, 0.45, 0.05], (3, 1))
    expected = np.sum(total * weights[:, :-1], axis=1) + weights[:, -1] * rf
    np.testing.assert_allclose(portfolio_returns(actual, rf, weights), expected, rtol=0, atol=1e-14)


def test_monthly_rebalance_is_not_buy_and_hold_drift() -> None:
    """Each month uses the target weights on the entire current account value."""
    actual = np.array([[0.2, 0], [0.2, 0], [0.2, 0]])
    total = portfolio_returns(actual, np.zeros(3), np.tile([0.5, 0.5, 0], (3, 1)))
    path = wealth_path(total)
    np.testing.assert_allclose(path, [10000, 11000, 12100, 13310], rtol=0, atol=1e-9)
    assert path[-1] != pytest.approx(5000 * 1.2**3 + 5000, abs=1e-9)


def test_revision_uses_first_rebalance_after_publication() -> None:
    """Neither a future announcement nor same-day release is backfilled."""
    dates = ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"]
    revision = {
        "published_on": "2025-02-28",
        "effective_on": "2025-01-01",
        "weights": [0.6, 0.4],
        "source_url": "https://example.test/source",
    }
    result = scheduled_weights(dates, [0.5, 0.5], [revision])
    assert result["active_revision"] == [-1, -1, 0, 0]
    np.testing.assert_allclose(result["weights"][:2], [[0.5, 0.5]] * 2, rtol=0, atol=1e-14)


def test_announced_revision_waits_until_effective_date() -> None:
    """Early publication does not activate a future effective allocation."""
    revision = {
        "published_on": "2025-01-01",
        "effective_on": "2025-03-15",
        "weights": [0.6, 0.4],
        "source_url": "https://example.test/source",
    }
    result = scheduled_weights(["2025-01-31", "2025-02-28", "2025-03-31"], [0.5, 0.5], [revision])
    assert result["active_revision"] == [-1, -1, 0]


def test_no_invented_annual_weight_changes() -> None:
    """Without a verified revision, year boundaries do not alter target weights."""
    result = scheduled_weights(["2024-12-31", "2025-01-31", "2026-01-30"], [0.95, 0.05], [])
    np.testing.assert_array_equal(result["weights"], np.tile([0.95, 0.05], (3, 1)))
    assert result["applied_revision_count"] == 0


@pytest.mark.parametrize("initial", [[0.6, 0.5], [-0.1, 1.1], [np.nan, 0.5]])
def test_invalid_target_weights_are_not_normalized(initial: list[float]) -> None:
    """Bad target allocations fail instead of silently changing the benchmark."""
    with pytest.raises(ValueError):
        scheduled_weights(["2025-01-31"], initial, [])


def test_missing_monthly_returns_fail() -> None:
    """A missing observation cannot create a shortened or filled return history."""
    with pytest.raises(ValueError):
        portfolio_returns(np.array([[np.nan]]), np.array([0.001]), np.array([[0.95, 0.05]]))


def test_seeds_are_individual_paths_not_extra_months() -> None:
    """Three runs compound separately, preserving the initial investment each time."""
    total = np.array([[0.1, -0.1], [-0.1, 0.1], [0.05, 0.05]])
    result = wealth_path(total)
    assert result.shape == (3, 3)
    np.testing.assert_allclose(result[:, 0], 10000, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result[:, -1], [9900, 9900, 11025], rtol=0, atol=1e-9)


def test_all_modules_and_public_symbols_documented() -> None:
    """Cover experiment files excluded from the standard repository doc gate."""
    for path in Path(__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text())
        assert ast.get_docstring(tree), path
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                assert ast.get_docstring(node), (path, node.name)
