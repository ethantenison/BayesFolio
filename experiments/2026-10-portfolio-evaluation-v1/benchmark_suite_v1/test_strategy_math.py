"""Synthetic benchmark lookback, exposure, missing-data and causality checks.

Usage: .venv/bin/pytest -q <this_directory>. No network, fitting or study writes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from strategy_math import benchmark_panel


def _fixture() -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    months = np.arange(np.datetime64("2019-01", "M"), np.datetime64("2024-01", "M"))
    dates = (months + 1).astype("datetime64[D]") - 1
    rng = np.random.default_rng(20261024)
    monthly = rng.normal(0.006, 0.025, size=(len(dates) - 1, 15))
    prices = 100 * np.vstack([np.ones(15), np.cumprod(1 + monthly, axis=0)])
    assets = [f"ETF{i:02d}" for i in range(15)]
    return dates, prices, dates[36:].astype(str).tolist(), assets, assets[:8]


def test_static_capital_weights_and_no_added_constraints() -> None:
    """Equal weight and 80/20 preserve their distinct capital exposures."""
    result = benchmark_panel(*_fixture())
    np.testing.assert_allclose(result["equal_weight"], 1 / 15, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["fixed_80_20"][:, :8], 0.1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["fixed_80_20"][:, 8:], 0.2 / 7, rtol=0, atol=1e-14)
    for name in ["equal_weight", "fixed_80_20", "inverse_volatility", "momentum_top7"]:
        np.testing.assert_allclose(result[name].sum(axis=1), 1, rtol=0, atol=1e-14)
        assert (result[name] >= 0).all()


def test_volatility_is_exactly_36_trailing_total_returns() -> None:
    """An independent loop verifies window endpoints, ddof and inverse weights."""
    dates, prices, origins, assets, equities = _fixture()
    result = benchmark_panel(dates, prices, origins, assets, equities)
    returns = prices[1:] / prices[:-1] - 1
    for offset in range(len(origins)):
        std = returns[offset : offset + 36].std(axis=0, ddof=1)
        np.testing.assert_allclose(result["volatility"][offset], std, rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            result["inverse_volatility"][offset], (1 / std) / (1 / std).sum(), rtol=0, atol=1e-14
        )
    np.testing.assert_array_equal(result["volatility_start_price_date"], dates[:24])
    np.testing.assert_array_equal(result["volatility_end_price_date"], dates[36:])


def test_momentum_12_minus_1_uses_11_returns_and_seven_holdings() -> None:
    """Momentum excludes the latest month and selects the seven largest scores."""
    dates, prices, origins, assets, equities = _fixture()
    result = benchmark_panel(dates, prices, origins, assets, equities)
    expected = prices[35:-1] / prices[24:-12] - 1
    np.testing.assert_allclose(result["momentum_score"], expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(result["momentum_selected"].sum(axis=1), np.full(24, 7))
    np.testing.assert_allclose(result["momentum_top7"][result["momentum_selected"]], 1 / 7, rtol=0, atol=1e-14)
    for scores, chosen in zip(expected, result["momentum_selected"], strict=True):
        assert scores[chosen].min() >= scores[~chosen].max()


def test_future_price_poison_does_not_change_earlier_decisions() -> None:
    """Values in later months, including the next-month outcome, cannot leak."""
    dates, prices, origins, assets, equities = _fixture()
    original = benchmark_panel(dates, prices, origins, assets, equities)
    changed = prices.copy()
    changed[37:] *= np.linspace(2, 30, len(prices) - 37)[:, None]
    poisoned = benchmark_panel(dates, changed, origins, assets, equities)
    for key in original:
        np.testing.assert_array_equal(original[key][0], poisoned[key][0])


def test_latest_month_does_not_enter_momentum_but_does_enter_volatility() -> None:
    """Skipping one month is distinct from the volatility availability rule."""
    dates, prices, origins, assets, equities = _fixture()
    original = benchmark_panel(dates, prices, origins, assets, equities)
    changed = prices.copy()
    changed[36] *= np.linspace(0.8, 1.2, 15)
    updated = benchmark_panel(dates, changed, origins, assets, equities)
    np.testing.assert_array_equal(original["momentum_score"][0], updated["momentum_score"][0])
    assert not np.allclose(original["volatility"][0], updated["volatility"][0], rtol=0, atol=1e-8)


def test_ties_are_ticker_sorted_and_invariant_to_input_order() -> None:
    """Tied momentum cannot depend on covariance or arbitrary asset ordering."""
    dates, prices, origins, assets, equities = _fixture()
    prices[:] = prices[:, :1]
    result = benchmark_panel(dates, prices, origins, assets, equities)
    assert result["momentum_selected"][0].tolist() == [True] * 7 + [False] * 8
    order = np.arange(15)[::-1]
    shuffled = benchmark_panel(dates, prices[:, order], origins, [assets[i] for i in order], equities)
    np.testing.assert_array_equal(result["momentum_top7"], shuffled["momentum_top7"][:, order])


@pytest.mark.parametrize("problem", ["missing_month", "nonfinite", "zero_price", "not_available", "short_history"])
def test_bad_history_is_rejected_without_repairs(problem: str) -> None:
    """No observations are filled, dates shifted, or lookbacks shortened."""
    dates, prices, origins, assets, equities = _fixture()
    if problem == "missing_month":
        dates, prices = np.delete(dates, 20), np.delete(prices, 20, axis=0)
    elif problem == "nonfinite":
        prices[20, 0] = np.nan
    elif problem == "zero_price":
        prices[20, 0] = 0
    elif problem == "not_available":
        origins[0] = "2022-01-30"
    else:
        origins = dates[35:].astype(str).tolist()
    with pytest.raises(ValueError):
        benchmark_panel(dates, prices, origins, assets, equities)


def test_zero_volatility_fails_instead_of_imposing_a_floor() -> None:
    """A constant-price asset does not get an invented or infinite weight."""
    dates, prices, origins, assets, equities = _fixture()
    prices[:, 0] = 100
    with pytest.raises(ValueError, match="volatility"):
        benchmark_panel(dates, prices, origins, assets, equities)


def test_negative_momentum_keeps_the_best_seven_without_cash_timing() -> None:
    """Long-only relative strength has no unapproved absolute-momentum filter."""
    dates, prices, origins, assets, equities = _fixture()
    slope = np.arange(len(prices))[:, None] * np.linspace(0.01, 0.02, 15)
    prices = 100 * np.exp(-slope + 0.001 * np.sin(np.arange(len(prices)))[:, None])
    result = benchmark_panel(dates, prices, origins, assets, equities)
    assert (result["momentum_score"] < 0).all()
    np.testing.assert_allclose(result["momentum_top7"].sum(axis=1), 1, rtol=0, atol=1e-14)


def test_public_symbols_and_modules_documented() -> None:
    """Cover experiment modules outside the standard package documentation gate."""
    for path in Path(__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text())
        assert ast.get_docstring(tree), path
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                assert ast.get_docstring(node), (path, node.name)
