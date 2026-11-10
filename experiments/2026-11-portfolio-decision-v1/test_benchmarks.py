"""Synthetic causal timing, ties, negative-score and cash accounting tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmarks  # noqa: E402
from benchmarks import momentum_from_prices, total_returns  # noqa: E402


def test_momentum_uses_skip_month_and_alphabetical_ties() -> None:
    """Ties favor ticker names and the decision month's price does not enter score."""
    dates = pd.date_range("2021-05-31", periods=14, freq="BME").to_numpy("datetime64[D]")
    assets = ["Z", "Y", "X", "G", "F", "E", "D", "C", "B", "A"]
    prices = np.full((14, 10), 100.)
    prices[11] = 90.
    prices[12] = 1.
    first = momentum_from_prices(dates, prices, [str(dates[12])], assets)
    assert sorted(np.array(assets)[first["selected"][0]].tolist()) == ["A", "B", "C", "D", "E", "F", "G"]
    np.testing.assert_allclose(first["scores"], -.1, rtol=0, atol=1e-14)
    prices[12:] = 9000.
    second = momentum_from_prices(dates, prices, [str(dates[12])], assets)
    np.testing.assert_array_equal(first["weights"], second["weights"])
    assert first["start_price_dates"][0] == dates[0]
    assert first["end_price_dates"][0] == dates[11]


def test_negative_momentum_still_holds_seven_assets() -> None:
    """The benchmark does not invent a cash timing rule during negative markets."""
    dates = pd.date_range("2021-05-31", periods=13, freq="BME").to_numpy("datetime64[D]")
    assets = [f"ETF{i}" for i in range(15)]
    prices = np.repeat(np.linspace(100., 70., 13)[:, None], 15, axis=1)
    result = momentum_from_prices(dates, prices, [str(dates[12])], assets)
    assert result["selected"].sum() == 7
    assert np.all(result["scores"] < 0)
    np.testing.assert_allclose(result["weights"].sum(1), 1., rtol=0, atol=1e-14)


def test_cash_earns_rf_and_weighted_excess_uses_existing_formula() -> None:
    """Schwab's 5% cash and negative ETF returns follow the shared return convention."""
    actual = np.array([[-.1, .2]])
    rf = np.array([.005])
    weights = np.array([[.5, .45, .05]])
    expected = (1 + .5 * -.1 + .45 * .2) * 1.005 - 1
    np.testing.assert_allclose(total_returns(actual, rf, weights), expected, rtol=0, atol=1e-14)


def test_missing_prices_and_insufficient_warmup_are_rejected() -> None:
    """No source filling or shortened lookback is allowed."""
    dates = pd.date_range("2021-05-31", periods=13, freq="BME").to_numpy("datetime64[D]")
    assets = [f"ETF{i}" for i in range(15)]
    prices = np.full((13, 15), 100.)
    with pytest.raises(ValueError):
        momentum_from_prices(dates, prices, [str(dates[11])], assets)
    prices[0, 0] = np.nan
    with pytest.raises(ValueError):
        momentum_from_prices(dates, prices, [str(dates[12])], assets)


def test_changed_historical_weights_raise_instead_of_substituting(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reconstructed benchmark disagreement must stop, never reuse saved weights."""
    if not benchmarks.PRIOR_PATHS.exists():
        pytest.skip("Local sealed benchmark fixture is unavailable")
    dates = pd.date_range("2022-05-31", periods=53, freq="BME").strftime("%Y-%m-%d").tolist()
    observed = pd.date_range("2021-05-31", periods=65, freq="BME").to_numpy("datetime64[D]")
    with np.load(benchmarks.PRIOR_PATHS, allow_pickle=False) as prior:
        assets = prior["assets"].tolist()
        actual = np.zeros((53, 15))
        risk_free = np.zeros(53)
        positions = [dates.index(date) for date in prior["dates"]]
        actual[positions], risk_free[positions] = prior["actual"], prior["risk_free"]
    # All-tied prices force an alphabetical allocation that disagrees with the
    # real frozen momentum fixture. The adapter must report that disagreement.
    monkeypatch.setattr(benchmarks, "_history", lambda *args: (observed, np.full((65, 15), 100.), {}))
    with pytest.raises(ValueError, match="never substitute saved weights"):
        benchmarks.build_benchmarks(dates, actual, risk_free, assets)
