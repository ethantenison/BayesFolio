"""Local synthetic cutoff and target-availability checks for November data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_extension import MARKET_ORIGIN, _splice_dividend_revision, cutoffs, validate_panel  # noqa: E402


def _panel() -> tuple[pd.DataFrame, list[str]]:
    tickers = [f"ETF{i:02d}" for i in range(15)]
    dates = pd.date_range("2017-05-31", MARKET_ORIGIN, freq="BME")
    frame = pd.DataFrame([(date, ticker, .01, 1.) for date in dates for ticker in tickers],
                         columns=["date", "asset_id", "y_excess_lead", "input"])
    frame.loc[frame.date.eq(pd.Timestamp(MARKET_ORIGIN)), "y_excess_lead"] = np.nan
    return frame, tickers


def test_october31_asof_uses_october30_business_origin() -> None:
    """Saturday as-of must not invent a Saturday market observation."""
    values = cutoffs()
    assert values["as_of_date"] == "2026-10-31"
    assert values["forecast_origin"] == "2026-10-30"
    assert values["last_labeled_origin"] == "2026-09-30"
    assert values["target_month_end"] == "2026-11-30"


def test_complete_data_requires_113_labeled_origins_and_one_live_origin() -> None:
    """The October origin is unlabeled; September's October return is known."""
    frame, tickers = _panel()
    result = validate_panel(frame, tickers)
    assert result["passed"]
    assert result["rows"] == 1710
    assert result["labeled_origins_per_etf"] == 113
    assert result["unavailable_november_targets"] == 15


@pytest.mark.parametrize("fault", ["future_target", "missing_feature", "missing_old_target", "missing_task"])
def test_invalid_data_cannot_pass(fault: str) -> None:
    """No future label, imputed input, missing target or omitted task is accepted."""
    frame, tickers = _panel()
    if fault == "future_target":
        frame.loc[frame.date.eq(pd.Timestamp(MARKET_ORIGIN)), "y_excess_lead"] = 0.
    elif fault == "missing_feature":
        frame.loc[0, "input"] = np.nan
    elif fault == "missing_old_target":
        frame.loc[0, "y_excess_lead"] = np.nan
    else:
        frame = frame.iloc[:-1]
    with pytest.raises(ValueError):
        validate_panel(frame, tickers)


def _dividend_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2026-08-03", MARKET_ORIGIN)
    frame = pd.DataFrame(100., index=dates, columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    frame["Volume"] = 1000
    old = frame.loc[:"2026-09-30"].copy()
    fresh = frame.copy()
    fresh.loc[fresh.index < "2026-09-28", "Adj Close"] = 99.
    evidence = frame.copy()
    evidence["Dividends"] = 0.
    evidence["Stock Splits"] = 0.
    evidence.loc["2026-09-28", "Dividends"] = 1.
    return old, fresh, evidence


def test_dividend_verified_append_keeps_old_vintage_bitwise_unchanged() -> None:
    """Recognized pre-anchor dividend corrections do not rewrite frozen history."""
    old, fresh, evidence = _dividend_fixture()
    combined, audit = _splice_dividend_revision(old, fresh, evidence)
    pd.testing.assert_frame_equal(combined.loc[old.index], old, check_exact=True)
    assert audit["historical_dividend_revision_recognized_not_applied"]
    assert audit["corporate_action_checks"][0]["ex_date"] == "2026-09-28"


@pytest.mark.parametrize("fault", ["wrong_dividend", "changed_raw_price", "split"])
def test_unknown_revisions_are_not_accepted(fault: str) -> None:
    """The audited exception is not a general relaxation of overlap verification."""
    old, fresh, evidence = _dividend_fixture()
    if fault == "wrong_dividend":
        evidence.loc["2026-09-28", "Dividends"] = 2.
    elif fault == "changed_raw_price":
        fresh.loc[fresh.index[2], "Close"] = 100.5
    else:
        evidence.loc["2026-09-28", "Stock Splits"] = 2.
    with pytest.raises((ValueError, AssertionError)):
        _splice_dividend_revision(old, fresh, evidence)
