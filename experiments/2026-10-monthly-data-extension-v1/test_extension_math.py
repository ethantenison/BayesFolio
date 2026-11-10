"""Offline regressions for source continuity, vintage preservation and cutoffs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from extension_math import append_macro, monthly_cutoffs, splice_market


def test_only_completed_months_have_resolved_targets() -> None:
    result = monthly_cutoffs("2026-10-24")
    assert result == {
        "completed_month": "2026-09-30",
        "last_scored_origin": "2026-08-31",
        "source_end_exclusive": "2026-10-01",
    }


def test_weekend_calendar_end_uses_the_previous_business_day() -> None:
    assert monthly_cutoffs("2026-06-01")["completed_month"] == "2026-05-29"


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2026-04-01", "2026-10-23")
    price = 100 + np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {
            "Adj Close": price,
            "Close": price + 5,
            "High": price + 7,
            "Low": price + 3,
            "Volume": np.ones(len(index)) * 1000,
        },
        index=index,
    )
    old = frame.loc[:"2026-04-30"].copy()
    old["Volume"] = old["Volume"].astype("int64")
    fresh = frame.copy()
    fresh["Adj Close"] *= 0.8
    return old, fresh


def test_dividend_vintage_is_anchored_without_changing_old_observations() -> None:
    old, fresh = _frames()
    result, audit = splice_market(old, fresh, symbol="SPY", completed="2026-09-30")
    pd.testing.assert_frame_equal(result.loc[old.index], old, check_exact=True)
    assert audit["adjusted_price_scale"] == pytest.approx(1.25)
    assert result.index[-1] == pd.Timestamp("2026-09-30")
    assert result.loc["2026-05-01", "Adj Close"] == pytest.approx(fresh.loc["2026-05-01", "Adj Close"] / 0.8)


def test_nonconstant_vintage_revision_is_not_silently_spliced() -> None:
    old, fresh = _frames()
    fresh.loc["2026-04-02", "Adj Close"] *= 1.05
    with pytest.raises(ValueError, match="Nonconstant"):
        splice_market(old, fresh, symbol="SPY", completed="2026-09-30")


def test_yield_levels_are_never_rescaled() -> None:
    old, fresh = _frames()
    _, audit = splice_market(old, fresh, symbol="^IRX", completed="2026-09-30")
    assert audit["adjusted_price_scale"] == pytest.approx(1)


def test_macro_revisions_are_reported_without_rewriting_old_vintages() -> None:
    old = pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-06-01", "2026-07-01"]))
    fresh = pd.Series(
        [100.1, 101.0, 102.0, 103.0], index=pd.to_datetime(["2026-06-01", "2026-07-01", "2026-08-01", "2026-10-01"])
    )
    result, audit = append_macro(old, fresh, completed="2026-09-30")
    assert result.tolist() == [100.0, 101.0, 102.0]
    assert audit["revised_overlap_observations"] == 1
    assert audit["revisions_applied"] is False
