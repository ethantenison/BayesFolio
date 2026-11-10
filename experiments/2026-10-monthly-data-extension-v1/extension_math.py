"""Pure chronological cutoffs and append-only source splicing for monthly data.

Research numerical layer; no file/network I/O or fitting. Adjusted prices are
vendor price units, yields annual percentage points and return targets decimals.
Historical rows are preserved; new adjusted prices use an audited overlap anchor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_cutoffs(as_of: str) -> dict[str, str]:
    """Separate completed outcomes from the still-unresolved forecast origin.

    Args:
        as_of: User's calendar date, ISO YYYY-MM-DD; no intraday close assumed.

    Returns:
        Last completed business-month bucket, last scoreable forecast origin,
        and exclusive source endpoint. The current calendar month is excluded.
    """
    first = pd.Timestamp(as_of).normalize().replace(day=1)
    completed = pd.offsets.BMonthEnd().rollback(first - pd.Timedelta(days=1))
    origin = completed - pd.offsets.BMonthEnd(1)
    return {
        "completed_month": completed.date().isoformat(),
        "last_scored_origin": origin.date().isoformat(),
        "source_end_exclusive": first.date().isoformat(),
    }


def splice_market(old: pd.DataFrame, fresh: pd.DataFrame, *, symbol: str, completed: str) -> tuple[pd.DataFrame, dict]:
    """Append new daily observations in the frozen historical adjusted-price scale.

    Args:
        old: Original frozen daily OHLCV rows; never modified.
        fresh: New daily vendor snapshot with an overlapping anchor observation.
        symbol: Yahoo symbol; ^IRX/^TNX yields must never be rescaled.
        completed: Latest fully completed monthly endpoint, ISO date.

    Returns:
        Combined raw OHLCV plus anchor/overlap diagnostics. Only Adj Close is
        rescaled for price series: the existing feature builder subsequently
        adjusts OHLC using Adj Close / Close. Yield levels remain unchanged.
    """
    cutoff = pd.Timestamp(completed)
    if old.index.has_duplicates or fresh.index.has_duplicates:
        raise ValueError("Duplicate daily observations")
    if not old.index.is_monotonic_increasing or not fresh.index.is_monotonic_increasing:
        raise ValueError("Daily sources must be chronological")
    fresh = fresh.loc[fresh.index <= cutoff].copy()
    anchor = old.index[-1]
    if anchor not in fresh.index or fresh.index[-1] != cutoff:
        raise ValueError(f"Missing overlap anchor or completed-month endpoint: {symbol}")
    overlap = old.index.intersection(fresh.index)
    if len(overlap) < 10:
        raise ValueError(f"Insufficient overlap for source-vintage check: {symbol}")
    yield_series = symbol in {"^IRX", "^TNX"}
    factor = 1.0 if yield_series else float(old.loc[anchor, "Adj Close"] / fresh.loc[anchor, "Adj Close"])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Invalid adjustment-vintage anchor")
    adjusted = fresh["Adj Close"] * factor
    relative = ((adjusted.loc[overlap] / old.loc[overlap, "Adj Close"]) - 1).abs()
    if not yield_series and relative.max() > 1e-4:
        raise ValueError(f"Nonconstant historical price revisions require investigation: {symbol}, {relative.max()}")
    tail = fresh.loc[fresh.index > anchor, old.columns].copy()
    tail["Adj Close"] *= factor
    # Batch downloads may promote integer Volume to float because other symbols
    # have different trading calendars. Restore its exact, checked integer type
    # before concatenation so historical rows retain both values and dtypes.
    for column, dtype in old.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            values = tail[column].to_numpy(float)
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                raise ValueError(f"Nonintegral or missing integer source field: {symbol}, {column}")
        tail[column] = tail[column].astype(dtype)
    if tail.empty:
        raise ValueError("No new daily data appended")
    combined = pd.concat([old, tail])
    pd.testing.assert_frame_equal(combined.loc[old.index], old, check_exact=True)
    if combined.index[-1] > cutoff:
        raise ValueError("Incomplete/future month leaked into daily sources")
    return combined, {
        "anchor_date": anchor.date().isoformat(),
        "adjusted_price_scale": factor,
        "yield_rescaled": False,
        "overlap_rows": len(overlap),
        "max_overlap_relative_difference": float(relative.max()),
        "appended_daily_rows": len(tail),
        "old_rows_unchanged": True,
    }


def append_macro(old: pd.Series, fresh: pd.Series, *, completed: str) -> tuple[pd.Series, dict]:
    """Preserve frozen macro vintages and append only genuinely new observations.

    Args:
        old: Existing CPI or HY series, potentially extending beyond old model data.
        fresh: Current official observations, used also to identify revisions.
        completed: Calendar cutoff excluding the current incomplete month.

    Returns:
        Existing vintage through the cutoff plus later observations if needed.
        Overlap revisions are reported but never silently applied to old history.
    """
    cutoff = pd.Timestamp(completed)
    old, fresh = old.loc[old.index <= cutoff].copy(), fresh.loc[fresh.index <= cutoff].copy()
    shared = old.index.intersection(fresh.index)
    differences = (old.loc[shared] - fresh.loc[shared]).abs()
    tail = fresh.loc[fresh.index > old.index.max()]
    result = pd.concat([old, tail]).sort_index()
    if result.index.has_duplicates or not np.isfinite(result.to_numpy()).all():
        raise ValueError("Invalid appended macro source")
    return result, {
        "overlap_observations": len(shared),
        "revised_overlap_observations": int((differences > 1e-10).sum()),
        "maximum_overlap_revision": float(differences.max()) if len(shared) else None,
        "revisions_applied": False,
        "appended_observations": len(tail),
    }
