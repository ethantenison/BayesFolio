"""Shared cache-frame operations for IO providers.

This internal module centralizes date/index normalization, coverage checks,
and frame merge/deduplication used by IO provider adapters. It contains
pure dataframe utilities and no business logic.
"""

from __future__ import annotations

import pandas as pd


def concat_frames(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two frames while preserving empty-frame semantics.

    Args:
        left: Left dataframe.
        right: Right dataframe.

    Returns:
        Concatenated dataframe copy.
    """

    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()
    return pd.concat([left, right], ignore_index=True)


def normalize_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a normalized datetime ``date`` column when present.

    Args:
        frame: Input dataframe.

    Returns:
        Frame copy with ``date`` converted via ``pd.to_datetime`` when present.
    """

    output = frame.copy()
    if "date" in output.columns:
        output["date"] = pd.to_datetime(output["date"])
    return output


def normalize_asset_id_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with upper-cased ``asset_id`` values when present.

    Args:
        frame: Input dataframe.

    Returns:
        Frame copy with ``asset_id`` converted to upper-case strings.
    """

    output = frame.copy()
    if "asset_id" in output.columns:
        output["asset_id"] = output["asset_id"].astype(str).str.upper()
    return output


def slice_requested(
    *,
    frame: pd.DataFrame,
    start: str,
    end: str,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Slice a frame by date range and optional ticker list.

    Args:
        frame: Input dataframe.
        start: Inclusive ISO start date.
        end: Inclusive ISO end date.
        tickers: Optional list of normalized upper-case asset tickers.

    Returns:
        Filtered dataframe copy.
    """

    if frame.empty:
        return frame.copy()

    output = normalize_date_column(frame)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    date_mask = (output["date"] >= start_ts) & (output["date"] <= end_ts)

    if tickers is None:
        return output[date_mask].copy().reset_index(drop=True)

    output = normalize_asset_id_column(output)
    ticker_mask = output["asset_id"].isin(tickers)
    return output[ticker_mask & date_mask].copy().reset_index(drop=True)


def missing_tickers(
    *,
    cache_frame: pd.DataFrame,
    tickers: list[str],
    start: str,
    end: str,
    freq: str,
) -> list[str]:
    """Find tickers missing full date coverage in a cached long panel.

    Args:
        cache_frame: Cached long-format frame with ``date`` and ``asset_id``.
        tickers: Normalized upper-case tickers requested.
        start: Inclusive ISO start date.
        end: Inclusive ISO end date.
        freq: Pandas frequency string for expected coverage dates.

    Returns:
        Tickers that are missing at least one expected date.
    """

    if cache_frame.empty:
        return tickers

    frame = normalize_asset_id_column(normalize_date_column(cache_frame))
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    expected_dates = pd.DatetimeIndex(pd.date_range(start=start_ts, end=end_ts, freq=freq))

    missing: list[str] = []
    for ticker in tickers:
        per_ticker = frame.loc[
            (frame["asset_id"] == ticker) & (frame["date"] >= start_ts) & (frame["date"] <= end_ts),
            "date",
        ]
        if per_ticker.empty:
            missing.append(ticker)
            continue

        if len(expected_dates) == 0:
            continue
        available = pd.DatetimeIndex(per_ticker.drop_duplicates().sort_values())
        if not expected_dates.isin(available).all():
            missing.append(ticker)

    return missing


def has_date_coverage(*, frame: pd.DataFrame, start: str, end: str, freq: str) -> bool:
    """Whether a frame has complete expected ``date`` coverage.

    Args:
        frame: Input frame with a ``date`` column.
        start: Inclusive ISO start date.
        end: Inclusive ISO end date.
        freq: Pandas frequency string for expected coverage dates.

    Returns:
        ``True`` when all expected dates are present in ``frame``.
    """

    if frame.empty or "date" not in frame.columns:
        return False

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    expected_dates = pd.DatetimeIndex(pd.date_range(start=start_ts, end=end_ts, freq=freq))
    if len(expected_dates) == 0:
        return True

    available = pd.DatetimeIndex(pd.to_datetime(frame["date"]).drop_duplicates().sort_values())
    return bool(expected_dates.isin(available).all())


def dedupe_rows(frame: pd.DataFrame, *, subset: list[str], sort_by: list[str]) -> pd.DataFrame:
    """Drop duplicate keys and return a stable, sorted frame copy.

    Args:
        frame: Input dataframe.
        subset: Column names used for deduplication keys.
        sort_by: Column names used for output ordering.

    Returns:
        Deduplicated and sorted frame copy.
    """

    if frame.empty:
        return frame

    output = normalize_date_column(frame)
    output = normalize_asset_id_column(output)
    output = output.drop_duplicates(subset=subset, keep="last")
    return output.sort_values(sort_by).reset_index(drop=True)
