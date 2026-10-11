"""Compare Schwab and Yahoo daily market data for VTV.

This experiment performs read-only retrieval from Schwab Market Data and
Yahoo Finance, aligns daily observations, and measures raw OHLCV agreement,
calendar coverage, corporate-action adjustments, and monthly return drift.
Returns are decimal units (``0.02`` means 2%).

Usage:
    uv run --locked python experiments/2026-10-schwab-market-data-audit/compare_vtv.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from bayesfolio.application import make_default_schwab_account_access

SCHWAB_MARKET_DATA_BASE_URL = "https://api.schwabapi.com/marketdata/v1"


def main() -> int:
    """Run the comparison and write aligned daily, monthly, and summary outputs."""

    parser = argparse.ArgumentParser(description="Compare Schwab and Yahoo market data for one ETF")
    parser.add_argument("--symbol", default="VTV")
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/2026-10-schwab-market-data-audit/outputs"),
    )
    args = parser.parse_args()

    symbol = args.symbol.upper().strip()
    if not symbol:
        raise ValueError("symbol must not be empty")
    if args.years not in {1, 2, 3, 5, 10, 15, 20}:
        raise ValueError("years must be one of 1, 2, 3, 5, 10, 15, or 20")

    schwab = _fetch_schwab_history(symbol=symbol, years=args.years)
    yahoo = _fetch_yahoo_history(
        symbol=symbol,
        start=schwab.index.min(),
        end=schwab.index.max(),
    )
    aligned = schwab.join(yahoo, how="outer")
    aligned.index.name = "date"
    monthly = _build_monthly_comparison(aligned)
    summary = _build_summary(symbol=symbol, aligned=aligned, monthly=monthly)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = args.output_dir / f"{symbol.lower()}_schwab_yahoo_daily.csv"
    monthly_path = args.output_dir / f"{symbol.lower()}_schwab_yahoo_monthly.csv"
    summary_path = args.output_dir / f"{symbol.lower()}_schwab_yahoo_summary.json"
    aligned.reset_index().to_csv(daily_path, index=False)
    monthly.reset_index().to_csv(monthly_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"daily comparison: {daily_path}")
    print(f"monthly comparison: {monthly_path}")
    print(f"summary: {summary_path}")
    return 0


def _fetch_schwab_history(*, symbol: str, years: int) -> pd.DataFrame:
    access = make_default_schwab_account_access()
    response = requests.get(
        f"{SCHWAB_MARKET_DATA_BASE_URL}/pricehistory",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {access.oauth.get_access_token()}",
        },
        params={
            "symbol": symbol,
            "periodType": "year",
            "period": str(years),
            "frequencyType": "daily",
            "frequency": "1",
            "needExtendedHoursData": "false",
            "needPreviousClose": "true",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    candles = payload.get("candles")
    if not isinstance(candles, list) or not candles:
        raise ValueError("Schwab returned no price-history candles")
    frame = pd.DataFrame(candles)
    required = {"datetime", "open", "high", "low", "close", "volume"}
    if required - set(frame.columns):
        raise ValueError("Schwab price-history candles are missing required fields")
    eastern_dates = pd.to_datetime(frame.pop("datetime"), unit="ms", utc=True).dt.tz_convert("America/New_York")
    frame.index = pd.DatetimeIndex(eastern_dates.dt.date)
    return frame.rename(columns={column: f"schwab_{column}" for column in frame.columns}).sort_index()


def _fetch_yahoo_history(*, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    yahoo_end = pd.Timestamp(end) + pd.Timedelta(days=1)
    frame = yf.Ticker(symbol).history(
        start=pd.Timestamp(start).date().isoformat(),
        end=yahoo_end.date().isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=False,
    )
    if frame.empty:
        raise ValueError("Yahoo returned no price history")
    frame.index = pd.DatetimeIndex(frame.index.date)
    columns = {
        "Open": "yahoo_open",
        "High": "yahoo_high",
        "Low": "yahoo_low",
        "Close": "yahoo_close",
        "Adj Close": "yahoo_adj_close",
        "Volume": "yahoo_volume",
        "Dividends": "yahoo_dividends",
        "Stock Splits": "yahoo_stock_splits",
        "Capital Gains": "yahoo_capital_gains",
    }
    return frame.rename(columns=columns).reindex(columns=list(columns.values())).sort_index()


def _build_monthly_comparison(aligned: pd.DataFrame) -> pd.DataFrame:
    common = aligned.dropna(subset=["schwab_close", "yahoo_close", "yahoo_adj_close"])
    month_end = common[["schwab_close", "yahoo_close", "yahoo_adj_close"]].resample("ME").last()
    month_end["schwab_raw_return"] = month_end["schwab_close"].pct_change()
    month_end["yahoo_raw_return"] = month_end["yahoo_close"].pct_change()
    month_end["yahoo_total_return"] = month_end["yahoo_adj_close"].pct_change()
    month_end["schwab_minus_yahoo_raw"] = month_end["schwab_raw_return"] - month_end["yahoo_raw_return"]
    month_end["schwab_raw_minus_yahoo_total"] = month_end["schwab_raw_return"] - month_end["yahoo_total_return"]
    return month_end


def _build_summary(*, symbol: str, aligned: pd.DataFrame, monthly: pd.DataFrame) -> dict[str, Any]:
    common = aligned.dropna(subset=["schwab_close", "yahoo_close"])
    schwab_dates = aligned["schwab_close"].notna()
    yahoo_dates = aligned["yahoo_close"].notna()
    raw_ohlc_agreement = {
        field: _difference_summary((common[f"schwab_{field}"] - common[f"yahoo_{field}"]).abs())
        for field in ("open", "high", "low", "close")
    }
    close_difference = (common["schwab_close"] - common["yahoo_close"]).abs()
    volume_common = aligned.dropna(subset=["schwab_volume", "yahoo_volume"])
    volume_denominator = volume_common[["schwab_volume", "yahoo_volume"]].max(axis=1).replace(0, np.nan)
    volume_relative_difference = (
        volume_common["schwab_volume"] - volume_common["yahoo_volume"]
    ).abs() / volume_denominator
    total_return_drift = monthly["schwab_raw_minus_yahoo_total"].dropna().abs()
    raw_return_drift = monthly["schwab_minus_yahoo_raw"].dropna().abs()
    dividends = aligned["yahoo_dividends"].fillna(0)
    splits = aligned["yahoo_stock_splits"].fillna(0)
    schwab_only_index = aligned.index[schwab_dates & ~yahoo_dates]
    yahoo_only_index = aligned.index[~schwab_dates & yahoo_dates]

    raw_ohlc_agreement["close"]["days_over_1_cent"] = int((close_difference > 0.01).sum())
    raw_ohlc_agreement["close"]["days_over_5_cents"] = int((close_difference > 0.05).sum())
    volume_summary = _difference_summary(volume_relative_difference)
    volume_summary["days_over_1_percent"] = int((volume_relative_difference > 0.01).sum())
    volume_summary["days_over_10_percent"] = int((volume_relative_difference > 0.10).sum())

    return {
        "symbol": symbol,
        "coverage": {
            "start": common.index.min().date().isoformat(),
            "end": common.index.max().date().isoformat(),
            "schwab_rows": int(schwab_dates.sum()),
            "yahoo_rows": int(yahoo_dates.sum()),
            "common_rows": int((schwab_dates & yahoo_dates).sum()),
            "schwab_only_dates": int((schwab_dates & ~yahoo_dates).sum()),
            "yahoo_only_dates": int((~schwab_dates & yahoo_dates).sum()),
            "schwab_only_date_values": [date.date().isoformat() for date in schwab_only_index],
            "schwab_only_weekend_dates": [date.date().isoformat() for date in schwab_only_index if date.weekday() >= 5],
            "yahoo_only_date_values": [date.date().isoformat() for date in yahoo_only_index],
        },
        "raw_ohlc_agreement_usd": raw_ohlc_agreement,
        "volume_relative_difference": volume_summary,
        "corporate_actions": {
            "dividend_event_count": int((dividends > 0).sum()),
            "dividends_per_share_total_usd": _finite_float(dividends.sum()),
            "split_event_count": int((splits > 0).sum()),
        },
        "monthly_return_drift_decimal": {
            "raw_schwab_vs_raw_yahoo": _difference_summary(raw_return_drift),
            "raw_schwab_vs_adjusted_yahoo": _difference_summary(total_return_drift),
            "months_over_10_basis_points_due_to_adjustment": int((total_return_drift > 0.001).sum()),
        },
        "endpoint_return_comparison_decimal": _endpoint_return_summary(common),
    }


def _difference_summary(values: pd.Series) -> dict[str, float | int | str | None]:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None, "max_date": None}
    max_date = pd.Timestamp(finite.idxmax()).date().isoformat()
    return {
        "count": int(finite.size),
        "mean": _finite_float(finite.mean()),
        "median": _finite_float(finite.median()),
        "p95": _finite_float(finite.quantile(0.95)),
        "max": _finite_float(finite.max()),
        "max_date": max_date,
    }


def _endpoint_return_summary(common: pd.DataFrame) -> dict[str, float]:
    first = common.iloc[0]
    last = common.iloc[-1]
    elapsed_years = (common.index[-1] - common.index[0]).days / 365.2425
    schwab_growth = last["schwab_close"] / first["schwab_close"]
    yahoo_raw_growth = last["yahoo_close"] / first["yahoo_close"]
    yahoo_adjusted_growth = last["yahoo_adj_close"] / first["yahoo_adj_close"]
    return {
        "elapsed_years": _finite_float(elapsed_years),
        "schwab_raw_price_return": _finite_float(schwab_growth - 1),
        "schwab_raw_price_cagr": _finite_float(schwab_growth ** (1 / elapsed_years) - 1),
        "yahoo_raw_price_return": _finite_float(yahoo_raw_growth - 1),
        "yahoo_raw_price_cagr": _finite_float(yahoo_raw_growth ** (1 / elapsed_years) - 1),
        "yahoo_adjusted_total_return": _finite_float(yahoo_adjusted_growth - 1),
        "yahoo_adjusted_total_return_cagr": _finite_float(yahoo_adjusted_growth ** (1 / elapsed_years) - 1),
    }


def _finite_float(value: object) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("comparison metric must be finite")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
