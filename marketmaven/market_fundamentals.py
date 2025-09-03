"""
Module for fetching fundamental financial data for assets.

"""
import numpy as np
import pandas as pd
import yfinance as yf

def fetch_vix_term_structure(start="2010-01-01", end=None, freq="M"):
    """
    Fetch and compute VIX term structure features (VIX vs VIX3M) for use in
    asset return forecasting and portfolio allocation models.

    Args:
        start (str): Start date for data download (default "2010-01-01").
        end (str, optional): End date for data download. Defaults to today if None.
        freq (str): Resampling frequency, e.g. "M" for monthly, "W-FRI" for weekly.

    Returns:
        DataFrame with columns:
            - date: Resampled calendar date (month- or week-end).
            - vix: 1-month implied volatility index level (VIX).
            - vix3m: 3-month implied volatility index level (VIX3M).
            - vix_ts_level: Log difference between VIX3M and VIX
                • > 0 (contango): market expects future volatility to be higher than near-term → risk-on, carry/momentum strategies often work.
                • < 0 (backwardation): near-term volatility higher than long-term → risk-off regime, drawdowns more likely.
            - vix_ts_chg_1m: 1-period change in term structure slope
                • Captures regime shifts (e.g., transition from contango to backwardation).
                • Useful as a short-term predictor for heightened risk or opportunity.
            - vix_ts_z_12m: 12-period rolling z-score of the slope
                • Standardized measure of how extreme the slope is relative to its recent history.
                • Helps flag unusually steep or inverted curves as distinct volatility regimes.

    Notes:
        - The VIX term structure is widely used as a forward-looking "fear vs. complacency" indicator.
        - These features are especially relevant at **monthly or weekly horizons** for ETF or portfolio return forecasting.
        - Requires both ^VIX and ^VIX3M from Yahoo Finance; data may be missing before ~2010.
    """
    # ^VIX = 1M implied vol, ^VIX3M = 3M implied vol (Yahoo symbols)
    px = yf.download(["^VIX", "^VIX3M"], start=start, end=end, interval="1d",
                     auto_adjust=False, progress=False, group_by="ticker")
    # normalize across yfinance versions
    if isinstance(px.columns, pd.MultiIndex):
        vix   = px[("^VIX",   "Adj Close")].rename("vix")
        vix3m = px[("^VIX3M", "Adj Close")].rename("vix3m")
    else:
        vix   = px["Adj Close"].rename("vix")      # single ticker fallback
        vix3m = None                                # (but we need both)

    df = pd.concat([vix, vix3m], axis=1).dropna().sort_index()
    # monthly period-end
    df_m = df.resample(freq).last().dropna()
    # term structure slope (log difference); ratio is fine too
    df_m["vix_ts_level"] = np.log(df_m["vix3m"]) - np.log(df_m["vix"])
    df_m["vix_ts_chg_1m"] = df_m["vix_ts_level"].diff(1)

    # rolling z-score over 12 months
    m = df_m["vix_ts_level"].rolling(12, min_periods=12).mean()
    s = df_m["vix_ts_level"].rolling(12, min_periods=12).std()
    df_m["vix_ts_z_12m"] = (df_m["vix_ts_level"] - m) / s.replace(0, np.nan)

    return df_m.reset_index().rename(columns={"index":"date"})