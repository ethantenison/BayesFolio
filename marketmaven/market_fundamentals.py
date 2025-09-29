"""
Module for fetching fundamental financial data for assets.

"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
import pandas_datareader.data as pdr
from marketmaven.configs import Interval, Horizon

def fetch_vix_term_structure(start="2010-01-01", end=None, horizon: Horizon = Horizon.MONTHLY):
    """
    Fetch and compute VIX term structure features (VIX vs VIX3M) for use in
    asset return forecasting and portfolio allocation models.

    Args:
        start (str): Start date for data download (default "2010-01-01").
        end (str, optional): End date for data download. Defaults to today if None.
        horizon (str): Resampling frequency, e.g. "M" for monthly, "W-FRI" for weekly.

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
    px = yf.download(["^VIX", "^VIX3M"], start=start, end=end, interval=Interval.DAILY,
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
    df_m = df.resample(horizon).last().dropna()
    # term structure slope (log difference); ratio is fine too
    df_m["vix_ts_level"] = np.log(df_m["vix3m"]) - np.log(df_m["vix"])
    df_m["vix_ts_chg_1m"] = df_m["vix_ts_level"].diff(1)

    # rolling z-score over 12 months
    m = df_m["vix_ts_level"].rolling(12, min_periods=12).mean()
    s = df_m["vix_ts_level"].rolling(12, min_periods=12).std()
    df_m["vix_ts_z_12m"] = (df_m["vix_ts_level"] - m) / s.replace(0, np.nan)

    return df_m.reset_index().rename(columns={"index":"date"})


def fetch_term_spread(start="2010-01-01", end=None, horizon: Horizon = Horizon.MONTHLY):
    """
    Fetch term spread = 10Y Treasury - 3M T-bill.
    Uses Yahoo Finance ^TNX (10Y) and ^IRX (3M).
    Falls back to FRED DTB3 for 3M if ^IRX not available.
    """
    tickers = ["^TNX", "^IRX"]
    px = yf.download(tickers, start=start, end=end, interval=Interval.DAILY,
                     auto_adjust=False, progress=False, group_by="ticker")

    # Normalize MultiIndex or single-index
    def _safe_col(df, ticker):
        if isinstance(df.columns, pd.MultiIndex):
            try:
                return df[(ticker, "Adj Close")]
            except KeyError:
                return None
        else:
            return df.get("Adj Close", None)

    tnote10y = _safe_col(px, "^TNX")
    tbill3m  = _safe_col(px, "^IRX")

    # Fallback for missing ^IRX (3M T-bill)
    if tbill3m is None or tbill3m.isna().all():
        print("⚠️ ^IRX not found — falling back to FRED DTB3 (3M T-bill).")
        tbill3m = yf.download("DTB3", start=start, end=end, interval=Interval.DAILY)["Adj Close"]

    df = pd.concat([tnote10y, tbill3m], axis=1)
    df.columns = ["tnote10y", "tbill3m"]

    # Convert Yahoo % yields to decimals
    df["tnote10y"] = df["tnote10y"] / 100.0
    df["tbill3m"]  = df["tbill3m"]  / 100.0

    # Resample
    df_m = df.resample(horizon).last().dropna()
    df_m["term_spread"] = df_m["tnote10y"] - df_m["tbill3m"]

    return df_m.reset_index().rename(columns={"index":"date"})


def fetch_credit_spread(start="2010-01-01", end=None, horizon: Horizon = Horizon.MONTHLY):
    """
    Yahoo-based credit spread proxy using HYG (high yield) vs LQD (investment grade).

    Returns
    -------
    DataFrame with:
      - date
      - credit_spread         = log(HYG) - log(LQD)  (level proxy)
      - credit_spread_chg_1p  = 1-period change of the above (same horizon as `horizon`)
    """
    import numpy as np
    import pandas as pd
    import yfinance as yf

    px = yf.download(["HYG", "LQD"], start=start, end=end, interval=Interval.DAILY,
                     auto_adjust=False, progress=False, group_by="ticker")

    # Normalize across yfinance versions
    def _get_col(df, ticker):
        if isinstance(df.columns, pd.MultiIndex):
            if (ticker, "Adj Close") in df.columns:
                return df[(ticker, "Adj Close")].rename(ticker)
            if (ticker, "Close") in df.columns:
                return df[(ticker, "Close")].rename(ticker)
            return None
        else:
            # Single-ticker case (unlikely here), but keep for completeness
            base = "Adj Close" if "Adj Close" in df.columns else "Close"
            s = df[base].rename(ticker) if base in df.columns else None
            return s

    hyg = _get_col(px, "HYG")
    lqd = _get_col(px, "LQD")
    if hyg is None or lqd is None:
        raise RuntimeError("Could not find HYG/LQD prices from yfinance response.")

    df = pd.concat([hyg, lqd], axis=1).dropna().sort_index()
    df.index = pd.to_datetime(df.index)

    # Resample to requested calendar (e.g., BM or W-FRI) using period-end closes
    df_m = df.resample(horizon).last().dropna()

    # Level proxy and its 1-period change
    spread_lvl = np.log(df_m["HYG"]) - np.log(df_m["LQD"])
    out = pd.DataFrame(index=df_m.index)
    out["credit_spread"] = spread_lvl
    out["credit_spread_chg_1p"] = out["credit_spread"].diff(1)

    return out.reset_index().rename(columns={"index": "date"})


def fetch_tbill_rate(start="2010-01-01", end=None, horizon: Horizon = Horizon.MONTHLY):
    """
    Fetch 3-month Treasury bill yield (^IRX from Yahoo).
    Falls back to FRED's DTB3 if ^IRX is unavailable.

    Returns
    -------
    DataFrame with columns:
      - date
      - tbill3m (decimal yield)
    """
    import yfinance as yf
    import pandas as pd

    px = yf.download("^IRX", start=start, end=end, interval=Interval.DAILY, progress=False)

    if px.empty:
        print("⚠️ ^IRX not available from Yahoo, falling back to FRED DTB3.")
        fred = yf.download("DTB3", start=start, end=end, interval=Interval.DAILY, progress=False)
        base = "Adj Close" if "Adj Close" in fred.columns else "Close"
        tbill3m = fred[base] / 100.0
    else:
        base = "Adj Close" if "Adj Close" in px.columns else "Close"
        tbill3m = px[base] / 100.0

    # Ensure DataFrame with correct column name
    df = tbill3m.resample(horizon).last().dropna()
    return df.reset_index().rename(columns={"index": "date", "^IRX" : "tbill3m"})

def fetch_dxy(start="2010-01-01", end=None, horizon: Horizon = Horizon.MONTHLY):
    """
    Fetch U.S. Dollar Index (DXY) proxy.
    Falls back to UUP ETF if direct DXY tickers are unavailable.
    """
    tickers = ["DX-Y.NYB", "DX=F", "UUP"]
    for tk in tickers:
        px = yf.download(tk, start=start, end=end, interval=Interval.DAILY, progress=False)
        if px.empty:
            continue

        base = "Adj Close" if "Adj Close" in px.columns else "Close"
        s = px[base]

        if isinstance(s, pd.Series):
            dxy_df = s.to_frame(name="dxy")
        else:
            dxy_df = s.rename(columns={s.columns[0]: "dxy"})

        dxy_df = dxy_df.resample(horizon).last().dropna()
        return dxy_df.reset_index().rename(columns={"index": "date"})

    raise ValueError("No DXY data available (DX-Y.NYB, DX=F, UUP all failed).")


def fetch_yield_curve_pcs(start="2010-01-01", end=None, horizon: Horizon = Horizon.MONTHLY, n_components=3):
    """
    Fetch Treasury yields across maturities, extract PCA components.
    Uses FRED if Yahoo fails.
    """
    tickers_yahoo = {"^IRX": "3m", "^FVX": "5y", "^TNX": "10y", "^TYX": "30y"}
    tickers_fred  = {"DTB3": "3m", "DGS5": "5y", "DGS10": "10y", "DGS30": "30y"}

    # Try Yahoo first
    px = yf.download(list(tickers_yahoo.keys()), start=start, end=end, interval=Interval.DAILY, progress=False)
    cols = []

    for tk, label in tickers_yahoo.items():
        try:
            if isinstance(px.columns, pd.MultiIndex):
                s = px[(tk, "Adj Close")] if (tk, "Adj Close") in px.columns else px[(tk, "Close")]
            else:
                s = px["Adj Close"] if "Adj Close" in px.columns else px["Close"]
            cols.append(s.rename(label) / 100.0)
        except Exception:
            continue

    if not cols:
        # fallback: fetch from FRED
        fred_data = {label: pdr.DataReader(tk, "fred", start, end) for tk, label in tickers_fred.items()}
        df = pd.concat(fred_data.values(), axis=1)
        df.columns = tickers_fred.values()
    else:
        df = pd.concat(cols, axis=1)

    df = df.dropna().resample(horizon).last()

    # PCA
    pca = PCA(n_components=min(n_components, df.shape[1]))
    pcs = pca.fit_transform(df.values)
    pcs_df = pd.DataFrame(pcs, index=df.index, columns=[f"yc_pc{i+1}" for i in range(pcs.shape[1])])

    return pcs_df.reset_index().rename(columns={"DATE": "Date"})

def fetch_macro_features(start="2010-01-01", end=None, horizon: Horizon = Horizon.MONTHLY):
    """
    Fetch and merge a standard set of macro-finance features for ETF return prediction.
    
    Features included:
        - VIX term structure (level, 1m change, 12m z-score)
        - Term spread (10Y – 3M Treasury yield)
        - Credit spread (BAA – AAA corporate yields)
        - 3M T-bill yield
        - Dollar Index (DXY)
        - Yield curve principal components (PC1 = level, PC2 = slope, PC3 = curvature)

    Args:
        start (str): Start date for data.
        end (str): End date. Defaults to today if None.
        horizon (str): Resample frequency ("M" = monthly, "W-FRI" = weekly).

    Returns:
        pd.DataFrame with all features aligned by date.
    """
    # Individual feature fetchers
    vix_df   = fetch_vix_term_structure(start=start, end=end, horizon=horizon)
    term_df  = fetch_term_spread(start=start, end=end, horizon=horizon)
    cred_df  = fetch_credit_spread(start=start, end=end, horizon=horizon)
    tbill_df = fetch_tbill_rate(start=start, end=end, horizon=horizon)
    dxy_df   = fetch_dxy(start=start, end=end, horizon=horizon)
    yc_df    = fetch_yield_curve_pcs(start=start, end=end, horizon=horizon, n_components=3)


    # Sort and forward-fill small gaps (e.g., missing DXY data)
        # Merge on date
    dfs = [vix_df, term_df, cred_df, tbill_df, dxy_df, yc_df]
    merged = dfs[0]
    for df in dfs[1:]:
        print(df.head(2))
        merged = pd.merge(merged, df, on="Date", how="left")

    # Sort and forward-fill small gaps (e.g., missing DXY data)
    merged = merged.sort_values("Date").ffill().fillna(0)

    return merged