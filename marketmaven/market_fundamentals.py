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

    return df_m.reset_index().rename(columns={"index":"date", "Date":"date"})


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

    return pcs_df.reset_index().rename(columns={"DATE": "date"})

def fetch_high_yield_spread(start="2010-01-01", end=None, horizon: Horizon = Horizon.MONTHLY):
    """
    Fetch global high-yield credit spreads from FRED as a proxy for EM sovereign risk.

    Primary series:
        - BAMLH0A0HYM2 : ICE BofA US High Yield Index Option-Adjusted Spread

    Fallback series:
        - BAMLH0A1HYBB : BB-rated high yield spread
        - BAMLH0A3HYCE : CCC-rated high yield spread

    Returns
    -------
    DataFrame with columns:
        - date
        - hy_spread          (decimal, e.g. 0.045 → 450 bps)
        - hy_spread_chg_1m   (1-period diff)
        - hy_spread_z_12m    (12-period rolling z-score)

    Notes
    -----
    HY spreads are a strong proxy for risk sentiment in EM sovereign debt
    when EM-specific spread indices are unavailable.
    """
    import pandas as pd
    import pandas_datareader.data as pdr

    # (series, column_name)
    fred_series = [
        ("BAMLH0A0HYM2", "hy_spread"),  # Primary HY OAS
        ("BAMLH0A1HYBB", "hy_spread"),  # BB HY OAS
        ("BAMLH0A3HYCE", "hy_spread"),  # CCC HY OAS
    ]

    df = None
    for fred_code, colname in fred_series:
        try:
            raw = pdr.DataReader(fred_code, "fred", start, end)
            raw = raw.rename(columns={fred_code: colname})
            # Convert basis points → decimals
            raw[colname] = raw[colname] / 100.0
            df = raw
            break
        except Exception:
            continue

    if df is None:
        raise RuntimeError("Could not fetch any High-Yield spread series from FRED.")

    # Resample to requested horizon
    df = df.resample(horizon).last()

    # Compute derivative features
    df["hy_spread_chg_1m"] = df["hy_spread"].diff(1)

    # 12-period rolling z-score
    mean = df["hy_spread"].rolling(12, min_periods=12).mean()
    std = df["hy_spread"].rolling(12, min_periods=12).std()
    df["hy_spread_z_12m"] = (df["hy_spread"] - mean) / std

    # Clean final output
    df = df.dropna()
    df = df.reset_index().rename(columns={"index": "date", "DATE":"date"})
    print(df.columns)
    df["date"] = pd.to_datetime(df["date"])

    return df

def fetch_earnings_yield(start, end, horizon):
    """
    Earnings Yield = 1 / Shiller CAPE ratio (CAPE from FRED)
    """
    import pandas as pd
    import pandas_datareader.data as pdr

    try:
        cape = pdr.DataReader("CAPE", "fred", start, end)  # Shiller PE
        cape = cape.resample(horizon).last().dropna()
        cape["earnings_yield"] = 1.0 / cape["CAPE"]
        df = cape[["earnings_yield"]].reset_index().rename(columns={"index": "date", "DATE":"date", "Date":"date"})
        return df
    except Exception:
        return None


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
    #tbill_df = fetch_tbill_rate(start=start, end=end, horizon=horizon)
    dxy_df   = fetch_dxy(start=start, end=end, horizon=horizon)
    yc_df    = fetch_yield_curve_pcs(start=start, end=end, horizon=horizon, n_components=3)


    # Merge on date
    dfs = [vix_df, term_df, cred_df, dxy_df, yc_df] #tbill_df,

    # Ensure all have lowercase column names and 'date'
    for i, df in enumerate(dfs):
        df.columns = [str(c).lower() for c in df.columns]
        if "date" not in df.columns:
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            elif "DATE" in df.columns:
                df = df.rename(columns={"DATE": "date"})
            else:
                df = df.reset_index().rename(columns={df.index.name or "index": "date"})
        dfs[i] = df
    merged = dfs[0]
    for df in dfs[1:]:

        merged = pd.merge(merged, df, on="date", how="left")

    # Sort and forward-fill small gaps (e.g., missing DXY data)
    merged = merged.sort_values("date").ffill().fillna(0)

    return merged


import yfinance as yf
import pandas as pd
import numpy as np

def fetch_core_global_macro(start="2010-01-01", end=None, horizon="M"):
    """
    Fetch the MOST IMPORTANT macro predictors using only Yahoo sources.

    Includes:
        oil (CL=F)
        copper (HG=F)
        gold (GC=F)
        10y nominal (^TNX)
        TIPS ETF (SCHP)
        EM FX proxy (CEW)
    """

    def _yf_series(ticker, name):
        px = yf.download(ticker, start=start, end=end, progress=False, group_by="ticker")

        if px.empty:
            return None

        # ---- Handle MultiIndex columns ----
        if isinstance(px.columns, pd.MultiIndex):
            # try common patterns
            candidates = [
                (name, "Adj Close"),
                (name, "Close"),
                ("Adj Close", ticker),
                ("Close", ticker),
            ]
            series = None
            for a, b in candidates:
                if (a, b) in px.columns:
                    series = px[(a, b)].rename(name)
                    break
            # fallback: pick first column that is numeric
            if series is None:
                first = px.columns[0]
                series = px[first].rename(name)
        else:
            # ---- Single index ----
            if "Adj Close" in px.columns:
                series = px["Adj Close"].rename(name)
            else:
                series = px["Close"].rename(name)

        # Resample monthly (or weekly)
        series = series.resample(horizon).last().dropna()
        return series.to_frame().reset_index().rename(columns={"Date": "date"})

    # --- Fetch Yahoo Series ---
    oil_df    = _yf_series("CL=F", "oil")
    copper_df = _yf_series("HG=F", "copper")
    gold_df   = _yf_series("GC=F", "gold")

    tnx_df    = _yf_series("^TNX", "y10_nominal")
    if tnx_df is not None:
        tnx_df["y10_nominal"] = tnx_df["y10_nominal"] / 100.0

    schp_df = _yf_series("SCHP", "schp")
    if schp_df is not None:
        schp_df["schp_ret"] = np.log(schp_df["schp"] / schp_df["schp"].shift(1))

    cew_df = _yf_series("CEW", "em_fx")
    if cew_df is not None:
        cew_df["em_fx_ret"] = np.log(cew_df["em_fx"] / cew_df["em_fx"].shift(1))

    # --- Merge ---
    dfs = [oil_df, copper_df, gold_df, tnx_df, schp_df, cew_df]
    dfs = [d for d in dfs if d is not None]

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")

    merged = merged.sort_values("date").ffill().dropna()

    # --- Derived predictors ---
    merged["oil_ret"] = merged["oil"].pct_change()
    merged["copper_ret"] = merged["copper"].pct_change()
    merged["gold_crude_ratio"] = merged["gold"] / merged["oil"]

    merged["y10_real_proxy"] = merged["schp_ret"].rolling(3).mean()
    merged["breakeven_proxy"] = (
        merged["y10_nominal"].diff() - merged["y10_real_proxy"].diff()
    )

    merged["em_fx_ret"] = merged["em_fx_ret"].fillna(0)

    return merged.reset_index(drop=True)




def fetch_enhanced_macro_features(start="2010-01-01", end=None, horizon: Horizon = Horizon.MONTHLY):
    """
    Enhanced macro-finance feature set, following the SAME merging pattern
    as fetch_macro_features(). All feature sources are converted into
    DataFrames with a `date` column, then merged cleanly on `date`.

    Added features:
        - ERP via SPY (ETF-based, stable)
        - VVIX, SKEW, MOVE treasury volatility
        - Market breadth (RSP/SPY, % above 50dma, ADD proxy)
        - ACM term premium from FRED
    """

    import pandas as pd
    import numpy as np
    import yfinance as yf
    import pandas_datareader.data as pdr
    from marketmaven.asset_prices import fetch_etf_features

    # ------------------------------------------------------------
    # Helper to convert ANY Series into a 2-col DataFrame with date
    # ------------------------------------------------------------
    def _series_to_df(series, name):
        if isinstance(series, pd.Series):
            df = series.to_frame(name=name).reset_index()
            df.columns = ["date", name]
            return df
        raise ValueError(f"Expected Series for {name}")

    # ------------------------------------------------------------
    # 1. Existing stable macro components (your original fetchers)
    # ------------------------------------------------------------
    vix_df  = fetch_vix_term_structure(start=start, end=end, horizon=horizon)
    term_df = fetch_term_spread(start=start, end=end, horizon=horizon)
    cred_df = fetch_credit_spread(start=start, end=end, horizon=horizon)
    dxy_df  = fetch_dxy(start=start, end=end, horizon=horizon)
    yc_df   = fetch_yield_curve_pcs(start=start, end=end, horizon=horizon, n_components=3)
    high_y_spread = fetch_high_yield_spread(start=start, end=end, horizon=horizon)
    #earnings_yield = fetch_earnings_yield(start=start, end=end, horizon=horizon)

    # ------------------------------------------------------------
    # 2. ERP using your ETF fetcher (no Yahoo weirdness)
    # ------------------------------------------------------------
    spy_df = fetch_etf_features(["SPY"], start, end, Horizon.DAILY)
    spy = spy_df[["date", "price"]].copy()
    spy = spy.sort_values("date")
    spy["price"] = spy["price"].replace(0, np.nan)
    spy["price"] = spy["price"].ffill().bfill()
    spy["spy_ret"] = np.log(spy["price"] / spy["price"].shift(1))
    spy_m = spy[["date", "spy_ret"]].set_index("date").resample(horizon).last().dropna()
    spy_df2 = spy_m.reset_index()

    # -- T-bill using your own function (stabilized wrapper)
    tbill = term_df[["Date", "tbill3m"]].copy()
    tbill.columns = ["date", "tbill3m"]

    # -- ERP = SPY excess return over T-bill
    erp_df = pd.merge(spy_df2, tbill, on="date", how="left")
    erp_df["erp"] = erp_df["spy_ret"] - erp_df["tbill3m"]
    erp_df = erp_df[["date", "erp"]]

    # ------------------------------------------------------------
    # 3. Additional volatility / risk signals (wrapped safely)
    # ------------------------------------------------------------
    def _safe_yf(ticker, name):
        try:
            px = yf.download(ticker, start=start, end=end, progress=False)
            if px.empty:
                return None
            base = "Adj Close" if "Adj Close" in px.columns else "Close"
            ser = px[base].rename(name)
            ser = ser.resample(horizon).last().dropna()
            return _series_to_df(ser, name)
        except Exception:
            return None


    # MOVE proxy using TLT
    tlt = fetch_etf_features(["TLT"], start, end, Horizon.DAILY)
    df = tlt[["date","price"]].copy()
    df["tlt_ret"] = np.log(df["price"] / df["price"].shift(1))
    df["move_proxy"] = df["tlt_ret"].rolling(21).std() * np.sqrt(12)
    move_df = df[["date","move_proxy"]].dropna()

    # 3c. SKEW proxy (VIX3M / VIX)
    if all(col in vix_df.columns for col in ["vix", "vix3m"]):
        skew_df = vix_df[["date", "vix", "vix3m"]].copy()
        skew_df["skew_proxy"] = skew_df["vix3m"] / skew_df["vix"]
        skew_df = skew_df[["date", "skew_proxy"]]
    else:
        skew_df = None

    # 3d. VIX slope (already have both series)
    if all(col in vix_df.columns for col in ["vix", "vix3m"]):
        vix_slope_df = vix_df[["date", "vix", "vix3m"]].copy()
        vix_slope_df["vix_slope"] = np.log(vix_slope_df["vix3m"]) - np.log(vix_slope_df["vix"])
        vix_slope_df = vix_slope_df[["date", "vix_slope"]]
    else:
        vix_slope_df = None
    # ------------------------------------------------------------
    # 4. Breadth signals: RSP/SPY, % above 50dma, ADD
    # ------------------------------------------------------------
    try:
        rsp_px = fetch_etf_features(["RSP"], start, end, Horizon.DAILY)
        spy_px = fetch_etf_features(["SPY"], start, end, Horizon.DAILY)

        rsp_px = rsp_px[["date", "price"]].rename(columns={"price": "rsp"})
        spy_px = spy_px[["date", "price"]].rename(columns={"price": "spy"})

        merged_breadth = pd.merge(rsp_px, spy_px, on="date", how="inner")
        merged_breadth["rsp_spy"] = merged_breadth["rsp"] / merged_breadth["spy"]

        rsp_spy_df = merged_breadth[["date", "rsp_spy"]]
    except Exception as e:
        rsp_spy_df = None

    spx = fetch_etf_features(["SPY"], start, end, Horizon.DAILY)
    spx = spx[["date","price"]].copy()
    spx["ma50"] = spx["price"].rolling(50).mean()
    spx["pct_above_50dma"] = (spx["price"] / spx["ma50"]) * 100
    spx = spx[["date", "pct_above_50dma"]].dropna()
    pct50_df = spx

    # ------------------------------------------------------------
    # 5. ACM term premium (FRED)
    # ------------------------------------------------------------
    try:
        acm = pdr.DataReader("TE10", "fred", start, end)
        acm = acm.resample(horizon).last().rename(columns={"TE10": "term_premium"})
        acm_df = acm.reset_index().rename(columns={"index": "date"})
    except Exception:
        acm_df = None

    # ------------------------------------------------------------
    # 6. Merge EVERYTHING (same pattern as your original function)
    # ------------------------------------------------------------
    dfs = [
        vix_df, term_df, cred_df, dxy_df, yc_df,          # existing macro
        spy_df2, erp_df,                       # ERP block
        skew_df, move_df,    # vol signals
        vix_slope_df, rsp_spy_df, pct50_df,   # breadth
        acm_df, high_y_spread, #earnings_yield                                   
    ]

    # Filter out None
    dfs = [df for df in dfs if df is not None]

    # Standardize to lowercase + ensure date column
    cleaned = []
    for df in dfs:
        df.columns = [str(c).lower() for c in df.columns]
        if "date" not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or "index": "date"})
        cleaned.append(df)

    # Merge the same way your macro function does
    merged = cleaned[0]
    for df in cleaned[1:]:
        merged = pd.merge(merged, df, on="date", how="left")

    # Forward-fill and fill remaining NaNs
    merged = merged.sort_values("date").ffill().fillna(0)

    return merged


