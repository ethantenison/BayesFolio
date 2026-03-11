"""
Module for fetching fundamental financial data for assets.

"""

import io
import os
import sys
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf
from pandas_datareader.fred import FredReader
from sklearn.decomposition import PCA

from bayesfolio.core.settings import Horizon, Interval

# Optional: only needed if you set use_tradingeconomics=True
try:
    import tradingeconomics as te  # pyright: ignore[reportMissingImports]
except Exception:
    te = None

import requests

FRED_TIMEOUT_SECONDS = int(float(os.getenv("BAYESFOLIO_FRED_TIMEOUT_SECONDS", "30")))
FRED_RETRY_COUNT = int(os.getenv("BAYESFOLIO_FRED_RETRY_COUNT", "1"))
FRED_RETRY_PAUSE_SECONDS = float(os.getenv("BAYESFOLIO_FRED_RETRY_PAUSE_SECONDS", "0.1"))


def _read_fred(symbols: str | list[str], start: str, end: str | None) -> pd.DataFrame:
    """Read FRED series with configurable timeout/retry behavior."""

    reader = FredReader(
        symbols=symbols,
        start=start,
        end=end,
        retry_count=FRED_RETRY_COUNT,
        pause=FRED_RETRY_PAUSE_SECONDS,
        timeout=FRED_TIMEOUT_SECONDS,
    )
    data = reader.read()
    return cast(pd.DataFrame, data)


def _download_frame(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Download market data and return a DataFrame for type-safe processing."""

    kwargs.setdefault("auto_adjust", False)
    data = yf.download(*args, **kwargs)
    if data is None:
        return pd.DataFrame()
    return cast(pd.DataFrame, data)


def fetch_global_yields(
    start: str = "2010-01-01",
    end: str | None = None,
    horizon: str = "M",  # accepts your Horizon enum (we read .value if present)
    countries: list[str] | None = None,
    transform: Literal["level", "diff_1p", "z12"] = "level",
    fred_codes: dict[str, list[str]] | None = None,
    stooq_map: dict[str, str] | None = None,
    use_tradingeconomics: bool = False,
    te_country_map: dict[str, tuple[str, str]] | None = None,  # (country, category)
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Fetch 10Y sovereign yields (decimals) for DE/JP/UK/CN with robust fallbacks.

    Pipeline per country (first success wins):
      1) FRED OECD long-term yields (monthly)  -> DE/JP/UK
      2) Stooq CSV (https://stooq.com then https://stooq.pl) -> CN & others
      3) pandas_datareader('stooq') -> best effort
      4) TradingEconomics (optional; requires package/API access)

    Returns
    -------
    DataFrame with columns: ['date', <keys...>], values in decimals (0.025 = 2.5%).
    If a country fails across all sources, the column is kept with NaNs (so your
    downstream merge logic remains stable).
    """
    # --- normalize horizon (supports your Horizon enum) ---
    horizon = getattr(horizon, "value", horizon)

    # --- defaults ---
    default_fred = {
        "de10y": ["IRLTLT01DEM156N"],  # Germany
        "jp10y": ["IRLTLT01JPM156N"],  # Japan
        "uk10y": ["IRLTLT01GBM156N"],  # UK
        "cn10y": [],  # not on FRED OECD 10Y
    }
    default_stooq = {
        "de10y": "10YDEY.B",
        "jp10y": "10YJPY.B",
        "uk10y": "10YUKY.B",
        "cn10y": "10YCNY.B",
    }
    # TradingEconomics mapping (country name must match TE’s catalog)
    default_te = {
        "de10y": ("Germany", "government bond 10y"),
        "jp10y": ("Japan", "government bond 10y"),
        "uk10y": ("United Kingdom", "government bond 10y"),
        "cn10y": ("China", "government bond 10y"),
    }

    code_map = {**default_fred, **(fred_codes or {})}
    stq_map = {**default_stooq, **(stooq_map or {})}
    te_map = {**default_te, **(te_country_map or {})}
    keys = countries or ["de10y", "jp10y", "uk10y", "cn10y"]

    # --- http session with UA (Stooq sometimes dislikes default UA) ---
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; MarketMavenBot/1.0; +https://example.com)"})

    def _log(msg):
        if verbose:
            print(msg, file=sys.stderr)

    # ---------- Source helpers ----------
    def _fred_one(symbols: list[str]):
        for sym in symbols:
            try:
                s = _read_fred(sym, start=start, end=end)[sym]
                s = s / 100.0
                s.index = pd.to_datetime(s.index)
                _log(f"FRED ok for {sym}")
                return s
            except Exception as e:
                _log(f"FRED fail {sym}: {e}")
        return None

    def _stooq_csv_one(ticker: str | None):
        if not ticker:
            return None
        # Try .com first, then .pl
        for base in ("https://stooq.com", "https://stooq.pl"):
            url = f"{base}/q/d/l/?s={ticker.lower()}&i=d"
            try:
                r = session.get(url, timeout=15)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                if df.empty or "Date" not in df.columns:
                    _log(f"Stooq CSV empty or schema mismatch for {ticker} via {base}")
                    continue
                # Some variants provide only Date/Close; some have OHLCV
                close_col = "Close" if "Close" in df.columns else df.columns[-1]
                df["Date"] = pd.to_datetime(df["Date"])
                s = df.set_index("Date")[close_col].astype(float)
                # Percent → decimal (heuristic if someone returns already-decimals)
                if s.dropna().abs().median() > 5:
                    s = s / 100.0
                s.index = pd.to_datetime(s.index)
                _log(f"Stooq CSV ok for {ticker} via {base}")
                return s
            except Exception as e:
                _log(f"Stooq CSV fail {ticker} via {base}: {e}")
        return None

    def _stooq_pdr_one(ticker: str | None):
        if not ticker:
            return None
        # pandas_datareader 'stooq' sometimes works for bonds; try anyway
        try:
            df = pdr.DataReader(ticker, "stooq", start, end)
            if df is None or df.empty:
                return None
            df = df.sort_index()
            # pick 'Close' if present, otherwise first numeric column
            close_col = "Close" if "Close" in df.columns else df.select_dtypes("number").columns[0]
            s = df[close_col].astype(float)
            if s.dropna().abs().median() > 5:
                s = s / 100.0
            _log(f"pandas_datareader('stooq') ok for {ticker}")
            return s
        except Exception as e:
            _log(f"pandas_datareader('stooq') fail {ticker}: {e}")
            return None

    def _te_one(country: str, category: str):
        if not use_tradingeconomics or te is None:
            return None
        try:
            # TE returns a list of dicts; we filter category
            te.login()  # relies on env vars if set; silently uses guest in some setups
            data = te.getMarketsData(country=country, category=category)
            if not data:
                return None
            df = pd.DataFrame(data)
            # Expect 'Date' and 'Close' or 'Value'
            date_col = "Date" if "Date" in df.columns else "date"
            val_col = "Close" if "Close" in df.columns else ("Value" if "Value" in df.columns else None)
            if val_col is None or date_col not in df.columns:
                return None
            ser = pd.Series(df[val_col].values, index=pd.to_datetime(df[date_col].values), dtype=float)
            # TE values are typically in percent; convert if needed
            if ser.dropna().abs().median() > 5:
                ser = ser / 100.0
            _log(f"TradingEconomics ok for {country} / {category}")
            return ser
        except Exception as e:
            _log(f"TradingEconomics fail {country} / {category}: {e}")
            return None

    # ---------- Build all series ----------
    series = []
    present_cols = []
    for k in keys:
        s = _fred_one(code_map.get(k, []))
        if s is None:
            # Stooq CSV (com then pl)
            s = _stooq_csv_one(stq_map.get(k))
        if s is None:
            # pandas_datareader('stooq')
            s = _stooq_pdr_one(stq_map.get(k))
        if s is None:
            # TradingEconomics (optional)
            country, category = te_map.get(k, (None, None))
            if country and category:
                s = _te_one(country, category)

        if s is not None:
            s = s.sort_index()
            s = s.resample(horizon).last().rename(k)
            series.append(s)
            present_cols.append(k)
            _log(f"{k}: SUCCESS")
        else:
            # keep the column with NaNs so the shape is stable
            idx = pd.date_range(pd.to_datetime(start), pd.to_datetime(end) if end else pd.Timestamp.today(), freq="M")
            series.append(pd.Series(index=idx, dtype="float64", name=k))
            _log(f"{k}: FAILED across all sources")

    # ---------- Align, name 'date', and transform ----------
    df = pd.concat(series, axis=1)
    df = df[sorted(df.columns, key=lambda c: keys.index(c))]  # preserve key order
    df = df.dropna(how="all")
    df = df.sort_index()

    # Make sure we output an explicit 'date' column (works across pandas versions)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "date", "DATE": "date", "index": "date"})

    # Optional transforms
    if transform == "diff_1p":
        for k in keys:
            if k in df.columns:
                df[k] = df[k].diff(1)
        df = df.dropna()
    elif transform == "z12":
        for k in keys:
            if k in df.columns:
                m = df[k].rolling(12, min_periods=12).mean()
                s = df[k].rolling(12, min_periods=12).std()
                df[k] = (df[k] - m) / s
        df = df.dropna()

    cols = ["date"] + [k for k in keys if k in df.columns]
    return df[cols]


def fetch_dealer_gamma_proxy(start="2010-01-01", end=None, horizon=Horizon.MONTHLY):
    """
    Dealer gamma proxy = realized volatility - implied volatility (VIX).
    Negative → dealers long gamma (trend-friendly)
    Positive → dealers short gamma (choppy / crash-prone)
    """

    def _safe_adj_close(px, ticker=None):
        """
        Robustly extract Adjusted Close or Close from yfinance output,
        handling both single-index and MultiIndex cases.
        """
        if px.empty:
            return None

        if isinstance(px.columns, pd.MultiIndex):
            # Try common patterns
            candidates = []
            if ticker is not None:
                candidates.extend(
                    [
                        (ticker, "Adj Close"),
                        (ticker, "Close"),
                    ]
                )
            candidates.extend(
                [
                    ("Adj Close", ticker),
                    ("Close", ticker),
                ]
            )

            for c in candidates:
                if c in px.columns:
                    return px[c]

            # Fallback: pick first numeric column
            return px.select_dtypes("number").iloc[:, 0]

        else:
            if "Adj Close" in px.columns:
                return px["Adj Close"]
            if "Close" in px.columns:
                return px["Close"]

            # Fallback
            return px.select_dtypes("number").iloc[:, 0]

    # --- Download data ---
    vix_px = _download_frame("^VIX", start=start, end=end, progress=False, group_by="ticker")
    spy_px = _download_frame("SPY", start=start, end=end, progress=False, group_by="ticker")

    vix = _safe_adj_close(vix_px, "^VIX")
    spy = _safe_adj_close(spy_px, "SPY")

    if vix is None or spy is None:
        raise RuntimeError("Could not extract VIX or SPY prices for dealer gamma proxy.")

    vix = vix.astype(float) / 100.0
    spy = spy.astype(float)

    # --- Realized volatility (21-day) ---
    log_spy = pd.Series(np.log(spy.to_numpy(dtype=float)), index=spy.index)
    rv = log_spy.diff().rolling(21).std() * np.sqrt(252)

    df = pd.DataFrame(
        {
            "rv": rv,
            "vix": vix,
        }
    ).dropna()

    df["dealer_gamma_proxy"] = df["rv"] - df["vix"]

    # --- Resample to forecast horizon ---
    df_m = (
        df["dealer_gamma_proxy"]
        .resample(getattr(horizon, "value", horizon))
        .last()
        .dropna()
        .to_frame()
        .reset_index()
        .rename(columns={"index": "date"})
    )

    return df_m


def fetch_put_call_ratio(start, end=None, horizon=Horizon.MONTHLY):
    px = _download_frame("^PPC", start=start, end=end, progress=False)

    if px.empty:
        return None

    base = "Adj Close" if "Adj Close" in px.columns else "Close"
    s = px[base].rename("put_call_ratio")

    s = s.resample(horizon).last().dropna()
    return s.to_frame().reset_index()


def fetch_spy_flow_proxy(start="2010-01-01", end=None, horizon=Horizon.MONTHLY):
    """
    Proxy for SPY passive flows using signed dollar volume,
    standardized as a 12m z-score.
    """

    px = _download_frame(
        "SPY",
        start=start,
        end=end,
        interval=Interval.DAILY,
        progress=False,
        group_by="ticker",
    )

    # ------------------------------------------------------------
    # Robust extraction of price and volume (Series, not DataFrames)
    # ------------------------------------------------------------
    price, volume = None, None

    if isinstance(px.columns, pd.MultiIndex):
        # Try (ticker, field)
        if ("SPY", "Adj Close") in px.columns:
            price = px[("SPY", "Adj Close")]
        elif ("SPY", "Close") in px.columns:
            price = px[("SPY", "Close")]

        if ("SPY", "Volume") in px.columns:
            volume = px[("SPY", "Volume")]

        # Fallback: search by field name
        if price is None:
            for c in px.columns:
                if c[1] in ("Adj Close", "Close"):
                    price = px[c]
                    break
        if volume is None:
            for c in px.columns:
                if c[1] == "Volume":
                    volume = px[c]
                    break
    else:
        base = "Adj Close" if "Adj Close" in px.columns else "Close"
        price = px[base]
        volume = px["Volume"]

    if price is None or volume is None:
        raise RuntimeError("Could not extract SPY price/volume for flow proxy.")

    price = price.astype(float)
    volume = volume.astype(float)

    # ------------------------------------------------------------
    # Build flow proxy
    # ------------------------------------------------------------
    df = pd.DataFrame(
        {"price": price, "volume": volume},
        index=price.index,
    ).dropna()

    # Signed dollar volume
    df["flow_proxy"] = np.sign(df["price"].diff()) * df["price"] * df["volume"]

    # ------------------------------------------------------------
    # Monthly aggregation + z-score
    # ------------------------------------------------------------
    df_m = df["flow_proxy"].resample(getattr(horizon, "value", horizon)).sum().to_frame("flow_proxy")

    m = df_m["flow_proxy"].rolling(12, min_periods=12).mean()
    s = df_m["flow_proxy"].rolling(12, min_periods=12).std()

    df_m["spy_flow_z_12m"] = (df_m["flow_proxy"] - m) / s

    return df_m[["spy_flow_z_12m"]].dropna().reset_index().rename(columns={"index": "date"})


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
                                • > 0 (contango): market expects future volatility to be
                                    higher than near-term → risk-on, carry/momentum
                                    strategies often work.
                                • < 0 (backwardation): near-term volatility higher than
                                    long-term → risk-off regime, drawdowns more likely.
            - vix_ts_chg_1m: 1-period change in term structure slope
                • Captures regime shifts (e.g., transition from contango to backwardation).
                • Useful as a short-term predictor for heightened risk or opportunity.
            - vix_ts_z_12m: 12-period rolling z-score of the slope
                • Standardized measure of how extreme the slope is relative to its recent history.
                                • Helps flag unusually steep or inverted curves as
                                    distinct volatility regimes.

    Notes:
        - The VIX term structure is widely used as a forward-looking "fear vs. complacency" indicator.
                - These features are especially relevant at **monthly or weekly
                    horizons** for ETF or portfolio return forecasting.
        - Requires both ^VIX and ^VIX3M from Yahoo Finance; data may be missing before ~2010.
    """
    # ^VIX = 1M implied vol, ^VIX3M = 3M implied vol (Yahoo symbols)
    px = _download_frame(
        ["^VIX", "^VIX3M"],
        start=start,
        end=end,
        interval=Interval.DAILY,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )
    # normalize across yfinance versions
    if isinstance(px.columns, pd.MultiIndex):
        vix = px[("^VIX", "Adj Close")].rename("vix")
        vix3m = px[("^VIX3M", "Adj Close")].rename("vix3m")
    else:
        vix = px["Adj Close"].rename("vix")  # single ticker fallback
        vix3m = None  # (but we need both)

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

    return df_m.reset_index().rename(columns={"index": "date", "Date": "date"})


def fetch_term_spread(start="2010-01-01", end=None, horizon: Horizon = Horizon.MONTHLY):
    """
    Fetch term spread = 10Y Treasury - 3M T-bill.
    Uses Yahoo Finance ^TNX (10Y) and ^IRX (3M).
    Falls back to FRED DTB3 for 3M if ^IRX not available.
    """
    tickers = ["^TNX", "^IRX"]
    px = _download_frame(
        tickers, start=start, end=end, interval=Interval.DAILY, auto_adjust=False, progress=False, group_by="ticker"
    )

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
    tbill3m = _safe_col(px, "^IRX")

    # Fallback for missing ^IRX (3M T-bill)
    if tbill3m is None or tbill3m.isna().all():
        print("⚠️ ^IRX not found — falling back to FRED DTB3 (3M T-bill).")
        dtb3 = _download_frame("DTB3", start=start, end=end, interval=Interval.DAILY)
        base = "Adj Close" if "Adj Close" in dtb3.columns else "Close"
        tbill3m = dtb3[base]

    df = pd.concat([tnote10y, tbill3m], axis=1)
    df.columns = ["tnote10y", "tbill3m"]

    # Convert Yahoo % yields to decimals
    df["tnote10y"] = df["tnote10y"] / 100.0
    df["tbill3m"] = df["tbill3m"] / 100.0

    # Resample
    df_m = df.resample(horizon).last().dropna()
    df_m["term_spread"] = df_m["tnote10y"] - df_m["tbill3m"]

    return df_m.reset_index().rename(columns={"index": "date"})


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

    px = _download_frame(
        ["HYG", "LQD"],
        start=start,
        end=end,
        interval=Interval.DAILY,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

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
    px = _download_frame("^IRX", start=start, end=end, interval=Interval.DAILY, progress=False)

    if px.empty:
        print("⚠️ ^IRX not available from Yahoo, falling back to FRED DTB3.")
        fred = _download_frame("DTB3", start=start, end=end, interval=Interval.DAILY, progress=False)
        base = "Adj Close" if "Adj Close" in fred.columns else "Close"
        tbill3m = fred[base] / 100.0
    else:
        base = "Adj Close" if "Adj Close" in px.columns else "Close"
        tbill3m = px[base] / 100.0

    # Ensure DataFrame with correct column name
    df = tbill3m.resample(horizon).last().dropna()
    return df.reset_index().rename(columns={"index": "date", "^IRX": "tbill3m"})


def fetch_dxy(start="2010-01-01", end=None, horizon: Horizon = Horizon.MONTHLY):
    """
    Fetch U.S. Dollar Index (DXY) proxy.
    Falls back to UUP ETF if direct DXY tickers are unavailable.
    """
    tickers = ["DX-Y.NYB", "DX=F", "UUP"]
    for tk in tickers:
        px = _download_frame(tk, start=start, end=end, interval=Interval.DAILY, progress=False)
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
    tickers_fred = {"DTB3": "3m", "DGS5": "5y", "DGS10": "10y", "DGS30": "30y"}

    # Try Yahoo first
    px = _download_frame(list(tickers_yahoo.keys()), start=start, end=end, interval=Interval.DAILY, progress=False)
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

    if cols:
        df = pd.concat(cols, axis=1)
    else:
        # Fallback: fetch from FRED symbol-by-symbol so one timeout does not
        # fail the entire yield-curve feature block.
        fred_cols: list[pd.Series] = []
        for tk, label in tickers_fred.items():
            try:
                series_df = _read_fred(tk, start=start, end=end)
                if tk not in series_df.columns:
                    continue
                series = series_df[tk].rename(label)
                fred_cols.append(series)
            except Exception:
                continue

        if not fred_cols:
            msg = "No yield-curve series available from Yahoo or FRED."
            raise RuntimeError(msg)

        df = pd.concat(fred_cols, axis=1)

    df = df.dropna().resample(horizon).last()

    if df.empty or df.shape[1] == 0:
        msg = "Yield-curve series are unavailable after alignment/resampling."
        raise RuntimeError(msg)

    # PCA
    pca = PCA(n_components=min(n_components, df.shape[1]))
    pcs = pca.fit_transform(df.values)
    pcs_df = pd.DataFrame(pcs, index=df.index, columns=[f"yc_pc{i + 1}" for i in range(pcs.shape[1])])

    return pcs_df.reset_index().rename(columns={"index": "date", "DATE": "date"})


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
    # (series, column_name)
    fred_series = [
        ("BAMLH0A0HYM2", "hy_spread"),  # Primary HY OAS
        ("BAMLH0A1HYBB", "hy_spread"),  # BB HY OAS
        ("BAMLH0A3HYCE", "hy_spread"),  # CCC HY OAS
    ]

    df = None
    for fred_code, colname in fred_series:
        try:
            raw = _read_fred(fred_code, start=start, end=end)
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
    df = df.reset_index().rename(columns={"index": "date", "DATE": "date"})
    print(df.columns)
    df["date"] = pd.to_datetime(df["date"])

    return df


def fetch_earnings_yield(start, end, horizon):
    """
    Earnings Yield = 1 / Shiller CAPE ratio (CAPE from FRED)
    """

    try:
        cape = _read_fred("CAPE", start=start, end=end)  # Shiller PE
        cape = cape.resample(horizon).last().dropna()
        cape["earnings_yield"] = 1.0 / cape["CAPE"]
        df = cape[["earnings_yield"]].reset_index().rename(columns={"index": "date", "DATE": "date", "Date": "date"})
        return df
    except Exception:
        return None


def fetch_cpi_inflation(start="2010-01-01", end=None, horizon: Horizon = Horizon.MONTHLY):

    cpi = _read_fred("CPIAUCSL", start=start, end=end)  # headline CPI level
    cpi = cpi.resample(horizon).last().dropna()  # monthly

    cpi["cpi_yoy"] = cpi["CPIAUCSL"].pct_change(12)
    cpi["cpi_mom"] = cpi["CPIAUCSL"].pct_change(1)

    return cpi.reset_index().rename(columns={"index": "date"})


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
    vix_df = fetch_vix_term_structure(start=start, end=end, horizon=horizon)
    term_df = fetch_term_spread(start=start, end=end, horizon=horizon)
    cred_df = fetch_credit_spread(start=start, end=end, horizon=horizon)
    # tbill_df = fetch_tbill_rate(start=start, end=end, horizon=horizon)
    dxy_df = fetch_dxy(start=start, end=end, horizon=horizon)
    try:
        yc_df = fetch_yield_curve_pcs(start=start, end=end, horizon=horizon, n_components=3)
    except Exception:
        yc_df = pd.DataFrame(columns=["date", "yc_pc1", "yc_pc2", "yc_pc3"])

    # Merge on date
    dfs = [vix_df, term_df, cred_df, dxy_df, yc_df]  # tbill_df,

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

    # Sort and forward-fill small gaps (e.g., missing DXY data).
    # Fill residual gaps on value columns only to avoid mixed-dtype downcasting.
    merged = merged.sort_values("date")
    value_columns = [column for column in merged.columns if column != "date"]
    merged[value_columns] = merged[value_columns].apply(pd.to_numeric, errors="coerce")
    merged[value_columns] = merged[value_columns].ffill().fillna(0.0)

    return merged


def fetch_core_global_macro(start="2010-01-01", end=None, horizon: Horizon = Horizon.MONTHLY):
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
        px = _download_frame(ticker, start=start, end=end, progress=False, group_by="ticker")

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
    oil_df = _yf_series("CL=F", "oil")
    copper_df = _yf_series("HG=F", "copper")
    gold_df = _yf_series("GC=F", "gold")

    tnx_df = _yf_series("^TNX", "y10_nominal")
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
    # The log shift caused predictions to collapse....
    # merged["oil_ret"] = np.log(merged["oil"] / merged["oil"].shift(1))
    # merged["copper_ret"] = np.log(merged["copper"] / merged["copper"].shift(1))
    merged["gold_crude_ratio"] = merged["gold"] / merged["oil"]

    merged["y10_real_proxy"] = merged["schp_ret"].rolling(3).mean()
    merged["breakeven_proxy"] = merged["y10_nominal"].diff() - merged["y10_real_proxy"].diff()

    # merged["em_fx_ret"] = merged["em_fx_ret"].fillna(0)
    merged = merged.sort_values("date").ffill().fillna(0)

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

    import numpy as np
    import pandas as pd

    from bayesfolio.engine.features.asset_prices import fetch_etf_features

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
    vix_df = fetch_vix_term_structure(start=start, end=end, horizon=horizon)
    term_df = fetch_term_spread(start=start, end=end, horizon=horizon)
    cred_df = fetch_credit_spread(start=start, end=end, horizon=horizon)
    dxy_df = fetch_dxy(start=start, end=end, horizon=horizon)
    yc_df = fetch_yield_curve_pcs(start=start, end=end, horizon=horizon, n_components=3)
    high_y_spread = fetch_high_yield_spread(start=start, end=end, horizon=horizon)
    global_macro = fetch_core_global_macro(start=start, end=end, horizon=horizon)
    cpi_df = fetch_cpi_inflation(start=start, end=end, horizon=horizon)
    global_yields = fetch_global_yields(start=start, end=end, horizon=horizon, transform="diff_1p")
    dealer_gamma = fetch_dealer_gamma_proxy(start=start, end=end, horizon=horizon)

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
            px = _download_frame(ticker, start=start, end=end, progress=False)
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
    df = tlt[["date", "price"]].copy()
    df["tlt_ret"] = np.log(df["price"] / df["price"].shift(1))
    df["move_proxy"] = df["tlt_ret"].rolling(21).std() * np.sqrt(12)
    move_df = df[["date", "move_proxy"]].dropna()

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

        breadth_m = merged_breadth.set_index("date").resample(horizon).last().dropna()

        breadth_m["rsp_spy_roc_1m"] = breadth_m["rsp_spy"].pct_change(1)

        rsp_spy_df = breadth_m[["rsp_spy", "rsp_spy_roc_1m"]].reset_index()

    except Exception:
        rsp_spy_df = None

    spx = fetch_etf_features(["SPY"], start, end, Horizon.DAILY)
    spx = spx[["date", "price"]].copy()
    spx["ma50"] = spx["price"].rolling(50).mean()
    spx["pct_above_50dma"] = (spx["price"] / spx["ma50"]) * 100
    spx = spx[["date", "pct_above_50dma"]].dropna()
    pct50_df = spx

    spy_flow = fetch_spy_flow_proxy(start, end, horizon)

    # ------------------------------------------------------------
    # 5. ACM term premium (FRED)
    # ------------------------------------------------------------
    try:
        acm = _read_fred("TE10", start=start, end=end)
        acm = acm.resample(horizon).last().rename(columns={"TE10": "term_premium"})
        acm_df = acm.reset_index().rename(columns={"index": "date"})
    except Exception:
        acm_df = None

    # ------------------------------------------------------------
    # 6. Merge EVERYTHING (same pattern as your original function)
    # ------------------------------------------------------------
    dfs = [
        vix_df,
        term_df,
        cred_df,
        dxy_df,
        yc_df,  # existing macro
        spy_df2,
        erp_df,  # ERP block
        skew_df,
        move_df,  # vol signals
        vix_slope_df,
        rsp_spy_df,
        spy_flow,
        dealer_gamma,
        pct50_df,  # breadth
        acm_df,
        high_y_spread,
        global_macro,
        cpi_df,
        global_yields,  # earnings_yield
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


def macro_predictive_screening(
    df: pd.DataFrame,
    macro_cols: list[str],
    target_col: str = "y_excess_lead",
    min_periods: int = 60,
    window: int = 12,
):
    """
    Time-series macro screening using rolling predictive IC.

    Macro variables are evaluated on:
    - absolute mean IC (strength, not sign)
    - IC IR (stability)
    - hit rate (optional diagnostic)
    """

    results = []

    for col in macro_cols:
        tmp = df[["date", "asset_id", col, target_col]].dropna()
        if tmp["date"].nunique() < min_periods:
            continue

        # Collapse cross-section → market-level return
        ts = tmp.groupby("date").agg({col: "first", target_col: "mean"}).dropna()

        if len(ts) < min_periods:
            continue

        rolling_ic = ts[col].rolling(window).corr(ts[target_col]).dropna()

        if len(rolling_ic) < min_periods // 2:
            continue

        results.append(
            {
                "feature": col,
                "mean_ic": rolling_ic.mean(),
                "abs_mean_ic": rolling_ic.abs().mean(),
                "ic_std": rolling_ic.std(ddof=1),
                "ic_ir": rolling_ic.mean() / (rolling_ic.std(ddof=1) + 1e-12),
                "hit_rate": (rolling_ic > 0).mean(),
                "n_periods": len(rolling_ic),
            }
        )

    return pd.DataFrame(results).sort_values("abs_mean_ic", ascending=False).reset_index(drop=True)
