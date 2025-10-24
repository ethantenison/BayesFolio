"""
Module to fetch asset prices and compute future excess returns over risk-free rate.

"""
import yfinance as yf
import pandas as pd
import numpy as np
from marketmaven.configs import Interval, Horizon

def fetch_prices(tickers, start, end=None, interval: Interval = Interval.DAILY):
    """
    Returns a tidy df with MultiIndex (date, ticker) and column 'Adj Close'.
    Adjusted Close includes dividends & splits => total-return compatible.
    """
    px = yf.download(
        tickers=tickers, start=start, end=end, interval=interval,
        group_by="ticker", auto_adjust=False, progress=False
    )
    # Normalize shape across yfinance versions
    if isinstance(px.columns, pd.MultiIndex):
        # Multi-index columns: level 0 = ticker, level 1 = field
        adj = []
        for tk in tickers:
            if (tk, 'Adj Close') in px.columns:
                s = px[(tk, 'Adj Close')].rename(tk)
                adj.append(s)
        adj = pd.concat(adj, axis=1)
    else:
        # Single-index columns: use 'Adj Close' directly (single ticker)
        adj = px[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
    adj = adj.dropna(how="all")
    adj.index = pd.to_datetime(adj.index)
    return adj

def fetch_rf_daily(start, end=None, interval: Interval = Interval.DAILY):
    """
    Fetch ^IRX (13-week T-bill), robust to MultiIndex columns and auto_adjust True/False.
    Returns a daily series of *continuous* daily rate ≈ (annual_fraction / 252).
    """
    rf = yf.download("^IRX", start=start, end=end, interval=interval, progress=False, auto_adjust=False)

    if rf.empty:
        raise RuntimeError("Could not download ^IRX. Try expanding the date range.")

    # Handle both single-index and MultiIndex column shapes
    if isinstance(rf.columns, pd.MultiIndex):
        col = None
        candidates = [
            ("Adj Close", "^IRX"), ("Close", "^IRX"),
            ("^IRX", "Adj Close"), ("^IRX", "Close")
        ]
        for c in candidates:
            if c in rf.columns:
                col = rf.loc[:, c].astype(float)
                break
        if col is None:
            # Fallback: slice by level name, then pick the first column
            if "Adj Close" in rf.columns.get_level_values(0):
                col = rf.xs("Adj Close", level=0, axis=1).iloc[:, 0].astype(float)
            else:
                col = rf.xs("Close", level=0, axis=1).iloc[:, 0].astype(float)
    else:
        if "Adj Close" in rf.columns:
            col = rf["Adj Close"].astype(float)
        elif "Close" in rf.columns:
            col = rf["Close"].astype(float)
        else:
            raise KeyError(f"^IRX: Neither 'Adj Close' nor 'Close' in columns: {rf.columns.tolist()}")

    # Convert annualized percent -> fraction
    ann_frac = col / 100.0
    rf_daily_cont = ann_frac / 252.0
    rf_daily_cont.index = pd.to_datetime(rf_daily_cont.index)

    # Fill to business-day grid and forward-fill small gaps
    all_bd = pd.date_range(rf_daily_cont.index.min(), rf_daily_cont.index.max(), freq="B")
    rf_daily_cont = rf_daily_cont.reindex(all_bd).ffill()
    rf_daily_cont.name = "rf_daily_cont"
    return rf_daily_cont

def compute_excess_future_return_calendar(adj_close: pd.Series,
                                          rf_daily_cont: pd.Series,
                                          horizon: Horizon= Horizon.MONTHLY):
    """
    Calendar-aligned future excess return for one asset.
    y_excess = exp( log(P_{t1}) - log(P_t) - ∑_{(t,t1]} rf_daily_cont ) - 1
    Returns a Series indexed by the start-of-period date t.
    """
    # Resample to period-end closes
    px_period = adj_close.resample(horizon).last().dropna()
    period_end = px_period.index
    px_fwd = px_period.shift(-1)

    # Price log-return for t -> t1
    log_r_price = (np.log(px_fwd) - np.log(px_period)).iloc[:-1]

    # Integrate RF over (t, t1] by summing continuous daily rates
    rf_log = []
    for t, t1 in zip(period_end[:-1], period_end[1:]):
        days = rf_daily_cont.loc[(rf_daily_cont.index > t) & (rf_daily_cont.index <= t1)]
        rf_log.append(days.sum() if not days.empty else 0.0)
    rf_log = pd.Series(rf_log, index=log_r_price.index)

    # Use numpy arrays to avoid dtype gotchas, then rebuild Series
    delta = (log_r_price.values - rf_log.values)
    y_excess_vals = np.exp(delta) - 1.0
    y_excess = pd.Series(y_excess_vals, index=log_r_price.index, name="y_excess_lead")
    return y_excess


def build_long_panel(tickers, start, end=None, horizon: Horizon = Horizon.MONTHLY):
    prices = fetch_prices(tickers, start, end)
    rf_daily_cont = fetch_rf_daily(start, end)

    # Choose calendar frequency
    def compute_fn(s):
        return compute_excess_future_return_calendar(s, rf_daily_cont, horizon=horizon)

    rows = []
    for tk in prices.columns:
        try:
            y_excess = compute_fn(prices[tk].dropna())
            df_tk = y_excess.to_frame()
            df_tk["asset_id"] = tk
            df_tk = df_tk.rename_axis("date").reset_index()[["date", "asset_id", "y_excess_lead"]]
            rows.append(df_tk)
        except Exception as e:
            print(f"[WARN] Skipping {tk}: {e}")

    panel = pd.concat(rows, axis=0).sort_values(["date", "asset_id"]).reset_index(drop=True)
    return panel


def fetch_etf_features(
    tickers: list[str],
    start: str,
    end: str = None,
    horizon: Horizon = Horizon.MONTHLY,
):
    """
    Fetch ETF-level features (liquidity, momentum, volatility, etc.) for one or more tickers.

    Parameters
    ----------
    tickers : list[str]
        List of ETF tickers (e.g., ["SPY", "QQQ", "ESGD"])
    start : str
        Start date for fetching data.
    end : str, optional
        End date (default: today).
    horizon : Horizon
        Resampling frequency (e.g., Horizon.MONTHLY or Horizon.WEEKLY).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ['Date', 'asset_id', 'price', 'volume', 'log_ret', 'mom1m', 'mom6m', 'mom12m',
         'mom36m', 'chmom', 'dolvol', 'turnover', 'sd_turn', 'ill', 'vol_1m']
        Missing values are filled with 0 (neutral imputation) as in the Ensemble GP paper.
    """
    
    # --- Resample (monthly, weekly, etc.) with feature-specific rules ---
    agg_map = {
        "price": "last",
        "log_ret": "sum",
        "mom1m": "last",
        "mom6m": "last",
        "mom12m": "last",
        "mom36m": "last",
        "chmom": "last",
        "volume": "sum",
        "dolvol": "sum",
        "turnover": "mean",
        "sd_turn": "mean",
        "ill": "mean",
        "vol_1m": "mean",
        "baspread": "mean",
    }

    if isinstance(tickers, str):
        tickers = [tickers]

    # Download all tickers together for efficiency
    df_raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=Interval.DAILY,
        progress=False,
        group_by="ticker",
    )
    print(f"\n>>> Columns: {df_raw.columns.tolist()[:8]} ...")  # Debug check

    def _extract_single_ticker_features(df, ticker):
        """Extract and compute features for one ticker."""
        px, vol = None, None
        if isinstance(df.columns, pd.MultiIndex):
            # Try both (field, ticker) and (ticker, field) orders
            if ("Adj Close", ticker) in df.columns:
                px = df[("Adj Close", ticker)]
            elif ("Close", ticker) in df.columns:
                px = df[("Close", ticker)]
            elif (ticker, "Adj Close") in df.columns:
                px = df[(ticker, "Adj Close")]
            elif (ticker, "Close") in df.columns:
                px = df[(ticker, "Close")]

            if ("Volume", ticker) in df.columns:
                vol = df[("Volume", ticker)]
            elif (ticker, "Volume") in df.columns:
                vol = df[(ticker, "Volume")]
        else:
            # Single ticker fallback
            base = "Adj Close" if "Adj Close" in df.columns else "Close"
            px = df[base]
            vol = df["Volume"]

        if px is None or vol is None:
            raise ValueError(f"Could not extract price/volume for {ticker}")

        # Compute features
        data = pd.concat({"price": px, "volume": vol}, axis=1).dropna()
        data.index = pd.to_datetime(data.index)
        data["log_ret"] = np.log(data["price"]).diff()

        # Momentum (price trends)
        data["mom1m"] = data["price"].pct_change(21)
        data["mom6m"] = data["price"].pct_change(126)
        data["mom12m"] = data["price"].pct_change(252)
        data["mom36m"] = data["price"].pct_change(756)
        data["chmom"] = data["mom12m"] - data["mom6m"]

        # Liquidity metrics
        data["dolvol"] = data["price"] * data["volume"]
        data["turnover"] = data["volume"] / data["volume"].rolling(21).mean() - 1
        data["sd_turn"] = data["turnover"].rolling(63).std()
        data["ill"] = data["log_ret"].abs() / data["dolvol"]

        # Rolling volatility
        data["vol_1m"] = data["log_ret"].rolling(21).std()

        # --- Approximate Bid-Ask Spread (Corwin-Schultz 2012) ---
        if ('High', ticker) in df.columns and ('Low', ticker) in df.columns:
            high = df[('High', ticker)]
            low = df[('Low', ticker)]
        elif (ticker, 'High') in df.columns and (ticker, 'Low') in df.columns:
            high = df[(ticker, 'High')]
            low = df[(ticker, 'Low')]
        else:
            high = low = None

        if high is not None and low is not None:
            log_hl = np.log(high / low)
            beta = (log_hl.rolling(2).sum() ** 2).rolling(21).mean()
            alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
            data["baspread"] = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
            data["baspread"] = data["baspread"].fillna(method="bfill").fillna(0)
        else:
            data["baspread"] = 0.0  # Default if High/Low not available

        # Resample (monthly, weekly, etc.)
        data = data.resample(horizon).agg(agg_map)

        # Fill missing with 0 (neutral imputation, paper-style)
        data = data.fillna(0)

        # Metadata
        data = data.reset_index().rename(columns={data.index.name or "index": "date"})
        data.columns = [c.lower() for c in data.columns]  # normalize column names
        data["asset_id"] = ticker
        return data

    # Compute for all tickers
    results = []
    for tk in tickers:
        try:
            res = _extract_single_ticker_features(df_raw, tk)
            results.append(res)
        except Exception as e:
            print(f"[WARN] Skipping {tk}: {e}")

    # Combine into one long DataFrame
    df_all = (
        pd.concat(results, axis=0)
        .sort_values(["date", "asset_id"])
        .reset_index(drop=True)
    )

    return df_all

