"""
Module to fetch asset prices and compute future excess returns over risk-free rate.

"""
import yfinance as yf
import pandas as pd
import numpy as np


def fetch_prices(tickers, start, end=None):
    """
    Returns a tidy df with MultiIndex (date, ticker) and column 'Adj Close'.
    Adjusted Close includes dividends & splits => total-return compatible.
    """
    px = yf.download(
        tickers=tickers, start=start, end=end, interval="1d",
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

def fetch_rf_daily(start, end=None):
    """
    Fetch ^IRX (13-week T-bill), robust to MultiIndex columns and auto_adjust True/False.
    Returns a daily series of *continuous* daily rate ≈ (annual_fraction / 252).
    """
    rf = yf.download("^IRX", start=start, end=end, interval="1d", progress=False, auto_adjust=False)

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
                                          freq: str = "W-FRI"):
    """
    Calendar-aligned future excess return for one asset.
    y_excess = exp( log(P_{t1}) - log(P_t) - ∑_{(t,t1]} rf_daily_cont ) - 1
    Returns a Series indexed by the start-of-period date t.
    """
    # Resample to period-end closes
    px_period = adj_close.resample(freq).last().dropna()
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

def compute_excess_future_return_fixed(adj_close: pd.Series,
                                       rf_daily_cont: pd.Series,
                                       h_days: int = 5):
    """
    Fixed trading-day horizon (t -> t+h).
    y_excess = exp( log(P_{t+h}) - log(P_t) - sum_{k=0..h-1} rf_daily_cont[t+k] ) - 1
    """
    # Ensure we have RF on the same business-day index
    s = adj_close.dropna()
    idx = s.index
    rf_on_px = rf_daily_cont.reindex(idx).ffill()

    log_p = np.log(s)
    log_r_price = log_p.shift(-h_days * -1) - log_p  # log(P_{t+h}) - log(P_t); shift(-h) forward
    # Sum RF daily cont rates over next h days, aligned at t
    rf_log = rf_on_px.shift(-h_days + 1).rolling(h_days).sum()  # sum of future h days
    # Target
    y_excess = np.exp(log_r_price - rf_log) - 1.0
    y_excess = y_excess.iloc[:-h_days]  # drop last h because of lookahead
    y_excess.name = "y_excess_lead"
    return y_excess

def build_long_panel(tickers, start, end=None, horizon="weekly", fixed_h_days=5):
    prices = fetch_prices(tickers, start, end)
    rf_daily_cont = fetch_rf_daily(start, end)

    # Choose calendar frequency
    if horizon.lower() == "weekly":
        freq = "W-FRI"
        compute_fn = lambda s: compute_excess_future_return_calendar(s, rf_daily_cont, freq=freq)
    elif horizon.lower() == "monthly":
        freq = "BM" # Business Month-end
        compute_fn = lambda s: compute_excess_future_return_calendar(s, rf_daily_cont, freq=freq)
    elif horizon.lower() == "fixed":
        compute_fn = lambda s: compute_excess_future_return_fixed(s, rf_daily_cont, h_days=fixed_h_days)
    else:
        raise ValueError("horizon must be one of {'weekly','monthly','fixed'}")

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
