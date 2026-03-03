"""
Module to fetch asset prices and compute future excess returns over risk-free rate.

"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

from bayesfolio.core.settings import Horizon, Interval


def cross_sectional_zscore(
    df: pd.DataFrame,
    cols: list[str],
    date_col: str = "date",
    eps: float = 1e-8,
):
    """
    Cross-sectional z-score per date.

    Applied only to ETF-level predictors.
    """
    df = df.copy()

    for c in cols:
        df[c] = df.groupby(date_col)[c].transform(lambda x: (x - x.mean()) / (x.std() + eps)).fillna(0.0)

    return df


def cross_sectional_ic_screening(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "y_excess_lead",
    date_col: str = "date",
    asset_col: str = "asset_id",
    min_assets: int = 5,
    min_periods: int = 24,
):
    """
    Stage 1: Cross-sectional predictive screening using Rank IC.

    For each feature:
        - Compute monthly Spearman Rank IC vs next-period returns
        - Aggregate IC statistics across time

    Parameters
    ----------
    df : pd.DataFrame
        Long panel with [date, asset_id, features..., target]
    feature_cols : list[str]
        Candidate predictors to screen
    target_col : str
        Forward excess return (e.g. y_excess_lead)
    date_col : str
        Date column
    asset_col : str
        Asset identifier
    min_assets : int
        Minimum assets per month to compute IC
    min_periods : int
        Minimum months required for a feature to be evaluated

    Returns
    -------
    pd.DataFrame
        Feature screening table with IC statistics

    Example:

    screen_df = cross_sectional_ic_screening(
        df=df,
        feature_cols=etf_cols + macro_cols,
        target_col="y_excess_lead",
    )
    screen_df

    selected = screen_df[
    (screen_df["abs_mean_ic"] > 0.01) &     # weak but real signal
    (screen_df["ic_ir"] > 0.01) &            # persistence
    (screen_df["hit_rate"] > 0.45)          # directional consistency
    ]["feature"].tolist()
    selected
    """

    results = []

    grouped = df[[date_col, asset_col, target_col] + feature_cols].dropna(subset=[target_col]).groupby(date_col)

    for feat in feature_cols:
        ic_series = []

        for date, g in grouped:
            g = g[[feat, target_col]].dropna()

            if len(g) < min_assets:
                continue

            ic, _ = spearmanr(g[feat], g[target_col])
            if np.isfinite(ic):
                ic_series.append(ic)

        if len(ic_series) < min_periods:
            continue

        ic_series = np.array(ic_series)

        results.append(
            {
                "feature": feat,
                "mean_ic": ic_series.mean(),
                "abs_mean_ic": np.abs(ic_series).mean(),
                "ic_std": ic_series.std(ddof=1),
                "ic_ir": ic_series.mean() / (ic_series.std(ddof=1) + 1e-8),
                "hit_rate": (ic_series > 0).mean(),
                "n_periods": len(ic_series),
            }
        )

    res = pd.DataFrame(results)

    if res.empty:
        return res

    return res.sort_values(by=["abs_mean_ic", "ic_ir"], ascending=False).reset_index(drop=True)


def fetch_prices(tickers, start, end=None, interval: Interval = Interval.DAILY):
    """
    Returns a tidy df with MultiIndex (date, ticker) and column 'Adj Close'.
    Adjusted Close includes dividends & splits => total-return compatible.
    """
    px = yf.download(
        tickers=tickers, start=start, end=end, interval=interval, group_by="ticker", auto_adjust=False, progress=False
    )
    # Normalize shape across yfinance versions
    if isinstance(px.columns, pd.MultiIndex):
        # Multi-index columns: level 0 = ticker, level 1 = field
        adj = []
        for tk in tickers:
            if (tk, "Adj Close") in px.columns:
                s = px[(tk, "Adj Close")].rename(tk)
                adj.append(s)
        adj = pd.concat(adj, axis=1)
    else:
        # Single-index columns: use 'Adj Close' directly (single ticker)
        adj = px[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
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
        candidates = [("Adj Close", "^IRX"), ("Close", "^IRX"), ("^IRX", "Adj Close"), ("^IRX", "Close")]
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


def compute_excess_future_return_calendar(
    adj_close: pd.Series, rf_daily_cont: pd.Series, horizon: Horizon = Horizon.MONTHLY
):
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
    delta = log_r_price.values - rf_log.values
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


def compute_max_drawdown(log_price: pd.Series, window: int = 63):
    """
    Rolling max drawdown over a given window.
    Uses log-price to ensure scale invariance.
    """
    roll_max = log_price.rolling(window).max()
    drawdown = log_price - roll_max
    return drawdown.rolling(window).min()


def add_cross_sectional_momentum_rank(
    df: pd.DataFrame,
    momentum_col: str = "mom12m",
    out_col: str = "cs_mom_rank",
):
    """
    Compute cross-sectional momentum rank by date. Not only how good is asset
    to its past but how good is it relative to the others?

    Parameters
    ----------
    df : long-format DataFrame with ['date', 'asset_id', momentum_col]
    momentum_col : which momentum horizon to rank (default: 12m)
    out_col : output column name

    Returns
    -------
    DataFrame with cross-sectional momentum rank ∈ [0, 1]
    """

    df = df.copy()

    df[out_col] = df.groupby("date")[momentum_col].rank(pct=True, method="average")

    return df


def fetch_etf_features(
    tickers: list[str] | str,
    start: str,
    end: str = None,
    horizon: Horizon = Horizon.MONTHLY,  # or your Horizon enum
):
    """
    Fetch ETF-level features (liquidity, momentum, volatility, etc.) for one or more tickers.

    Parameters
    ----------
    tickers : list[str] or str
        List of ETF tickers (e.g., ["SPY", "QQQ"]) or a single ticker string.
    start : str
        Start date for fetching data.
    end : str, optional
        End date (default: today).
    horizon : Horizon or str
        Resampling frequency (e.g., Horizon.MONTHLY or Horizon.WEEKLY).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns including:
        ['date', 'asset_id', 'price', 'volume', 'log_ret',
         'mom1m', 'mom6m', 'mom12m', 'mom36m', 'chmom',
         'dolvol', 'turnover', 'sd_turn', 'ill',
         'vol_1w', 'vol_1m', 'vol_3m', 'vol_of_vol',
         'vol_z', 'vol_accel',
         'ma_1m', 'ma_3m', 'ma_signal', 'trend_slope',
         'overnight_gap', 'ret_autocorr', 'vol_autocorr',
         'ret_skew', 'ret_kurt', 'baspread']
        Missing values are filled with 0 (neutral imputation).
    """

    # If you use an enum, let it pass through; pandas just needs its .value
    resample_rule = getattr(horizon, "value", horizon)

    # --- Resample (monthly, weekly, etc.) with feature-specific rules ---
    agg_map = {
        "price": "last",
        "log_ret": "sum",  # cumulative log return over period
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
        "vol_1w": "mean",
        "vol_1m": "mean",
        "vol_3m": "mean",
        "vol_of_vol": "mean",
        "vol_z": "mean",
        "vol_accel": "mean",
        "ma_1m": "last",
        "ma_3m": "last",
        "ma_signal": "last",
        "ma_regime": "last",
        "trend_slope": "last",
        "overnight_gap": "mean",
        "ret_autocorr": "last",
        "vol_autocorr": "last",
        "ret_skew": "last",
        "ret_kurt": "last",
        "baspread": "mean",
        "max_dd_3m": "mean",
        "max_dd_6m": "mean",
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

    def _extract_single_ticker_features(df, ticker: str) -> pd.DataFrame:
        """Extract and compute ETF-local features for one ticker."""
        px, vol = None, None

        # ---- Extract price & volume ----
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

        data = pd.concat({"price": px, "volume": vol}, axis=1).dropna()
        data.index = pd.to_datetime(data.index)

        # ---- Basic returns ----
        data["log_ret"] = np.log(data["price"]).diff()

        # ---- Momentum (price trends) ----
        data["mom1m"] = data["price"].pct_change(21)
        data["mom6m"] = data["price"].pct_change(126)
        data["mom12m"] = data["price"].pct_change(252)
        data["mom36m"] = data["price"].pct_change(756)
        data["chmom"] = data["mom12m"] - data["mom6m"]

        # ---- Liquidity metrics ----
        data["dolvol"] = data["price"] * data["volume"]
        data["turnover"] = data["volume"] / data["volume"].rolling(21).mean() - 1
        data["sd_turn"] = data["turnover"].rolling(63).std()
        data["ill"] = data["log_ret"].abs() / data["dolvol"]

        # ---- Volatility at multiple horizons ----
        data["vol_1w"] = data["log_ret"].rolling(5).std()
        data["vol_1m"] = data["log_ret"].rolling(21).std()
        data["vol_3m"] = data["log_ret"].rolling(63).std()

        # Vol-of-vol (instability of volatility)
        data["vol_of_vol"] = data["vol_1m"].diff().abs().rolling(21).std()

        # ---- Volume-based features ----
        # Z-score of volume over 3 months
        vol_mean_63 = data["volume"].rolling(63).mean()
        vol_std_63 = data["volume"].rolling(63).std()
        data["vol_z"] = (data["volume"] - vol_mean_63) / vol_std_63

        # Volume acceleration (percentage change)
        v = data["volume"]
        data["vol_accel"] = (v - v.shift(1)) / v.shift(1).replace(0, np.nan)
        data["vol_accel"] = data["vol_accel"].fillna(0)

        # ---- Moving average features ----
        data["ma_1m"] = data["price"].rolling(21).mean()
        data["ma_3m"] = data["price"].rolling(63).mean()
        data["ma_signal"] = data["ma_1m"] / data["ma_3m"] - 1
        data["ma_regime"] = (data["ma_signal"] > 0).astype(int)

        # Trend slope (simple linear regression over last 21 days)
        def _slope(x):
            idx = np.arange(len(x))
            # np.polyfit returns [slope, intercept]
            return np.polyfit(idx, x, 1)[0] if np.all(np.isfinite(x)) else np.nan

        data["trend_slope"] = data["price"].rolling(21).apply(_slope, raw=False)

        # Max drawdown
        log_price = np.log(data["price"])
        data["max_dd_3m"] = compute_max_drawdown(log_price, window=63)
        data["max_dd_6m"] = compute_max_drawdown(log_price, window=126)

        # ---- Overnight / daily gap (log move day-over-day) ----
        data["overnight_gap"] = np.log(data["price"] / data["price"].shift(1))

        # ---- Autocorrelation & distributional features ----
        def _autocorr(x):
            s = pd.Series(x)
            return s.autocorr() if s.notna().sum() > 2 else np.nan

        data["ret_autocorr"] = data["log_ret"].rolling(21).apply(_autocorr, raw=False)
        data["vol_autocorr"] = data["vol_1m"].rolling(63).apply(_autocorr, raw=False)

        data["ret_skew"] = data["log_ret"].rolling(63).skew()
        data["ret_kurt"] = data["log_ret"].rolling(63).kurt()

        # --- Approximate Bid-Ask Spread (Corwin-Schultz 2012) ---
        high = low = None
        if isinstance(df.columns, pd.MultiIndex):
            if ("High", ticker) in df.columns and ("Low", ticker) in df.columns:
                high = df[("High", ticker)]
                low = df[("Low", ticker)]
            elif (ticker, "High") in df.columns and (ticker, "Low") in df.columns:
                high = df[(ticker, "High")]
                low = df[(ticker, "Low")]
        else:
            if "High" in df.columns and "Low" in df.columns:
                high = df["High"]
                low = df["Low"]

        if high is not None and low is not None:
            log_hl = np.log(high / low)
            beta = (log_hl.rolling(2).sum() ** 2).rolling(21).mean()
            alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
            bas = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
            data["baspread"] = bas.fillna(method="bfill").fillna(0)
        else:
            data["baspread"] = 0.0  # Default if High/Low not available

        # Resample (monthly, weekly, etc.)
        data_resampled = data.resample(resample_rule).agg(agg_map)

        # Fill missing with 0 (neutral imputation, paper-style)
        data_resampled = data_resampled.fillna(0)

        # Metadata & column normalization
        data_resampled = data_resampled.reset_index().rename(columns={data_resampled.index.name or "index": "date"})
        data_resampled.columns = [c.lower() for c in data_resampled.columns]
        data_resampled["asset_id"] = ticker

        return data_resampled

    # Compute for all tickers
    results = []
    for tk in tickers:
        try:
            res = _extract_single_ticker_features(df_raw, tk)
            results.append(res)
        except Exception as e:
            print(f"[WARN] Skipping {tk}: {e}")

    if not results:
        raise RuntimeError("No ETF features were generated for the requested tickers.")

    # Combine into one long DataFrame
    df_all = pd.concat(results, axis=0).sort_values(["date", "asset_id"]).reset_index(drop=True)

    return df_all
