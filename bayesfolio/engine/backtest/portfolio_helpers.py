import numpy as np
import pandas as pd
import riskfolio as rp

from bayesfolio.core.settings import RiskfolioConfig


def riskfolio_returns_rolling(
    y_true: pd.DataFrame,  # realized excess returns (T × N)
    y_pred: pd.DataFrame,  # predicted excess returns (T × N)
    y_unc: pd.DataFrame | None = None,  # optional uncertainty (T × N)
    window: int = 36,  # rolling window for risk estimation
    config=None,  # RiskfolioConfig
    rf: float = 0.0,
    long_only: bool = True,
    leverage: float = 1.0,
    unc_penalty: float = 0.0,  # if >0, shrinks mu by uncertainty
    min_assets: int = 3,
    min_hist_frac: float = 0.80,  # allow up to 20% missing in window
    eps: float = 1e-12,
):
    """
    Walk-forward backtest using Riskfolio where each period t:
      - risk is estimated from realized returns [t-window, ..., t-1]
      - expected returns are y_pred[t]
      - realized portfolio return uses y_true[t]

    Returns
    -------
    port_ret : pd.Series
        Portfolio realized returns (aligned to y_true index).
    w_df : pd.DataFrame
        Weights per period (index aligned to port_ret index, columns assets).
    """
    if config is None:
        config = RiskfolioConfig()

    # --- Align assets ---
    common_assets = y_true.columns.intersection(y_pred.columns)
    y_true = y_true.loc[:, common_assets].copy()
    y_pred = y_pred.loc[:, common_assets].copy()

    if y_unc is not None:
        y_unc = y_unc.loc[:, common_assets].copy()

    # --- Align time index (intersection) ---
    common_idx = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[common_idx].copy()
    y_pred = y_pred.loc[common_idx].copy()
    if y_unc is not None:
        y_unc = y_unc.loc[common_idx].copy()

    print(f"Riskfolio rolling backtest: T={len(y_true)}, N={len(common_assets)}")

    T, N = y_true.shape
    if T <= window:
        raise ValueError(f"Not enough rows for window={window}. Need > window, got T={T}.")

    rets: list[float] = []
    w_rows: list[pd.Series] = []
    idx: list[pd.Timestamp] = []

    # bounds
    l_bound, u_bound = (0.0, leverage) if long_only else (-leverage, leverage)

    for t in range(window, T):
        # 1) Historical returns window (no lookahead)
        R_hist = y_true.iloc[t - window : t].copy()
        mu_t = y_pred.iloc[t].copy()

        # Optional uncertainty-aware shrinkage
        if (y_unc is not None) and (unc_penalty > 0):
            sigma_t = y_unc.iloc[t].abs().clip(lower=eps)
            mu_t = mu_t - unc_penalty * sigma_t

        # 2) Determine valid assets
        # - must have mu at t
        # - must have enough non-NaNs in history window
        valid = mu_t.dropna().index

        hist_ok = R_hist.notna().mean(axis=0) >= float(min_hist_frac)
        valid = valid.intersection(R_hist.columns[hist_ok])

        # also require realized return exists at t (so dot-product is valid)
        valid = valid.intersection(y_true.columns[y_true.iloc[t].notna()])

        if len(valid) < min_assets:
            continue

        R_hist_v = R_hist.loc[:, valid].copy()

        # Fill remaining gaps inside window (Riskfolio/cov estimators hate NaNs)
        # Use column-wise ffill/bfill, then 0 as last resort.
        R_hist_v = R_hist_v.ffill().bfill().fillna(0.0)

        mu_v = mu_t.loc[valid].astype(float)
        print("mu_v:", mu_v)

        # 3) Build portfolio
        port = rp.Portfolio(returns=R_hist_v, nea=getattr(config, "nea", 10))

        # Provide expected returns with labels (Riskfolio expects indexed structure)
        port.mu = pd.DataFrame(mu_v, columns=["mu"])

        port.card = None

        # Estimate stats (this sets covariance etc.)
        port.assets_stats(
            method_mu=getattr(config, "method_mu", "hist"),
            method_cov=getattr(config, "method_cov", "hist"),
        )

        # 4) Optimize
        try:
            w = port.optimization(
                model=getattr(config, "model", "Classic"),
                rm=getattr(config, "rm", "MV"),
                obj=getattr(config, "obj", "Sharpe"),
                rf=rf,
                l=0,
                u=u_bound,
                hist=True,
            )
        except TypeError:
            # fallback for older versions
            w = port.optimization(
                model=getattr(config, "model", "Classic"),
                rm=getattr(config, "rm", "MV"),
                obj=getattr(config, "obj", "Sharpe"),
                rf=rf,
                hist=True,
            )
        except Exception:
            # solver / numerical failure at this time step
            continue

        if w is None or len(w) == 0:
            continue

        # Riskfolio returns weights as DataFrame indexed by asset names
        # Align and extract weights for valid assets only
        if isinstance(w, pd.DataFrame):
            # Often a single column like 'weights'
            w_ser = w.iloc[:, 0]
        else:
            # Sometimes it returns a Series
            w_ser = pd.Series(w)

        w_ser = w_ser.reindex(valid).astype(float)

        # If any weights missing -> skip
        if w_ser.isna().any():
            continue

        w_t = w_ser.values
        r_t = y_true.iloc[t].loc[valid].values.astype(float)

        print("w_t:", w_t)

        # Defensive normalization
        if long_only:
            s = float(np.sum(w_t))
            if (not np.isfinite(s)) or (abs(s) < eps):
                continue
            w_t = w_t / s * leverage
        else:
            gross = float(np.sum(np.abs(w_t)))
            if np.isfinite(gross) and gross > eps:
                w_t = w_t / gross * leverage
            else:
                continue

        port_ret_t = float(np.dot(w_t, r_t))
        if not np.isfinite(port_ret_t):
            continue

        rets.append(port_ret_t)
        w_rows.append(pd.Series(w_t, index=valid))
        idx.append(y_true.index[t])

    port_ret = pd.Series(rets, index=idx, name="riskfolio_return")
    w_df = pd.DataFrame(w_rows, index=idx).reindex(columns=common_assets).fillna(0.0)
    return port_ret, w_df


def long_short_returns(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    """
    Compute long–short returns based on predicted cross-sectional rankings.
    For each period, the asset with the highest predicted return is taken
    as a long position, and the asset with the lowest predicted return is
    taken as a short position. The resulting long–short return is:

        r_ls,t = r_true,long_asset,t − r_true,short_asset,t

    Parameters
    ----------
    y_true : pd.DataFrame
        Realized returns for each time period (T × N assets).
    y_pred : pd.DataFrame
        Model-predicted returns for each time period (T × N assets).

    Returns
    -------
    pd.Series
        A Series of long–short returns with one value per time period.

    Why This Matters
    ----------------
    Long–short portfolio performance is one of the most important ways to
    evaluate an asset-return forecasting model in empirical finance.
    It answers the key question:

        “Do the model’s predictions successfully rank assets by expected return?”

    This measure is directly connected to the core idea of factor investing
    and anomaly discovery: profitable long–short spreads indicate that
    the model extracts economically meaningful signals beyond noise.

    Pros
    ----
    • **Direct test of predictive skill**
      Long–short spreads reveal whether the model finds alpha, independent
      of portfolio optimization or risk-budgeting frameworks.

    • **Benchmarked to academic standards**
      Used universally in asset-pricing research (Fama–French portfolios,
      factor testing, Gu–Kelly–Xiu deep learning, anomaly studies).

    • **Robust to scale**
      Only requires correct *ranking*, not accurate magnitude of predictions.

    • **Model evaluation, not portfolio engineering**
      Avoids conflating predictive performance with optimization artifacts.

    Cons
    ----
    • **Concentrated and unrealistic to trade**
      A 1×1 long–short portfolio has high tracking error and is not meant to
      represent a real investable strategy.

    • **Ignores risk and diversification**
      Does not account for correlations, risk constraints, or turnover costs.

    • **Sensitive to small cross-sections**
      With few assets, a single ranking mistake has large impact.

    • **Not a substitute for portfolio optimization**
      After evaluating signal strength, real portfolios should still use
      optimization (e.g., Riskfolio, mean–variance, HRP, Black–Litterman).

    Notes
    -----
    This method is best interpreted as a **diagnostic of information content**
    rather than a practical trading rule. A strong long–short return series
    confirms that the forecasting model carries meaningful cross-sectional
    predictive power.
    """
    ls_returns = []

    for t in range(len(y_pred)):
        preds = y_pred.iloc[t]
        true = y_true.iloc[t]

        # Rank predicted returns
        ranked = preds.sort_values()

        short_asset = ranked.index[0]
        long_asset = ranked.index[-1]

        r_ls = true[long_asset] - true[short_asset]  # equal-weighted LS
        ls_returns.append(r_ls)

    return pd.Series(ls_returns, name="long_short")


def long_short_returns_topk(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    k: int | None = None,
    q: float | None = 0.2,
    min_assets: int = 5,
):
    """
    Compute long–short returns using Top-K / Bottom-K portfolios
    based on predicted cross-sectional rankings.

    Either `k` or `q` must be provided.

    Parameters
    ----------
    y_true : pd.DataFrame
        Realized returns (T × N).
    y_pred : pd.DataFrame
        Predicted returns (T × N).
    k : int, optional
        Number of assets in long and short legs.
    q : float, optional
        Fraction of universe in each leg (e.g., 0.2 → top/bottom quintile).
    min_assets : int, default=5
        Minimum cross-sectional size required to compute returns.

    Returns
    -------
    pd.Series
        Long–short return series.
    """
    if k is None and q is None:
        raise ValueError("Either k or q must be specified.")

    ls_returns = []

    for t in range(len(y_pred)):
        preds = y_pred.iloc[t]
        true = y_true.iloc[t]

        # Align & clean
        valid = preds.notna() & true.notna()
        preds = preds[valid]
        true = true[valid]

        n = len(preds)
        if n < min_assets:
            ls_returns.append(np.nan)
            continue

        # Determine K
        k_t = k if k is not None else max(1, int(np.floor(q * n)))
        k_t = min(k_t, n // 2)  # safety cap

        if k_t == 0:
            ls_returns.append(np.nan)
            continue

        ranked = preds.sort_values()

        short_assets = ranked.index[:k_t]
        long_assets = ranked.index[-k_t:]

        # Equal-weighted portfolios
        r_long = true.loc[long_assets].mean()
        r_short = true.loc[short_assets].mean()

        ls_returns.append(r_long - r_short)

    return pd.Series(ls_returns, name="long_short")


def portfolio_stats(ret: pd.Series, periods_per_year=12):
    """
    Compute standard performance statistics for a portfolio return series.

    Parameters
    ----------
    ret : pd.Series
        Series of periodic returns (e.g., monthly long–short returns).
    periods_per_year : int, default=12
        Number of return periods per year (12 for monthly data, 52 for weekly, 252 for daily).

    Returns
    -------
    dict
        A dictionary containing:

        📈 **cum_return** : float
            Total cumulative return over the sample
            `(1 + r_1)(1 + r_2)...(1 + r_T) - 1`

        🚀 **ann_return** : float
            Annualized geometric return
            `(1 + mean(r))**periods_per_year - 1`

        ⚠️ **ann_vol** : float
            Annualized volatility
            `std(r) * sqrt(periods_per_year)`

        ⚡ **sharpe** : float
            Annualized Sharpe ratio (risk-free assumed 0)
            `ann_return / ann_vol`

        📉 **max_drawdown** : float
            Maximum drawdown over the cumulative return path
            `min((cum_ret - peak) / peak)`

    Notes
    -----
    - NaNs in the input series are removed.
    - If the series is empty after cleaning, all returned metrics are NaN.
    """
    ret = ret.dropna()
    if len(ret) == 0:
        return {
            "cum_return": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    cum_return = (1 + ret).prod() - 1
    ann_return = (1 + ret.mean()) ** periods_per_year - 1
    ann_vol = ret.std() * np.sqrt(periods_per_year)

    sharpe = np.nan if ann_vol == 0 else ann_return / ann_vol

    # Max drawdown
    cumulative = (1 + ret).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()

    return {
        "cum_return": float(cum_return),
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def assessing_long_short_performance(y_true: pd.DataFrame, y_pred: pd.DataFrame, label: str = "model"):
    ls_ret = long_short_returns(y_true, y_pred)
    stats = portfolio_stats(ls_ret, periods_per_year=12)
    stats = {f"{label}/{k}": v for k, v in stats.items()}
    return stats


def assess_performance(
    ret: pd.Series,
    label: str,
    periods_per_year: int = 12,
):
    stats = portfolio_stats(ret, periods_per_year)
    return {f"{label}/{k}": v for k, v in stats.items()}
