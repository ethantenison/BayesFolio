import numpy as np
import pandas as pd



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
        long_asset  = ranked.index[-1]

        r_ls = true[long_asset] - true[short_asset]   # equal-weighted LS
        ls_returns.append(r_ls)

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
    ann_return = (1 + ret.mean())**periods_per_year - 1
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
    
def assessing_long_short_performance(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    label: str = "model"
):
    ls_ret = long_short_returns(y_true, y_pred)
    stats = portfolio_stats(ls_ret, periods_per_year=12)
    stats = {f"{label}/{k}": v for k, v in stats.items()}
    return stats