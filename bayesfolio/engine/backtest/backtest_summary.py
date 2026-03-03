"""Riskfolio Backtesting"""

import numpy as np
import pandas as pd
import riskfolio as rp


def summarize_backtest(bt_df: pd.DataFrame):
    """Compute summary performance metrics for the backtest."""
    ret = pd.to_numeric(bt_df["net_return"], errors="coerce").dropna()
    if len(ret) == 0:
        return {}

    total_growth = float(np.prod(1.0 + ret.to_numpy(dtype=float)))
    annualization = 12.0 / float(len(ret))
    cagr = float(np.power(total_growth, annualization) - 1.0)
    vol = ret.std() * np.sqrt(12)
    sharpe = cagr / vol if vol > 0 else np.nan
    sortino = cagr / (ret[ret < 0].std() * np.sqrt(12)) if (ret < 0).any() else np.nan
    dd = (bt_df["cum_return"].cummax() - bt_df["cum_return"]) / bt_df["cum_return"].cummax()
    max_dd = dd.max()
    calmar = cagr / max_dd if max_dd > 0 else np.nan
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": max_dd,
        "Calmar": calmar,
        "MeanTurnover": bt_df["turnover"].mean(),
    }


def opt_weights(excess_returns, risk_config):
    # Initialize Riskfolio portfolio
    port = rp.Portfolio(returns=excess_returns)
    port.assets_stats(method_mu=risk_config.method_mu, method_cov=risk_config.method_cov)
    port.upperlng = risk_config.upperlng
    port.nea = risk_config.nea  # Update to use risk_config.nea

    # Optimize portfolio
    weights_df = port.optimization(
        model=risk_config.model,
        rm=risk_config.rm,
        obj=risk_config.obj,
        rf=risk_config.rf,
        l=risk_config.ra,
        hist=risk_config.hist,
    )
    weights = np.ravel(weights_df.to_numpy())
    print(weights)
    shp = rp.Sharpe(
        w=weights_df, mu=port.mu, cov=port.cov, returns=excess_returns, rm=risk_config.rm, rf=risk_config.rf
    )

    return shp, weights


def backtest_portfolio(
    y: pd.DataFrame, ticker_config, risk_config, window: int = 24, cost_rate: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Config-driven backtest for Riskfolio portfolios.

    Parameters
    ----------
    y : pd.DataFrame
        Asset returns, indexed by date (aligned to ticker_config.horizon)
    ticker_config : TickerConfig
        Contains start_date, end_date, interval, tickers, and horizon (e.g., 'BM')
    risk_config : RiskfolioConfig
        Contains optimization and risk parameters for portfolio construction
    window : int
        Lookback window (in periods, matching y's frequency)
    cost_rate : float
        Proportional transaction cost per unit turnover (e.g., 0.001 = 10bps)

    Returns
    -------
    bt_results : pd.DataFrame
        Monthly portfolio performance and turnover
    weights_df : pd.DataFrame
        Asset weights per rebalance period
    final_weights : pd.Series
        Suggested portfolio for next period
    """

    # Ensure data integrity
    y = y.copy()
    y.index = pd.to_datetime(y.index)
    y = y.sort_index().apply(pd.to_numeric, errors="coerce")

    # Align frequency to data horizon (BM, W-FRI, etc.)
    freq = getattr(ticker_config.horizon, "value", "BM")
    y = y.asfreq(freq).dropna(how="all")
    rebal_dates = y.index[window:]

    results = []
    weights_over_time = []
    w_prev = None

    for date in rebal_dates:
        try:
            # Rolling training window
            train = y.loc[:date].iloc[-window:]
            train = train.dropna(axis=1, how="any")
            if train.empty:
                continue

            # Initialize Riskfolio portfolio
            port = rp.Portfolio(returns=train)
            port.assets_stats(method_mu=risk_config.method_mu, method_cov=risk_config.method_cov)
            port.upperlng = risk_config.upperlng
            port.nea = risk_config.nea  # Update to use risk_config.nea

            # Optimize portfolio
            w = port.optimization(
                model=risk_config.model,
                rm=risk_config.rm,
                obj=risk_config.obj,
                rf=risk_config.rf,
                l=risk_config.ra,
                hist=risk_config.hist,
            )

            # Handle failed optimizations
            if w is None or w.empty:
                w = pd.Series(1 / len(train.columns), index=train.columns)
            else:
                w = w.squeeze()  # Convert DataFrame to Series
                w = w.reindex(train.columns).fillna(0)

            # Store weights
            weights_over_time.append(w.rename(date))

            # Ensure next-period returns exist
            if date not in y.index:
                continue
            next_ret = y.loc[date].reindex(w.index).fillna(0)

            # Compute portfolio return
            port_ret = float(np.dot(next_ret, w))

            # Transaction cost and turnover
            if w_prev is not None:
                common = w.index.intersection(w_prev.index)
                turnover = float((w.loc[common] - w_prev.loc[common]).abs().sum())
                cost = turnover * cost_rate
            else:
                turnover, cost = np.nan, 0.0

            net_ret = port_ret - cost
            results.append(
                {
                    "date": date,
                    "portfolio_return": port_ret,
                    "net_return": net_ret,
                    "turnover": turnover,
                    "n_assets": len(train.columns),
                }
            )

            w_prev = w.copy()

        except Exception as e:
            print(f"[WARN] Skipping {date.date()} due to {type(e).__name__}: {e}")
            continue

    # Build DataFrames
    bt_results = pd.DataFrame(results).set_index("date")
    bt_results["cum_return"] = (1 + bt_results["net_return"]).cumprod()

    weights_df = pd.DataFrame(weights_over_time)
    weights_df.index.name = "date"

    # Equal-weight benchmark
    eq_w = pd.Series(1 / y.shape[1], index=y.columns)
    eq_returns = (y @ eq_w).loc[bt_results.index]
    bt_results["benchmark"] = (1 + eq_returns).cumprod()

    # === Schwab-aligned benchmark ===
    schwab_weights = pd.Series(
        {
            "AVEM": 0.04,
            "BCD": 0.02,
            "BYLD": 0.03,
            "ESGD": 0.2,
            "HYEM": 0.00,
            "IEF": 0.08,
            "ISCF": 0.04,
            "SNPE": 0.37,
            "VBK": 0.08,
            "VNQ": 0.028,
            "VNQI": 0.02,
            "VWOB": 0.03,
        }
    )
    schwab_weights = schwab_weights.reindex(y.columns).fillna(0)
    schwab_weights /= schwab_weights.sum()  # normalize

    schwab_returns = (y @ schwab_weights).loc[bt_results.index]
    bt_results["benchmark_schwab"] = (1 + schwab_returns).cumprod()

    bt_results["excess_return"] = bt_results["cum_return"] - bt_results["benchmark"]

    # Final weights (suggested next portfolio)
    final_weights = weights_df.iloc[-1].sort_values(ascending=False)

    return bt_results, weights_df, final_weights
