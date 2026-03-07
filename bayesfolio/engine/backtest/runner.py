from __future__ import annotations

import numpy as np
import pandas as pd

from bayesfolio.contracts.results.backtest import BacktestResult
from bayesfolio.contracts.results.optimize import OptimizeResult


def run_weighted_backtest(
    realized_returns: pd.DataFrame,
    optimization: OptimizeResult,
    periods_per_year: int = 12,
) -> BacktestResult:
    """Run a weighted portfolio backtest and return summary performance metrics.

    Args:
        realized_returns: Period returns by asset as decimals where each row is a period and
            each column corresponds to an asset in `optimization.asset_order`.
        optimization: Optimizer output containing asset order and portfolio weights.
        periods_per_year: Number of return periods per year (for annualization).

    Returns:
        BacktestResult with:
            - cumulative_return (decimal),
            - annualized_return as CAGR (decimal),
            - annualized_volatility (decimal),
            - sharpe_ratio (dimensionless),
            - max_drawdown (decimal, negative or zero),
            - calmar_ratio (dimensionless),
            - sortino_ratio (dimensionless).
    """

    aligned = realized_returns[optimization.asset_order]
    weight_vector = np.asarray(optimization.weights, dtype=float)
    portfolio_returns = aligned.to_numpy(dtype=float) @ weight_vector

    cumulative = float(np.prod(1.0 + portfolio_returns) - 1.0)
    annualized = float((1.0 + cumulative) ** (periods_per_year / max(len(portfolio_returns), 1)) - 1.0)
    annualized_vol = (
        float(np.std(portfolio_returns, ddof=1) * np.sqrt(periods_per_year)) if len(portfolio_returns) >= 2 else 0.0
    )
    sharpe = float(annualized / annualized_vol) if annualized_vol > 0 else 0.0

    wealth_curve = np.cumprod(1.0 + portfolio_returns)
    running_peak = np.maximum.accumulate(wealth_curve)
    drawdowns = (wealth_curve / running_peak) - 1.0
    max_drawdown = float(drawdowns.min()) if drawdowns.size > 0 else 0.0

    downside_returns = np.minimum(portfolio_returns, 0.0)
    downside_obs = int(np.count_nonzero(portfolio_returns < 0.0))
    downside_vol = float(np.std(downside_returns, ddof=1) * np.sqrt(periods_per_year)) if downside_obs >= 2 else 0.0
    sortino = float(annualized / downside_vol) if downside_vol > 0 else 0.0
    calmar = float(annualized / abs(max_drawdown)) if max_drawdown < 0 else 0.0

    return BacktestResult(
        cumulative_return=cumulative,
        annualized_return=annualized,
        annualized_volatility=annualized_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar,
        sortino_ratio=sortino,
    )
