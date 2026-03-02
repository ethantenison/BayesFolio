from __future__ import annotations

import numpy as np
import pandas as pd

from bayesfolio.schemas.common import SchemaMetadata
from bayesfolio.schemas.contracts.backtest import BacktestResult
from bayesfolio.schemas.contracts.optimize import OptimizationResult


def run_weighted_backtest(
    realized_returns: pd.DataFrame,
    optimization: OptimizationResult,
    periods_per_year: int = 12,
) -> BacktestResult:
    """Run weighted backtest and return summary metrics in decimal units."""

    aligned = realized_returns[optimization.asset_order]
    weight_vector = np.asarray(optimization.weights, dtype=float)
    portfolio_returns = aligned.to_numpy(dtype=float) @ weight_vector

    cumulative = float(np.prod(1.0 + portfolio_returns) - 1.0)
    annualized = float((1.0 + cumulative) ** (periods_per_year / max(len(portfolio_returns), 1)) - 1.0)
    annualized_vol = float(np.std(portfolio_returns, ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(annualized / annualized_vol) if annualized_vol > 0 else 0.0

    return BacktestResult(
        metadata=SchemaMetadata(**optimization.metadata.model_dump()),
        cumulative_return=cumulative,
        annualized_return=annualized,
        annualized_volatility=annualized_vol,
        sharpe_ratio=sharpe,
    )
