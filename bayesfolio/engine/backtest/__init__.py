"""Backtest stage adapters."""

from bayesfolio.engine.backtest.backtest_summary import (
    backtest_portfolio,
    opt_weights,
    summarize_backtest,
)
from bayesfolio.engine.backtest.evaluate_asset_pricing import evaluate_asset_pricing
from bayesfolio.engine.backtest.portfolio_helpers import (
    assess_performance,
    assessing_long_short_performance,
    long_short_returns,
    long_short_returns_topk,
    portfolio_stats,
    riskfolio_returns_rolling,
)
from bayesfolio.engine.backtest.runner import run_weighted_backtest

__all__ = [
    "run_weighted_backtest",
    "assess_performance",
    "assessing_long_short_performance",
    "backtest_portfolio",
    "evaluate_asset_pricing",
    "long_short_returns",
    "long_short_returns_topk",
    "opt_weights",
    "portfolio_stats",
    "riskfolio_returns_rolling",
    "summarize_backtest",
]
