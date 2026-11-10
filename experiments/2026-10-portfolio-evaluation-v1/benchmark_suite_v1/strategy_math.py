"""Pure deterministic monthly benchmark weights from adjusted ETF prices.

Research numerical layer only: no I/O, GP, optimizer, covariance or data refresh.
Prices are positive adjusted dollar levels; returns, volatility and weights are
decimals. Imported by run_benchmarks.py; synthetic tests document timing rules.
"""

from __future__ import annotations

import numpy as np


def benchmark_panel(
    price_dates: np.ndarray,
    prices: np.ndarray,
    origins: list[str],
    assets: list[str],
    equity_assets: list[str],
    *,
    volatility_months: int = 36,
    momentum_months: int = 12,
    momentum_skip: int = 1,
    top_k: int = 7,
    equity_fraction: float = 0.8,
) -> dict[str, np.ndarray]:
    """Build four monthly rebalanced, long-only, fully invested strategies.

    Args:
        price_dates: Complete consecutive months, dated at actual last observations.
        prices: (history_months, assets) finite positive adjusted price levels.
        origins: Strictly increasing monthly decision dates, ISO YYYY-MM-DD.
        assets: Unique tickers in the price and output-weight column order.
        equity_assets: Equity members; all remaining assets are fixed income.
        volatility_months: Trailing monthly total returns for sample volatility.
        momentum_months: Denominator price offset; 12 means price at t-12.
        momentum_skip: Numerator offset; 1 means price at t-1, excluding month t.
        top_k: Number of equally weighted momentum winners, including in downturns.
        equity_fraction: Fixed capital fraction equally distributed within equities.

    Returns:
        Four (origins, assets) weight matrices, monthly signal arrays, and explicit
        lookback endpoint dates. Ties use alphabetical ticker order, not input order.
        Volatility uses 36 returns through t; momentum uses P[t-1]/P[t-12]-1.
        No future price affects an earlier decision; no shorting, leverage, missing
        value filling, zero-volatility floor or weight fallback is permitted.
    """
    dates = np.asarray(price_dates, dtype="datetime64[D]")
    decisions = np.asarray(origins, dtype="datetime64[D]")
    if dates.ndim != 1 or decisions.ndim != 1 or not len(dates) or not len(decisions):
        raise ValueError("Nonempty one-dimensional date axes required")
    if np.isnat(dates).any() or np.isnat(decisions).any():
        raise ValueError("Finite dates required")
    months = dates.astype("datetime64[M]")
    decision_months = decisions.astype("datetime64[M]")
    if (np.diff(months).astype(int) != 1).any() or (np.diff(decision_months).astype(int) != 1).any():
        raise ValueError("Complete consecutive months required; no gaps or duplicate months")
    if not assets or len(set(assets)) != len(assets) or prices.shape != (len(dates), len(assets)):
        raise ValueError("Unique assets and matching monthly price matrix required")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Finite positive prices required; no filling or dropping")
    if not 0 < equity_fraction < 1 or not equity_assets or len(set(equity_assets)) != len(equity_assets):
        raise ValueError("Valid equity sleeve required")
    equity = np.isin(assets, equity_assets)
    if not set(equity_assets) < set(assets):
        raise ValueError("Equity and fixed-income sleeves must both be nonempty and known")
    if any(
        not isinstance(v, int) or isinstance(v, bool)
        for v in [volatility_months, momentum_months, momentum_skip, top_k]
    ):
        raise ValueError("Lookbacks and holding count must be integers")
    if volatility_months < 2 or not 0 <= momentum_skip < momentum_months or not 1 <= top_k <= len(assets):
        raise ValueError("Invalid lookbacks or momentum holding count")
    positions = np.searchsorted(months, decision_months)
    if (positions >= len(months)).any() or not np.array_equal(months[positions], decision_months):
        raise ValueError("Every decision requires its matching observed price month")
    if (dates[positions] > decisions).any():
        raise ValueError("A monthly close was not yet available at the decision")
    if (positions < max(volatility_months, momentum_months)).any():
        raise ValueError("Insufficient complete lookback history")

    returns = prices[1:] / prices[:-1] - 1
    offsets = positions[:, None] - volatility_months + np.arange(volatility_months)
    volatility = returns[offsets].std(axis=1, ddof=1)
    if not np.isfinite(volatility).all() or (volatility <= 0).any():
        raise ValueError("Positive finite historical volatility required; no floor or fallback")
    inverse = 1 / volatility
    inverse /= inverse.sum(axis=1, keepdims=True)
    momentum = prices[positions - momentum_skip] / prices[positions - momentum_months] - 1
    selected = np.zeros_like(momentum, dtype=bool)
    for index, scores in enumerate(momentum):
        ordered = np.lexsort((np.asarray(assets), -scores))
        selected[index, ordered[:top_k]] = True
    fixed = np.where(equity, equity_fraction / equity.sum(), (1 - equity_fraction) / (~equity).sum())
    return {
        "equal_weight": np.full((len(decisions), len(assets)), 1 / len(assets)),
        "fixed_80_20": np.broadcast_to(fixed, momentum.shape).copy(),
        "inverse_volatility": inverse,
        "momentum_top7": selected.astype(float) / top_k,
        "volatility": volatility,
        "momentum_score": momentum,
        "momentum_selected": selected,
        "volatility_start_price_date": dates[positions - volatility_months],
        "volatility_end_price_date": dates[positions],
        "momentum_start_price_date": dates[positions - momentum_months],
        "momentum_end_price_date": dates[positions - momentum_skip],
    }
