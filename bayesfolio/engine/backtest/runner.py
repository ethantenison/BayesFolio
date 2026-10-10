from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from bayesfolio.contracts.commands.backtest import WalkForwardBacktestCommand, WalkForwardPeriod
from bayesfolio.contracts.results.backtest import (
    BacktestResult,
    WalkForwardBacktestResult,
    WalkForwardPeriodResult,
)
from bayesfolio.contracts.results.monthly_portfolio import MonthlyPortfolioResult
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

    return _summarize_returns(portfolio_returns, periods_per_year=periods_per_year)


def build_walk_forward_period(
    candidate: MonthlyPortfolioResult,
    *,
    realization_date: date,
    realized_returns: dict[str, float],
) -> WalkForwardPeriod:
    """Pair one historical monthly candidate with its subsequent returns.

    Args:
        candidate: Historical monthly result whose candidate weights were fixed
            on ``as_of_date``.
        realization_date: Later date on which the subsequent returns became
            observable.
        realized_returns: Decimal returns keyed by the candidate asset order.

    Returns:
        Validated walk-forward period ready for a backtest command.
    """

    optimization = candidate.optimize_result
    weights = dict(zip(optimization.asset_order, optimization.weights, strict=True))
    return WalkForwardPeriod(
        strategy_id=candidate.strategy_id,
        source_run_label=candidate.run_label,
        decision_date=candidate.as_of_date,
        realization_date=realization_date,
        weights=weights,
        realized_returns=realized_returns,
    )


def run_walk_forward_backtest(command: WalkForwardBacktestCommand) -> WalkForwardBacktestResult:
    """Evaluate dated weights only against their subsequent realized returns.

    Each command period carries the decision date, a strictly later realization
    date, weights fixed at the decision date, and the subsequent realized asset
    returns. This explicit pairing prevents the runner from guessing date shifts
    and makes look-ahead alignment reviewable in the serialized contract.

    Args:
        command: Validated walk-forward periods and evaluation assumptions.

    Returns:
        Period-level outcomes and aggregate performance metrics after transaction
        costs.

    Raises:
        ValueError: If transaction costs make a period net return less than or
            equal to ``-1.0``.
    """

    previous_weights: dict[str, float] | None = None
    period_results: list[WalkForwardPeriodResult] = []
    net_returns: list[float] = []
    turnovers: list[float] = []
    total_transaction_cost_amount = 0.0
    portfolio_value = command.starting_value
    cost_per_unit_turnover = command.transaction_cost_bps / 10_000.0

    for period in command.periods:
        gross_return = float(sum(period.weights[asset] * period.realized_returns[asset] for asset in period.weights))
        if previous_weights is None:
            turnover = sum(abs(weight) for weight in period.weights.values()) if command.charge_initial_trade else 0.0
        else:
            all_assets = set(previous_weights) | set(period.weights)
            turnover = 0.5 * sum(
                abs(period.weights.get(asset, 0.0) - previous_weights.get(asset, 0.0)) for asset in all_assets
            )

        transaction_cost_rate = turnover * cost_per_unit_turnover
        transaction_cost_amount = portfolio_value * transaction_cost_rate
        net_return = gross_return - transaction_cost_rate
        if net_return <= -1.0:
            raise ValueError("Transaction costs and realized return produce a net return less than or equal to -1.0.")
        portfolio_value *= 1.0 + net_return
        period_results.append(
            WalkForwardPeriodResult(
                source_run_label=period.source_run_label,
                decision_date=period.decision_date,
                realization_date=period.realization_date,
                gross_return=gross_return,
                turnover=turnover,
                transaction_cost_rate=transaction_cost_rate,
                transaction_cost_amount=transaction_cost_amount,
                net_return=net_return,
                ending_value=portfolio_value,
            )
        )
        net_returns.append(net_return)
        turnovers.append(turnover)
        total_transaction_cost_amount += transaction_cost_amount
        previous_weights = period.weights

    return WalkForwardBacktestResult(
        strategy_id=command.strategy_id,
        summary=_summarize_returns(np.asarray(net_returns), periods_per_year=command.periods_per_year),
        periods=period_results,
        periods_per_year=command.periods_per_year,
        starting_value=command.starting_value,
        terminal_value=portfolio_value,
        transaction_cost_bps=command.transaction_cost_bps,
        average_turnover=float(np.mean(turnovers)),
        total_transaction_cost_amount=total_transaction_cost_amount,
        hit_rate=float(np.mean(np.asarray(net_returns) > 0.0)),
    )


def _summarize_returns(portfolio_returns: np.ndarray, *, periods_per_year: float) -> BacktestResult:
    """Summarize a decimal portfolio-return sequence consistently."""

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
