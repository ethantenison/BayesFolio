"""Pure aggregation of externally measured portfolio and benchmark returns."""

from __future__ import annotations

import numpy as np

from bayesfolio.contracts.commands.performance import TrackPortfolioPerformanceCommand
from bayesfolio.contracts.results.performance import (
    PerformancePeriodResult,
    PortfolioPerformanceResult,
)


def track_portfolio_performance(
    command: TrackPortfolioPerformanceCommand,
) -> PortfolioPerformanceResult:
    """Aggregate realized returns, P/L, active risk, beta, and Jensen alpha.

    The input returns must already be measured correctly for external cash-flow
    timing. Jensen alpha is a descriptive single-benchmark estimate and must not
    be interpreted as causal evidence of investment skill.

    Args:
        command: Ordered measured periods and annualization assumptions.

    Returns:
        Period-level cumulative values and benchmark-relative summary metrics.
    """

    portfolio_returns = np.asarray([period.portfolio_return for period in command.periods], dtype=float)
    benchmark_returns = np.asarray([period.benchmark_return for period in command.periods], dtype=float)
    risk_free_returns = np.asarray([period.risk_free_return for period in command.periods], dtype=float)
    active_returns = portfolio_returns - benchmark_returns

    portfolio_growth = np.cumprod(1.0 + portfolio_returns)
    benchmark_growth = np.cumprod(1.0 + benchmark_returns)
    period_count = len(command.periods)
    annualization_exponent = command.periods_per_year / period_count
    cumulative_portfolio_return = float(portfolio_growth[-1] - 1.0)
    cumulative_benchmark_return = float(benchmark_growth[-1] - 1.0)
    annualized_portfolio_return = float(portfolio_growth[-1] ** annualization_exponent - 1.0)
    annualized_benchmark_return = float(benchmark_growth[-1] ** annualization_exponent - 1.0)
    annualized_excess_return = annualized_portfolio_return - annualized_benchmark_return

    if period_count >= 2:
        tracking_error = float(np.std(active_returns, ddof=1) * np.sqrt(command.periods_per_year))
    else:
        tracking_error = 0.0
    annualized_mean_active = float(np.mean(active_returns) * command.periods_per_year)
    information_ratio = annualized_mean_active / tracking_error if tracking_error > 0 else None

    portfolio_excess = portfolio_returns - risk_free_returns
    benchmark_excess = benchmark_returns - risk_free_returns
    beta: float | None = None
    annualized_jensen_alpha: float | None = None
    if period_count >= 2:
        benchmark_variance = float(np.var(benchmark_excess, ddof=1))
        if benchmark_variance > 0:
            covariance = float(np.cov(portfolio_excess, benchmark_excess, ddof=1)[0, 1])
            beta = covariance / benchmark_variance
            period_alpha = float(np.mean(portfolio_excess) - beta * np.mean(benchmark_excess))
            annualized_jensen_alpha = period_alpha * command.periods_per_year

    period_results = [
        PerformancePeriodResult(
            start_date=period.start_date,
            end_date=period.end_date,
            portfolio_return=period.portfolio_return,
            benchmark_return=period.benchmark_return,
            active_return=period.portfolio_return - period.benchmark_return,
            cumulative_portfolio_return=float(portfolio_growth[index] - 1.0),
            cumulative_benchmark_return=float(benchmark_growth[index] - 1.0),
            realized_gain_loss=period.realized_gain_loss,
            unrealized_gain_loss_change=period.unrealized_gain_loss_change,
            fees=period.fees,
        )
        for index, period in enumerate(command.periods)
    ]

    return PortfolioPerformanceResult(
        account_reference=command.account_reference,
        strategy_id=command.strategy_id,
        benchmark_id=command.benchmark_id,
        currency=command.currency,
        return_measurement_method=command.return_measurement_method,
        periods=period_results,
        period_count=period_count,
        cumulative_portfolio_return=cumulative_portfolio_return,
        cumulative_benchmark_return=cumulative_benchmark_return,
        annualized_portfolio_return=annualized_portfolio_return,
        annualized_benchmark_return=annualized_benchmark_return,
        annualized_excess_return=annualized_excess_return,
        annualized_tracking_error=tracking_error,
        information_ratio=information_ratio,
        beta=beta,
        annualized_jensen_alpha=annualized_jensen_alpha,
        total_realized_gain_loss=float(sum(period.realized_gain_loss for period in command.periods)),
        total_unrealized_gain_loss_change=float(sum(period.unrealized_gain_loss_change for period in command.periods)),
        total_fees=float(sum(period.fees for period in command.periods)),
    )
