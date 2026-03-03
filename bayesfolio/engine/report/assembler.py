from __future__ import annotations

from bayesfolio.contracts.results.backtest import BacktestResult
from bayesfolio.contracts.results.report import ReportResult


def assemble_report(backtest_result: BacktestResult) -> ReportResult:
    """Build report result from a backtest result."""

    return ReportResult(
        headline_metrics={
            "cumulative_return": backtest_result.cumulative_return,
            "annualized_return": backtest_result.annualized_return,
            "annualized_volatility": backtest_result.annualized_volatility,
            "sharpe_ratio": backtest_result.sharpe_ratio,
        },
    )
