from __future__ import annotations

from bayesfolio.schemas.contracts.backtest import BacktestResult
from bayesfolio.schemas.contracts.report import ReportBundle


def assemble_report(backtest_result: BacktestResult) -> ReportBundle:
    """Build report bundle from a backtest result."""

    return ReportBundle(
        metadata=backtest_result.metadata,
        headline_metrics={
            "cumulative_return": backtest_result.cumulative_return,
            "annualized_return": backtest_result.annualized_return,
            "annualized_volatility": backtest_result.annualized_volatility,
            "sharpe_ratio": backtest_result.sharpe_ratio,
        },
    )
