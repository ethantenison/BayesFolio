from __future__ import annotations

import pandas as pd

from bayesfolio.contracts.results.optimize import OptimizeResult
from bayesfolio.engine.backtest.runner import run_weighted_backtest


def test_run_weighted_backtest_includes_new_metrics_and_non_positive_drawdown() -> None:
    realized_returns = pd.DataFrame(
        {
            "SPY": [0.02, -0.01, 0.015, -0.02, 0.01],
            "QQQ": [0.01, -0.015, 0.02, -0.01, 0.005],
        }
    )
    optimization = OptimizeResult(asset_order=["SPY", "QQQ"], weights=[0.6, 0.4])

    result = run_weighted_backtest(realized_returns=realized_returns, optimization=optimization, periods_per_year=12)

    assert isinstance(result.max_drawdown, float)
    assert isinstance(result.calmar_ratio, float)
    assert isinstance(result.sortino_ratio, float)
    assert result.max_drawdown <= 0.0
