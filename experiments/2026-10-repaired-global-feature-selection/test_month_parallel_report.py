"""Guard conventional Sharpe and excess-path diagnostics in the parallel report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from report_month_parallel import conventional_stats


def test_conventional_sharpe_not_cagr_ratio() -> None:
    """Reporting uses monthly arithmetic excess mean and sample standard deviation."""
    returns = pd.Series([0.02, -0.01, 0.04, 0.00])
    result = conventional_stats(returns)
    assert result["sharpe"] == pytest.approx(returns.mean() / returns.std(ddof=1) * np.sqrt(12))
    assert result["cumulative_excess_return_diagnostic_only"] == pytest.approx((1 + returns).prod() - 1)


def test_first_month_loss_is_drawdown() -> None:
    """The initial unit capital belongs in the running-peak calculation."""
    result = conventional_stats(pd.Series([-0.10, 0.02, 0.03]))
    assert result["max_drawdown_excess_path_diagnostic_only"] == pytest.approx(-0.10)


def test_constant_return_sharpe_fails_closed() -> None:
    """Undefined Sharpe cannot be silently reported as zero or infinity."""
    with pytest.raises(ValueError):
        conventional_stats(pd.Series([0.01] * 24))
