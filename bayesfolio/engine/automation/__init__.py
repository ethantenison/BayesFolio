"""Automation workflows for scheduled BayesFolio execution."""

from bayesfolio.engine.automation.monthly_portfolio import (
    MonthlyPortfolioRuntimeDependencies,
    execute_monthly_portfolio,
    run_monthly_portfolio,
)
from bayesfolio.engine.automation.monthly_ports import MonthlyPortfolioRunIO

__all__ = [
    "MonthlyPortfolioRunIO",
    "MonthlyPortfolioRuntimeDependencies",
    "execute_monthly_portfolio",
    "run_monthly_portfolio",
]
