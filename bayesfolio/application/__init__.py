"""Application composition roots for supported BayesFolio workflows."""

from bayesfolio.application.feature_providers import make_default_feature_providers
from bayesfolio.application.monthly_decision import finalize_monthly_portfolio
from bayesfolio.application.monthly_portfolio import (
    build_default_monthly_portfolio_dependencies,
    complete_monthly_portfolio_dependencies,
    run_monthly_portfolio,
)

__all__ = [
    "build_default_monthly_portfolio_dependencies",
    "complete_monthly_portfolio_dependencies",
    "finalize_monthly_portfolio",
    "make_default_feature_providers",
    "run_monthly_portfolio",
]
