"""Application composition roots for supported BayesFolio workflows."""

from bayesfolio.application.feature_providers import make_default_feature_providers
from bayesfolio.application.monthly_decision import finalize_monthly_portfolio
from bayesfolio.application.monthly_portfolio import (
    build_default_monthly_portfolio_dependencies,
    complete_monthly_portfolio_dependencies,
    run_monthly_portfolio,
)
from bayesfolio.application.schwab_account import (
    DEFAULT_SCHWAB_REDIRECT_URI,
    SchwabAccountAccess,
    make_default_schwab_account_access,
)

__all__ = [
    "build_default_monthly_portfolio_dependencies",
    "complete_monthly_portfolio_dependencies",
    "finalize_monthly_portfolio",
    "make_default_feature_providers",
    "make_default_schwab_account_access",
    "run_monthly_portfolio",
    "DEFAULT_SCHWAB_REDIRECT_URI",
    "SchwabAccountAccess",
]
