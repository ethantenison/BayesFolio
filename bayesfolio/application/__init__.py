"""Application composition roots for supported BayesFolio workflows."""

from bayesfolio.application.feature_providers import make_default_feature_providers
from bayesfolio.application.monthly_decision import finalize_monthly_portfolio
from bayesfolio.application.monthly_portfolio import (
    build_default_monthly_portfolio_dependencies,
    complete_monthly_portfolio_dependencies,
    run_monthly_portfolio,
)
from bayesfolio.application.portfolio_adjustment import (
    DEFAULT_LOCAL_QWEN_BASE_URL,
    DEFAULT_LOCAL_QWEN_MODEL,
    make_local_qwen_adjustment_model,
    run_portfolio_adjustment,
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
    "make_local_qwen_adjustment_model",
    "make_default_feature_providers",
    "make_default_schwab_account_access",
    "run_monthly_portfolio",
    "run_portfolio_adjustment",
    "DEFAULT_SCHWAB_REDIRECT_URI",
    "DEFAULT_LOCAL_QWEN_BASE_URL",
    "DEFAULT_LOCAL_QWEN_MODEL",
    "SchwabAccountAccess",
]
