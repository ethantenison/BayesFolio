"""Deprecated compatibility path for application-level provider composition."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from bayesfolio.engine.features.dataset_builder import FeatureProviders
from bayesfolio.engine.features.market_fundamentals import fetch_enhanced_macro_features


def make_default_feature_providers(
    *,
    cache_root: str | Path | None = "artifacts/cache",
    macro_fetcher: Callable[..., pd.DataFrame] = fetch_enhanced_macro_features,
    macro_cache_subdir: str = "macro_enhanced_periodic",
    macro_max_retries: int = 2,
    macro_retry_backoff_seconds: float = 1.0,
) -> FeatureProviders:
    """Delegate legacy engine imports to the application composition root.

    Args:
        cache_root: Base cache directory, or ``None`` to disable caching.
        macro_fetcher: Macro feature fetch callable.
        macro_cache_subdir: Cache subdirectory for macro features.
        macro_max_retries: Maximum transient macro-fetch retries.
        macro_retry_backoff_seconds: Linear retry backoff base in seconds.

    Returns:
        Application-composed concrete feature providers.

    Warns:
        DeprecationWarning: Always; import this function from
            ``bayesfolio.application`` instead.
    """

    from bayesfolio.application.feature_providers import make_default_feature_providers as compose_providers

    warnings.warn(
        "bayesfolio.engine.features.make_default_feature_providers is deprecated; "
        "use bayesfolio.application.make_default_feature_providers instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compose_providers(
        cache_root=cache_root,
        macro_fetcher=macro_fetcher,
        macro_cache_subdir=macro_cache_subdir,
        macro_max_retries=macro_max_retries,
        macro_retry_backoff_seconds=macro_retry_backoff_seconds,
    )
