"""Canonical feature-provider wiring for BayesFolio workflows.

Layer responsibility: this module centralizes the standard composition of IO
providers used by feature-building flows, while keeping business logic in the
engine layer and retrieval/caching behavior in injected IO adapters. Inputs are
cache-root configuration and provider retry settings; outputs are configured
providers for decimal-unit return labels and ETF/macro feature tables.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from bayesfolio.engine.features.asset_prices import build_long_panel, fetch_etf_features
from bayesfolio.engine.features.dataset_builder import FeatureProviders
from bayesfolio.engine.features.market_fundamentals import fetch_enhanced_macro_features
from bayesfolio.io import EtfFeaturesProvider, MacroProvider, ReturnsProvider

DEFAULT_MACRO_CACHE_SUBDIR = "macro_enhanced"


def make_default_feature_providers(
    *,
    cache_root: str | Path | None = "artifacts/cache",
    macro_fetcher: Callable[..., pd.DataFrame] = fetch_enhanced_macro_features,
    macro_cache_subdir: str = DEFAULT_MACRO_CACHE_SUBDIR,
    macro_max_retries: int = 2,
    macro_retry_backoff_seconds: float = 1.0,
) -> FeatureProviders:
    """Build the standard provider container for BayesFolio feature workflows.

    Args:
        cache_root: Base cache directory. When provided, return labels, ETF
            features, and macro features are cached under dataset-specific
            subdirectories beneath this root. When ``None``, all providers run
            without local parquet caching.
        macro_fetcher: Macro feature fetch callable used by the macro provider.
            The default is the enhanced macro feature set.
        macro_cache_subdir: Cache subdirectory name for macro features.
            Defaults to ``"macro_enhanced"`` to keep the richer schema
            isolated from legacy reduced-schema caches.
        macro_max_retries: Maximum retry attempts for transient macro fetch
            failures.
        macro_retry_backoff_seconds: Linear retry backoff base in seconds for
            transient macro fetch failures.

    Returns:
        FeatureProviders configured with the shared BayesFolio defaults for
        decimal-unit return labels, ETF features, and enhanced macro features.
    """

    cache_base = Path(cache_root) if cache_root is not None else None
    return FeatureProviders(
        returns_provider=ReturnsProvider(
            fetcher=build_long_panel,
            cache_dir=_resolve_cache_dir(cache_base, "returns"),
        ),
        macro_provider=MacroProvider(
            fetcher=macro_fetcher,
            max_retries=macro_max_retries,
            retry_backoff_seconds=macro_retry_backoff_seconds,
            cache_dir=_resolve_cache_dir(cache_base, macro_cache_subdir),
        ),
        etf_features_provider=EtfFeaturesProvider(
            fetcher=fetch_etf_features,
            cache_dir=_resolve_cache_dir(cache_base, "etf_features"),
        ),
    )


def _resolve_cache_dir(cache_root: Path | None, dataset: str) -> str | None:
    if cache_root is None:
        return None
    return str(cache_root / dataset)
