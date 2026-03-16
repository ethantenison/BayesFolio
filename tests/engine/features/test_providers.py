"""Tests for canonical feature-provider wiring.

These tests verify the shared provider factory used by BayesFolio feature
workflows. Inputs are cache-root and retry configuration; outputs are provider
instances with the expected enhanced-macro defaults and cache locations.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from bayesfolio.engine.features import make_default_feature_providers
from bayesfolio.engine.features.market_fundamentals import fetch_enhanced_macro_features
from bayesfolio.io import EtfFeaturesProvider, MacroProvider, ReturnsProvider


def test_make_default_feature_providers_uses_enhanced_macro_defaults() -> None:
    """Build providers with the shared enhanced-macro default configuration."""

    providers = make_default_feature_providers(cache_root="artifacts/cache")
    returns_provider = cast(ReturnsProvider, providers.returns_provider)
    etf_features_provider = cast(EtfFeaturesProvider, providers.etf_features_provider)
    macro_provider = cast(MacroProvider, providers.macro_provider)

    assert isinstance(returns_provider, ReturnsProvider)
    assert isinstance(etf_features_provider, EtfFeaturesProvider)
    assert isinstance(macro_provider, MacroProvider)
    assert returns_provider._cache_dir == Path("artifacts/cache/returns")
    assert etf_features_provider._cache_dir == Path("artifacts/cache/etf_features")
    assert macro_provider._cache_dir == Path("artifacts/cache/macro_enhanced")
    assert macro_provider._fetcher is fetch_enhanced_macro_features
    assert macro_provider._max_retries == 2
    assert macro_provider._retry_backoff_seconds == 1.0


def test_make_default_feature_providers_allows_cacheless_runtime() -> None:
    """Disable local caches while preserving the shared enhanced-macro default."""

    providers = make_default_feature_providers(
        cache_root=None,
        macro_max_retries=1,
        macro_retry_backoff_seconds=0.0,
    )
    returns_provider = cast(ReturnsProvider, providers.returns_provider)
    etf_features_provider = cast(EtfFeaturesProvider, providers.etf_features_provider)
    macro_provider = cast(MacroProvider, providers.macro_provider)

    assert returns_provider._cache_dir is None
    assert etf_features_provider._cache_dir is None
    assert macro_provider._cache_dir is None
    assert macro_provider._fetcher is fetch_enhanced_macro_features
    assert macro_provider._max_retries == 1
    assert macro_provider._retry_backoff_seconds == 0.0
