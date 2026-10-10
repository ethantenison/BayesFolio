"""Tests for the deprecated engine feature-provider compatibility path."""

import pytest

from bayesfolio.engine.features import make_default_feature_providers


def test_legacy_provider_factory_delegates_with_a_warning() -> None:
    with pytest.warns(DeprecationWarning, match="bayesfolio.application"):
        providers = make_default_feature_providers(cache_root=None)

    assert providers.returns_provider is not None
    assert providers.macro_provider is not None
    assert providers.etf_features_provider is not None
