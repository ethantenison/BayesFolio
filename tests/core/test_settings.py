"""Tests for shared BayesFolio settings models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bayesfolio.core.settings import RiskfolioConfig


def test_riskfolio_config_alpha_default_and_explicit_value() -> None:
    assert RiskfolioConfig().alpha == 0.05
    assert RiskfolioConfig(alpha=0.5).alpha == 0.5


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.1])
def test_riskfolio_config_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValidationError):
        RiskfolioConfig(alpha=alpha)
