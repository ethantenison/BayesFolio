"""Tests for deterministic monthly portfolio command defaults."""

import pytest
from pydantic import ValidationError

from bayesfolio.contracts.commands.monthly_portfolio import MonthlyForecastConfig


def test_monthly_forecast_uses_a_deterministic_default_seed() -> None:
    assert MonthlyForecastConfig().seed == 0


def test_monthly_forecast_rejects_an_absent_seed() -> None:
    with pytest.raises(ValidationError, match="seed"):
        MonthlyForecastConfig.model_validate({"seed": None})
