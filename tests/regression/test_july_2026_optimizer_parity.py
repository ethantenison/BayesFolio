"""Regression gate for the July 2026 optimizer-stage portfolio weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bayesfolio.contracts.commands.optimize import OptimizeCommand
from bayesfolio.engine.asset_allocation.riskfolio_adapter import optimize_from_historical_returns

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "regression" / "july_2026"


def test_july_2026_optimizer_weights_match_golden_fixture() -> None:
    scenarios = pd.read_csv(FIXTURE_DIR / "scenarios.csv")
    expected = pd.read_csv(FIXTURE_DIR / "expected_weights.csv")
    expected_assets = expected["asset"].tolist()

    assert scenarios.columns.tolist() == expected_assets
    request = OptimizeCommand(
        objective="Sharpe",
        risk_measure="CVaR",
        model="Classic",
        rf=0.0,
        alpha=0.5,
        kelly=None,
        min_weight=0.0,
        max_weight=0.25,
        nea=10,
        hist=True,
    )

    result = optimize_from_historical_returns(scenarios, request)
    actual_weights = np.asarray(result.weights, dtype=float)
    expected_weights = expected["weight"].to_numpy(dtype=float)

    assert result.asset_order == expected_assets
    assert np.isfinite(actual_weights).all()
    assert (actual_weights >= 0.0).all()
    assert abs(float(actual_weights.sum()) - 1.0) <= 1e-12
    np.testing.assert_allclose(actual_weights, expected_weights, rtol=0.0, atol=1e-10)
    assert float(np.abs(actual_weights - expected_weights).sum()) <= 1e-9
