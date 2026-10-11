"""Focused tests for the Riskfolio optimization adapter."""

from __future__ import annotations

import pandas as pd
import pytest

from bayesfolio.contracts.commands.optimize import OptimizeCommand
from bayesfolio.engine.asset_allocation import riskfolio_adapter


def test_historical_optimizer_forwards_alpha_and_optimizer_settings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakePortfolio:
        def __init__(self, *, returns: pd.DataFrame, nea: int) -> None:
            captured["returns"] = returns
            captured["nea"] = nea
            self.alpha = 0.05
            self.card = object()
            self.upperlng = None
            self.lowerlng = None

        def assets_stats(self, *, method_mu: str, method_cov: str) -> None:
            captured["alpha_at_assets_stats"] = self.alpha
            captured["method_mu"] = method_mu
            captured["method_cov"] = method_cov
            captured["card"] = self.card
            captured["upperlng"] = self.upperlng
            captured["lowerlng"] = self.lowerlng

        def optimization(self, **kwargs: object) -> pd.DataFrame:
            captured["optimization"] = kwargs
            return pd.DataFrame(
                {"weights": [0.5, 0.3, 0.2]},
                index=["SPY", "VTV", "IEF"],
            )

    monkeypatch.setattr(riskfolio_adapter.rp, "Portfolio", _FakePortfolio)
    returns = pd.DataFrame(
        {
            "SPY": [0.01, -0.02, 0.03],
            "VTV": [0.02, -0.01, 0.01],
            "IEF": [-0.01, 0.02, 0.00],
        }
    )
    request = OptimizeCommand(
        objective="Sharpe",
        risk_measure="CVaR",
        model="Classic",
        rf=0.01,
        alpha=0.5,
        kelly="approx",
        min_weight=0.01,
        max_weight=0.5,
        nea=8,
        hist=False,
    )

    result = riskfolio_adapter.optimize_from_historical_returns(returns, request)

    assert captured["nea"] == 2
    assert captured["alpha_at_assets_stats"] == 0.5
    assert captured["method_mu"] == "hist"
    assert captured["method_cov"] == "hist"
    assert captured["card"] is None
    assert captured["upperlng"] == 0.5
    assert captured["lowerlng"] == 0.01
    assert captured["optimization"] == {
        "model": "Classic",
        "rm": "CVaR",
        "obj": "Sharpe",
        "kelly": "approx",
        "rf": 0.01,
        "hist": False,
    }
    assert result.asset_order == ["SPY", "VTV", "IEF"]
    assert result.weights == pytest.approx([0.5, 0.3, 0.2])
