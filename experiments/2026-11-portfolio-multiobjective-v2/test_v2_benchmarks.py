"""Synthetic-only benchmark, cost accounting, and static stress verification."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import v2_benchmarks as benchmarks
from v2_common import ASSETS, ELIGIBLE


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _snapshot(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


@pytest.fixture
def synthetic_context(tmp_path, monkeypatch):
    """Replace every source/outcome with artificial local arrays and metadata."""
    source, _ = benchmarks._v1_benchmarks()
    current, old = tmp_path / "v2", tmp_path / "v1"
    current.mkdir()
    (current / "SHORTLIST_FROZEN.json").write_text("{}")
    (current / "v2_metrics.py").write_text("synthetic metrics provenance")
    monkeypatch.setattr(benchmarks, "HERE", current)
    monkeypatch.setattr(benchmarks, "V1", old)
    monkeypatch.setattr(benchmarks, "save", _write)
    monkeypatch.setattr(benchmarks, "snapshot", _snapshot)
    dates = pd.date_range("2022-05-31", periods=53, freq="BME").strftime("%Y-%m-%d").to_numpy(dtype=str)
    actual = np.tile(np.linspace(-.03, .03, 15), (53, 1))
    actual += .01 * np.sin(np.arange(53))[:, None]
    rf = np.full(53, .002)
    context = {"dates": dates, "assets": np.array(ASSETS), "actual": actual, "rf": rf}
    weights = np.zeros((53, 16))
    for asset, weight in [("SPY", .5), ("IWM", .2), ("VEA", .25)]:
        weights[:, ASSETS.index(asset)] = weight
    weights[:, -1] = .05
    total = source.total_returns(actual, rf, weights)
    prior = old / "outputs/benchmarks/schwab_aggressive.npz"
    _snapshot(prior, {"dates": dates, "weights": weights, "total_returns": total,
                      "wealth": 10000 * np.r_[1., np.cumprod(1 + total)]})
    audit = old / "outputs/benchmarks/AUDIT.json"
    _write(audit, {"input_sha256": {}, "vintage_note": "synthetic frozen vintage"})
    schedule = old / "weights_schedule.json"
    _write(schedule, {"proxy_order": ["SPY", "IWM", "VEA", "BND", "CASH"], "profiles": {"aggressive": {
        "initial_weights": [.5, .2, .25, 0, .05], "verified_revisions": []}}})
    seal = {"passed": True, "artifact_sha256": {str(path.resolve()): benchmarks.digest(path)
                                               for path in [prior, audit]}}
    _write(old / "COMPLETED.json", seal)
    observed = pd.date_range("2021-05-31", periods=65, freq="BME").to_numpy("datetime64[D]")
    prices = np.repeat(np.linspace(100., 80., 65)[:, None], 14, axis=1)
    called = []

    def history(requested, assets):
        called.append(assets)
        assert requested == dates.tolist()
        return observed, prices, {}

    module = SimpleNamespace(SCHEDULE=schedule, _history=history,
        momentum_from_prices=source.momentum_from_prices, total_returns=source.total_returns,
        _scheduled=lambda origins, initial, revisions: {"weights": np.tile(initial, (len(origins), 1)),
                                                       "applied_revision_count": 0})
    monkeypatch.setattr(benchmarks, "_v1_benchmarks", lambda: (module, {}))
    return context, current, old, called


def test_confirmation_gate_blocks_benchmarks_and_stress(tmp_path, monkeypatch):
    """Both entry points reject scoring before the frozen shortlist exists."""
    monkeypatch.setattr(benchmarks, "HERE", tmp_path)
    with pytest.raises(ValueError, match="Freeze the final shortlist"):
        benchmarks.build_benchmarks({})
    with pytest.raises(ValueError, match="Freeze the final shortlist"):
        benchmarks.stress(np.ones(15) / 15)


def test_synthetic_benchmark_parity_momentum_exclusion_and_saved_paths(synthetic_context):
    """Two benchmarks retain Schwab parity and rank exactly seven eligible assets."""
    context, current, _old, called = synthetic_context
    result = benchmarks.build_benchmarks(context)
    assert result["passed"]
    assert list(result["rows"]) == ["schwab_aggressive", "momentum_top7"]
    assert called == [ELIGIBLE]
    momentum = result["paths"]["momentum_top7"]
    np.testing.assert_array_equal(momentum["weights"][:, ASSETS.index("MGK")], 0)
    np.testing.assert_array_equal(momentum["weights"][:, -1], 0)
    assert (np.count_nonzero(momentum["weights"], axis=1) == 7).all()
    assert np.all(momentum["momentum_scores"] < 0)
    assert sorted(np.array(ASSETS)[momentum["weights"][0, :-1] > 0].tolist()) == sorted(ELIGIBLE)[:7]
    for name, values in result["paths"].items():
        assert values["weights"].shape == (53, 16)
        assert values["wealth"].shape == values["drawdown"].shape == (54,)
        assert result["rows"][name]["confirmation12"]["months"] == 12
        assert (current / "outputs/benchmarks" / f"{name}.npz").exists()
        np.testing.assert_allclose(values["cost_0bps_total_returns"], values["total_returns"], rtol=0, atol=1e-14)
    assert result["audit"]["schwab_exact_frozen53_weight_return_wealth_parity"]


def test_changed_schwab_frozen_returns_are_not_substituted(synthetic_context):
    """A protected-but-inconsistent prior benchmark fails exact accounting parity."""
    context, _current, old, _called = synthetic_context
    prior_path = old / "outputs/benchmarks/schwab_aggressive.npz"
    values = benchmarks.arrays(prior_path)
    values["total_returns"][0] += .01
    _snapshot(prior_path, values)
    seal_path = old / "COMPLETED.json"
    seal = benchmarks.read(seal_path)
    seal["artifact_sha256"][str(prior_path)] = benchmarks.digest(prior_path)
    _write(seal_path, seal)
    with pytest.raises(AssertionError):
        benchmarks.build_benchmarks(context)


def test_transaction_costs_charge_etf_notional_but_no_cash_fee():
    """Cash earns RF fee-free; initial ETF purchase obeys self-financing arithmetic."""
    returns = np.array([[.03, .002], [-.1, .002]])
    cash = benchmarks._cost_path(returns, np.array([[0., 1.], [0., 1.]]), 20)
    np.testing.assert_allclose(cash["total_returns"], .002, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(cash["cost_fraction"], 0)
    invested = benchmarks._cost_path(returns, np.array([[.8, .2], [.8, .2]]), 10)
    expected_fraction = 1 / (1 + .001 * .8)
    assert invested["post_cost_wealth_fraction"][0] == pytest.approx(expected_fraction, abs=1e-14)
    assert invested["ETF_traded_notional_fraction"][0] == pytest.approx(.8 * expected_fraction, abs=1e-14)
    assert invested["total_returns"][0] == pytest.approx(expected_fraction * (1 + .8 * .03 + .2 * .002) - 1,
                                                         abs=1e-14)


def test_static_stress_uses_fixed_targets_and_explicit_shock_buckets(tmp_path, monkeypatch):
    """Static path stress includes2020 and cannot be confused with GP decisions."""
    monkeypatch.setattr(benchmarks, "HERE", tmp_path)
    monkeypatch.setattr(benchmarks, "save", _write)
    monkeypatch.setattr(benchmarks, "snapshot", _snapshot)
    (tmp_path / "SHORTLIST_FROZEN.json").write_text("{}")
    dates = pd.date_range("2019-12-31", periods=14, freq="BME").strftime("%Y-%m-%d").to_numpy(dtype=str)
    returns = np.full((14, 14), .01)
    returns[np.char.startswith(dates.astype(str), "2020-03")] = -.2
    monkeypatch.setattr(benchmarks, "_stress_history", lambda: (dates, returns, {"synthetic": True}))
    weights = np.zeros(15)
    weights[ASSETS.index("SPY")], weights[ASSETS.index("HYG")] = .8, .2
    result = benchmarks.stress(weights)
    assert result["march2020"]["available"]
    assert result["march2020"]["return"] == pytest.approx(-.2, abs=1e-14)
    assert result["worst_windows"]["3"]["return"] == pytest.approx(.8 * 1.01**2 - 1, abs=1e-14)
    assert result["hypothetical_shocks"]["equity20_FI5"]["portfolio_return"] == pytest.approx(-.17, abs=1e-14)
    assert result["hypothetical_shocks"]["equity30_HY15_otherFI5"]["portfolio_return"] == pytest.approx(-.27,
                                                                                                        abs=1e-14)
    assert Path(result["artifact_path"]).exists()
    assert any("no historical GP forecasts" in text for text in result["assumptions"])
    weights[ASSETS.index("MGK")] = .01
    with pytest.raises(ValueError, match="exactlyzero MGK"):
        benchmarks.stress(weights)


def test_stress_history_rejects_changed_sealed_prices(tmp_path, monkeypatch):
    """Static stress history uses exact sealed cells and preserves common months."""
    monkeypatch.setattr(benchmarks, "V1", tmp_path)
    dates = pd.date_range("2019-01-31", periods=15, freq="BME")
    hashes = {}
    for asset in ELIGIBLE:
        path = tmp_path / "data/raw/combined" / f"{asset}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Adj Close": np.linspace(100., 110., 15)}, index=dates).to_parquet(path)
        hashes[str(path.resolve())] = benchmarks.digest(path)
    _write(tmp_path / "data/COMPLETED.json", {"passed": True, "artifact_sha256": hashes})
    observed, returns, audit = benchmarks._stress_history()
    assert returns.shape == (14, 14)
    assert observed[-1].startswith("2020-03")
    assert audit["no_interior_months_dropped"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed or unsealed"):
        benchmarks._stress_history()
