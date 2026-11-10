"""Offline checks of trailing auxiliary signals and immutable cross-version data.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_feature_pool.py
"""

from __future__ import annotations

import json

import extend_feature_pool as extension
import numpy as np
import pandas as pd
import pytest
import workflow


def test_additional_macro_signals_do_not_use_future_prices() -> None:
    """Changing future price observations cannot alter earlier monthly signals."""
    dates = pd.bdate_range("2016-01-01", "2018-12-31")
    sources = {}
    for index, name in enumerate(["SPY", "^VIX", "^VIX3M", "RSP", "CL=F", "HG=F", "GC=F", "CEW"]):
        values = 20 + index + np.arange(len(dates)) / 100 + np.sin(np.arange(len(dates)) / 15)
        sources[name] = pd.DataFrame({"Adj Close": values}, index=dates)
    original = extension.extra_macro(sources)
    changed = {name: frame.copy() for name, frame in sources.items()}
    for frame in changed.values():
        frame.loc["2018-01-01":, "Adj Close"] *= 3
    modified = extension.extra_macro(changed)
    pd.testing.assert_frame_equal(original[original.date.lt("2018-01-01")],
                                  modified[modified.date.lt("2018-01-01")])


def test_gamma_proxy_uses_annual_realized_minus_decimal_implied_vol() -> None:
    """Existing proxy units/formula are retained instead of substituting another gamma measure."""
    dates = pd.bdate_range("2017-01-01", "2017-03-31")
    prices = pd.Series(100 * np.exp(np.arange(len(dates)) * 0.001), index=dates)
    names = ["SPY", "^VIX", "^VIX3M", "RSP", "CL=F", "HG=F", "GC=F", "CEW"]
    sources = {name: pd.DataFrame({"Adj Close": prices}) for name in names}
    sources["^VIX"]["Adj Close"] = 20.0
    result = extension.extra_macro(sources)
    assert result.dealer_gamma_proxy.iloc[-1] == pytest.approx(-0.2, abs=1e-10)


def test_frozen_feature_extension_retains_every_shared_cell() -> None:
    """The actual experiment versions share identical labels, features, dates, and task rows."""
    v1 = pd.read_parquet(extension.data.ARTIFACTS / "global.parquet")
    v2 = pd.read_parquet(extension.OUTPUT / "global.parquet")
    pd.testing.assert_frame_equal(v2[v1.columns], v1, check_exact=True)
    meta = json.loads((extension.OUTPUT / "dataset.manifest.json").read_text())
    assert meta["base_global_sha256"] == extension.data.digest(extension.data.ARTIFACTS / "global.parquet")
    available = workflow.validate_frozen_data()
    assert {*extension.EXTRA_ETF, *extension.EXTRA_MACRO}.issubset(available)


def test_local_predictor_cannot_be_relabelled_as_macro() -> None:
    """Fixed kernel-block membership also applies to the expanded volatility inputs."""
    from register_pool_history import pool_candidate, pool_config

    item = pool_candidate("us_equity", ["mom6m"], ["vol_1m"], ["Synthetic invalid block assignment"])
    config = pool_config("us_equity", [item], "test")
    with pytest.raises(ValueError, match="local lag cannot be injected"):
        workflow.validate_plan(config, extension.data.HERE / "plans/00_test_invalid.json",
                               workflow.validate_frozen_data())
