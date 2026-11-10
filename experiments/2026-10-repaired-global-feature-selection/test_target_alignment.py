"""Offline integration checks that validation-only future prices never alter inputs.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_target_alignment.py
"""

from __future__ import annotations

import audit_target_alignment as audit
import numpy as np
import pandas as pd


def test_target_endpoint_tail_preserves_frozen_feature_prices() -> None:
    """Extra outcome observations are isolated from the predictor source snapshots."""
    sources = audit.data.market_sources()
    prices, complete, scales = audit.audit_price_series(sources)
    assert complete
    assert scales["^IRX"] == 1.0
    for ticker in [*audit.data.TICKERS, "^IRX"]:
        original = sources[ticker]["Adj Close"].dropna()
        np.testing.assert_array_equal(prices[ticker].loc[original.index].to_numpy(), original.to_numpy())
        assert original.index.max() == pd.Timestamp("2026-04-30")
        assert prices[ticker].index.max() == pd.Timestamp("2026-05-29")


def test_full_reconstruction_audit_covers_all_preserved_targets() -> None:
    """No final-month cache-only labels are silently omitted from source validation."""
    import json

    evidence = json.loads((audit.data.HERE / "results/target_alignment_audit.json").read_text())
    assert evidence["passed"]
    assert evidence["labels_unchanged"]
    assert evidence["independent_reconstruction_total"] == 1620
    assert evidence["remaining_cache_only_forward_targets"] == 0
    assert all(row["independently_rebuilt_observations"] == 108 for row in evidence["by_asset"])
