"""Regression checks that historical discovery masks are not lost to date-panel changes.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_history_inventory.py
"""

from __future__ import annotations

import json
from pathlib import Path

import inventory_history as history
import pandas as pd


def historical_run(tmp_path: Path, grouping: str = "default") -> tuple[dict, Path]:
    """Create complete synthetic 12-window international discovery evidence."""
    dates = pd.date_range("2025-07-31", periods=12, freq="BME").date.astype(str).tolist()
    tickers = history.data.FAMILIES["international_equity"]
    manifest = {"data": {"training_universe": tickers}, "rebalance_dates": dates,
                "us_equity_first_round_wrapper": {"time_index_included": True,
                    "covar_grouping_proposal": grouping, "etf_columns": ["mom6m"], "macro_columns": ["oil_ret"]}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    rows = [{"date": date, "asset_id": ticker, "y_true": 0.01, "y_pred": 0.005, "y_std": 0.02}
            for date in dates for ticker in tickers]
    pd.DataFrame(rows).to_csv(tmp_path / "gp_window_predictions.csv", index=False)
    return manifest, path


def test_full_twelve_month_mask_is_eligible(tmp_path: Path) -> None:
    """Changing to a 24-month evaluation panel does not invalidate an old feature mask."""
    manifest, path = historical_run(tmp_path)
    family, reason = history.historical_mask(manifest, path, {"mom6m", "oil_ret"})
    assert family == "international_equity"
    assert reason is None


def test_international_hardcoded_split_screen_excluded(tmp_path: Path) -> None:
    """Known invalid historical grouping cannot support feature-importance evidence."""
    manifest, path = historical_run(tmp_path, "split_groups")
    _, reason = history.historical_mask(manifest, path, {"mom6m", "oil_ret"})
    assert reason is not None and "hard-coded" in reason


def test_unsupported_column_is_quarantined_not_substituted(tmp_path: Path) -> None:
    """A missing feature requires an explicit disposition, not an altered mask."""
    manifest, path = historical_run(tmp_path)
    _, reason = history.historical_mask(manifest, path, {"mom6m"})
    assert reason is not None and reason.startswith("quarantined") and "oil_ret" in reason


def test_macro_only_history_is_not_invalidated_by_an_invented_local_block_rule(tmp_path: Path) -> None:
    """The user's historical rebaseline scope includes valid macro-only configurations."""
    manifest, path = historical_run(tmp_path)
    manifest["us_equity_first_round_wrapper"]["etf_columns"] = []
    family, reason = history.historical_mask(manifest, path, {"oil_ret"})
    assert family == "international_equity"
    assert reason is None
