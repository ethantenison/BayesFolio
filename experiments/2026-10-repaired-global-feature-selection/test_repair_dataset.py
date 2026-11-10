"""Synthetic offline checks for repaired-data causality, units, and completeness.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_repair_dataset.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("repaired_dataset", Path(__file__).with_name("repair_dataset.py"))
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def test_credit_units_and_trailing_transform() -> None:
    """Credit transforms use percent-to-decimal units and only past observations."""
    dates = pd.date_range("2015-01-30", periods=30, freq="BME")
    series = pd.Series(np.arange(30, dtype=float) + 3, index=dates)
    result = repair.monthly_hy(series, dates)
    assert result.iloc[0].hy_spread == pytest.approx(0.03)
    assert result.iloc[1].hy_spread_chg_1p == pytest.approx(0.01)
    assert result.hy_spread_z_12p.iloc[:11].isna().all()
    changed = series.copy()
    changed.iloc[20:] *= 10
    pd.testing.assert_frame_equal(result.iloc[:20], repair.monthly_hy(changed, dates).iloc[:20])


def test_alignment_shifts_each_task_once_and_keeps_labels() -> None:
    """Task boundaries, alias columns, and target lags remain causally aligned."""
    dates = pd.date_range("2017-01-31", "2026-04-30", freq="BME")
    rows = [{"date": date, "asset_id": ticker, "y_excess_lead": index / 1000}
            for ticker in repair.TICKERS for index, date in enumerate(dates)]
    labels = pd.DataFrame(rows)
    etf = labels[["date", "asset_id"]].copy()
    etf["mom12m_skip1m"] = np.tile(np.arange(len(dates)), len(repair.TICKERS))
    macro = pd.DataFrame({"date": dates})
    for original in repair.ALIASES.values():
        macro[original] = np.arange(len(dates), dtype=float)
    panel = repair.align_panel(etf, macro, labels)
    first = panel[panel.date.eq(repair.DATES[0])]
    assert len(first) == 15
    assert first.mom12m_skip1m.eq(3).all()
    assert first.y_excess_lead.eq(0.004).all()
    assert first.lag_y_excess_lead.eq(0.003).all()
    assert first.hy_spread_z_12p.eq(3).all()
    np.testing.assert_array_equal(panel.hy_spread_z_12m, panel.hy_spread_z_12p)
    assert panel.groupby("asset_id").size().eq(108).all()


def complete_panel() -> pd.DataFrame:
    """Return a varying, complete synthetic 15-task monthly panel."""
    index = pd.MultiIndex.from_product([repair.DATES, sorted(repair.TICKERS)], names=["date", "asset_id"])
    frame = index.to_frame(index=False)
    frame["x"] = np.arange(len(frame), dtype=float)
    frame["y_excess_lead"] = frame.x / 100000
    return frame


def test_coverage_requires_all_84_training_months() -> None:
    """A nominal date span cannot conceal an omitted training observation."""
    frame = complete_panel()
    assert repair.validate_panel(frame)["passed"]
    with pytest.raises(ValueError, match="108 dates"):
        repair.validate_panel(frame.iloc[1:])


def test_nonfinite_and_placeholder_columns_rejected() -> None:
    """Unknown values and wholly zero columns cannot pass the fit gate."""
    frame = complete_panel()
    frame.loc[5, "x"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        repair.validate_panel(frame)
    frame = complete_panel()
    frame["hy_spread"] = 0.0
    with pytest.raises(ValueError, match="constant"):
        repair.validate_panel(frame)


def test_global_limit_and_requested_family_round_size() -> None:
    """The accepted goal amendment does not expand the global loop limit."""
    assert repair.STUDY["max_new_family_configurations_per_round"] == 12
    assert repair.STUDY["max_new_global_configurations_per_round"] == 6
    assert repair.STUDY["max_workers_total"] == 15


def test_missing_inflation_is_not_fabricated_or_interpolated() -> None:
    """An unavailable CPI month holds a prior signal and never uses a future level."""
    dates = pd.date_range("2023-01-01", "2026-03-01", freq="MS")
    series = pd.Series(np.arange(len(dates), dtype=float) + 250, index=dates).drop(pd.Timestamp("2025-10-01"))
    signals, audit = repair.inflation_signals(series)
    assert signals.loc["2025-10-31", "cpi_chg_1p"] == pytest.approx(signals.loc["2025-09-30", "cpi_chg_1p"])
    assert signals.loc["2025-11-28", "cpi_chg_1p"] == pytest.approx(signals.loc["2025-09-30", "cpi_chg_1p"])
    assert audit["missing_source_months"] == ["2025-10-31"]
    changed = series.copy()
    changed.loc["2025-11-01":] *= 5
    changed_signals, _ = repair.inflation_signals(changed)
    pd.testing.assert_frame_equal(signals.loc[:"2025-10-31"], changed_signals.loc[:"2025-10-31"])
