"""Deterministic guardrail tests for shared GP tuning and causal training windows."""

from __future__ import annotations

import campaign
import numpy as np
import pandas as pd
import pytest
from tuning_worker import training_panel


def test_rolling_training_retains_original_global_month_index() -> None:
    """A rolling window changes history, not the global seed/date key."""
    dates = pd.date_range("2017-05-01", periods=112, freq="BME")
    assets = [f"ETF{i:02d}" for i in range(15)]
    frame = pd.DataFrame([(date, asset, float(index)) for index, date in enumerate(dates) for asset in assets],
                         columns=["date", "asset_id", "y_excess_lead"])
    for index in [0, 27]:
        for window in [None, 48, 60, 84]:
            train, evaluation = training_panel(frame, str(dates[84 + index].date()), index, assets, window)
            assert train.date.nunique() == (window or 84 + index)
            assert train.date.max() < evaluation.date.min()
            assert train.y_excess_lead.max() == 83 + index
            assert np.isfinite(train.y_excess_lead).all()


def test_incomplete_or_renumbered_panels_are_rejected() -> None:
    """No gaps, unapproved windows or reset rolling indices may hide in a fit."""
    dates = pd.date_range("2017-05-01", periods=112, freq="BME")
    frame = pd.DataFrame({"date": dates, "asset_id": "ETF", "y_excess_lead": 1.0})
    with pytest.raises(ValueError, match="chronology"):
        training_panel(frame, str(dates[-1].date()), 0, ["ETF"], 48)
    with pytest.raises(ValueError, match="rolling"):
        training_panel(frame, str(dates[-1].date()), 27, ["ETF"], 36)
    with pytest.raises(ValueError, match="Incomplete"):
        training_panel(frame, str(dates[-1].date()), 27, ["ETF", "missing"], 48)


def test_worker_recycling_is_not_reenabled() -> None:
    """The shared executor must avoid this runtime's worker-recycling deadlock."""
    import inspect

    source = inspect.getsource(campaign.fit_campaign)
    assert "max_tasks_per_child=" not in source
