"""Test paired scientific-plot inputs and decimal forecast diagnostics without fits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from plot_international_family_vs_joint import forecast_metrics, paired_panels


def _panel():
    return pd.DataFrame({"date": pd.to_datetime(["2024-05-31", "2024-06-28"]), "asset_id": ["VEA", "VEA"],
                         "y_true": [0.01, -0.02], "y_pred": [0.005, -0.01], "y_std": [0.02, 0.02]})


def test_identical_dates_and_labels_align_without_smoothing():
    frame = _panel()
    left, right = paired_panels(frame.iloc[::-1], frame, "VEA", ["2024-05-31", "2024-06-28"])
    pd.testing.assert_frame_equal(left.reset_index(drop=True), frame)
    pd.testing.assert_frame_equal(right.reset_index(drop=True), frame)


def test_changed_target_rejected():
    frame, changed = _panel(), _panel()
    changed.loc[0, "y_true"] += 0.001
    with pytest.raises(AssertionError):
        paired_panels(frame, changed, "VEA", ["2024-05-31", "2024-06-28"])


@pytest.mark.parametrize("bad", [0, np.nan, -0.01])
def test_invalid_predictive_spread_rejected(bad):
    frame = _panel()
    frame.loc[0, "y_std"] = bad
    with pytest.raises(ValueError, match="Invalid saved"):
        paired_panels(frame, frame, "VEA", ["2024-05-31", "2024-06-28"])


def test_duplicate_date_rejected():
    frame = _panel()
    frame.loc[1, "date"] = frame.loc[0, "date"]
    with pytest.raises(ValueError, match="duplicated"):
        paired_panels(frame, frame, "VEA", ["2024-05-31", "2024-06-28"])


def test_forecast_mean_variability_is_not_predictive_uncertainty():
    frame = _panel()
    metrics = forecast_metrics(frame)
    assert metrics["rmse"] == pytest.approx(np.sqrt((0.005**2 + 0.01**2) / 2))
    assert metrics["forecast_time_sd"] == pytest.approx(frame.y_pred.std(ddof=1))
    assert metrics["mean_predictive_sd"] == pytest.approx(0.02)
