"""Deterministic independent raw-weight and final-target availability audits."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_validation import (  # noqa: E402
    ASSETS,
    validate_portfolio,
    validate_training_availability,
    validate_weights_csv,
)


def _portfolio() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    means = np.linspace(-.05, .05, 15)
    covariance = np.eye(15) * .002
    weights = np.zeros(15)
    scores = np.arange(7, 16, dtype=float)
    weights[6:] = 1 / 9 + np.sqrt((1 / 7 - 1 / 9) / np.sum((scores - scores.mean())**2)) * (
        scores - scores.mean())
    return means, covariance, .001, weights


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2017-05-31", "2026-10-30", freq="BME")
    frame = pd.DataFrame([(date, ticker, .01, month) for month, date in enumerate(dates) for ticker in ASSETS],
                         columns=["date", "asset_id", "y_excess_lead", "t_index"])
    frame.loc[frame.date.eq(pd.Timestamp("2026-10-30")), "y_excess_lead"] = np.nan
    return frame


def test_known_analytic_weights_match_original_strict_conic() -> None:
    """Independent closed-form NE7 solution agrees with the frozen conic solver."""
    mean, covariance, budget, weights = _portfolio()
    before = weights.copy()
    audit = validate_portfolio(mean, covariance, budget, weights, ASSETS)
    assert audit["passed"]
    assert audit["original_conic"]["solver_status"] == "optimal"
    assert audit["maximum_absolute_weight_difference"] < 1e-6
    assert audit["raw_checks"]["effective_assets"] == pytest.approx(7, abs=1e-12)
    np.testing.assert_array_equal(weights, before)


@pytest.mark.parametrize("fault", ["sum", "negative", "concentration", "variance", "nan", "asymmetric"])
def test_raw_constraint_failures_are_not_repaired(fault: str) -> None:
    """Invalid raw weights/moments fail rather than being clipped or normalized."""
    mean, covariance, budget, weights = _portfolio()
    if fault == "sum":
        weights *= .9
    elif fault == "negative":
        weights[0], weights[-1] = -.001, weights[-1] + .001
    elif fault == "concentration":
        weights = np.r_[1., np.zeros(14)]
    elif fault == "variance":
        budget = 1e-7
    elif fault == "nan":
        mean[0] = np.nan
    else:
        covariance[0, 1] = .0001
    with pytest.raises(ValueError):
        validate_portfolio(mean, covariance, budget, weights, ASSETS)


def test_feasible_but_nonoptimal_weights_fail_original_parity() -> None:
    """Feasibility alone cannot qualify a substituted equal-weight portfolio."""
    mean, covariance, budget, _ = _portfolio()
    with pytest.raises(ValueError, match="parity"):
        validate_portfolio(mean, covariance, budget, np.full(15, 1 / 15), ASSETS)


def test_csv_preserves_all_assets_raw_precision_and_cutoff(tmp_path: Path) -> None:
    """The delivered CSV, not just its unrounded in-memory predecessor, is checked."""
    mean, covariance, budget, weights = _portfolio()
    path = tmp_path / "NOVEMBER_WEIGHTS.csv"
    frame = pd.DataFrame({"asset_id": ASSETS, "weight": weights, "market_origin": "2026-10-30"})
    frame.to_csv(path, index=False, float_format="%.17g")
    result = validate_weights_csv(path, mean, covariance, budget, weights, ASSETS,
                                  expected_metadata={"market_origin": "2026-10-30"})
    assert result["passed"]
    assert len(result["zero_weight_assets"]) == 6
    frame.loc[0, "weight"] = .001
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="differ"):
        validate_weights_csv(path, mean, covariance, budget, weights, ASSETS)


def test_october_market_origin_keeps_november_targets_unknown() -> None:
    """The Saturday decision as-of uses Friday data and excludes November labels."""
    result = validate_training_availability(_panel(), ASSETS, ["t_index"])
    assert result["passed"]
    assert result["training_months_per_asset"] == 113
    assert result["unavailable_live_targets"] == 15


@pytest.mark.parametrize("fault", ["future_label", "missing_target", "missing_asset", "HY_input", "Saturday_bar"])
def test_label_and_input_availability_gates(fault: str) -> None:
    """Unavailable labels, missing tasks and excluded inputs cannot pass."""
    frame = _panel()
    columns = ["t_index"]
    if fault == "future_label":
        frame.loc[frame.date.eq(pd.Timestamp("2026-10-30")), "y_excess_lead"] = .1
    elif fault == "missing_target":
        frame.loc[0, "y_excess_lead"] = np.nan
    elif fault == "missing_asset":
        frame = frame.iloc[:-1]
    elif fault == "HY_input":
        frame["hy_spread"] = .01
        columns.append("hy_spread")
    else:
        frame.loc[frame.date.eq(pd.Timestamp("2026-10-30")), "date"] = pd.Timestamp("2026-10-31")
    with pytest.raises(ValueError):
        validate_training_availability(frame, ASSETS, columns)
