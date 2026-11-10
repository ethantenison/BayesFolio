"""Synthetic deterministic November decision, pooling and causal-window tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from decision_math import candidates, pool, promotion
from independent_validation import ASSETS
from live_worker import training_panel
from policy import allocate


def test_candidate_menu_is_static_bounded_and_retains_controls() -> None:
    """Five old controls and25 new stable recipes fit the approved batch budget."""
    first, second = candidates(), candidates()
    assert first == second
    assert len(first) == 30
    assert sum(item["control"] for item in first) == 5
    for item in first:
        weights = np.array(item["fractions"])
        assert weights.shape == (4, 3)
        np.testing.assert_allclose(weights.sum(1), 1., atol=1e-12, rtol=0)
        assert np.min(weights) >= 0


def test_seed_moments_are_pooled_before_allocation() -> None:
    """Predictions are averaged, not ranks or seed-specific portfolio weights."""
    means = np.zeros((3, 3, 2, 15))
    for expert in range(3):
        for seed in range(3):
            means[expert, seed] = expert * 10 + seed
    cov = np.broadcast_to(np.eye(15), (3, 3, 2, 15, 15)).copy()
    cov[1] *= 3
    cov[2] *= 99
    fractions = np.tile([.25, .5, .25], (4, 1))
    mean, covariance = pool(means, cov, fractions)
    np.testing.assert_allclose(mean, 11., atol=1e-14, rtol=0)
    np.testing.assert_allclose(covariance, np.broadcast_to(2 * np.eye(15), covariance.shape), atol=1e-14)


@pytest.mark.parametrize(("date", "index", "months"), [("2026-09-30", 28, 112), ("2026-10-30", 29, 113)])
def test_live_window_excludes_unknown_november_target(date: str, index: int, months: int) -> None:
    """Fixed date indexes retain correct available train history and unknown tail."""
    dates = pd.date_range("2017-05-31", "2026-10-30", freq="BME")
    frame = pd.DataFrame([(d, asset, .01 if d < dates[-1] else np.nan) for d in dates for asset in ASSETS],
                         columns=["date", "asset_id", "y_excess_lead"])
    train, evaluation = training_panel(frame, date, index, ASSETS, None)
    assert train.date.nunique() == months
    assert train.date.max() < pd.Timestamp(date)
    assert np.isfinite(train.y_excess_lead).all()
    if index == 29:
        assert evaluation.y_excess_lead.isna().all()
    rolling, _ = training_panel(frame, date, index, ASSETS, 48)
    assert rolling.date.nunique() == 48


def test_single_pool_allocator_is_feasible_without_repairs() -> None:
    """One set of pooled moments produces exact KKT weights and effective count7."""
    result = allocate(np.arange(15) / 1000, np.eye(15) * .001, .001)
    weights = np.array(result["weights"])
    assert result["solver_status"] == "analytic_kkt"
    assert weights.sum() == pytest.approx(1., abs=1e-12)
    assert weights.min() >= 0
    assert 1 / (weights @ weights) == pytest.approx(7., abs=1e-10)
    assert not result["fallback_used"]


def test_incumbent_does_not_promote_itself_without_joint_improvement() -> None:
    """No improvement is a valid retained-incumbent outcome, not a search trigger."""
    record = {"all52": {"sharpe": 1., "rank_ic": .2, "total_return": .5,
                         "max_drawdown": -.1, "family_ic": [.1] * 4},
              "later28": {"sharpe": 1.2, "rank_ic": .3, "total_return": .4},
              "leave_one_out_sharpe": [1.] * 52, "block_mean_excess": [.01] * 6}
    decision = promotion(record, record)
    assert not decision["eligible"]
    assert not decision["gates"]["all52_sharpe_improvement_0p05"]
    assert not decision["gates"]["all52_IC_improvement_0p01"]
