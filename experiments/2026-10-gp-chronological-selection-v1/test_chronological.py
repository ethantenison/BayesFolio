"""Deterministic causality, fixed-recipe, exclusion and selection tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from selection_math import RICH_ETF, RICH_MACRO, blend, catalog, recipe, select_static, training_panel


def test_catalog_is_bounded_and_honors_existing_exclusions() -> None:
    """Nineteen hypotheses include rich inputs without restoring excluded HY inputs."""
    specs = catalog()
    assert len(specs) == len({row["id"] for row in specs}) == 19
    assert len(RICH_ETF) + len(RICH_MACRO) == 25
    assert sum(row["reuse"] is not None for row in specs) == 9
    assert sum((24 if row["reuse"] else 52) * 3 for row in specs) == 2208
    for row in specs:
        assert row["rank"] in [1, 2]
        assert row["maxiter"] in [5, 15]
        assert not set(row.get("macro", [])) & {"hy_spread", "hy_spread_chg_1p"}
    no_lag = next(row for row in specs if row["id"] == "rich_without_lag_r2_i15_exp")
    assert "lag_y_excess_lead" not in no_lag["etf"]


@pytest.mark.parametrize(("index", "expected"), [(-24, 60), (0, 84), (27, 111)])
def test_training_history_and_original_seed_index(index: int, expected: int) -> None:
    """Earlier forecasts expand from60 months and exactly match later old chronology."""
    dates = pd.date_range("2017-05-31", periods=112, freq="BME")
    tickers = [f"ETF_{i}" for i in range(15)]
    frame = pd.DataFrame([(date, asset, .01) for date in dates for asset in tickers],
                         columns=["date", "asset_id", "y_excess_lead"])
    origin = str(dates[84 + index].date())
    train, evaluation = training_panel(frame, origin, index, tickers, None)
    assert train.date.nunique() == expected
    assert len(evaluation) == 15
    assert train.date.max() < pd.Timestamp(origin)
    altered = frame.copy()
    altered.loc[altered.date >= origin, "y_excess_lead"] = -999.
    train2, _ = training_panel(altered, origin, index, tickers, None)
    pd.testing.assert_frame_equal(train, train2)
    rolling, _ = training_panel(frame, origin, index, tickers, 48)
    assert rolling.date.nunique() == 48


def test_recipe_is_static_and_jointly_weights_three_experts() -> None:
    """All three weights can change and are constant across seeds and dates."""
    counts = np.tile([16, 40, 8, 0], (4, 1))
    case = recipe(counts)
    means = np.stack([np.full((3, 24, 15), value) for value in [1., 2., 4., 9.]])
    result = blend(means, case)
    np.testing.assert_allclose(result, (16 + 80 + 32) / 64, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="simplexes"):
        recipe(np.tile([16, 16, 16, 16], (4, 1)))
    with pytest.raises(ValueError, match="simplexes"):
        recipe(np.ones((3, 4, 4)))


def _selection_fixture() -> tuple:
    rng = np.random.default_rng(71)
    actual = rng.normal(.003, .035, (38, 15))
    means = rng.normal(0, .01, (4, 3, 38, 15)) + actual[None, None] * .1
    rf = np.full(38, .002)
    covariance = np.broadcast_to(np.eye(15) * .002, (3, 38, 15, 15)).copy()
    budget = np.full(38, .003)
    incumbent = recipe(np.tile([32, 32, 0, 0], (4, 1)))
    return means, actual, rf, covariance, budget, incumbent


def _allocate(mean: np.ndarray, covariance: np.ndarray, budget: np.ndarray) -> tuple:
    del covariance, budget
    return np.full_like(mean, 1 / 15), 0


def test_future_outcome_perturbation_cannot_change_frozen_selection() -> None:
    """Changing every future return leaves all earlier-selected recipes identical."""
    means, actual, rf, covariance, budget, incumbent = _selection_fixture()
    first, evidence = select_static(means[:, :, :24], actual[:24], rf[:24], covariance[:, :24], budget[:24],
                                     incumbent, _allocate, random_recipes=4)
    actual[24:] = -999
    means[:, :, 24:] = 999
    second, evidence2 = select_static(means[:, :, :24], actual[:24], rf[:24], covariance[:, :24], budget[:24],
                                      incumbent, _allocate, random_recipes=4)
    assert first == second
    assert evidence == evidence2
    for row in first.values():
        assert np.max(np.count_nonzero(row["recipe"]["counts"], axis=1)) <= 3


def test_selection_rejects_short_or_misaligned_outcome_ledgers() -> None:
    """Selection cannot silently drop months or use too little meta-training history."""
    means, actual, rf, covariance, budget, incumbent = _selection_fixture()
    with pytest.raises(ValueError, match="earlier ledgers"):
        select_static(means[:, :, :12], actual[:12], rf[:12], covariance[:, :12], budget[:12],
                      incumbent, _allocate)
    with pytest.raises(ValueError, match="earlier ledgers"):
        select_static(means[:, :, :25], actual[:24], rf[:24], covariance[:, :24], budget[:24],
                      incumbent, _allocate)
