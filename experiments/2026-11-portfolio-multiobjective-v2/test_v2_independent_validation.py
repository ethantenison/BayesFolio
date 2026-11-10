"""Synthetic fault-injection checks for independent v2 acceptance.

No historical candidate is scored and no GP is fitted. Test returns are decimal,
covariance squared decimals and portfolio weights fractions.
Usage: uv run --locked --no-sync pytest -q <this_file>
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import v2_independent_validation as validation
from scipy.stats import rankdata
from v2_independent_validation import (
    ASSETS,
    ELIGIBLE_INDEX,
    validate_fit_record,
    validate_gp_spec,
    validate_portfolio,
    validate_training_availability,
    validate_weights_csv,
)


def _moments() -> tuple[np.ndarray, np.ndarray]:
    mean = np.linspace(-.04, .02, 15)
    mean[ASSETS.index("MGK")] = .005
    return mean, np.eye(15) * .0001


def _kkt(mean: np.ndarray, minimum: float, *, include_MGK_in_ranks: bool = False) -> np.ndarray:
    if minimum == 14:
        result = np.zeros(15)
        result[ELIGIBLE_INDEX] = 1 / 14
        return result
    scores = (rankdata(mean)[ELIGIBLE_INDEX] / 15 if include_MGK_in_ranks
              else rankdata(mean[ELIGIBLE_INDEX]) / 14)
    order = np.argsort(-scores)
    for count in range(int(np.floor(minimum)) + 1, 15):
        active = scores[order[:count]]
        centered = active - active.mean()
        scale = np.sqrt((1 / minimum - 1 / count) / np.sum(centered**2))
        candidate = np.full(count, 1 / count) + scale * centered
        threshold = active.mean() - 1 / (count * scale)
        if candidate.min() >= 0 and (count == 14 or scores[order[count]] <= threshold):
            result = np.zeros(15)
            result[ELIGIBLE_INDEX[order[:count]]] = candidate
            return result
    raise AssertionError("Synthetic rank KKT fixture has no active set")


def _numerics() -> dict:
    return {"solver_status": "analytic_kkt", "analytic_certificate": "Independent simplex/NE KKT fixture"}


def _spec(*, broad: bool = False, rolling: bool = False) -> dict:
    inputs = ["t_index", "mom12m_skip1m", "mom6m", "cpi_chg_12p", "hy_spread_z_12p", "term_spread"]
    if broad:
        inputs = ["t_index", "lag_y_excess_lead", "mom12m_skip1m", "mom6m", "trend_slope",
                  "cpi_chg_12p", "dealer_gamma_proxy", "dxy", "em_fx_ret", "gold_crude_ratio",
                  "hy_spread_z_12p", "oil_ret", "spy_ret", "term_spread", "vix_ts_z_12p"]
    return {"inputs": inputs, "tickers": ASSETS.copy(), "rank": 2, "maxiter": 15,
            "rolling_months": 48 if rolling else None, "kernel_variant": "signed_lkj_eta_2",
            "time_modulation_mode": "neither"}


def _frame() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2017-05-31", "2026-10-30", freq="BME")
    for index, date in enumerate(dates):
        for asset in ASSETS:
            rows.append({"date": date, "asset_id": asset, "t_index": float(index),
                         "y_excess_lead": np.nan if index == len(dates) - 1 else -.01})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("minimum", [3, 4.5, 7, 10, 13, 14])
def test_arbitrary_effective_floor_matches_independent_conic(minimum):
    """Eligible-rank closed forms match strict conic optima, including NE14."""
    mean, covariance = _moments()
    result = validate_portfolio(mean, covariance, _kkt(mean, minimum), ASSETS,
                                min_effective_assets=minimum, **_numerics())
    assert result["passed"]
    assert result["independent_conic"]["solver_status"] == "optimal"
    assert result["raw_checks"]["MGK_weight"] == pytest.approx(0, abs=0)


def test_MGK_prediction_never_participates_in_allocation_ranking():
    """Changing the excluded forecast cannot alter the eligible optimum."""
    mean, covariance = _moments()
    weights = _kkt(mean, 7)
    mean[ASSETS.index("MGK")] = -1
    validate_portfolio(mean, covariance, weights, ASSETS, min_effective_assets=7, **_numerics())
    mean[ASSETS.index("MGK")] = 1
    validate_portfolio(mean, covariance, weights, ASSETS, min_effective_assets=7, **_numerics())


def test_using_fifteen_ranks_is_rejected_even_when_weights_are_feasible():
    """A zero MGK weight alone does not remove its effect on rank spacing."""
    mean, covariance = _moments()
    incorrect = _kkt(mean, 7, include_MGK_in_ranks=True)
    with pytest.raises(ValueError, match="parity"):
        validate_portfolio(mean, covariance, incorrect, ASSETS, min_effective_assets=7, **_numerics())


def test_representation_precision_does_not_relax_raw_or_objective_gates(monkeypatch):
    """Small conic weight noise passes; material, objective and raw faults fail."""
    mean, covariance = _moments()
    weights = _kkt(mean, 7)
    ordered = ELIGIBLE_INDEX[np.argsort(mean[ELIGIBLE_INDEX])[-3:]]
    reference = weights.copy()
    reference[ordered] += np.array([1e-6, -2e-6, 1e-6])
    monkeypatch.setattr(validation, "_conic_reference", lambda *args: {
        "weights": reference, "solver_status": "optimal", "attempts": [{"status": "optimal"}]})
    result = validate_portfolio(mean, covariance, weights, ASSETS, min_effective_assets=7, **_numerics())
    assert 1e-6 < result["maximum_absolute_weight_difference"] < 5e-6
    reference[:] = weights
    reference[ordered] += np.array([1e-4, -2e-4, 1e-4])
    with pytest.raises(ValueError, match="parity"):
        validate_portfolio(mean, covariance, weights, ASSETS, min_effective_assets=7, **_numerics())
    reference[:] = weights
    reference[ordered[0]] += 4e-6
    reference[ordered[-1]] -= 4e-6
    with pytest.raises(ValueError, match="objective_gap"):
        validate_portfolio(mean, covariance, weights, ASSETS, min_effective_assets=7, **_numerics())
    invalid = weights.copy()
    invalid[1] = 1e-15
    with pytest.raises(ValueError, match="exactly zero"):
        validate_portfolio(mean, covariance, invalid, ASSETS, min_effective_assets=7, **_numerics())


@pytest.mark.parametrize("status", ["optimal_inaccurate", "infeasible", "failed", "unknown"])
def test_unaccepted_status_is_rejected(status):
    mean, covariance = _moments()
    with pytest.raises(ValueError, match="statuses"):
        validate_portfolio(mean, covariance, _kkt(mean, 7), ASSETS,
                           min_effective_assets=7, solver_status=status)


def test_numerical_status_requires_actual_attempts_and_analytic_requires_certificate():
    mean, covariance = _moments()
    with pytest.raises(ValueError, match="actual last optimal"):
        validate_portfolio(mean, covariance, _kkt(mean, 7), ASSETS,
                           min_effective_assets=7, solver_status="optimal")
    with pytest.raises(ValueError, match="certificate"):
        validate_portfolio(mean, covariance, _kkt(mean, 7), ASSETS,
                           min_effective_assets=7, solver_status="analytic_kkt")
    validate_portfolio(mean, covariance, _kkt(mean, 7), ASSETS, min_effective_assets=7,
                       solver_status="optimal", solver_attempts=[{"status": "optimal"}])


@pytest.mark.parametrize("fault", ["MGK", "sum", "negative", "concentration", "nonfinite", "covariance"])
def test_unrepaired_raw_faults_are_rejected(fault):
    mean, covariance = _moments()
    weights = _kkt(mean, 7)
    if fault == "MGK":
        weights[1] = 1e-15
    elif fault == "sum":
        weights *= .99
    elif fault == "negative":
        weights[0] = -.001
    elif fault == "concentration":
        weights = _kkt(mean, 3)
    elif fault == "nonfinite":
        mean[0] = np.nan
    else:
        covariance[0, 3] = .00001
    with pytest.raises(ValueError):
        validate_portfolio(mean, covariance, weights, ASSETS, min_effective_assets=7, **_numerics())


def test_optional_caps_are_checked_in_raw_units_without_return_floors():
    mean, covariance = _moments()
    mean -= 1
    weights = _kkt(mean, 14)
    result = validate_portfolio(mean, covariance, weights, ASSETS, min_effective_assets=14,
                                variance_ceiling=.0001, equity_cap=.5, **_numerics())
    assert result["raw_checks"]["equity_weight"] == pytest.approx(.5)
    with pytest.raises(ValueError, match="equity cap"):
        validate_portfolio(mean, covariance, weights, ASSETS, min_effective_assets=14,
                           equity_cap=.49, **_numerics())
    with pytest.raises(ValueError, match="variance ceiling"):
        validate_portfolio(mean, covariance, weights, ASSETS, min_effective_assets=14,
                           variance_ceiling=.000001, **_numerics())


def test_binding_variance_ceiling_has_independent_conic_parity():
    mean, covariance = _moments()
    weights = _kkt(mean, 10)
    result = validate_portfolio(mean, covariance, weights, ASSETS, min_effective_assets=7,
                                variance_ceiling=.00001, **_numerics())
    assert result["raw_checks"]["variance_to_ceiling"] == pytest.approx(1, abs=1e-10)


def test_binding_equity_cap_has_independent_conic_parity():
    eligible_mean = np.r_[np.arange(8, 15), np.arange(1, 8)] / 1000
    mean = np.zeros(15)
    mean[ELIGIBLE_INDEX] = eligible_mean
    score = rankdata(eligible_mean) / 14
    group_centered = np.r_[score[:7] - score[:7].mean(), score[7:] - score[7:].mean()]
    base = np.r_[np.full(7, .35 / 7), np.full(7, .65 / 7)]
    scale = np.sqrt((1 / 12 - np.sum(base**2)) / np.sum(group_centered**2))
    weights = np.zeros(15)
    weights[ELIGIBLE_INDEX] = base + scale * group_centered
    result = validate_portfolio(mean, np.eye(15) * .0001, weights, ASSETS,
                                min_effective_assets=12, equity_cap=.35, **_numerics())
    assert result["raw_checks"]["equity_weight"] == pytest.approx(.35)


def test_serialized_weights_keep_all_tasks_MGK_zero_and_metadata(tmp_path):
    mean, covariance = _moments()
    weights = _kkt(mean, 7)
    path = tmp_path / "weights.csv"
    frame = pd.DataFrame({"asset_id": ASSETS, "weight": weights, "forecast_month": "2026-11"})
    frame.to_csv(path, index=False, float_format="%.17g")
    result = validate_weights_csv(path, mean, covariance, weights, ASSETS, min_effective_assets=7,
                                  expected_metadata={"forecast_month": "2026-11"}, **_numerics())
    assert result["csv_rows"] == 15
    frame = frame.loc[frame.asset_id != "MGK"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="fifteen ETFs"):
        validate_weights_csv(path, mean, covariance, weights, ASSETS, min_effective_assets=7, **_numerics())


def test_all_training_tasks_are_retained_and_November_is_unknown():
    result = validate_training_availability(_frame(), ASSETS, ["t_index"])
    assert result["expanding_training_months"] == 113
    assert result["unknown_live_targets"] == 15
    assert "MGK" in result["GP_tasks"]


@pytest.mark.parametrize("fault", ["November", "missing_task", "missing_target", "Saturday", "HY"])
def test_unavailable_targets_or_changed_population_are_rejected(fault):
    frame = _frame()
    columns = ["t_index"]
    if fault == "November":
        frame.loc[frame.date == pd.Timestamp("2026-10-30"), "y_excess_lead"] = -.01
    elif fault == "missing_task":
        frame = frame.loc[frame.asset_id != "MGK"]
    elif fault == "missing_target":
        frame.loc[0, "y_excess_lead"] = np.nan
    elif fault == "Saturday":
        frame.loc[frame.date == pd.Timestamp("2026-10-30"), "date"] = pd.Timestamp("2026-10-31")
    else:
        frame["hy_spread"] = 1
        columns.append("hy_spread")
    with pytest.raises(ValueError):
        validate_training_availability(frame, ASSETS, columns)


def test_exact_admitted_compact_rank1_and_prior_hypotheses():
    reference = _spec()
    candidate = deepcopy(reference)
    candidate["rank"] = 1
    candidate["prior_overrides"] = {"lkj_eta": 4}
    assert validate_gp_spec(candidate, reference, deepcopy(candidate), allow_compact_rank1=True)["passed"]
    with pytest.raises(ValueError, match="admission"):
        validate_gp_spec(candidate, reference, reference, allow_compact_rank1=True)


@pytest.mark.parametrize("fault", ["features", "window", "maxiter", "rank", "population", "modulation", "target"])
def test_admission_does_not_override_protected_GP_settings(fault):
    reference = _spec(broad=True)
    candidate = deepcopy(reference)
    if fault == "features":
        candidate["inputs"].append("hy_spread")
    elif fault == "window":
        candidate["rolling_months"] = 36
    elif fault == "maxiter":
        candidate["maxiter"] = 36
    elif fault == "rank":
        candidate["rank"] = 1
    elif fault == "population":
        candidate["tickers"].remove("MGK")
    elif fault == "modulation":
        candidate["time_modulation_mode"] = "both"
    else:
        candidate["target"] = "new_target"
    with pytest.raises(ValueError):
        validate_gp_spec(candidate, reference, deepcopy(candidate), allow_compact_rank1=True)


@pytest.mark.parametrize("rolling,months,start", [(False, 113, "2017-05-31"), (True, 48, "2022-10-31")])
def test_actual_fit_uses_only_pre_origin_labeled_months(rolling, months, start):
    spec = _spec(broad=rolling, rolling=rolling)
    record = {"status": "fit", "date": "2026-10-30", "seed": 27, "rank": 2, "maxiter": 15,
              "train_start": start, "train_end": "2026-09-30", "train_months": months}
    assert validate_fit_record(record, spec, origin="2026-10-30", seed=27)["passed"]
    record["train_end"] = "2026-10-30"
    with pytest.raises(ValueError, match="window"):
        validate_fit_record(record, spec, origin="2026-10-30", seed=27)
