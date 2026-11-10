"""Independent synthetic numerical checks for pure November v2 metrics.

No cached candidate returns or confirmation outcomes are read. Financial arrays
use decimal returns and fractional weights. Run with:
    uv run --locked --no-sync pytest -q experiments/2026-11-portfolio-multiobjective-v2/test_v2_metrics.py
"""

from __future__ import annotations

import numpy as np
import pytest
from v2_metrics import FAMILY_ASSETS, paired_block_bootstrap, ranking_diagnostics, score_portfolio, score_returns

ASSETS = ["SPY", "MGK", "VTV", "IWM", "VEA", "VWO", "VSS", "EWX", "BND", "IEF", "LQD", "HYG",
          "BNDX", "VWOB", "HYEM"]


def _panel(months: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(627)
    actual = rng.normal(.005, .035, size=(months, 15))
    means = .5 * actual + rng.normal(0, .01, size=actual.shape)
    weights = np.ones_like(actual) / 14
    weights[:, 1] = 0
    return actual, np.full(months, .002), means, weights


def test_cagr_compounding_and_conventional_sharpe() -> None:
    """Annualization uses geometric total return and arithmetic rf subtraction."""
    total = np.tile([.02, -.01, .04], 8)
    rf = np.full(24, .003)
    summary, paths = score_returns(total, rf)
    expected = (1.02 * .99 * 1.04) ** 8
    assert summary["total_return"] == pytest.approx(expected - 1, abs=1e-14)
    assert summary["cagr"] == pytest.approx(expected**.5 - 1, abs=1e-14)
    assert summary["sharpe"] == pytest.approx(np.sqrt(12) * (total - rf).mean() / (total - rf).std(ddof=1))
    assert paths["wealth"][0] == pytest.approx(10000)
    assert len(paths["wealth"]) == 25


def test_initial_loss_is_in_stitched_drawdown_and_recovery() -> None:
    """The initial investment peak must survive return-panel stitching."""
    summary, paths = score_returns(np.array([-.2, .1, .15, -.05]), np.zeros(4))
    assert summary["max_drawdown"] == pytest.approx(-.2)
    assert summary["max_drawdown_recovery_months"] == 3
    assert summary["max_drawdown_recovery_censored"] is False
    assert summary["underwater_at_end"] is True
    assert summary["ongoing_underwater_months"] == 1
    assert paths["drawdown_episodes"][0]["peak_index"] == 0
    assert paths["drawdown_episodes"][0]["recovery_index"] == 3


def test_negative_path_and_censored_recovery_are_not_floored() -> None:
    """Negative observed market returns stay negative and retain censoring."""
    total = np.array([-.1, -.2, -.1])
    summary, paths = score_returns(total, np.zeros(3))
    assert summary["total_return"] == pytest.approx(-.352)
    assert summary["max_drawdown"] == pytest.approx(-.352)
    assert summary["worst_month"] == pytest.approx(-.2)
    assert summary["max_drawdown_recovery_months"] == 3
    assert summary["max_drawdown_recovery_censored"] is True
    np.testing.assert_array_equal(paths["total_returns"], total)


def test_twelve_month_stress_includes_each_window_start() -> None:
    """A twelve-month loss streak has the same total loss and initial-peak DD."""
    summary, _ = score_returns(np.r_[np.full(12, -.01), .2], np.zeros(13))
    assert summary["worst_12_month_return"] == pytest.approx(.99**12 - 1)
    assert summary["worst_12_month_drawdown"] == pytest.approx(.99**12 - 1)


def test_no_drawdown_and_undefined_zero_variance_are_explicit() -> None:
    """Zero excess variance does not manufacture an infinite Sharpe."""
    summary, _ = score_returns(np.zeros(12), np.zeros(12))
    assert summary["sharpe"] is None
    assert summary["max_drawdown"] == pytest.approx(0)
    assert summary["max_drawdown_recovery_months"] == 0
    assert summary["max_drawdown_recovery_censored"] is False
    positive, _ = score_returns(np.full(12, .01), np.zeros(12))
    assert positive["sharpe"] is None
    assert positive["annualized_excess_vol"] == pytest.approx(0)


def test_equal_peak_recovers_instead_of_extending_episode() -> None:
    """Returning to the exact previous high ends an underwater episode."""
    summary, paths = score_returns(np.array([-.2, .25, -.1]), np.zeros(3))
    assert summary["max_drawdown_recovery_months"] == 2
    assert paths["drawdown_episodes"][0]["censored"] is False
    assert len(paths["drawdown_episodes"]) == 2


def test_mgk_does_not_change_pooled_or_family_rank_ic() -> None:
    """All fourteen eligible targets remain, while extreme MGK changes vanish."""
    actual, _, means, _ = _panel()
    baseline, baseline_paths = ranking_diagnostics(actual, means, assets=ASSETS)
    actual[:, 1], means[:, 1] = -100, 100
    changed, changed_paths = ranking_diagnostics(actual, means, assets=ASSETS)
    assert baseline == changed
    assert len(changed["ranking_assets"]) == 14
    assert [len(members) for members in changed["family_assets"].values()] == [3, 4, 4, 3]
    np.testing.assert_array_equal(baseline_paths["rank_ic"], changed_paths["rank_ic"])


def test_family_mapping_can_include_mgk_but_ranking_cannot() -> None:
    """The delivered fifteen-column family manifest is safely explicit."""
    actual, _, means, _ = _panel()
    groups = {key: list(value) for key, value in FAMILY_ASSETS.items()}
    groups["us_equity"].insert(1, "MGK")
    summary, _ = ranking_diagnostics(actual, means, assets=ASSETS, families=groups)
    assert summary["family_assets"]["us_equity"] == ["SPY", "VTV", "IWM"]


def test_ties_use_average_ranks() -> None:
    """Identical nonconstant tied target/prediction ranks have IC one."""
    actual = np.tile(np.repeat([-.02, .01, .03], 5), (2, 1))
    # Each family must remain nonconstant in the fixture as well as the universe.
    actual[:, [2, 5, 9, 13]] += .001
    summary, _ = ranking_diagnostics(actual, actual.copy(), assets=ASSETS)
    assert summary["rank_ic"] == pytest.approx(1)
    assert all(value == pytest.approx(1) for value in summary["family_ic"].values())


def test_constant_ranking_fails_instead_of_dropping_month() -> None:
    """Undefined ranks cannot silently make the evaluated market sample smaller."""
    actual, _, means, _ = _panel()
    means[0] = 0
    with pytest.raises(ValueError, match="constant monthly vector"):
        ranking_diagnostics(actual, means, assets=ASSETS)


def test_discounted_excess_conversion_and_family_contributions() -> None:
    """Weighted asset totals match the established GP-target conversion."""
    actual, rf, means, weights = _panel()
    summary, paths = score_portfolio(actual, rf, means, weights, assets=ASSETS)
    expected = (1 + np.sum(weights * actual, 1)) * (1 + rf) - 1
    np.testing.assert_allclose(paths["total_returns"], expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(np.sum(list(paths["family_arithmetic_return_contributions"].values()), 0),
                               expected, atol=1e-14, rtol=0)
    assert summary["effective_assets"] == pytest.approx(14)
    assert sum(summary["family_weights"].values()) == pytest.approx(1)
    assert set(summary["cost_cases"]) == {"0", "5", "10", "20"}
    np.testing.assert_allclose(paths["cost_cases"]["0"]["total_returns"], expected, atol=1e-14, rtol=0)


def test_initial_purchase_cost_is_self_financing_and_not_halved() -> None:
    """First purchases cost c/(1+c), with all post-cost wealth invested."""
    actual, rf, means, weights = _panel()
    _, paths = score_portfolio(actual, rf, means, weights, assets=ASSETS)
    details = paths["cost_cases"]["20"]
    assert details["post_cost_wealth_fraction"][0] == pytest.approx(1 / 1.002, abs=1e-13)
    assert details["cost_fraction"][0] == pytest.approx(.002 / 1.002, abs=1e-13)
    assert details["traded_notional_fraction"][0] == pytest.approx(1 / 1.002, abs=1e-13)


def test_subsequent_cost_uses_drifted_holdings_and_exact_funding() -> None:
    """A winning asset drifts above target and incurs a real rebalance sale."""
    actual, rf, means, weights = _panel(3)
    actual[0, 0] = .5
    _, paths = score_portfolio(actual, rf, means, weights, assets=ASSETS)
    details = paths["cost_cases"]["10"]
    holdings = weights[0] * (1 + paths["asset_total_returns"][0])
    expected_drift = holdings / holdings.sum()
    np.testing.assert_allclose(details["pretrade_asset_fractions"][1], expected_drift, atol=1e-14, rtol=0)
    assert expected_drift[0] > weights[1, 0]
    for month in range(3):
        fraction = details["post_cost_wealth_fraction"][month]
        notional = np.abs(fraction * weights[month] - details["pretrade_asset_fractions"][month]).sum()
        assert fraction + .001 * notional == pytest.approx(1, abs=1e-13)


def test_costs_do_not_make_negative_market_returns_nonnegative() -> None:
    """A losing market path remains losing under every trading-cost case."""
    actual, rf, means, weights = _panel(3)
    actual -= .2
    summary, paths = score_portfolio(actual, rf, means, weights, assets=ASSETS)
    assert np.all(paths["total_returns"] < 0)
    for key, case in summary["cost_cases"].items():
        assert case["total_return"] <= summary["total_return"] + 1e-14
        assert np.all(paths["cost_cases"][key]["total_returns"] < 0)


def test_nonzero_mgk_weight_is_rejected() -> None:
    """MGK exclusion applies to the actual delivered allocator weights."""
    actual, rf, means, weights = _panel()
    weights[:, 1] = .01
    weights[:, 0] -= .01
    with pytest.raises(ValueError, match="exactly zero"):
        score_portfolio(actual, rf, means, weights, assets=ASSETS)


def test_raw_weight_numerical_residuals_are_accepted_unchanged() -> None:
    """Validator-consistent residuals survive returns, costs and input evidence."""
    actual, rf, means, weights = _panel(3)
    weights[:, 0] = -1e-11
    weights[:, 2] += 1 / 14 + 1e-11
    weights[:, 8] += 5e-8
    saved = weights.copy()
    summary, paths = score_portfolio(actual, rf, means, weights, assets=ASSETS)
    np.testing.assert_array_equal(weights, saved)
    assert summary["raw_minimum_weight"] == pytest.approx(-1e-11, abs=1e-20)
    assert summary["raw_maximum_weight_sum_error"] == pytest.approx(5e-8, abs=1e-15)
    np.testing.assert_allclose(paths["total_returns"], np.sum(saved * paths["asset_total_returns"], 1),
                               atol=1e-15, rtol=0)
    for key, details in paths["cost_cases"].items():
        bps = float(key)
        for month in range(3):
            fraction = details["post_cost_wealth_fraction"][month]
            notional = np.abs(fraction * saved[month] - details["pretrade_asset_fractions"][month]).sum()
            assert fraction + bps / 10000 * notional == pytest.approx(1, abs=1e-13)
        assert np.isfinite(details["total_returns"]).all()
    np.testing.assert_allclose(paths["cost_cases"]["0"]["total_returns"], paths["total_returns"],
                               atol=1e-15, rtol=0)


def test_material_negative_weight_is_rejected_even_with_exact_funding() -> None:
    """A meaningful short is not classified as an accepted solver residual."""
    actual, rf, means, weights = _panel(3)
    weights[:, 0] = -1e-6
    weights[:, 2] += 1 / 14 + 1e-6
    with pytest.raises(ValueError, match="long-only fully invested"):
        score_portfolio(actual, rf, means, weights, assets=ASSETS)


def test_paired_block_bootstrap_is_deterministic_and_paired() -> None:
    """Identical policies keep exactly zero differences under every shared draw."""
    returns = np.array([.1, -.1, .02, .03, -.04, .01])
    ic = np.linspace(-.3, .4, 6)
    kwargs = {"candidate_ic": ic, "reference_ic": ic, "block_length": 3, "draws": 100, "seed": 29}
    summary, paths = paired_block_bootstrap(returns, returns, np.zeros(6), **kwargs)
    second, second_paths = paired_block_bootstrap(returns, returns, np.zeros(6), **kwargs)
    assert summary == second
    np.testing.assert_array_equal(paths["sampled_indices"], second_paths["sampled_indices"])
    for values in paths["differences"].values():
        np.testing.assert_allclose(values, 0, atol=1e-14, rtol=0)
    assert np.all((paths["sampled_indices"][:, 1] - paths["sampled_indices"][:, 0]) % 6 == 1)
    assert np.all((paths["sampled_indices"][:, 2] - paths["sampled_indices"][:, 1]) % 6 == 1)


def test_bootstrap_preserves_undefined_sharpe_count() -> None:
    """Constant return sensitivity never invents a finite Sharpe advantage."""
    summary, _ = paired_block_bootstrap(np.zeros(6), np.zeros(6), np.zeros(6),
                                         candidate_ic=np.zeros(6), reference_ic=np.zeros(6),
                                         block_length=2, draws=20, seed=7)
    assert summary["metrics"]["sharpe"]["point_difference"] is None
    assert summary["metrics"]["sharpe"]["undefined_replicates"] == 20
    assert summary["metrics"]["sharpe"]["fraction_above_zero"] is None


@pytest.mark.parametrize("case", ["nonfinite", "short_rf", "negative_weight", "not_invested", "bankrupt"])
def test_invalid_inputs_fail_without_repair(case: str) -> None:
    """Missing periods and invalid financing cannot be silently substituted."""
    actual, rf, means, weights = _panel()
    if case == "nonfinite":
        actual[0, 0] = np.nan
    elif case == "short_rf":
        rf = rf[:-1]
    elif case == "negative_weight":
        weights[0, 0] = -.1
    elif case == "not_invested":
        weights[0] *= .9
    else:
        actual[0] = -1.1
    with pytest.raises(ValueError):
        score_portfolio(actual, rf, means, weights, assets=ASSETS)
