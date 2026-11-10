"""Synthetic formula, Monte Carlo, invariance and unchanged-allocator tests.

Usage: .venv/bin/pytest -q <this_directory>. No GP fits, network or study writes.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest
import run_uncertainty_rank as study
from expected_rank_math import METHODS, allocation_scores, gaussian_expected_rank, pairwise_diagnostics
from scipy.special import ndtr
from scipy.stats import rankdata


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    mean = np.array([0.02, 0, -0.01, 0.04])
    covariance = 0.0004 * np.array([[1, 0.6, 0.1, 0.2], [0.6, 2, 0.3, 0.1], [0.1, 0.3, 1, 0.4], [0.2, 0.1, 0.4, 1.5]])
    return mean, covariance


def test_two_asset_closed_form() -> None:
    """A correlated pair agrees with the univariate Gaussian difference formula."""
    mean = np.array([0.03, 0.01])
    covariance = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
    result = gaussian_expected_rank(mean, covariance)
    probability = ndtr(0.02 / np.sqrt(0.0011))
    assert result["probability"][0, 1] == pytest.approx(probability, abs=1e-14)
    np.testing.assert_allclose(result["scores"], [(1 + probability) / 2, (2 - probability) / 2], rtol=0, atol=1e-14)


def test_pairwise_complements_and_fixed_total_rank() -> None:
    """Self-comparisons are excluded and expected ranks sum to the rank total."""
    mean, covariance = _inputs()
    result = gaussian_expected_rank(np.broadcast_to(mean, (3, 24, 4)), np.broadcast_to(covariance, (3, 24, 4, 4)))
    np.testing.assert_allclose(result["probability"] + result["probability"].swapaxes(-1, -2), 1, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result["scores"].sum(axis=-1), 2.5, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.diagonal(result["probability"], axis1=-2, axis2=-1), 0.5, rtol=0, atol=1e-15)


def test_zero_noise_limit_matches_old_hard_ranks() -> None:
    """Unique and tied means converge to the original normalized rank convention."""
    for mean in [np.array([0.01, -0.02, 0.03, 0.04]), np.array([0.01, 0.01, -0.03, 0.02])]:
        scores = gaussian_expected_rank(mean, np.eye(4) * 1e-12)["scores"]
        np.testing.assert_allclose(scores, rankdata(mean) / 4, rtol=0, atol=1e-14)


def test_tied_means_do_not_create_low_variance_preference() -> None:
    """Different marginal uncertainty alone gives no preference at equal means."""
    result = gaussian_expected_rank(np.zeros(4), np.diag([0.0001, 0.001, 0.01, 0.1]))
    np.testing.assert_allclose(result["scores"], 0.625, rtol=0, atol=1e-14)


def test_return_shift_and_unit_invariance() -> None:
    """Common return shifts and coherent decimal-to-percent conversion preserve ranks."""
    mean, covariance = _inputs()
    expected = gaussian_expected_rank(mean, covariance)["scores"]
    shifted = gaussian_expected_rank(mean + 1, covariance)["scores"]
    scaled = gaussian_expected_rank(mean * 100, covariance * 10000)["scores"]
    np.testing.assert_allclose(expected, shifted, rtol=0, atol=1e-14)
    np.testing.assert_allclose(expected, scaled, rtol=0, atol=1e-14)


def test_common_market_shock_cancels_in_relative_returns() -> None:
    """A shared additive Gaussian shock changes portfolio risk but not pair differences."""
    mean, covariance = _inputs()
    left = gaussian_expected_rank(mean, covariance)
    right = gaussian_expected_rank(mean, covariance + 0.0005 * np.ones((4, 4)))
    np.testing.assert_allclose(left["probability"], right["probability"], rtol=0, atol=1e-14)


def test_asset_permutation_equivariance() -> None:
    """Reordering ETF axes only reorders the output scores and probability axes."""
    mean, covariance = _inputs()
    order = [2, 0, 3, 1]
    expected = gaussian_expected_rank(mean, covariance)
    result = gaussian_expected_rank(mean[order], covariance[np.ix_(order, order)])
    np.testing.assert_allclose(result["scores"], expected["scores"][order], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["probability"], expected["probability"][np.ix_(order, order)], rtol=0, atol=1e-14)


def test_expected_ranks_agree_with_joint_gaussian_simulation() -> None:
    """A deterministic Monte Carlo check independently verifies the expected-rank identity."""
    mean, covariance = _inputs()
    samples = np.random.default_rng(27).multivariate_normal(mean, covariance, size=200000)
    empirical = (rankdata(samples, axis=-1) / 4).mean(axis=0)
    np.testing.assert_allclose(empirical, gaussian_expected_rank(mean, covariance)["scores"], rtol=0, atol=0.003)


def test_heterogeneous_uncertainty_can_change_mean_order() -> None:
    """Expected ranks can demote a high-mean but highly uncertain return forecast."""
    mean = np.array([0, 0.01, 0.02])
    result = gaussian_expected_rank(mean, np.diag([1e-8, 1e-8, 0.1]))
    assert mean[2] > mean[1] and result["scores"][2] < result["scores"][1]


def test_diagonal_score_ablation_changes_no_input_risk_matrix() -> None:
    """The covariance ablation is confined to ranking, never mutated in place."""
    mean, covariance = _inputs()
    saved = covariance.copy()
    diagonal = allocation_scores(mean, covariance, "expected_diagonal")
    expected = gaussian_expected_rank(mean, np.diag(np.diag(covariance)))
    np.testing.assert_array_equal(covariance, saved)
    np.testing.assert_array_equal(diagonal["scores"], expected["scores"])


def test_reranking_only_changes_spacing_not_expected_order() -> None:
    """The registered diagnostic isolates ordinal changes from confidence spacing."""
    mean, covariance = _inputs()
    direct = allocation_scores(mean, covariance, "expected_full")
    reranked = allocation_scores(mean, covariance, "reranked_expected_full")
    np.testing.assert_array_equal(reranked["scores"], rankdata(direct["scores"]) / 4)
    np.testing.assert_array_equal(direct["probability"], reranked["probability"])


@pytest.mark.parametrize("failure", ["missing", "asymmetric", "indefinite", "axes"])
def test_invalid_gaussian_inputs_fail_without_repair(failure: str) -> None:
    """Invalid uncertainty cannot be clipped, repaired or silently omitted."""
    mean, covariance = _inputs()
    if failure == "missing":
        mean[0] = np.nan
    elif failure == "asymmetric":
        covariance[0, 1] += 0.1
    elif failure == "indefinite":
        covariance[0, 0] = -1
    else:
        covariance = covariance[:3, :3]
    with pytest.raises(ValueError):
        gaussian_expected_rank(mean, covariance)


def test_stable_log_loss_without_probability_clipping() -> None:
    """A wrong extreme forecast has finite exact log-CDF loss, not a clipped loss."""
    mean, actual = np.array([100.0, -100.0]), np.array([-0.01, 0.01])
    result = pairwise_diagnostics(actual, mean, np.eye(2))
    assert np.isfinite(result["log_loss"]) and result["log_loss"] > 9000
    assert result["brier"] == pytest.approx(1, abs=1e-14)


def test_observed_ties_and_pair_count_are_explicit() -> None:
    """Every unordered pair has a defined strict-event outcome, including ties."""
    result = pairwise_diagnostics(np.zeros(3), np.zeros(3), np.eye(3))
    assert result["observed_ties"] == 3
    assert result["brier"] == pytest.approx(0.25, abs=1e-14)
    assert result["log_loss"] == pytest.approx(np.log(2), abs=1e-14)


def test_registered_scope_has_no_new_allocation_risk_cases() -> None:
    """Two frozen recipes each have the exact two primary and two diagnostic scores."""
    cases = study._cases()
    assert len(cases) == len({case["id"] for case in cases}) == 8
    assert sum(case["role"] == "primary" for case in cases) == 4
    assert {case["covariance"] for case in cases} == {"half"}
    assert {case["decision"] for case in cases} == set(METHODS)


def test_worker_keeps_original_allocator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the output path and score panel are adapted inside isolated workers."""
    original_allocator = study.base.allocate_checked
    monkeypatch.setattr(study.base, "OUTPUT", study.base.OUTPUT)
    monkeypatch.setattr(study.base, "_panel", study.base._panel)
    monkeypatch.setattr(
        study.base,
        "_job",
        lambda case: {
            "allocator": study.base.allocate_checked,
            "output": study.base.OUTPUT,
            "panel": study.base._panel,
        },
    )
    result = study._worker(study._case("sharpe", "expected_full"))
    assert result["allocator"] is original_allocator
    assert result["output"] == study.OUTPUT and result["panel"] is study._panel


def test_unknown_score_variant_is_rejected() -> None:
    """Unregistered score methods cannot silently enter the experiment."""
    mean, covariance = _inputs()
    with pytest.raises(ValueError, match="Unknown score"):
        allocation_scores(mean, covariance, "unregistered")


def test_all_modules_and_public_symbols_are_documented() -> None:
    """Cover research files excluded from the repository's standard doc gate."""
    for path in study.HERE.glob("*.py"):
        tree = ast.parse(path.read_text())
        assert ast.get_docstring(tree), path
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                assert ast.get_docstring(node), (path, node.name)
