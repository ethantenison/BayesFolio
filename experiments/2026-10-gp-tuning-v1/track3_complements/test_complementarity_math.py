"""Deterministic tests for ensemble diversity, static recipes, and Pareto gains."""

from __future__ import annotations

import numpy as np
import pytest
from complementarity_math import blend_third_model, incremental_pareto, residual_diversity


def test_residual_diversity_identical_and_opposite_errors() -> None:
    """Identical errors add no diversity; opposite errors offer a correction."""
    actual = np.zeros((2, 3))
    reference = np.array([[[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]]])
    same = residual_diversity(actual, reference, reference)[0]
    opposite = residual_diversity(actual, reference, -reference)[0]
    assert same["residual_correlation"] == pytest.approx(1, abs=1e-12)
    assert same["forecast_disagreement_rmse"] == pytest.approx(0, abs=1e-12)
    assert opposite["residual_correlation"] == pytest.approx(-1, abs=1e-12)
    alignment = opposite["correction_error_alignment"]
    assert alignment is not None
    assert alignment > 0


def test_correction_alignment_matches_exact_blended_mse_change() -> None:
    """Diversity alone is not enough; report the actual correction direction."""
    generator = np.random.default_rng(17)
    actual = generator.normal(0, 0.03, (8, 5))
    reference = generator.normal(0, 0.02, (3, 8, 5))
    extra = generator.normal(0, 0.02, (3, 8, 5))
    reports = residual_diversity(actual, reference, extra)
    fraction = 0.125
    blend = (1 - fraction) * reference + fraction * extra
    for seed, report in enumerate(reports):
        exact_delta = np.mean((blend[seed] - actual) ** 2 - (reference[seed] - actual) ** 2)
        alignment = report["correction_error_alignment"]
        disagreement = report["squared_disagreement"]
        assert alignment is not None and disagreement is not None
        expected_delta = -2 * fraction * alignment + fraction**2 * disagreement
        assert exact_delta == pytest.approx(expected_delta, abs=1e-14)


def test_degenerate_residual_correlation_is_undefined_not_zero() -> None:
    """Perfect forecasts do not create an artificial zero-correlation result."""
    result = residual_diversity(np.zeros((2, 3)), np.zeros((1, 2, 3)), np.ones((1, 2, 3)))[0]
    assert result["residual_correlation"] is None
    assert result["cross_sectionally_demeaned_residual_correlation"] is None


def test_static_blending_separates_means_and_covariance_and_preserves_psd() -> None:
    """One global scalar is applied uniformly; covariance can remain unchanged."""
    reference = np.zeros((3, 4, 2))
    complement = np.ones_like(reference) * 0.02
    base_cov = np.broadcast_to(np.eye(2) * 0.01, (3, 4, 2, 2)).copy()
    extra_cov = np.broadcast_to(np.array([[0.03, 0.01], [0.01, 0.03]]), base_cov.shape).copy()
    mean, covariance = blend_third_model(
        reference, complement, base_cov, extra_cov, mean_fraction=0.25, covariance_fraction=0
    )
    np.testing.assert_allclose(mean, 0.005, rtol=0, atol=1e-14)
    np.testing.assert_allclose(covariance, base_cov, rtol=0, atol=1e-14)
    _, covariance = blend_third_model(
        reference, complement, base_cov, extra_cov, mean_fraction=0.125, covariance_fraction=0.5
    )
    assert np.linalg.eigvalsh(covariance).min() >= 0


def test_invalid_covariance_is_not_repaired() -> None:
    """Invalid inputs fail explicitly rather than clipping covariance values."""
    mean = np.zeros((1, 1, 2))
    valid = np.eye(2)[None, None]
    invalid = np.array([[[[1.0, 2.0], [2.0, 1.0]]]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        blend_third_model(mean, mean, valid, invalid, mean_fraction=0.25, covariance_fraction=0.25)


def test_frontier_increment_identifies_tradeoffs_dominance_and_duplicates() -> None:
    """All objectives maximize; a duplicate or dominated candidate adds nothing."""
    baseline = np.array([[1.5, 0.6, 0.3]])
    candidates = np.array([[1.5, 0.6, 0.3], [1.4, 0.5, 0.2], [1.6, 0.6, 0.3], [1.4, 0.7, 0.4]])
    result = incremental_pareto(baseline, candidates)
    assert result["not_dominated_by_baseline"] == [True, False, True, True]
    assert result["dominates_any_baseline"] == [False, False, True, False]
    assert result["on_combined_frontier"] == [False, False, True, True]
    assert result["distinct_frontier_expansion"] == [False, False, True, True]


@pytest.mark.parametrize("bad_fraction", [-0.1, 1.1, np.nan])
def test_static_recipe_rejects_invalid_fraction(bad_fraction: float) -> None:
    """No silent clipping of mixture parameters is allowed."""
    mean = np.zeros((1, 1, 2))
    covariance = np.eye(2)[None, None]
    with pytest.raises(ValueError, match="Fractions"):
        blend_third_model(mean, mean, covariance, covariance, mean_fraction=bad_fraction, covariance_fraction=0)


def test_diagnostics_reject_nonfinite_and_misaligned_arrays() -> None:
    """Missing/scattered rows cannot silently change the comparison sample."""
    with pytest.raises(ValueError, match="axes"):
        residual_diversity(np.zeros((4, 2)), np.zeros((3, 3, 2)), np.zeros((3, 3, 2)))
    with pytest.raises(ValueError, match="finite"):
        incremental_pareto(np.ones((1, 3)), np.array([[1, np.nan, 2]]))
