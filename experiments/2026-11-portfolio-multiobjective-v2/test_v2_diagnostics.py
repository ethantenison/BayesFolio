"""Pure synthetic fixtures for Gaussian density and projection diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal
from v2_diagnostics import gaussian_panel


def test_joint_density_matches_gaussian_reference_and_marginal_diagonal_case():
    """Joint NLL/d agrees with an independent density and includes negative NLL."""
    actual = np.array([[.01, -.02], [.03, .01]])
    mean = np.array([[.005, -.01], [.015, .005]])
    covariance = np.repeat(np.diag([.001, .002])[None], 2, axis=0)
    result = gaussian_panel(actual, mean, covariance, assets=["A", "B"])
    expected = np.array([-multivariate_normal.logpdf(y, mean=m, cov=c) / 2
                         for y, m, c in zip(actual, mean, covariance)])
    np.testing.assert_allclose(result["monthly_joint_NLL_per_asset"], expected, rtol=0, atol=1e-13)
    assert result["mean_joint_gaussian_NLL_per_asset"] < 0
    assert result["mean_marginal_gaussian_NLL"] == pytest.approx(expected.mean(), abs=1e-13)


def test_correlated_projection_and_fixed_reference_use_full_covariance():
    """Scalar scores use wΣw, retaining offdiagonal covariance and raw weights."""
    actual, mean = np.array([[.02, -.01]]), np.zeros((1, 2))
    covariance = np.array([[[.0004, .0001], [.0001, .0009]]])
    weights = np.array([[.75, .25]])
    result = gaussian_panel(actual, mean, covariance, assets=["A", "B"], portfolio_weights=weights,
                            reference_weights=np.array([.5, .5]))
    error = float(weights[0] @ actual[0])
    variance = float(weights[0] @ covariance[0] @ weights[0])
    assert result["allocated_portfolio_projection"]["mean_squared_standardized_error"] == pytest.approx(
        error**2 / variance, abs=1e-13)
    assert result["allocated_portfolio_projection"]["mean_gaussian_NLL"] == pytest.approx(
        .5 * (np.log(2 * np.pi * variance) + error**2 / variance), abs=1e-13)
    fixed_error, fixed_variance = .005, .000375
    assert result["fixed_eligible_EW_reference"]["mean_squared_standardized_error"] == pytest.approx(
        fixed_error**2 / fixed_variance, abs=1e-13)


def test_gaussian_scaling_changes_density_but_not_calibration_or_coverage():
    """Decimal-to-percent scaling adds log100/density dimension as expected."""
    actual = np.array([[.01, -.02], [-.01, .02]])
    mean = np.zeros_like(actual)
    covariance = np.repeat(np.diag([.0001, .0004])[None], 2, axis=0)
    first = gaussian_panel(actual, mean, covariance, assets=["A", "B"])
    second = gaussian_panel(actual * 100, mean * 100, covariance * 10000, assets=["A", "B"])
    assert second["mean_joint_gaussian_NLL_per_asset"] - first["mean_joint_gaussian_NLL_per_asset"] == pytest.approx(
        np.log(100), abs=1e-13)
    assert second["mean_joint_mahalanobis_per_dimension"] == pytest.approx(
        first["mean_joint_mahalanobis_per_dimension"], abs=1e-13)
    assert second["nominal95_marginal_coverage"] == first["nominal95_marginal_coverage"]


def test_invalid_covariance_or_missing_month_is_rejected_without_repair():
    """No month dropping, covariance jitter or silent NaN replacement is allowed."""
    actual, mean = np.zeros((2, 2)), np.zeros((2, 2))
    covariance = np.repeat(np.eye(2)[None], 2, axis=0)
    covariance[0, 0, 0] = -1
    with pytest.raises(np.linalg.LinAlgError):
        gaussian_panel(actual, mean, covariance, assets=["A", "B"])
    covariance[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Complete finite"):
        gaussian_panel(actual, mean, covariance, assets=["A", "B"])


def test_offdiagonal_discrepancy_is_reported_with_single_vector_limit():
    """Outer-product diagnostics retain units and avoid identifying true covariance."""
    actual = np.array([[.02, -.01]])
    covariance = np.array([[[.0004, .0001], [.0001, .0009]]])
    result = gaussian_panel(actual, np.zeros((1, 2)), covariance, assets=["A", "B"])
    discrepancy = result["covariance_error_outer_product_discrepancy"]
    assert discrepancy["offdiagonal_monthly_MSE_decimal_power4"] == pytest.approx((.0001 + .0002)**2, abs=1e-18)
    assert "not true covariance" in discrepancy["interpretation"]
