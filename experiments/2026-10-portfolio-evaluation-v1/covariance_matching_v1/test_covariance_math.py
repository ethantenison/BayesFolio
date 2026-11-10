"""Synthetic covariance hypothesis tests; no live data, fitting or external writes.

All variances/covariances are decimal squared. Usage: pytest this directory.
"""

from __future__ import annotations

import numpy as np
import pytest
from covariance_math import METHODS, _root, matched_covariance


def _sources() -> tuple[np.ndarray, np.ndarray]:
    return np.array([[0.04, 0.01], [0.01, 0.02]]), np.array([[0.02, -0.005], [-0.005, 0.03]])


def _correlation(matrix: np.ndarray) -> np.ndarray:
    standard = np.sqrt(np.diag(matrix))
    return matrix / standard[:, None] / standard[None, :]


@pytest.mark.parametrize("method", METHODS)
def test_every_method_is_symmetric_spd(method: str) -> None:
    """Sources remain unchanged and all output matrices are SPD."""
    broad, compact = _sources()
    result = matched_covariance(broad, compact, np.array([0.5, 0.25]), method)
    np.linalg.cholesky(result)
    np.testing.assert_allclose(result, result.T, atol=1e-14)
    np.testing.assert_allclose(broad, [[0.04, 0.01], [0.01, 0.02]], atol=1e-14)


def test_gate_scalar_fraction_and_pairwise_formula() -> None:
    """Constant fractions reduce to arithmetic pooling; pairs use both weights."""
    broad, compact = _sources()
    np.testing.assert_allclose(
        matched_covariance(broad, compact, np.full(2, 0.25), "family_gate"), 0.75 * broad + 0.25 * compact, atol=1e-14
    )
    result = matched_covariance(broad, compact, np.array([0.5, 0.25]), "family_gate")
    assert result[0, 1] == pytest.approx(
        np.sqrt(0.5 * 0.75) * broad[0, 1] + np.sqrt(0.5 * 0.25) * compact[0, 1], abs=1e-14
    )


def test_variance_correlation_factorial_controls() -> None:
    """Separate controls change only the intended diagonal/correlation component."""
    broad, compact = _sources()
    alpha = np.array([0.5, 0.25])
    half = matched_covariance(broad, compact, alpha, "half")
    gate = matched_covariance(broad, compact, alpha, "family_gate")
    variance = matched_covariance(broad, compact, alpha, "family_variance_only")
    correlation = matched_covariance(broad, compact, alpha, "family_correlation_only")
    np.testing.assert_allclose(np.diag(variance), np.diag(gate), atol=1e-14)
    np.testing.assert_allclose(_correlation(variance), _correlation(half), atol=1e-14)
    np.testing.assert_allclose(np.diag(correlation), np.diag(half), atol=1e-14)
    np.testing.assert_allclose(_correlation(correlation), _correlation(gate), atol=1e-14)


def test_shared_root_idempotence_and_gate_attenuation() -> None:
    """Shared roots preserve identical sources; gates can attenuate cross-family risk."""
    broad, _ = _sources()
    alpha = np.array([0.5, 0])
    np.testing.assert_allclose(matched_covariance(broad, broad, alpha, "root_shared_100"), broad, atol=1e-14)
    gate = matched_covariance(broad, broad, alpha, "family_gate")
    np.testing.assert_allclose(np.diag(gate), np.diag(broad), atol=1e-14)
    assert abs(gate[0, 1]) < abs(broad[0, 1])


@pytest.mark.parametrize(
    "rho,method", [(0, "root_independent"), (0.5, "root_shared_50"), (0.9, "root_shared_90"), (1, "root_shared_100")]
)
def test_root_coupling_matches_joint_covariance(rho: float, method: str) -> None:
    """The assumed block covariance is PSD and gives the declared linear fusion."""
    broad, compact = _sources()
    alpha = np.array([0.5, 0.25])
    cross = rho * _root(broad) @ _root(compact).T
    joint = np.block([[broad, cross], [cross.T, compact]])
    assert np.linalg.eigvalsh(joint).min() >= -1e-14
    transform = np.c_[np.diag(1 - alpha), np.diag(alpha)]
    np.testing.assert_allclose(
        matched_covariance(broad, compact, alpha, method), transform @ joint @ transform.T, atol=1e-14
    )


def test_independence_artificial_shrinkage_is_visible() -> None:
    """Averaging identical independent uncertainties halves variance at 50/50."""
    broad, _ = _sources()
    np.testing.assert_allclose(
        matched_covariance(broad, broad, np.full(2, 0.5), "root_independent"), 0.5 * broad, atol=1e-14
    )


def test_invalid_covariance_is_rejected_without_repair() -> None:
    """An indefinite source is never made positive by clipping or jitter."""
    broad, compact = _sources()
    broad[0, 0] = -1
    with pytest.raises(np.linalg.LinAlgError):
        matched_covariance(broad, compact, np.array([0.5, 0.25]), "family_gate")
