"""Pure covariance matching hypotheses for fixed family prediction ensembles.

Research numerical layer; no fitting, outcomes, allocation or I/O. Complete
observed-return covariance is decimal squared. Congruence gates and symmetric
matrix-root couplings preserve positive semidefiniteness; outputs must pass
strict positive-definiteness checks without jitter or eigenvalue clipping.
"""

from __future__ import annotations

import numpy as np

METHODS = [
    "broad",
    "compact",
    "half",
    "family_gate",
    "family_variance_only",
    "family_correlation_only",
    "root_shared_100",
    "root_shared_90",
    "root_shared_50",
    "root_independent",
]


def _check(matrix: np.ndarray) -> None:
    if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1] or not np.isfinite(matrix).all():
        raise ValueError("Complete finite square covariance required")
    if not np.allclose(matrix, matrix.swapaxes(-1, -2), rtol=1e-10, atol=1e-14):
        raise ValueError("Symmetric covariance required; no repair")
    np.linalg.cholesky(matrix)


def _root(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    if (values <= 0).any():
        raise ValueError("Positive eigenvalues required; no clipping")
    return (vectors * np.sqrt(values)[..., None, :]) @ vectors.swapaxes(-1, -2)


def _rescale_diagonal(matrix: np.ndarray, variance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(variance / np.diagonal(matrix, axis1=-2, axis2=-1))
    return matrix * scale[..., :, None] * scale[..., None, :]


def matched_covariance(broad: np.ndarray, compact: np.ndarray, alpha: np.ndarray, method: str) -> np.ndarray:
    """Build a declared covariance hypothesis without changing predicted means.

    Args:
        broad: Complete broad covariance, (..., assets, assets), decimal squared.
        compact: Same axes and units for the compact covariance.
        alpha: Fixed compact prediction fraction per asset, shape (assets,).
        method: A member of METHODS. Family gates use square-root fractions;
            root couplings use linear fractions and symmetric SPD matrix roots.

    Returns:
        Symmetric strictly positive-definite covariance. Variance-only retains
        half-blend correlations; correlation-only retains half-blend variances.
        Root methods assume cross-model covariance rho * root(B) * root(C).T,
        not an estimated forecast-error covariance. Independence is diagnostic
        only. No model is asserted to be a single fitted GP posterior.
    """
    if broad.shape != compact.shape or alpha.shape != broad.shape[-1:]:
        raise ValueError("Matched covariance axes and one fraction per asset required")
    if not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
        raise ValueError("Finite fractions between zero and one required")
    _check(broad)
    _check(compact)
    half = 0.5 * (broad + compact)
    if method == "broad":
        result = broad.copy()
    elif method == "compact":
        result = compact.copy()
    elif method == "half":
        result = half
    elif method in {"family_gate", "family_variance_only", "family_correlation_only"}:
        b, c = np.sqrt(1 - alpha), np.sqrt(alpha)
        gate = b[:, None] * broad * b[None, :] + c[:, None] * compact * c[None, :]
        if method == "family_gate":
            result = gate
        elif method == "family_variance_only":
            result = _rescale_diagonal(half, np.diagonal(gate, axis1=-2, axis2=-1))
        else:
            result = _rescale_diagonal(gate, np.diagonal(half, axis1=-2, axis2=-1))
    elif method in {"root_shared_100", "root_shared_90", "root_shared_50", "root_independent"}:
        rho = {"root_shared_100": 1.0, "root_shared_90": 0.9, "root_shared_50": 0.5, "root_independent": 0.0}[method]
        left, right = (1 - alpha)[:, None] * _root(broad), alpha[:, None] * _root(compact)
        cross = left @ right.swapaxes(-1, -2)
        result = left @ left.swapaxes(-1, -2) + right @ right.swapaxes(-1, -2)
        result += rho * (cross + cross.swapaxes(-1, -2))
    else:
        raise ValueError(f"Unknown covariance hypothesis: {method}")
    # Remove arithmetic roundoff asymmetry only, never alter eigenvalues.
    result = 0.5 * (result + result.swapaxes(-1, -2))
    _check(result)
    return result
