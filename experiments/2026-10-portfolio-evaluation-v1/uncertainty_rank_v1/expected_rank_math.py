"""Analytic Gaussian expected ranks and deterministic pairwise diagnostics.

Research-only pure numerical layer. No fitting, I/O, calibration or optimization.
Means and realized returns are decimals; covariance is decimal squared. Ranking
uses observed-return uncertainty, including noise, not latent-mean uncertainty.
Usage: imported by run_uncertainty_rank.py and synthetic tests.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr
from scipy.stats import rankdata

METHODS = ["hard_rank", "expected_full", "expected_diagonal", "reranked_expected_full"]


def gaussian_expected_rank(mean: np.ndarray, covariance: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate expected normalized ranks under a complete Gaussian hypothesis.

    Args:
        mean: Finite means with shape (..., assets), in decimal returns.
        covariance: Matched observed-return matrices (..., assets, assets),
            including observation noise, in decimal-return squared.

    Returns:
        Pairwise P(Y_i > Y_j), difference variances, standardized mean gaps and
        expected ranks divided by the asset count. Self-probabilities are 0.5
        only as bookkeeping; no self-comparison enters the expected rank.
        Scores share the old hard-rank normalization and its zero-noise limit.

    Raises:
        ValueError: Missing, asymmetric, nonpositive-definite or mismatched data.
            No variance floors, covariance repair or probability clipping.
    """
    if mean.ndim < 1 or mean.shape[-1] < 2 or covariance.shape != (*mean.shape, mean.shape[-1]):
        raise ValueError("Matched asset and covariance axes with at least two assets required")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Finite complete Gaussian inputs required")
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-14):
        raise ValueError("Symmetric covariance required; no repair")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("Positive definite observed covariance required; no repair") from error
    assets = mean.shape[-1]
    variance = np.diagonal(covariance, axis1=-2, axis2=-1)
    difference_variance = variance[..., :, None] + variance[..., None, :] - 2 * covariance
    off_diagonal = ~np.eye(assets, dtype=bool)
    if not (difference_variance[..., off_diagonal] > 0).all():
        raise ValueError("Strictly positive variance of every nonself difference required")
    mean_gap = mean[..., :, None] - mean[..., None, :]
    z = np.divide(mean_gap, np.sqrt(difference_variance), out=np.zeros_like(mean_gap), where=off_diagonal)
    probability = ndtr(z)
    # Expected rank is 1 + sum over OTHER assets. The diagonal contributes 0.5,
    # so adding 0.5 gives exactly the same result without counting a self-win.
    scores = (0.5 + probability.sum(axis=-1)) / assets
    return {
        "scores": scores,
        "probability": probability,
        "difference_variance": difference_variance,
        "standardized_mean_gap": z,
    }


def allocation_scores(mean: np.ndarray, covariance: np.ndarray, method: str) -> dict[str, np.ndarray]:
    """Create one registered score variant while preserving allocation risk.

    Args:
        mean: Means (..., assets), decimal returns, never realized outcomes.
        covariance: Complete matched observed covariance in decimal squared.
        method: hard_rank, expected_full, expected_diagonal or
            reranked_expected_full. Diagonal changes ranking uncertainty only.

    Returns:
        Scores and the underlying Gaussian pairwise calculation. Expected scores
        are used directly unless the explicit reranking diagnostic is selected.
        Hard-rank probability arrays describe the full Gaussian hypothesis,
        not degenerate probabilities inferred from the hard rank positions.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown score method: {method}")
    scoring_covariance = covariance
    if method == "expected_diagonal":
        # Validate the original joint input before this declared ablation.
        gaussian_expected_rank(mean, covariance)
        scoring_covariance = np.zeros_like(covariance)
        indices = np.arange(mean.shape[-1])
        scoring_covariance[..., indices, indices] = np.diagonal(covariance, axis1=-2, axis2=-1)
    result = gaussian_expected_rank(mean, scoring_covariance)
    result["expected_scores"] = result["scores"].copy()
    if method == "hard_rank":
        result["scores"] = rankdata(mean, axis=-1) / mean.shape[-1]
    elif method == "reranked_expected_full":
        result["scores"] = rankdata(result["scores"], axis=-1) / mean.shape[-1]
    return result


def pairwise_diagnostics(actual: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> dict[str, np.ndarray]:
    """Score pairwise event probabilities after allocations, without calibration.

    Args:
        actual: Realized decimal returns with the same shape as mean.
        mean: Matched ex-ante predictive means in decimal returns.
        covariance: Corresponding complete Gaussian covariance, decimal squared.

    Returns:
        One record per leading seed/month position: Brier loss, stable binary
        log loss, hard-mean-order Brier, and observed tie count. Each unordered
        pair is scored once in fixed ETF-axis order; the event is strictly Y_i
        greater than Y_j, including false when an observed pair ties. Pairwise
        log loss uses log-CDFs, never clipped probabilities. Pairs and seeds
        are not independent market observations.
    """
    if actual.shape != mean.shape or not np.isfinite(actual).all():
        raise ValueError("Complete matching realized returns required for diagnostics")
    result = gaussian_expected_rank(mean, covariance)
    left, right = np.triu_indices(mean.shape[-1], k=1)
    observed = actual[..., left] > actual[..., right]
    probability = result["probability"][..., left, right]
    z = result["standardized_mean_gap"][..., left, right]
    point = (mean[..., left] > mean[..., right]).astype(float)
    point += 0.5 * (mean[..., left] == mean[..., right])
    log_loss = np.where(observed, -log_ndtr(z), -log_ndtr(-z))
    return {
        "brier": np.mean((probability - observed) ** 2, axis=-1),
        "log_loss": log_loss.mean(axis=-1),
        "hard_order_brier": np.mean((point - observed) ** 2, axis=-1),
        "observed_ties": np.sum(actual[..., left] == actual[..., right], axis=-1),
    }
