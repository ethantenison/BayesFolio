"""Pure third-model diversity and incremental frontier diagnostics.

This experiment-local numerical module performs no I/O, fitting, or allocation.
Forecasts and actuals are decimal monthly returns with seed/month/asset axes;
Pareto objectives are conventional Sharpe, compounded return, and rank IC, all
maximized. Undefined correlations are returned as None rather than fake zeros.
"""

from __future__ import annotations

import numpy as np


def _finite_array(values: np.ndarray, *, dimensions: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != dimensions or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a nonempty finite {dimensions}-dimensional array")
    return result


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first_flat = first.ravel() - np.mean(first)
    second_flat = second.ravel() - np.mean(second)
    denominator = float(np.linalg.norm(first_flat) * np.linalg.norm(second_flat))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.clip(np.dot(first_flat, second_flat) / denominator, -1, 1))


def residual_diversity(
    actual: np.ndarray,
    reference_mean: np.ndarray,
    complement_mean: np.ndarray,
) -> list[dict[str, float | None]]:
    """Measure paired forecast-error diversity separately for each seed.

    Args:
        actual: Realized decimal returns, shape (months, assets).
        reference_mean: Two-model reference means, shape (seeds, months, assets).
        complement_mean: Third-model means with exactly the same shape/alignment.

    Returns:
        One dictionary per seed. Correlations are dimensionless and None when
        undefined. RMSE is in decimal returns; error products and squared
        disagreement are in squared decimal returns. Positive correction/error
        alignment means an infinitesimal move toward the complement reduces MSE.

    Raises:
        ValueError: Inputs are missing, nonfinite, empty, or differently shaped.
    """
    observed = _finite_array(actual, dimensions=2, label="actual")
    reference = _finite_array(reference_mean, dimensions=3, label="reference_mean")
    complement = _finite_array(complement_mean, dimensions=3, label="complement_mean")
    if reference.shape != complement.shape or reference.shape[1:] != observed.shape:
        raise ValueError("Forecast and realized-return axes must match exactly")
    reports: list[dict[str, float | None]] = []
    for base, extra in zip(reference, complement, strict=True):
        base_error = base - observed
        extra_error = extra - observed
        correction = extra - base
        reports.append(
            {
                "residual_correlation": _correlation(base_error, extra_error),
                "cross_sectionally_demeaned_residual_correlation": _correlation(
                    base_error - base_error.mean(axis=-1, keepdims=True),
                    extra_error - extra_error.mean(axis=-1, keepdims=True),
                ),
                "forecast_disagreement_rmse": float(np.sqrt(np.mean(correction**2))),
                "reference_rmse": float(np.sqrt(np.mean(base_error**2))),
                "complement_rmse": float(np.sqrt(np.mean(extra_error**2))),
                "correction_error_alignment": float(-np.mean(base_error * correction)),
                "squared_disagreement": float(np.mean(correction**2)),
                "error_product": float(np.mean(base_error * extra_error)),
            }
        )
    return reports


def blend_third_model(
    reference_mean: np.ndarray,
    complement_mean: np.ndarray,
    reference_covariance: np.ndarray,
    complement_covariance: np.ndarray,
    *,
    mean_fraction: float,
    covariance_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one static third-model recipe to all seeds and months.

    Args:
        reference_mean: Decimal reference returns, shape (seeds, months, assets).
        complement_mean: Decimal third-model means with identical shape.
        reference_covariance: Reference return covariance in squared decimal
            returns, shape (seeds, months, assets, assets).
        complement_covariance: Third-model covariance with identical shape.
        mean_fraction: Global fixed third-model mean fraction in [0, 1].
        covariance_fraction: Independently fixed covariance fraction in [0, 1].

    Returns:
        Convex blended means and observed covariance arrays. No between-model
        disagreement term, covariance repair, or month-adaptive weight is added.

    Raises:
        ValueError: Inputs differ in shape, contain nonfinite values, use invalid
            fractions, or fail covariance symmetry/positive-semidefinite checks.
    """
    base = _finite_array(reference_mean, dimensions=3, label="reference_mean")
    extra = _finite_array(complement_mean, dimensions=3, label="complement_mean")
    base_cov = _finite_array(reference_covariance, dimensions=4, label="reference_covariance")
    extra_cov = _finite_array(complement_covariance, dimensions=4, label="complement_covariance")
    expected_covariance_shape = (*base.shape, base.shape[-1])
    if base.shape != extra.shape or base_cov.shape != expected_covariance_shape or extra_cov.shape != base_cov.shape:
        raise ValueError("Mean/covariance axes must match exactly")
    for fraction in (mean_fraction, covariance_fraction):
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Fractions must be finite scalars in [0, 1]")
    for covariance in (base_cov, extra_cov):
        if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-12):
            raise ValueError("Covariance must be symmetric; no implicit repair is allowed")
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-12:
            raise ValueError("Covariance must be positive semidefinite")
    return (
        (1 - mean_fraction) * base + mean_fraction * extra,
        (1 - covariance_fraction) * base_cov + covariance_fraction * extra_cov,
    )


def incremental_pareto(
    baseline_objectives: np.ndarray,
    candidate_objectives: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> dict[str, list[bool]]:
    """Identify incremental candidates on a three-objective maximize frontier.

    Args:
        baseline_objectives: Finite baseline points, shape (baselines, 3), with
            columns conventional Sharpe, compounded return, and ranking IC.
        candidate_objectives: Candidate points with the same columns and units.
        tolerance: Nonnegative absolute tolerance for dominance comparisons.

    Returns:
        Per-candidate flags for nondomination by baselines, dominance of at least
        one baseline, nondomination by all points, and a distinct expansion of
        the baseline frontier. Exact/tolerance duplicates are not expansions.
        Candidate seed means must be computed before calling; this function does
        not treat seeds as independent market histories or assign significance.

    Raises:
        ValueError: Objectives are nonfinite, empty, not three-dimensional, or
            tolerance is invalid.
    """
    baseline = _finite_array(baseline_objectives, dimensions=2, label="baseline_objectives")
    candidate = _finite_array(candidate_objectives, dimensions=2, label="candidate_objectives")
    if baseline.shape[1] != 3 or candidate.shape[1] != 3:
        raise ValueError("Exactly three maximize objectives are required")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    all_points = np.concatenate([baseline, candidate], axis=0)
    reports: dict[str, list[bool]] = {
        "not_dominated_by_baseline": [],
        "dominates_any_baseline": [],
        "on_combined_frontier": [],
        "distinct_frontier_expansion": [],
    }
    for point in candidate:
        baseline_dominates = np.all(baseline >= point - tolerance, axis=1) & np.any(
            baseline > point + tolerance, axis=1
        )
        point_dominates = np.all(point >= baseline - tolerance, axis=1) & np.any(
            point > baseline + tolerance, axis=1
        )
        all_dominate = np.all(all_points >= point - tolerance, axis=1) & np.any(
            all_points > point + tolerance, axis=1
        )
        duplicates_baseline = np.any(np.all(np.abs(baseline - point) <= tolerance, axis=1))
        nondominated = not bool(np.any(baseline_dominates))
        reports["not_dominated_by_baseline"].append(nondominated)
        reports["dominates_any_baseline"].append(bool(np.any(point_dominates)))
        reports["on_combined_frontier"].append(not bool(np.any(all_dominate)))
        reports["distinct_frontier_expansion"].append(nondominated and not bool(duplicates_baseline))
    return reports
