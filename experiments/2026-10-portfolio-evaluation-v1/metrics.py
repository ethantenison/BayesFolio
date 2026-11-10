"""Pure, vectorized scoring for monthly ETF forecasts and portfolio decisions.

Research-only numerical layer; no files, fitting, network, optimizer, or global
RNG state. Returns and predictive standard deviations are decimals; covariance
is decimal squared. Gaussian scores are losses: smaller is better, including
CRPS (the negative of the reward convention in Gneiting and Raftery, 2007).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

Array = NDArray[np.float64]


def _finite(value: Array, name: str) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite {name}; missing observations cannot be dropped")


def gaussian_scores(actual: Array, mean: Array, std: Array) -> dict[str, Array]:
    """Score univariate Gaussian forecasts without fitting or recalibration.

    Args:
        actual: Realized decimal returns, arbitrary shape.
        mean: Matching predictive means in decimal returns.
        std: Matching strictly positive predictive standard deviations.

    Returns:
        Elementwise NLL, nonnegative CRPS loss, PIT, standardized errors,
        squared standardized errors, and central interval coverage/widths.
    """
    if actual.shape != mean.shape or std.shape != mean.shape:
        raise ValueError("Actual, mean and standard deviation shapes differ")
    for value, name in [(actual, "actual"), (mean, "mean"), (std, "std")]:
        _finite(value, name)
    if not (std > 0).all():
        raise ValueError("Predictive standard deviations must be positive")
    z = (actual - mean) / std
    pit = ndtr(z)
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    result = {"nll": 0.5 * np.log(2 * np.pi) + np.log(std) + 0.5 * z**2,
              "crps": std * (z * (2 * pit - 1) + 2 * pdf - 1 / np.sqrt(np.pi)),
              "pit": pit, "z": z, "z_squared": z**2,
              "squared_error": (actual - mean)**2, "predicted_variance": std**2}
    for level in [50, 80, 95]:
        quantile = ndtri((1 + level / 100) / 2)
        result[f"coverage_{level}"] = (np.abs(z) <= quantile).astype(float)
        result[f"interval_width_{level}"] = 2 * quantile * std
    return result


def joint_gaussian_scores(actual: Array, mean: Array, covariance: Array) -> dict[str, Array]:
    """Score complete observed-return covariance, including cross-ETF dependence.

    Args:
        actual: Shape (months, assets), decimal realized target returns.
        mean: Same shape, predictive target-return means.
        covariance: Shape (months, assets, assets), observed predictive covariance
            in raw target units, including likelihood noise, not task-kernel B.

    Returns:
        Monthly joint NLL, NLL per asset, and squared Mahalanobis error per asset.

    Raises:
        ValueError: Incomplete, asymmetric or non-positive-definite covariance.
            No diagonal replacement, eigenvalue clipping, or hidden jitter.
    """
    if actual.ndim != 2 or actual.shape != mean.shape or covariance.shape != (*mean.shape, mean.shape[1]):
        raise ValueError("Joint Gaussian shapes differ")
    for value, name in [(actual, "actual"), (mean, "mean"), (covariance, "covariance")]:
        _finite(value, name)
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-14):
        raise ValueError("Predictive covariance must be symmetric")
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("Predictive covariance must be positive definite; no repair") from error
    whitened = np.linalg.solve(factor, (actual - mean)[..., None])[..., 0]
    mahalanobis = np.sum(whitened**2, axis=1)
    logdet = 2 * np.log(np.diagonal(factor, axis1=-2, axis2=-1)).sum(axis=1)
    n = mean.shape[1]
    nll = 0.5 * (n * np.log(2 * np.pi) + logdet + mahalanobis)
    return {"joint_nll": nll, "joint_nll_per_asset": nll / n, "mahalanobis_squared_per_asset": mahalanobis / n}


def project_portfolios(actual: Array, mean: Array, covariance: Array, weights: Array) -> dict[str, Array]:
    """Project a joint Gaussian onto fixed or ex-ante-selected portfolios.

    Args:
        actual: Shape (months, assets), decimal realized target returns.
        mean: Matching predictive mean array.
        covariance: Complete predictive observed-return covariance array.
        weights: Shape (months, portfolios, assets); fixed probes must use the
            same weights for every model and seed. Signed probes are supported.

    Returns:
        Projected actuals, means, standard deviations and Gaussian scores.
        Scores for adaptive allocations are decision diagnostics, not a proper
        like-for-like score of the full distribution across different weights.
    """
    joint_gaussian_scores(actual, mean, covariance)
    if weights.ndim != 3 or weights.shape[0] != len(actual) or weights.shape[2] != actual.shape[1]:
        raise ValueError("Portfolio projection weights do not align")
    _finite(weights, "weights")
    realized = np.einsum("tpa,ta->tp", weights, actual)
    expected = np.einsum("tpa,ta->tp", weights, mean)
    variance = np.einsum("tpa,tab,tpb->tp", weights, covariance, weights, optimize=True)
    if not (variance > 0).all():
        raise ValueError("Nonpositive projected variance")
    std = np.sqrt(variance)
    return {"actual": realized, "mean": expected, "std": std, **gaussian_scores(realized, expected, std)}


def calibration_summary(scores: dict[str, Array]) -> dict[str, float | int]:
    """Summarize calibration using errors about forecast means, not return volatility.

    Args:
        scores: One-dimensional projected or marginal Gaussian score arrays.

    Returns:
        Bias, mean squared standardized error (target one), empirical error RMS
        divided by predicted RMS SD (target one), coverage, width and PIT bins.
        Forecast error bias and scale are reported separately.
    """
    z = scores["z"]
    if z.ndim != 1 or len(z) < 2:
        raise ValueError("Calibration needs at least two chronological observations")
    result: dict[str, float | int] = {"months": len(z), "mean_standardized_error": float(z.mean()),
        "mean_squared_standardized_error": float(np.mean(z**2)),
        "error_rms_to_predicted_sd_rms": float(np.sqrt(
            np.mean(scores["squared_error"]) / np.mean(scores["predicted_variance"]))),
        "mean_predicted_sd": float(np.sqrt(scores["predicted_variance"]).mean()),
        "mean_nll": float(scores["nll"].mean()), "mean_crps": float(scores["crps"].mean())}
    for key, value in scores.items():
        if key.startswith(("coverage_", "interval_width_")):
            result[key] = float(value.mean())
    counts, _ = np.histogram(scores["pit"], bins=np.linspace(0, 1, 6))
    result.update({f"pit_bin_{index}_count": int(count) for index, count in enumerate(counts)})
    return result


def rank_ic(actual: Array, mean: Array) -> Array:
    """Compute monthly cross-sectional Spearman IC with explicit undefined months.

    Args:
        actual: Shape (months, assets), realized decimal returns.
        mean: Same shape, forecast means.

    Returns:
        Shape (months,) IC; constant cross-sections are NaN, never silently zero.
        Consumers must report the defined count and may not promote partial IC.
    """
    if actual.ndim != 2 or actual.shape != mean.shape or actual.shape[1] < 2:
        raise ValueError("Rank IC needs aligned cross-sections with at least two assets")
    _finite(actual, "actual")
    _finite(mean, "mean")
    left, right = rankdata(actual, axis=1), rankdata(mean, axis=1)
    left -= left.mean(axis=1, keepdims=True)
    right -= right.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1))
    result = np.full(len(actual), np.nan)
    np.divide(np.sum(left * right, axis=1), denominator, out=result, where=denominator > 0)
    return result


def forecast_skill(actual: Array, mean: Array, baseline_mean: Array) -> dict[str, Array]:
    """Compare errors to a training-only historical-mean forecast, per ETF.

    Args:
        actual: Shape (months, assets), decimal realized target returns.
        mean: Matching forecast means.
        baseline_mean: Matching expanding historical means known at each origin.

    Returns:
        Per-ETF RMSE and out-of-sample R-squared relative to historical means.
        Zero baseline error yields undefined R-squared, not invented skill.
    """
    if actual.shape != mean.shape or actual.shape != baseline_mean.shape or actual.ndim != 2:
        raise ValueError("Forecast skill arrays do not align")
    for value in [actual, mean, baseline_mean]:
        _finite(value, "forecast skill input")
    error = np.square(actual - mean).sum(axis=0)
    baseline = np.square(actual - baseline_mean).sum(axis=0)
    ratio = np.full_like(error, np.nan)
    np.divide(error, baseline, out=ratio, where=baseline > 0)
    return {"rmse": np.sqrt(error / len(actual)), "oos_r_squared_vs_expanding_mean": 1 - ratio}


def portfolio_performance(total_returns: Array, risk_free: Array, costs: Array | None = None) -> dict:
    """Evaluate account wealth and conventional arithmetic-excess-return Sharpe.

    Args:
        total_returns: Shape (months,), gross arithmetic portfolio total returns.
        risk_free: Matching realized arithmetic risk-free returns, not yields.
        costs: Explicit monthly fraction of beginning wealth, deducted once.
            None means gross performance; it never means zero costs were verified.

    Returns:
        Sharpe, total return, actual-NAV drawdown, volatility, worst month, and
        Bartlett HAC long-run Sharpe sensitivity. No turnover objective or CER
        is introduced; investor risk aversion has not been specified.
    """
    if total_returns.ndim != 1 or len(total_returns) < 3 or risk_free.shape != total_returns.shape:
        raise ValueError("Portfolio performance needs aligned monthly returns and RF")
    _finite(total_returns, "total returns")
    _finite(risk_free, "risk-free returns")
    deduction = np.zeros_like(total_returns) if costs is None else costs
    if deduction.shape != total_returns.shape or not np.isfinite(deduction).all() or (deduction < 0).any():
        raise ValueError("Costs must be aligned nonnegative beginning-wealth fractions")
    total = total_returns - deduction
    if (total <= -1).any() or (risk_free <= -1).any():
        raise ValueError("Invalid arithmetic total returns")
    excess = total - risk_free
    deviation = float(excess.std(ddof=1))
    wealth = np.r_[1.0, np.cumprod(1 + total)]
    centered = excess - excess.mean()
    lag = min(3, len(excess) - 1)
    long_variance = float(centered @ centered / len(centered))
    for k in range(1, lag + 1):
        long_variance += 2 * (1 - k / (lag + 1)) * float(centered[k:] @ centered[:-k] / len(centered))
    return {"months": len(total), "cost_basis": "gross_costs_not_supplied" if costs is None else "net_explicit_costs",
            "conventional_sharpe": float(excess.mean() / deviation * np.sqrt(12)) if deviation > 0 else None,
            "hac_long_run_sharpe_sensitivity": float(excess.mean() / np.sqrt(long_variance) * np.sqrt(12))
            if long_variance > 0 else None, "hac_bartlett_lags": lag,
            "annualized_excess_vol": deviation * np.sqrt(12),
            "annualized_mean_arithmetic_excess": float(excess.mean() * 12),
            "compounded_total_return": float(wealth[-1] - 1),
            "max_drawdown_total_nav": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
            "worst_total_return_month": float(total.min())}


def paired_block_bootstrap(left: Array, right: Array, *, seed: int = 20261020,
                           samples: int = 2000, block: int = 3) -> dict:
    """Estimate paired mean and Sharpe differences without treating seeds as months.

    Args:
        left: Shape (months,) or (months, seeds), arithmetic excess returns/IC.
        right: Same shape, comparator; both use identical resampled month indices.
        seed: Local reproducible RNG seed, independent of model seeds.
        samples: Number of circular moving-block replicates.
        block: Consecutive months per block, fixed before looking at results.

    Returns:
        Conditional exploratory 95% percentile intervals; not selection-adjusted
        confidence or evidence that reused evaluation months are untouched holdout.
        Seed metrics are averaged, never concatenated into longer return paths.
    """
    if left.shape != right.shape or left.ndim not in (1, 2) or len(left) < 3:
        raise ValueError("Paired monthly arrays do not align")
    if not 1 <= block <= len(left) or samples < 100:
        raise ValueError("Invalid bootstrap budget")
    _finite(left, "bootstrap left")
    _finite(right, "bootstrap right")
    a = left[:, None] if left.ndim == 1 else left
    b = right[:, None] if right.ndim == 1 else right
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(a), size=(samples, int(np.ceil(len(a) / block))))
    indices = ((starts[..., None] + np.arange(block)) % len(a)).reshape(samples, -1)[:, :len(a)]
    x, y = a[indices], b[indices]
    differences = (x.mean(axis=1) - y.mean(axis=1)).mean(axis=1)
    sx, sy = x.std(axis=1, ddof=1), y.std(axis=1, ddof=1)
    valid = (sx > 0).all(axis=1) & (sy > 0).all(axis=1)
    sharpe = ((x[valid].mean(axis=1) / sx[valid] - y[valid].mean(axis=1) / sy[valid]) * np.sqrt(12)).mean(axis=1)
    return {"months": len(a), "seeds_not_independent_months": a.shape[1], "bootstrap_seed": seed,
            "samples": samples, "block_months": block, "mean_difference": float(np.mean(a - b)),
            "mean_difference_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
            "sharpe_difference_ci95": np.quantile(sharpe, [0.025, 0.975]).tolist() if len(sharpe) else None,
            "defined_sharpe_replicates": int(valid.sum()),
            "interpretation": "conditional_exploratory_not_selection_adjusted"}
