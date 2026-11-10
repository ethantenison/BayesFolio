"""Post-selection Gaussian predictive diagnostics for the frozen v2 decision.

This research-only adapter evaluates saved moments without fitting, allocating,
recalibrating or selecting. Targets/means are monthly discounted excess decimal
returns and observed covariance includes likelihood noise in squared decimals.
Every finalist uses its eligible14 principal matrix, excluding the historical
MGK storage placeholder. Expert15 scores are reported separately and use actual
GP covariance. Usage: uv run --locked --no-sync python <this_file> --diagnose
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm
from v2_common import ASSETS, ELIGIBLE, ELIGIBLE_INDEX, HERE, arrays, digest, read, save, verify_hashes

BLOCKS = {"development41": (0, 41), "confirmation12": (41, 53), "full53": (0, 53)}


def _projected(error: np.ndarray, covariance: np.ndarray, weights: np.ndarray) -> dict:
    projected_error = np.einsum("ta,ta->t", error, weights)
    variance = np.einsum("ta,tab,tb->t", weights, covariance, weights)
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("Positive projected variance required; no covariance repair or month dropping")
    standardized = projected_error / np.sqrt(variance)
    nll = .5 * (np.log(2 * np.pi * variance) + standardized**2)
    return {"mean_gaussian_NLL": float(nll.mean()),
            "mean_squared_standardized_error": float(np.mean(standardized**2)),
            "mean_standardized_error": float(standardized.mean()),
            "nominal95_coverage": float(np.mean(np.abs(standardized) <= norm.ppf(.975))),
            "RMSE_decimal": float(np.sqrt(np.mean(projected_error**2))),
            "mean_predicted_standard_deviation_decimal": float(np.sqrt(variance).mean()),
            "maximum_squared_standardized_error": float(np.max(standardized**2)),
            "monthly_squared_standardized_error": (standardized**2).tolist(),
            "monthly_nominal95_coverage": (np.abs(standardized) <= norm.ppf(.975)).tolist(),
            "monthly_gaussian_NLL": nll.tolist()}


def gaussian_panel(actual: np.ndarray, mean: np.ndarray, covariance: np.ndarray, *,
                   assets: list[str], portfolio_weights: np.ndarray | None = None,
                   reference_weights: np.ndarray | None = None) -> dict:
    """Score one complete Gaussian panel and optional portfolio/error projections.

    Args:
        actual: (months,tasks) decimal discounted excess realized returns.
        mean: Matching decimal predictive means.
        covariance: (months,tasks,tasks) positive-definite observed GP/risk recipe
            covariance in squared decimals, including modeled observation noise.
        assets: Exact unique aligned task identifiers.
        portfolio_weights: Optional(months,tasks) saved raw target fractions.
        reference_weights: Optional(tasks,) fixed eligible-EW reference fractions.

    Returns:
        Joint GaussianNLL per asset, marginalNLL, RMSE, descriptive calibration,
        projected Gaussian scores and explicitly noisy covariance/error-outer
        discrepancies. NLL is a decimal-unit density score and may be negative.

    Raises:
        ValueError: Nonfinite/misaligned input or invalid projected variance.
        numpy.linalg.LinAlgError: Non-SPD covariance; no repair is performed.
    """
    actual, mean, covariance = (np.asarray(value, dtype=float) for value in [actual, mean, covariance])
    if (actual.ndim != 2 or not len(actual) or mean.shape != actual.shape
            or covariance.shape != (*actual.shape, actual.shape[1]) or len(assets) != actual.shape[1]
            or len(set(assets)) != len(assets)
            or not all(np.isfinite(value).all() for value in [actual, mean, covariance])):
        raise ValueError("Complete finite aligned Gaussian mean/covariance/actual panels required")
    np.testing.assert_allclose(covariance, covariance.swapaxes(-1, -2), rtol=0, atol=1e-12)
    error = actual - mean
    cholesky = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(cholesky, error[..., None])[..., 0]
    mahalanobis = np.sum(whitened**2, axis=1)
    dimension = actual.shape[1]
    logdet = 2 * np.log(cholesky.diagonal(axis1=-2, axis2=-1)).sum(1)
    joint = .5 * (dimension * np.log(2 * np.pi) + logdet + mahalanobis)
    variance = covariance.diagonal(axis1=-2, axis2=-1)
    marginal = .5 * (np.log(2 * np.pi * variance) + error**2 / variance)
    standardized = error / np.sqrt(variance)
    offdiag = np.triu_indices(dimension, 1)
    outer = error[:, :, None] * error[:, None, :]
    predicted_offdiag = covariance[:, offdiag[0], offdiag[1]]
    realized_offdiag = outer[:, offdiag[0], offdiag[1]]
    discrepancy = predicted_offdiag - realized_offdiag
    result = {
        "months": len(actual), "tasks": dimension, "assets": assets,
        "mean_joint_gaussian_NLL_per_asset": float(np.mean(joint / dimension)),
        "mean_joint_gaussian_NLL_all_tasks": float(joint.mean()),
        "mean_joint_logdet_term_per_asset": float(.5 * np.mean(logdet) / dimension),
        "mean_joint_mahalanobis_term_per_asset": float(.5 * np.mean(mahalanobis) / dimension),
        "mean_marginal_gaussian_NLL": float(marginal.mean()),
        "RMSE_decimal": float(np.sqrt(np.mean(error**2))),
        "mean_marginal_squared_standardized_error": float(np.mean(standardized**2)),
        "mean_joint_mahalanobis_per_dimension": float(np.mean(mahalanobis / dimension)),
        "nominal95_joint_ellipsoid_coverage": float(np.mean(mahalanobis <= chi2.ppf(.95, dimension))),
        "nominal95_marginal_coverage": float(np.mean(np.abs(standardized) <= norm.ppf(.975))),
        "per_asset": {asset: {"RMSE_decimal": float(np.sqrt(np.mean(error[:, index]**2))),
            "mean_marginal_NLL": float(marginal[:, index].mean()),
            "mean_squared_standardized_error": float(np.mean(standardized[:, index]**2)),
            "nominal95_coverage": float(np.mean(np.abs(standardized[:, index]) <= norm.ppf(.975)))}
            for index, asset in enumerate(assets)},
        "covariance_error_outer_product_discrepancy": {
            "offdiagonal_monthly_MSE_decimal_power4": float(np.mean(discrepancy**2)) if len(offdiag[0]) else None,
            "aggregate_offdiagonal_gap_RMSE_decimal_squared": float(np.sqrt(np.mean(
                np.mean(discrepancy, axis=0)**2))) if len(offdiag[0]) else None,
            "mean_predicted_offdiagonal_decimal_squared": float(predicted_offdiag.mean())
                if len(offdiag[0]) else None,
            "mean_residual_cross_product_decimal_squared": float(realized_offdiag.mean())
                if len(offdiag[0]) else None,
            "interpretation": "One residual vector per month gives a noisy outer product, not true covariance. "
                "Mean error/misspecification contributes; averaging dates mixes market regimes. "
                "This discrepancy is descriptive, not an identified covariance accuracy estimate or proper score."},
        "monthly_joint_NLL_per_asset": (joint / dimension).tolist(),
        "monthly_joint_mahalanobis_per_dimension": (mahalanobis / dimension).tolist(),
    }
    if portfolio_weights is not None:
        weights = np.asarray(portfolio_weights, dtype=float)
        if weights.shape != actual.shape or not np.isfinite(weights).all():
            raise ValueError("Saved portfolio weights must align with the complete panel")
        result["allocated_portfolio_projection"] = _projected(error, covariance, weights)
    if reference_weights is not None:
        reference = np.asarray(reference_weights, dtype=float)
        if reference.shape != (dimension,) or not np.isfinite(reference).all():
            raise ValueError("Fixed reference weights must align with task dimensions")
        result["fixed_eligible_EW_reference"] = _projected(error, covariance, np.broadcast_to(reference, actual.shape))
    return result


def _blocks(actual: np.ndarray, mean: np.ndarray, covariance: np.ndarray, assets: list[str],
            dates: np.ndarray, weights: np.ndarray | None = None) -> dict:
    reference = np.array([0. if asset == "MGK" else 1 / 14 for asset in assets])
    result = {}
    for name, (start, end) in BLOCKS.items():
        score = gaussian_panel(actual[start:end], mean[start:end], covariance[start:end], assets=assets,
            portfolio_weights=None if weights is None else weights[start:end], reference_weights=reference)
        score.update(first_forecast_origin=str(dates[start]), last_forecast_origin=str(dates[end - 1]),
                     all_outcomes_previously_inspected=True, selection_input=False)
        result[name] = score
    return result


def produce_diagnostics() -> dict:
    """Write numerical diagnostics while proving frozen selection files unchanged.

    Returns:
        Passed post-selection diagnostic report with finalist14 and expert15
        sections, temporal blocks, exact source hashes and before/after hashes.
        No policy/fit/weight changes or uncertainty recalibration are performed.
    """
    protected = [HERE / name for name in ["FINAL_SELECTION.json", "CONFIG.json", "NOVEMBER_WEIGHTS.csv",
        "ROOT_VALIDATION.json", "COMPARISON.json", "SHORTLIST_FROZEN.json"]]
    before = {str(path): digest(path) for path in protected}
    selection, comparison = read(HERE / "FINAL_SELECTION.json"), read(HERE / "COMPARISON.json")
    if not selection["passed"] or comparison["selected_id"] != selection["selected_id"]:
        raise ValueError("Passed frozen selection and matched comparison required")
    context_path = HERE / "outputs/context.npz"
    context = arrays(context_path)
    if (context["assets"].tolist() != ASSETS or context["actual"].shape != (53, 15)
            or context["dates"].tolist() != comparison["dates53"]):
        raise ValueError("Complete canonical context required")
    source_hashes = {str(context_path): digest(context_path)}
    finalists, experts = {}, {}
    for record in comparison["records"]:
        path = Path(record["artifact_path"])
        fingerprint = digest(path)
        if fingerprint != record["full"]["arrays_sha256"]:
            raise ValueError("Frozen finalist moments changed")
        panel = arrays(path)
        np.testing.assert_array_equal(panel["assets"], ASSETS)
        np.testing.assert_array_equal(panel["dates"], context["dates"])
        np.testing.assert_array_equal(panel["actual"], context["actual"])
        if np.any(panel["weights"][:, ASSETS.index("MGK")] != 0):
            raise ValueError("Finalist has nonzero MGK weight")
        covariance = panel["covariance"][:, ELIGIBLE_INDEX][:, :, ELIGIBLE_INDEX]
        finalists[record["id"]] = {"policy_label": record["policy"]["label"],
            "selected": record["id"] == selection["selected_id"],
            "covariance_fractions": record["policy"]["covariance_fractions"],
            "covariance_experts": record["policy"]["covariance_experts"],
            "blocks": _blocks(panel["actual"][:, ELIGIBLE_INDEX], panel["mean"][:, ELIGIBLE_INDEX],
                              covariance, ELIGIBLE, context["dates"], panel["weights"][:, ELIGIBLE_INDEX])}
        source_hashes[str(path)] = fingerprint
    fit_seal = read(HERE / "FITS_COMPLETED.json")
    for path in sorted((HERE / "outputs/experts").glob("*.npz")):
        fingerprint = digest(path)
        if fit_seal["artifact_sha256"].get(str(path)) != fingerprint:
            raise ValueError("Unsealed expert moment panel")
        panel = arrays(path)
        np.testing.assert_array_equal(panel["assets"], ASSETS)
        np.testing.assert_array_equal(panel["seeds"], [27, 127, 227])
        np.testing.assert_array_equal(panel["dates"][:53], context["dates"])
        mean, covariance = panel["mean"][:, :53].mean(0), panel["covariance"][:, :53].mean(0)
        experts[path.stem] = {"seed_pooling": "Arithmetic mean of means and observed covariance across27/127/227; "
            "no between-seed disagreement covariance added", "portfolio_allocation": False,
            "all15_GP_tasks": _blocks(context["actual"], mean, covariance, ASSETS, context["dates"]),
            "eligible14_comparable_view": _blocks(context["actual"][:, ELIGIBLE_INDEX], mean[:, ELIGIBLE_INDEX],
                covariance[:, ELIGIBLE_INDEX][:, :, ELIGIBLE_INDEX], ELIGIBLE, context["dates"])}
        source_hashes[str(path)] = fingerprint
    if len(finalists) != 10 or len(experts) != 4:
        raise ValueError("Exactly ten frozen finalists and four GP experts required")
    verify_hashes(before)
    result = {"passed": True, "diagnostics_only_not_selection": True, "recalibration_performed": False,
        "selected_id_unchanged": selection["selected_id"], "final_selection_unchanged": True,
        "protected_sha256_before": before, "protected_sha256_after": {name: digest(Path(name)) for name in before},
        "source_sha256": source_hashes, "code_sha256": digest(Path(__file__)),
        "units": {"targets_and_means": "monthly discounted excess decimal return",
            "covariance": "observed predictive/risk recipe covariance in squared decimals",
            "NLL": "negative natural-log Gaussian density in decimal-return units; negative values are valid",
            "RMSE": "decimal return; multiply by100 for percentage points"},
        "interpretation": [
            "Every block is retrospective and previously inspected; these scores do not change the frozen decision.",
            "Joint NLL combines logdet and mean-error terms; it scores a Gaussian distribution, not covariance alone.",
            "Covariance recipes pool operational second moments, omitting between-model disagreement covariance.",
            "Marginal/portfolio squared standardized errors target1 and nominal95 coverage targets0.95 only under the "
                "stated predictive Gaussian assumptions. These are aggregate diagnostics, not recalibration.",
            "Portfolio total-return errors/standard deviations multiply by the same(1+RF); standardized errors and "
                "coverage are unchanged. Reported NLL remains in discounted-excess units.",
            "Each month supplies one residual vector. Offdiagonal/error-outer-product discrepancies cannot identify "
                "true time-varying covariance; allocation performance is not proof of covariance accuracy.",
            "Finalist scores use only14eligible tasks and exclude the MGK historical placeholder principal row/column. "
                "Separate15-task expert scores use full GP observed covariance and includeMGK as a prediction task.",
            "Fixed eligible-EW projection uses identical14-ETF weights across all candidates; allocated projections "
                "use each candidate's saved raw weights and are not interchangeable distribution comparisons.",
            "Coverage from12 confirmation months is coarse; seeds are not independent market evidence.",
        ], "finalists_eligible14": finalists, "experts": experts}
    save(HERE / "DISTRIBUTION_DIAGNOSTICS.json", result)
    save(HERE / "DISTRIBUTION_DIAGNOSTICS.md", _markdown(result).encode())
    verify_hashes(before)
    return result


def _markdown(report: dict) -> str:
    lines = ["# Post-selection predictive distribution diagnostics", "",
        "Numerical diagnostics only. Configuration and November weights were frozen first and remain unchanged. "
        "No fit, optimization, covariance repair or recalibration was performed.", "",
        "All53 outcomes and the development41/confirmation12 subsets were previously inspected. "
        "NLL uses natural-log Gaussian densities in decimal discounted-excess return units; negative NLL is valid. "
        "RMSE is in percentage points. Lower NLL/RMSE are descriptive diagnostics, not a new selection rule.", "",
        "## Frozen finalists: all53 months, eligible14 tasks", "",
        "| Policy | Joint NLL/asset | Marginal NLL | RMSE(pp) | Portfolio mean z² | Portfolio95 coverage | EW NLL |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in report["finalists_eligible14"].items():
        score = value["blocks"]["full53"]
        projection = score["allocated_portfolio_projection"]
        marker = " **selected**" if value["selected"] else ""
        lines.append(f"| `{name}`{marker} | {score['mean_joint_gaussian_NLL_per_asset']:.4f} | "
            f"{score['mean_marginal_gaussian_NLL']:.4f} | {100 * score['RMSE_decimal']:.3f} | "
            f"{projection['mean_squared_standardized_error']:.3f} | {projection['nominal95_coverage']:.1%} | "
            f"{score['fixed_eligible_EW_reference']['mean_gaussian_NLL']:.4f} |")
    lines += ["", "## Expert diagnostics: seed-pooled forecasts, no allocation", "",
        "Joint15 and joint14 scores use different task sets and should be compared within their own columns. "
        "Every expert retained all15 GP tasks; the14-task view simply excludesMGK for a matched comparison.", "",
        "| Expert | Joint15 NLL/asset | Joint14 NLL/asset | Marginal15 NLL | RMSE15(pp) | Joint15 mean Mahalanobis/d |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, value in report["experts"].items():
        score, eligible = value["all15_GP_tasks"]["full53"], value["eligible14_comparable_view"]["full53"]
        lines.append(f"| `{name}` | {score['mean_joint_gaussian_NLL_per_asset']:.4f} | "
            f"{eligible['mean_joint_gaussian_NLL_per_asset']:.4f} | {score['mean_marginal_gaussian_NLL']:.4f} | "
            f"{100 * score['RMSE_decimal']:.3f} | {score['mean_joint_mahalanobis_per_dimension']:.3f} |")
    lines += ["", "## Selected configuration across temporal blocks", "",
        "| Block | Months | Joint NLL/asset | Portfolio mean z² | Portfolio95 coverage | EW mean z² |",
        "|---|---:|---:|---:|---:|---:|"]
    chosen = report["finalists_eligible14"][report["selected_id_unchanged"]]
    for name, score in chosen["blocks"].items():
        allocated, fixed = score["allocated_portfolio_projection"], score["fixed_eligible_EW_reference"]
        lines.append(f"| {name} | {score['months']} | {score['mean_joint_gaussian_NLL_per_asset']:.4f} | "
            f"{allocated['mean_squared_standardized_error']:.3f} | {allocated['nominal95_coverage']:.1%} | "
            f"{fixed['mean_squared_standardized_error']:.3f} |")
    lines += ["", "## Interpretation limits", ""]
    lines += ["- " + text for text in report["interpretation"]]
    lines += ["", "The JSON includes every temporal block, monthly NLL/z²/coverage values, per-ETF metrics, "
        "offdiagonal discrepancies and source/decision hashes. Decision files matched before and after.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true", required=True)
    parser.parse_args()
    produce_diagnostics()
