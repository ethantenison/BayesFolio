"""Pure orchestration of ranking, uncertainty and portfolio evaluation arrays.

No fitting or I/O. Same fixed probes across models; adaptive-weight distribution
scores are separately labeled decision diagnostics. All forecasts remain in raw
discounted-excess decimal units; Sharpe and NAV use explicit return conversion.
"""

from __future__ import annotations

import numpy as np
from metrics import (
    calibration_summary,
    forecast_skill,
    gaussian_scores,
    joint_gaussian_scores,
    portfolio_performance,
    project_portfolios,
    rank_ic,
)
from scipy.stats import rankdata


def evaluate_model(run: dict, context: dict, covariance: np.ndarray | None, covariance_meta: dict) -> dict:
    """Evaluate one seed with explicit forecast, risk and economic-score layers.

    Args:
        run: Actual/mean/std arrays and validated saved legacy allocations.
        context: Dates/assets/families, causal baseline means/common covariance,
            fixed probe weights and resolved monthly arithmetic risk-free returns.
        covariance: Complete observed GP predictive covariance, or None.
        covariance_meta: Required source precision labels from artifact validation.

    Returns:
        JSON-compatible score records plus monthly return paths for paired tests.
        No automatic promotion, weighted composite score, hypervolume, turnover
        objective or inferred investor risk threshold is used.
    """
    actual, mean, std = run["actual"], run["mean"], run["std"]
    if actual.shape != (len(context["dates"]), len(context["assets"])):
        raise ValueError("Evaluation axes differ from context")
    marginal = gaussian_scores(actual, mean, std)
    skill = forecast_skill(actual, mean, context["baseline_mean"])
    etfs = []
    for index, asset in enumerate(context["assets"]):
        selected = {name: values[:, index] for name, values in marginal.items()}
        etfs.append({"asset": asset, **calibration_summary(selected),
                     **{key: float(values[index]) if np.isfinite(values[index]) else None
                        for key, values in skill.items()}})
    scopes = {"all": context["assets"], **context["families"]}
    ranking, monthly_ranking, family_ic = [], [], []
    for scope, assets in scopes.items():
        indices = [context["assets"].index(asset) for asset in assets]
        values = rank_ic(actual[:, indices], mean[:, indices])
        defined = np.isfinite(values)
        complete = bool(defined.all())
        ranking.append({"scope": scope, "assets": len(assets), "months": len(values),
            "defined_ic_months": int(defined.sum()), "complete_ic": complete,
            "mean_ic": float(values.mean()) if complete else None,
            "first_12_mean_ic": float(values[:12].mean()) if defined[:12].all() else None,
            "last_12_mean_ic": float(values[12:].mean()) if defined[12:].all() else None,
            "positive_ic_fraction": float((values > 0).mean()) if complete else None})
        monthly_ranking.extend({"date": date, "scope": scope, "ic": float(value) if np.isfinite(value) else None}
                               for date, value in zip(context["dates"], values, strict=True))
        if scope != "all":
            family_ic.append(values)
    balanced = np.mean(family_ic, axis=0)
    ranking.append({"scope": "equal_family_average", "assets": len(context["assets"]),
                    "months": len(actual), "defined_ic_months": int(np.isfinite(balanced).sum()),
                    "complete_ic": bool(np.isfinite(balanced).all()),
                    "mean_ic": float(balanced.mean()) if np.isfinite(balanced).all() else None})
    if covariance is None:
        joint = {"status": "unavailable_no_complete_covariance", **covariance_meta}
    else:
        joint_scores = joint_gaussian_scores(actual, mean, covariance)
        joint = {"status": "scored", **covariance_meta,
                 **{key: float(value.mean()) for key, value in joint_scores.items()},
                 "monthly": {key: value.tolist() for key, value in joint_scores.items()}}
    weights = {name: np.broadcast_to(value, actual.shape) for name, value in context["fixed_probes"].items()}
    ranks = rankdata(mean, axis=1)
    weights.update(linear_rank=ranks / ranks.sum(axis=1, keepdims=True), legacy_maxret=run["legacy_weights"])
    names = list(weights)
    stacked = np.stack(list(weights.values()), axis=1)
    if (not np.isfinite(stacked).all() or stacked.min() < -1e-9
            or not np.allclose(stacked.sum(axis=-1), 1, rtol=0, atol=1e-7)):
        raise ValueError("Invalid portfolio weights; no normalization or fallback")
    distribution_rows = []
    covariance_options = [(context["historical_covariance"], "common_training_ledoit_wolf_not_gp")]
    if covariance is not None:
        covariance_options.append((covariance, covariance_meta["source"]))
    for cov, source in covariance_options:
        projected = project_portfolios(actual, mean, cov, stacked)
        for index, name in enumerate(names):
            selection = {key: values[:, index] for key, values in projected.items()}
            role = "fixed_reference" if name in context["fixed_probes"] else "adaptive_decision_diagnostic"
            distribution_rows.append({"portfolio": name,
                "comparison_role": role,
                "covariance_source": source, **calibration_summary(selection)})
    portfolios, paths = [], {}
    for name, allocation in weights.items():
        discounted_excess = np.sum(allocation * actual, axis=1)
        # Fully invested portfolios preserve this affine target conversion.
        total = (1 + discounted_excess) * (1 + context["risk_free"]) - 1
        arithmetic_excess = total - context["risk_free"]
        paths[name] = {"total": total.tolist(), "arithmetic_excess": arithmetic_excess.tolist(),
                       "discounted_excess_legacy": discounted_excess.tolist()}
        portfolios.append({"portfolio": name, **portfolio_performance(total, context["risk_free"]),
            "mean_effective_asset_count": float(np.mean(1 / np.sum(allocation**2, axis=1))),
            "largest_weight": float(allocation.max()),
            "role": "fixed_benchmark" if name in context["fixed_probes"] else "forecast_driven",
            "allocator": "saved_return_maximization_not_risk_budgeted" if name == "legacy_maxret" else name})
    return {"ranking": ranking, "monthly_ranking": monthly_ranking, "etfs": etfs,
            "joint_distribution": joint, "portfolio_distributions": distribution_rows,
            "portfolios": portfolios, "monthly_portfolio_paths": paths,
            "mean_marginal_nll": float(marginal["nll"].mean()),
            "mean_oos_r_squared_equal_etf": float(np.mean(skill["oos_r_squared_vs_expanding_mean"]))
            if np.isfinite(skill["oos_r_squared_vs_expanding_mean"]).all() else None}
