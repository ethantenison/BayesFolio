"""No-position-cap allocation and complete per-configuration research scoring.

Usage: imported by full_rerun.py and its spawn-safe evaluation workers. Forecast
targets are discounted-excess decimal returns; covariance is decimal squared.
GP-native and historical-risk policies use the same absolute historical-EW risk
budget. Never construct cross-family GP covariance or replace failed weights.
"""

from __future__ import annotations

import numpy as np
from metrics import calibration_summary, gaussian_scores, project_portfolios
from refresh_math import _density, _failed_portfolio, _forecast, _portfolio, _ranking
from scipy.stats import rankdata


class NoCapAllocator:
    """Cache strict rank-allocation problems without an ETF upper-weight bound.

    Attributes:
        covariance: Monthly risk matrices, decimal squared, shape (24, 15, 15).
        budget: Fixed absolute monthly variance limits from historical EW risk.
        problems: DPP problems, decision variables and rank-score parameters.
    """

    def __init__(self, covariance: np.ndarray, budget: np.ndarray) -> None:
        import cvxpy as cp

        if (
            covariance.shape != (24, 15, 15)
            or budget.shape != (24,)
            or not np.isfinite(covariance).all()
            or not np.isfinite(budget).all()
            or not (budget > 0).all()
        ):
            raise ValueError("Incomplete risk matrices or absolute variance budgets")
        if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-14):
            raise ValueError("Asymmetric covariance; no repair")
        np.linalg.cholesky(covariance)
        self.covariance, self.budget, self.problems = covariance, budget, []
        for index, matrix in enumerate(covariance):
            weights, scores = cp.Variable(15), cp.Parameter(15)
            # No weights <= cap constraint. Diversification is a separate,
            # inherited policy and still limits concentrated allocations.
            constraints = [
                weights >= 0,
                cp.sum(weights) == 1,
                cp.sum_squares(weights) <= 1 / 3,
                cp.quad_form(weights, cp.psd_wrap(matrix / budget[index])) <= 1,
            ]
            problem = cp.Problem(cp.Maximize(scores @ weights - 1e-8 * cp.sum_squares(weights)), constraints)
            if not problem.is_dpp():
                raise ValueError("Expected reusable parameterized allocation")
            self.problems.append((problem, weights, scores))

    def solve(self, mean: np.ndarray) -> np.ndarray:
        """Solve all months without position caps, retries, or fallback weights.

        Args:
            mean: Shape (24, 15), finite raw expected-return forecasts.

        Returns:
            Raw solver weights, shape (24, 15), with no clipping/normalization.

        Raises:
            RuntimeError: Any failed solve or violated inherited constraint.
        """
        import cvxpy as cp

        if mean.shape != (24, 15) or not np.isfinite(mean).all():
            raise ValueError("Incomplete forecast panel")
        ranks = rankdata(mean, axis=1) / 15
        rows = []
        for index, (problem, variable, parameter) in enumerate(self.problems):
            parameter.value = ranks[index]
            try:
                problem.solve(
                    solver="CLARABEL",
                    warm_start=False,
                    enforce_dpp=True,
                    tol_gap_abs=1e-9,
                    tol_gap_rel=1e-9,
                    tol_feas=1e-9,
                )
            except cp.error.SolverError as error:
                raise RuntimeError(f"Month index {index}: solver error; no fallback: {error}") from error
            if problem.status != cp.OPTIMAL or variable.value is None:
                raise RuntimeError(f"Month index {index}: {problem.status}; no fallback")
            weights = np.asarray(variable.value).copy()
            variance = weights @ self.covariance[index] @ weights
            if (
                not np.isfinite(weights).all()
                or weights.min() < -1e-8
                or abs(weights.sum() - 1) > 1e-7
                or weights @ weights > 1 / 3 + 1e-7
                or variance > self.budget[index] * (1 + 1e-6)
            ):
                raise RuntimeError(f"Month index {index}: constraint violation; no repair")
            rows.append(weights)
        return np.asarray(rows)


def _fixed_distributions(
    actual: np.ndarray, mean: np.ndarray, covariance: np.ndarray, probes: dict[str, np.ndarray]
) -> list[dict]:
    rows = []
    for name, allocation in probes.items():
        weights = np.broadcast_to(allocation, (len(actual), 1, len(allocation)))
        projected = project_portfolios(actual, mean, covariance, weights)
        rows.append(
            {
                "portfolio": name,
                "role": "fixed_reference",
                "covariance_source": "exact_observed_GP",
                **calibration_summary({key: value[:, 0] for key, value in projected.items()}),
            }
        )
    return rows


def _score_seed(
    values: dict, candidate: dict, context: dict, anchor: np.ndarray, common_allocator: NoCapAllocator
) -> dict:
    indices = [context["assets"].index(a) for a in candidate["tickers"]]
    actual = context["actual"][:, indices]
    mean, std, covariance = values["mean"], values["std"], values["covariance"]
    forecast = _forecast(actual, mean, std, context["baseline_mean"][:, indices])
    density = _density(actual, mean, covariance)
    marginal = gaussian_scores(actual, mean, std)
    etfs = [
        {"asset": asset, **calibration_summary({key: value[:, offset] for key, value in marginal.items()})}
        for offset, asset in enumerate(candidate["tickers"])
    ]
    probes = (
        context["fixed_probes"]
        if candidate["family"] == "joint"
        else {"equal_weight_within_family": np.ones(len(indices)) / len(indices)}
    )
    fixed_gp = _fixed_distributions(actual, mean, covariance, probes)
    full = anchor.copy()
    full[:, indices] = mean
    scopes = {"all": context["assets"], **context["families"]}
    ranking = {
        name: _ranking(
            context["actual"][:, [context["assets"].index(a) for a in assets]],
            full[:, [context["assets"].index(a) for a in assets]],
        )
        for name, assets in scopes.items()
    }
    allocations, portfolios = {}, {}
    solvers = {"historical_risk_no_cap": common_allocator}
    if candidate["family"] == "joint":
        solvers["gp_risk_no_cap"] = NoCapAllocator(covariance, context["risk_budget"])
    else:
        portfolios["gp_risk_no_cap"] = {
            "status": "not_applicable_no_cross_family_GP_covariance",
            "fallback_used": False,
        }
    for policy, solver in solvers.items():
        try:
            weights = solver.solve(full)
            allocations[policy] = weights
            portfolios[policy] = _portfolio(context, weights)
        except RuntimeError as error:
            portfolios[policy] = _failed_portfolio(error)
    ranks = rankdata(full, axis=1)
    allocations["linear_rank"] = ranks / ranks.sum(axis=1, keepdims=True)
    allocations.update(
        {name: np.broadcast_to(weights, full.shape) for name, weights in context["fixed_probes"].items()}
    )
    for name, weights in allocations.items():
        if name not in portfolios:
            portfolios[name] = _portfolio(context, weights)
    distributions = []
    covariances = [(context["historical_covariance"], "common_training_LedoitWolf_not_GP")]
    if candidate["family"] == "joint":
        covariances.append((covariance, "exact_observed_GP"))
    for risk, source in covariances:
        for policy, weights in allocations.items():
            projected = project_portfolios(context["actual"], full, risk, weights[:, None, :])
            distributions.append(
                {
                    "portfolio": policy,
                    "covariance_source": source,
                    "role": "fixed_reference" if policy in context["fixed_probes"] else "adaptive_decision_diagnostic",
                    **calibration_summary({key: value[:, 0] for key, value in projected.items()}),
                }
            )
    return {
        "forecast": forecast,
        "joint_distribution": density,
        "etfs": etfs,
        "fixed_GP_portfolio_distributions": fixed_gp,
        "full_portfolio_ranking": ranking,
        "equal_family_ic": float(np.mean([ranking[name]["mean_ic"] for name in context["families"]])),
        "portfolios": portfolios,
        "portfolio_distributions": distributions,
        "comparison": "direct_joint" if candidate["family"] == "joint" else "conditional_family_replacement",
        "ETF_position_cap": None,
    }


def _aggregate_results(candidate: dict, results: list[dict]) -> dict:
    result = {
        "configuration": candidate["configuration"],
        "family": candidate["family"],
        "inputs": candidate["inputs"],
        "input_count_excluding_time": len(candidate["inputs"]) - 1,
        "seed_count": len(results),
        "ETF_position_cap": None,
        "task_count": len(candidate["tickers"]),
        "portfolio_comparison": "direct_joint" if candidate["family"] == "joint" else "conditional_family_replacement",
    }
    metrics = {
        "mean_ic": [r["forecast"]["mean_ic"] for r in results],
        "pooled_15ETF_ic": [r["full_portfolio_ranking"]["all"]["mean_ic"] for r in results],
        "equal_family_ic": [r["equal_family_ic"] for r in results],
        **{
            key: [r["forecast"][key] for r in results]
            for key in [
                "mean_marginal_nll",
                "mean_crps",
                "rmse",
                "coverage_95",
                "mean_predicted_sd",
                "error_rms_to_predicted_sd_rms",
                "mean_oos_r_squared",
            ]
        },
        **{
            key: [r["joint_distribution"][key] for r in results]
            for key in ["joint_nll", "joint_nll_per_asset", "mahalanobis_squared_per_asset"]
        },
    }
    for key, values in metrics.items():
        result[key] = float(np.mean(values))
        result[f"seed_sd_{key}"] = float(np.std(values, ddof=1))
    for policy in ["historical_risk_no_cap", "gp_risk_no_cap", "linear_rank", "equal_weight", "fixed_60_40"]:
        rows = [r["portfolios"][policy] for r in results]
        complete = all(row["status"] == "solved" for row in rows)
        result[f"{policy}_complete_seeds"] = sum(row["status"] == "solved" for row in rows)
        for key in [
            "conventional_sharpe",
            "compounded_total_return",
            "max_drawdown_total_nav",
            "annualized_excess_vol",
            "first_12_sharpe",
            "last_12_sharpe",
        ]:
            result[f"{policy}_{key}"] = float(np.mean([row[key] for row in rows])) if complete else None
    return result
