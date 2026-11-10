"""Deterministic allocation and diagnostics for approved follow-up experiments.

Research-only numerical layer. Forecasts and errors are discounted-excess decimal
returns; covariance and risk budgets are decimal squared. No fitting, I/O, CVaR,
weight caps, fallback allocations, clipping or normalization of solved weights.
Usage: imported by run_study.py and study_workers.py.
"""

from __future__ import annotations

import numpy as np
from metrics import calibration_summary, gaussian_scores, joint_gaussian_scores, project_portfolios, rank_ic
from refresh_math import _portfolio
from scipy.stats import rankdata, spearmanr


def _online_scale(actual: np.ndarray, mean: np.ndarray, covariance: np.ndarray, prior: float = 6.0) -> np.ndarray:
    # EW calibration is prequential: origin t uses errors only at origins < t.
    # No months are dropped and the initial multiplier is the explicit prior 1.
    weights = np.full(actual.shape[1], 1 / actual.shape[1])
    variance = np.einsum("a,tab,b->t", weights, covariance, weights)
    if not (variance > 0).all() or prior <= 0:
        raise ValueError("Positive calibration variances/prior required")
    z_squared = ((actual - mean) @ weights) ** 2 / variance
    accumulated = np.r_[0.0, np.cumsum(z_squared[:-1])]
    return (prior + accumulated) / (prior + np.arange(len(actual)))


def _risk(source: str, actual: np.ndarray, mean: np.ndarray, native: np.ndarray | None, history: np.ndarray) -> tuple:
    if source == "historical":
        return history, np.ones(len(actual))
    if native is None:
        raise ValueError("No learned cross-family GP covariance is available")
    if source == "GP":
        return native, np.ones(len(actual))
    if source == "GP_prequential_scale":
        factors = _online_scale(actual, mean, native)
        return native * factors[:, None, None], factors
    if source.startswith("GP_blend_"):
        fraction = float(source.removeprefix("GP_blend_"))
        if not 0 < fraction < 1:
            raise ValueError("Invalid predeclared historical blend")
        return (1 - fraction) * native + fraction * history, np.ones(len(actual))
    if source == "GP_diagonal":
        matrix = np.zeros_like(native)
        indices = np.arange(native.shape[-1])
        matrix[:, indices, indices] = np.diagonal(native, axis1=-2, axis2=-1)
        return matrix, np.ones(len(actual))
    raise ValueError(f"Unknown risk source: {source}")


def _scores(mean: np.ndarray, historical_mean: np.ndarray, objective: str) -> tuple:
    if objective in {"rank", "smooth_rank"}:
        return rankdata(mean, axis=1) / mean.shape[1], 0.25 if objective == "smooth_rank" else 1e-8
    if objective == "shrunk_return":
        values = 0.5 * mean + 0.5 * historical_mean
        centered = values - values.mean(axis=1, keepdims=True)
        scale = np.max(np.abs(centered), axis=1, keepdims=True)
        return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0), 1e-8
    raise ValueError(f"Unknown allocation objective: {objective}")


class ScoreAllocator:
    """Reuse strict convex allocations for explicit return or rank scores.

    Attributes:
        covariance: Observed risk matrices, shape (24, 15, 15), decimal squared.
        budget: Positive variance limits, shape (24,), decimal squared.
        minimum: Optional effective-count floor, never an ETF position cap.
        problems: Month-specific CVXPY problems, weight variables and parameters.
    """

    def __init__(self, covariance: np.ndarray, budget: np.ndarray, minimum: int | None, penalty: float) -> None:
        import cvxpy as cp

        if covariance.shape != (24, 15, 15) or budget.shape != (24,):
            raise ValueError("Exact month and asset axes required")
        if not np.isfinite(covariance).all() or not np.isfinite(budget).all() or not (budget > 0).all():
            raise ValueError("Finite covariance and positive variance budgets required")
        if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-14):
            raise ValueError("Asymmetric covariance; no repair")
        np.linalg.cholesky(covariance)
        if minimum is not None and (not isinstance(minimum, int) or not 1 <= minimum <= 15):
            raise ValueError("Invalid effective-count minimum")
        if penalty < 0 or not np.isfinite(penalty):
            raise ValueError("Invalid concentration penalty")
        self.covariance, self.budget, self.minimum = covariance, budget, minimum
        self.problems = []
        for index, matrix in enumerate(covariance):
            weights, scores = cp.Variable(15), cp.Parameter(15)
            constraints = [
                weights >= 0,
                cp.sum(weights) == 1,
                cp.quad_form(weights, cp.psd_wrap(matrix / budget[index])) <= 1,
            ]
            if minimum is not None:
                constraints.append(cp.sum_squares(weights) <= 1 / minimum)
            problem = cp.Problem(cp.Maximize(scores @ weights - penalty * cp.sum_squares(weights)), constraints)
            if not problem.is_dpp():
                raise ValueError("Allocation problem is not reusable")
            self.problems.append((problem, weights, scores))

    def allocate(self, scores: np.ndarray) -> list[dict]:
        """Record all 24 solves without filling failed weights or skipping months.

        Args:
            scores: Finite dimensionless objective coefficients, shape (24, 15).

        Returns:
            Monthly status, raw weights or null, concentration and variance.
        """
        import cvxpy as cp

        if scores.shape != (24, 15) or not np.isfinite(scores).all():
            raise ValueError("Incomplete objective scores")
        records = []
        for index, (problem, variable, parameter) in enumerate(self.problems):
            parameter.value = scores[index]
            record = {
                "index": index,
                "weights": None,
                "minimum_effective": self.minimum,
                "ETF_position_cap": None,
                "fallback_used": False,
                "constraint_relaxation": False,
                "variance_budget": float(self.budget[index]),
            }
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
                records.append({**record, "status": "solver_error", "error": str(error)})
                continue
            if problem.status != cp.OPTIMAL or variable.value is None:
                records.append({**record, "status": str(problem.status)})
                continue
            weights = np.asarray(variable.value).copy()
            squared = float(weights @ weights)
            variance = float(weights @ self.covariance[index] @ weights)
            if (
                not np.isfinite(weights).all()
                or weights.min() < -1e-8
                or abs(weights.sum() - 1) > 1e-7
                or variance > self.budget[index] * (1 + 1e-6)
                or (self.minimum is not None and squared > 1 / self.minimum + 1e-7)
            ):
                records.append({**record, "status": "constraint_violation"})
                continue
            records.append(
                {
                    **record,
                    "status": "solved",
                    "weights": weights.tolist(),
                    "predicted_variance": variance,
                    "effective_count": 1 / squared,
                    "largest_weight": float(weights.max()),
                }
            )
        return records


def _score(context: dict, mean: np.ndarray, covariance: np.ndarray, records: list[dict]) -> dict:
    if len(records) != 24 or [row["index"] for row in records] != list(range(24)):
        raise ValueError("Missing or reordered monthly status")
    result = {
        "months": [{**row, "date": context["dates"][row["index"]]} for row in records],
        "solved_months": sum(row["status"] == "solved" for row in records),
        "performance": None,
        "adaptive_calibration": None,
    }
    if result["solved_months"] != 24:
        return result
    weights = np.asarray([row["weights"] for row in records])
    result["performance"] = _portfolio(context, weights)
    result["performance"].update(
        mean_effective_count=float(np.mean(1 / np.sum(weights**2, axis=1))), largest_weight=float(weights.max())
    )
    projected = project_portfolios(context["actual"], mean, covariance, weights[:, None, :])
    result["adaptive_calibration"] = calibration_summary({k: v[:, 0] for k, v in projected.items()})
    return result


def _forecast_rows(
    context: dict, name: str, seed: int, mean: np.ndarray, covariance: np.ndarray, assets: list[str]
) -> tuple[list[dict], list[dict]]:
    indices = [context["assets"].index(asset) for asset in assets]
    actual = context["actual"][:, indices]
    std = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    marginal = gaussian_scores(actual, mean, std)
    rows, ranking = [], []
    for index, asset in enumerate(assets):
        prediction, observed = mean[:, index], actual[:, index]
        correlation = float(spearmanr(observed, prediction).statistic) if np.std(prediction) > 0 else None
        rows.append(
            {
                "configuration": name,
                "seed": seed,
                "asset": asset,
                "time_series_spearman": correlation,
                "prediction_sd_to_actual_sd": float(np.std(prediction, ddof=1) / np.std(observed, ddof=1)),
                "prediction_bias": float(np.mean(prediction - observed)),
                "rmse": float(np.sqrt(np.mean((prediction - observed) ** 2))),
                **calibration_summary({key: value[:, index] for key, value in marginal.items()}),
            }
        )
    scopes = {
        "pooled": assets,
        **{family: tickers for family, tickers in context["families"].items() if set(tickers).issubset(assets)},
    }
    for scope, tickers in scopes.items():
        local = [assets.index(asset) for asset in tickers]
        monthly = rank_ic(actual[:, local], mean[:, local])
        if not np.isfinite(monthly).all():
            raise ValueError("Undefined ranking month; no silent omission")
        ranking.append(
            {
                "configuration": name,
                "seed": seed,
                "scope": scope,
                "mean_ic": float(monthly.mean()),
                "monthly_ic": monthly.tolist(),
            }
        )
    return rows, ranking


def _fixed(context: dict, name: str, source: str, seed: int, mean: np.ndarray, covariance: np.ndarray) -> list[dict]:
    scores = []
    for label, fixed in context["fixed_probes"].items():
        weights = np.broadcast_to(fixed, (24, 1, 15))
        projected = project_portfolios(context["actual"], mean, covariance, weights)
        scores.append(
            {
                "configuration": name,
                "risk_source": source,
                "seed": seed,
                "portfolio": label,
                **calibration_summary({key: value[:, 0] for key, value in projected.items()}),
            }
        )
    density = joint_gaussian_scores(context["actual"], mean, covariance)
    scores.append(
        {
            "configuration": name,
            "risk_source": source,
            "seed": seed,
            "portfolio": "full_joint_density",
            **{key: float(value.mean()) for key, value in density.items()},
        }
    )
    return scores
