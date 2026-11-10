"""Paired risk-budget-only and effective-count portfolio allocation.

Usage: imported by run_comparison.py and policy_workers.py. This research-only
module does not fit GPs or perform I/O. Means are discounted-excess decimals;
covariance and fixed historical-EW budgets are decimal squared. No ETF cap,
CVaR, constraint relaxation, clipping, or previous-weight fallback is present.
"""

from __future__ import annotations

import numpy as np
from metrics import calibration_summary, project_portfolios
from refresh_math import _portfolio
from scipy.stats import rankdata


class PolicyAllocator:
    """Cache 24 independent allocations with an optional effective-count floor.

    Attributes:
        covariance: Complete risk matrices, shape (24, 15, 15), decimal squared.
        budget: Fixed variance limits, shape (24,), identical across policies.
        minimum_effective: None for no floor, otherwise a value between 1 and 15.
        problems: Month-specific problem, weights variable and rank parameter.
    """

    def __init__(self, covariance: np.ndarray, budget: np.ndarray, minimum_effective: int | None = None) -> None:
        import cvxpy as cp

        if (
            covariance.shape != (24, 15, 15)
            or budget.shape != (24,)
            or not np.isfinite(covariance).all()
            or not np.isfinite(budget).all()
            or not (budget > 0).all()
        ):
            raise ValueError("Incomplete covariance or positive absolute variance budgets")
        if minimum_effective is not None and not 1 <= minimum_effective <= 15:
            raise ValueError("Invalid effective-count floor")
        if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-14):
            raise ValueError("Asymmetric covariance; no repair")
        np.linalg.cholesky(covariance)
        self.covariance, self.budget, self.minimum_effective = covariance, budget, minimum_effective
        self.problems = []
        for index, matrix in enumerate(covariance):
            weights, parameter = cp.Variable(15), cp.Parameter(15)
            constraints = [weights >= 0, cp.sum(weights) == 1]
            if minimum_effective is not None:
                constraints.append(cp.sum_squares(weights) <= 1 / minimum_effective)
            constraints.append(cp.quad_form(weights, cp.psd_wrap(matrix / budget[index])) <= 1)
            # Same deterministic tiny tie-break in both policies; it is not a
            # hard count floor and does not prevent a feasible 100% position.
            objective = cp.Maximize(parameter @ weights - 1e-8 * cp.sum_squares(weights))
            problem = cp.Problem(objective, constraints)
            if not problem.is_dpp():
                raise ValueError("Expected reusable allocation problems")
            self.problems.append((problem, weights, parameter))

    def allocate(self, mean: np.ndarray) -> list[dict]:
        """Record every month, including explicit failed solves without weights.

        Args:
            mean: Finite expected returns, shape (24, 15), decimal target units.

        Returns:
            Twenty-four monthly records. Failed months contain weights=None;
            later months are still solved, not dropped or filled from the past.
        """
        import cvxpy as cp

        if mean.shape != (24, 15) or not np.isfinite(mean).all():
            raise ValueError("Incomplete forecast panel")
        ranks, records = rankdata(mean, axis=1) / 15, []
        for index, (problem, variable, parameter) in enumerate(self.problems):
            parameter.value = ranks[index]
            record = {
                "index": index,
                "weights": None,
                "minimum_effective": self.minimum_effective,
                "ETF_position_cap": None,
                "variance_budget": float(self.budget[index]),
                "fallback_used": False,
                "constraint_relaxation": False,
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
                records.append({**record, "status": str(problem.status), "error": "Strict solve failed; no fallback"})
                continue
            weights = np.asarray(variable.value).copy()
            squared = float(weights @ weights)
            variance = float(weights @ self.covariance[index] @ weights)
            if (
                not np.isfinite(weights).all()
                or weights.min() < -1e-8
                or abs(weights.sum() - 1) > 1e-7
                or variance > self.budget[index] * (1 + 1e-6)
                or (self.minimum_effective is not None and squared > 1 / self.minimum_effective + 1e-7)
            ):
                records.append({**record, "status": "constraint_violation", "error": "No clipping or normalization"})
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


def _policies(family: str) -> dict[str, int | None]:
    return {"risk_budget_only": None, "risk_budget_plus_effective": 10 if family == "joint" else 3}


def _score_allocations(
    context: dict, mean: np.ndarray, records: list[dict], covariances: dict[str, np.ndarray]
) -> dict:
    if len(records) != 24 or [record["index"] for record in records] != list(range(24)):
        raise ValueError("Every original month must have exactly one allocation status")
    complete = len(records) == 24 and all(record["status"] == "solved" for record in records)
    output = {
        "status": "solved" if complete else "incomplete_no_performance_no_fallback",
        "solved_months": sum(record["status"] == "solved" for record in records),
        "months": [{**row, "date": context["dates"][row["index"]]} for row in records],
        "ETF_position_cap": None,
        "performance": None,
        "risk_calibration": [],
    }
    if not complete:
        return output
    weights = np.array([row["weights"] for row in records])
    output["performance"] = _portfolio(context, weights)
    output["performance"].update(
        mean_effective_count=float(np.mean(1 / np.sum(weights**2, axis=1))), largest_weight=float(weights.max())
    )
    for source, covariance in covariances.items():
        projected = project_portfolios(context["actual"], mean, covariance, weights[:, None, :])
        output["risk_calibration"].append(
            {
                "covariance_source": source,
                "role": "adaptive_decision_diagnostic_not_fixed_portfolio_comparison",
                **calibration_summary({key: values[:, 0] for key, values in projected.items()}),
            }
        )
    return output


def _summary(candidate: dict, rows: list[dict]) -> list[dict]:
    results = []
    for source in ["historical", "GP"] if candidate["family"] == "joint" else ["historical"]:
        for variant, minimum in _policies(candidate["family"]).items():
            key = f"{source}__{variant}"
            selected = [row["portfolios"][key] for row in rows]
            complete = all(row["status"] == "solved" for row in selected)
            result = {
                "configuration": candidate["configuration"],
                "family": candidate["family"],
                "covariance_policy": source,
                "variant": variant,
                "minimum_effective": minimum,
                "ETF_position_cap": None,
                "complete_seeds": sum(row["status"] == "solved" for row in selected),
                "solved_months": sum(row["solved_months"] for row in selected),
                "comparison": "direct_joint" if candidate["family"] == "joint" else "conditional_family_replacement",
            }
            for metric in [
                "conventional_sharpe",
                "compounded_total_return",
                "max_drawdown_total_nav",
                "annualized_excess_vol",
                "first_12_sharpe",
                "last_12_sharpe",
                "mean_effective_count",
                "largest_weight",
            ]:
                values = [row["performance"][metric] for row in selected] if complete else []
                result[metric] = float(np.mean(values)) if complete else None
                result[f"seed_sd_{metric}"] = float(np.std(values, ddof=1)) if complete else None
            results.append(result)
    return results
