"""Deterministic ranking metrics and common-risk allocation for GP research.

All returns are decimal monthly excess returns. Covariance uses only resolved
training labels. Split GP scenario matrices are never concatenated to fabricate
cross-family dependence. Solver failures stop the experiment; no fallback.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
from audit_joint_objectives import spearman
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf


def validate_predictions(frame: pd.DataFrame, tickers: list[str], dates: list[str]) -> None:
    """Require finite complete uniquely keyed forecasts and positive uncertainties.

    Args:
        frame: Decimal forecast/actual panel.
        tickers: Expected task universe.
        dates: Expected forecast-origin dates.
    """
    expected = pd.MultiIndex.from_product([dates, sorted(tickers)], names=["date", "asset_id"])
    indexed = frame.set_index(["date", "asset_id"]).sort_index()
    if indexed.index.has_duplicates or not indexed.index.equals(expected):
        raise ValueError("Missing/duplicate/unexpected forecast keys")
    if not np.isfinite(frame[["y_true", "y_pred", "y_std"]]).all().all() or not frame.y_std.gt(0).all():
        raise ValueError("Invalid forecasts or uncertainties")


def forecast_metrics(frame: pd.DataFrame, families: dict[str, list[str]]) -> tuple[list, list]:
    """Compute pooled and within-family IC, calibration and error diagnostics.

    Args:
        frame: Complete prediction panel for one architecture/seed.
        families: Predeclared family mapping; pooled metrics are also emitted.

    Returns:
        Aggregate scope records and corresponding monthly IC records.
    """
    rows, months = [], []
    for scope, tickers in {"all": sorted(frame.asset_id.unique()), **families}.items():
        panel = frame[frame.asset_id.isin(tickers)]
        ic = panel.groupby("date").apply(
            lambda x: spearman(x.y_pred.to_numpy(), x.y_true.to_numpy()), include_groups=False)
        if len(ic) != 24 or not np.isfinite(ic).all():
            raise ValueError("Undefined monthly ranking; no silent omission")
        error = panel.y_true - panel.y_pred
        nll = 0.5 * np.log(2 * np.pi * panel.y_std**2) + 0.5 * (error / panel.y_std)**2
        rows.append({"scope": scope, "ic": ic.mean(), "first_12_ic": ic.iloc[:12].mean(),
                     "last_12_ic": ic.iloc[12:].mean(), "positive_ic_months": int(ic.gt(0).sum()),
                     "nll": nll.mean(), "rmse": np.sqrt(np.square(error).mean()),
                     "coverage_95": error.abs().le(1.959963984540054 * panel.y_std).mean(),
                     "mean_predictive_sd": panel.y_std.mean()})
        months.extend({"scope": scope, "date": date, "ic": value} for date, value in ic.items())
    return rows, months


def common_covariance(frame: pd.DataFrame, date: str, tickers: list[str], index: int) -> np.ndarray:
    """Estimate Ledoit-Wolf covariance on the same resolved history for all models.

    Args:
        frame: Frozen global raw panel with forecast-origin dates and labels.
        date: Current forecast origin; the row at this date is not eligible.
        tickers: Exact common 15-ETF allocation universe.
        index: Original index in the scored month grid.

    Returns:
        Positive-definite monthly covariance estimated from 84+index rows.
    """
    history = frame[frame.date.lt(pd.Timestamp(date))].pivot(
        index="date", columns="asset_id", values="y_excess_lead").reindex(columns=tickers)
    if len(history) != 84 + index or not np.isfinite(history.to_numpy()).all():
        raise ValueError("Covariance history is incomplete or leaks across the origin")
    covariance = LedoitWolf().fit(history.to_numpy()).covariance_
    if not np.linalg.eigvalsh(covariance).min() > 0:
        raise ValueError("Common covariance is not positive definite")
    return covariance


def allocate(predictions: np.ndarray, covariance: np.ndarray, policy: str) -> tuple[np.ndarray, dict]:
    """Allocate ranks under the frozen common universe and risk/diversification rules.

    Args:
        predictions: Raw decimal predicted excess returns, same ordered 15 ETFs.
        covariance: Training-only common covariance, independent of architecture.
        policy: equal_weight, linear_rank, risk_rank_cap35, or risk_rank_no_cap.

    Returns:
        Weights and constraint diagnostics. Risk target equals training equal-
        weight variance; effective asset count is at least three for risk rules.

    Raises:
        RuntimeError: Solver failure or invalid weights; never substitutes weights.
    """
    n = len(predictions)
    if n != 15 or covariance.shape != (n, n) or not np.isfinite(predictions).all():
        raise ValueError("Allocator requires the complete common 15-ETF universe")
    equal = np.ones(n) / n
    reference_variance = float(equal @ covariance @ equal)
    if reference_variance <= 0 or not np.isfinite(covariance).all():
        raise ValueError("Invalid common covariance")
    ranks = rankdata(predictions, method="average")
    cap, status = 1.0, "closed_form"
    if policy == "equal_weight":
        weights = equal
    elif policy == "linear_rank":
        weights = ranks / ranks.sum()
    elif policy in {"risk_rank_cap35", "risk_rank_no_cap"}:
        cap = 0.35 if policy == "risk_rank_cap35" else 1.0
        variable = cp.Variable(n)
        scaled = covariance / reference_variance
        constraints = [variable >= 0, variable <= cap, cp.sum(variable) == 1,
                       cp.sum_squares(variable) <= 1 / 3,
                       cp.quad_form(variable, cp.psd_wrap(scaled)) <= 1]
        # Tiny fixed strictly concave tie-breaker, not an optimized risk penalty.
        problem = cp.Problem(cp.Maximize((ranks / n) @ variable - 1e-8 * cp.sum_squares(variable)), constraints)
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_feas=1e-9, tol_gap_rel=1e-9)
        status = problem.status
        if status != cp.OPTIMAL or variable.value is None:
            raise RuntimeError(f"Strict rank/risk solve failed: {status}; no fallback")
        weights = np.asarray(variable.value).reshape(-1)
    else:
        raise ValueError(f"Unknown registered allocation policy: {policy}")
    variance = float(weights @ covariance @ weights)
    nea = float(1 / (weights @ weights))
    if (not np.isfinite(weights).all() or weights.min() < -1e-7 or weights.max() > cap + 1e-7
            or not np.isclose(weights.sum(), 1, atol=1e-7, rtol=0)):
        raise RuntimeError("Invalid weights; no clipping, renormalization or fallback")
    if policy.startswith("risk_") and (nea < 3 - 1e-6 or variance / reference_variance > 1 + 1e-6):
        raise RuntimeError("Risk/diversification constraint violated; no relaxation")
    return weights, {"solver_status": status, "effective_assets": nea, "max_weight": weights.max(),
                     "variance_ratio_to_equal_weight": variance / reference_variance,
                     "forecast_annualized_vol": np.sqrt(12 * variance), "fallback_used": False}


def portfolio_metrics(returns: np.ndarray) -> dict[str, Any]:
    """Compute conventional Sharpe and diagnostic wealth/risk from monthly excess returns.

    Args:
        returns: Complete 24-month realized excess-return path.

    Returns:
        Sharpe, volatility, compounded excess-return wealth and drawdown diagnostics.
        Compounding excess returns is not the actual total-return account NAV.
    """
    if len(returns) != 24 or not np.isfinite(returns).all() or np.std(returns, ddof=1) <= 0:
        raise ValueError("Invalid monthly portfolio path")
    wealth = np.r_[1.0, np.cumprod(1 + returns)]
    return {"sharpe": float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(12)),
            "annualized_vol": float(np.std(returns, ddof=1) * np.sqrt(12)),
            "compounded_excess_return": float(wealth[-1] - 1),
            "max_drawdown_excess_wealth": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
            "worst_month": float(np.min(returns))}
