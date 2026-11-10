"""Independent acceptance checks for the v2 November decision.

This experiment-only audit layer never fits, searches, repairs or writes files.
It independently reconstructs eligible-14 rank scores and a strict conic optimum
from the original 15-task moments. Returns and weights are decimal fractions;
covariance and optional variance ceilings are squared decimal returns.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ASSETS = ["SPY", "MGK", "VTV", "IWM", "VEA", "VWO", "VSS", "EWX",
          "BND", "IEF", "LQD", "HYG", "BNDX", "VWOB", "HYEM"]
ELIGIBLE_INDEX = np.array([index for index, asset in enumerate(ASSETS) if asset != "MGK"])
ELIGIBLE = [ASSETS[index] for index in ELIGIBLE_INDEX]
_MGK_INDEX = ASSETS.index("MGK")
_EQUITY_MASK = np.array([index < 8 for index in ELIGIBLE_INDEX])
_WEIGHT_TOLERANCE = 5e-6
_EXCLUDED_INPUTS = {"hy_spread", "hy_spread_chg", "hy_spread_chg_1p", "hy_spread_chg_1m"}


def _assets(assets: list[str]) -> None:
    if list(assets) != ASSETS:
        raise ValueError("The exact original fifteen-task order, including MGK and IWM, is required")


def _limits(min_effective_assets: float, variance_ceiling: float | None,
            equity_cap: float | None) -> None:
    if not np.isfinite(min_effective_assets) or not 3 <= min_effective_assets <= 14:
        raise ValueError("The static effective-count minimum must be within 3..14")
    if variance_ceiling is not None and (not np.isfinite(variance_ceiling) or variance_ceiling <= 0):
        raise ValueError("A registered variance ceiling must be positive or explicitly None")
    if equity_cap is not None and (not np.isfinite(equity_cap) or not 0 <= equity_cap <= 1):
        raise ValueError("A static total-equity cap must be within 0..1 or explicitly None")


def _raw(mean: np.ndarray, covariance: np.ndarray, weights: np.ndarray,
         min_effective_assets: float, variance_ceiling: float | None,
         equity_cap: float | None) -> dict:
    if (mean.shape != (15,) or covariance.shape != (15, 15) or weights.shape != (15,)
            or not all(np.isfinite(value).all() for value in [mean, covariance, weights])):
        raise ValueError("Finite original15 mean/covariance/raw-weight arrays are required")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("Observed covariance must be symmetric without repair")
    np.linalg.cholesky(covariance)
    if np.count_nonzero(weights[_MGK_INDEX:_MGK_INDEX + 1]):
        raise ValueError("MGK must have exactly zero allocation, not an approximately zero solver residual")
    eligible_weights = weights[ELIGIBLE_INDEX]
    principal = covariance[np.ix_(ELIGIBLE_INDEX, ELIGIBLE_INDEX)]
    np.linalg.cholesky(principal)
    total, minimum = float(weights.sum()), float(weights.min())
    concentration = float(eligible_weights @ eligible_weights)
    variance = float(eligible_weights @ principal @ eligible_weights)
    equity = float(eligible_weights[_EQUITY_MASK].sum())
    if (minimum < -1e-8 or abs(total - 1) > 1e-7 or concentration <= 0
            or concentration > 1 / min_effective_assets + 1e-7):
        raise ValueError("Unmodified weights violate long-only, full-investment or effective-count constraints")
    if variance_ceiling is not None and variance > variance_ceiling * (1 + 1e-6):
        raise ValueError("Unmodified weights violate their registered variance ceiling")
    if equity_cap is not None and equity > equity_cap + 1e-7:
        raise ValueError("Unmodified weights violate their registered total-equity cap")
    scores = rankdata(mean[ELIGIBLE_INDEX], method="average") / 14
    return {"weight_sum": total, "minimum_weight": minimum, "maximum_weight": float(weights.max()),
            "MGK_weight": float(weights[_MGK_INDEX]), "effective_assets": 1 / concentration,
            "minimum_effective_assets": float(min_effective_assets), "sum_squared_weights": concentration,
            "equity_weight": equity, "equity_cap": equity_cap, "predicted_variance": variance,
            "variance_ceiling": variance_ceiling,
            "variance_to_ceiling": None if variance_ceiling is None else variance / variance_ceiling,
            "rank_objective": float(scores @ eligible_weights - 1e-8 * concentration),
            "eligible_score_ranks": (scores * 14).tolist(), "eligible_assets": ELIGIBLE,
            "position_cap": None, "raw_weights_changed": False}


def _conic_reference(mean: np.ndarray, covariance: np.ndarray, min_effective_assets: float,
                     variance_ceiling: float | None, equity_cap: float | None) -> dict:
    scores = rankdata(mean[ELIGIBLE_INDEX], method="average") / 14
    principal = covariance[np.ix_(ELIGIBLE_INDEX, ELIGIBLE_INDEX)]
    weight = cp.Variable(14)
    constraints = [weight >= 0, cp.sum(weight) == 1]
    if min_effective_assets == 14:
        # Cauchy--Schwarz proves NE>=14 uniquely fixes the fourteen weights.
        # This equivalent equality avoids a cone with no strictly feasible point.
        constraints.append(weight == np.full(14, 1 / 14))
    else:
        constraints.append(cp.norm(weight, 2) <= 1 / np.sqrt(min_effective_assets))
    if variance_ceiling is not None:
        factor = np.linalg.cholesky(principal).T
        constraints.append(cp.norm(factor @ weight / np.sqrt(variance_ceiling), 2) <= 1)
    if equity_cap is not None:
        constraints.append(cp.sum(weight[_EQUITY_MASK]) <= equity_cap)
    centered = scores - scores.mean()
    problem = cp.Problem(cp.Maximize(centered @ weight - 1e-8 * cp.sum_squares(weight)), constraints)
    attempts = []
    for regularization in [1e-12, 1e-13]:
        options = {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9,
                   "max_iter": 500, "static_regularization_enable": True,
                   "static_regularization_constant": regularization, "max_step_fraction": .9}
        try:
            problem.solve(solver="CLARABEL", **options)
            attempt = {"status": problem.status, "iterations": problem.solver_stats.num_iters,
                       "options": options}
        except cp.error.SolverError as error:
            attempt = {"status": "solver_error", "error": str(error), "options": options}
        attempts.append(attempt)
        if problem.status == cp.OPTIMAL and weight.value is not None:
            full = np.zeros(15)
            full[ELIGIBLE_INDEX] = np.asarray(weight.value, dtype=float).reshape(14)
            return {"weights": full, "attempts": attempts, "solver_status": problem.status}
    raise ValueError(f"Independent conic reference did not return strict optimal: {attempts}")


def validate_portfolio(mean: np.ndarray, covariance: np.ndarray, weights: np.ndarray, assets: list[str], *,
                       min_effective_assets: float, variance_ceiling: float | None = None,
                       equity_cap: float | None = None, solver_status: str,
                       solver_attempts: list[dict] | None = None,
                       analytic_certificate: str | None = None) -> dict:
    """Audit eligible14 raw weights against a separately constructed conic solve.

    Args:
        mean: Original shape (15,) pooled decimal GP forecasts, including MGK.
        covariance: Original shape (15,15) observed covariance, squared decimals.
        weights: Shape (15,) raw fractional weights with MGK exactly zero.
        assets: Exact original fifteen-task order; GP task population is unchanged.
        min_effective_assets: Registered static effective-count floor within 3..14.
        variance_ceiling: Registered causal risk ceiling, squared decimals, or None.
        equity_cap: Optional registered aggregate equity fraction, not an ETF cap.
        solver_status: Strict "optimal", or explicitly certified "analytic_kkt".
        solver_attempts: Actual numerical attempts; last must be optimal for a solve.
        analytic_certificate: Nonempty mathematical certificate for analytic_kkt.

    Returns:
        Raw feasibility, independent strict-optimal solve, and weight/objective
        parity evidence. Independent conic representation parity uses 5e-6 in
        fractional weights; exact delivered-policy reproduction is a separate
        1e-12 root gate. Raw constraints are not relaxed. No clipping,
        normalization, fallback or covariance repair occurs.

    Raises:
        ValueError: Axes, constraints, numerical acceptance or independent parity fail.
    """
    _assets(assets)
    _limits(min_effective_assets, variance_ceiling, equity_cap)
    if solver_status == "optimal":
        if not solver_attempts or solver_attempts[-1].get("status") != "optimal":
            raise ValueError("Strict optimal requires its actual last optimal numerical attempt")
    elif solver_status == "analytic_kkt":
        if not analytic_certificate:
            raise ValueError("Analytical allocation requires an explicit mathematical certificate")
    else:
        raise ValueError("Inaccurate, failed and unaccepted solver statuses are ineligible")
    mean, covariance, weights = (np.asarray(value, dtype=float) for value in [mean, covariance, weights])
    raw = _raw(mean, covariance, weights, min_effective_assets, variance_ceiling, equity_cap)
    reference = _conic_reference(mean, covariance, min_effective_assets, variance_ceiling, equity_cap)
    original = _raw(mean, covariance, reference["weights"], min_effective_assets, variance_ceiling, equity_cap)
    weight_gap = float(np.max(np.abs(weights - reference["weights"])))
    objective_gap = abs(raw["rank_objective"] - original["rank_objective"])
    if weight_gap > _WEIGHT_TOLERANCE or objective_gap > 1e-7:
        raise ValueError(f"Independent14-rank parity failed: weight_gap={weight_gap}, objective_gap={objective_gap}")
    return {"passed": True, "raw_checks": raw, "supplied_status": solver_status,
            "supplied_numerical_attempts": list(solver_attempts or []),
            "supplied_analytic_certificate": analytic_certificate,
            "independent_conic": {"solver_status": reference["solver_status"],
                                  "attempts": reference["attempts"], "raw_checks": original,
                                  "weights": reference["weights"].tolist()},
            "maximum_absolute_weight_difference": weight_gap,
            "weight_parity_absolute_tolerance": _WEIGHT_TOLERANCE,
            "absolute_objective_difference": objective_gap, "no_postsolve_repairs": True}


def validate_weights_csv(path: Path, mean: np.ndarray, covariance: np.ndarray, weights: np.ndarray,
                         assets: list[str], *, expected_metadata: dict[str, str] | None = None,
                         **portfolio_options) -> dict:
    """Audit the unchanged fifteen-row decimal output and independent allocation.

    Args:
        path: Existing asset_id/weight CSV; no file is written.
        mean: Original fifteen pooled decimal forecasts.
        covariance: Original observed covariance in squared-decimal units.
        weights: Expected raw fifteen-element fractional optimizer weights.
        assets: Exact original fifteen-task order, including MGK at zero.
        expected_metadata: Optional columns whose value must match every row.
        **portfolio_options: Registered constraints and actual numerical evidence
            required by validate_portfolio.

    Returns:
        CSV fingerprint, serialization parity and independent portfolio audit.
    """
    _assets(assets)
    frame = pd.read_csv(path, dtype={"asset_id": str})
    if (not {"asset_id", "weight"}.issubset(frame.columns) or frame["asset_id"].tolist() != assets
            or frame["asset_id"].duplicated().any()):
        raise ValueError("CSV must contain all fifteen ETFs once in the exact task order")
    serialized = pd.to_numeric(frame["weight"], errors="raise").to_numpy(dtype=float)
    expected = np.asarray(weights, dtype=float)
    if expected.shape != (15,) or not np.isfinite(serialized).all():
        raise ValueError("Finite decimal weights are required")
    gap = float(np.max(abs(serialized - expected)))
    if gap > 1e-12:
        raise ValueError("Serialized weights differ from the raw allocation")
    for column, expected_value in (expected_metadata or {}).items():
        if column not in frame or not frame[column].astype(str).eq(str(expected_value)).all():
            raise ValueError(f"CSV metadata mismatch: {column}")
    result = validate_portfolio(mean, covariance, serialized, assets, **portfolio_options)
    result.update(csv=str(path), csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), csv_rows=15,
                  metadata_checked=dict(expected_metadata or {}), raw_serialized_maximum_difference=gap)
    return result


def validate_training_availability(panel: pd.DataFrame, assets: list[str], input_columns: list[str], *,
                                   origin: str = "2026-10-30", as_of: str = "2026-10-31") -> dict:
    """Check all fifteen tasks, completed October data and unknown November labels.

    Args:
        panel: Frozen monthly frame with date/asset_id/y_excess_lead and inputs.
        assets: Unchanged fifteen-task order; MGK training is retained.
        input_columns: Exact union of registered model inputs, including t_index.
        origin: October30 business-month-end market/forecast origin.
        as_of: October31 logical information date, not a Saturday market bar.

    Returns:
        Complete panel, finite inputs and target-availability evidence. These are
        row-timing checks, not certification of historical macro vintages.
    """
    _assets(assets)
    if (origin != "2026-10-30" or as_of != "2026-10-31"
            or pd.Timestamp(origin) != pd.Timestamp(as_of) + pd.offsets.BMonthEnd(-1)):
        raise ValueError("The October31 logical as-of must map to the October30 market origin")
    if not {"date", "asset_id", "y_excess_lead", *input_columns}.issubset(panel.columns):
        raise ValueError("Missing registered task/date/target/input columns")
    if "t_index" not in input_columns or _EXCLUDED_INPUTS.intersection(input_columns):
        raise ValueError("Retain t_index and exclude raw HY level/change inputs")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any() or frame.duplicated(["date", "asset_id"]).any():
        raise ValueError("Missing dates or duplicate task-month rows")
    dates = pd.date_range("2017-05-31", origin, freq="BME")
    expected = pd.MultiIndex.from_product([dates, sorted(assets)], names=["date", "asset_id"])
    observed = pd.MultiIndex.from_frame(frame[["date", "asset_id"]]).sort_values()
    if not observed.equals(expected):
        raise ValueError("All114 dates and all15 tasks are required, without future or omitted rows")
    training = frame.loc[frame["date"] < pd.Timestamp(origin)]
    live = frame.loc[frame["date"] == pd.Timestamp(origin)]
    if (not np.isfinite(training["y_excess_lead"].to_numpy(dtype=float)).all()
            or not live["y_excess_lead"].isna().all()
            or not np.isfinite(frame[input_columns].to_numpy(dtype=float)).all()
            or ((training["date"] + pd.offsets.BMonthEnd(1)) > pd.Timestamp(origin)).any()):
        raise ValueError("Training labels/inputs must be available; all November targets must remain unknown")
    return {"passed": True, "GP_tasks": ASSETS, "allocatable_assets": ELIGIBLE,
            "as_of_date": as_of, "market_origin": origin, "forecast_month": "2026-11",
            "expanding_training_months": 113, "last_training_origin": "2026-09-30",
            "last_training_target_available_on": "2026-10-30", "unknown_live_targets": len(live),
            "inputs_checked": list(input_columns), "row_timing_only_not_macro_vintage_certification": True}


def validate_gp_spec(spec: dict, reference_spec: dict, registered_spec: dict, *,
                     allow_compact_rank1: bool = False) -> dict:
    """Check an exact admitted specification against the immutable reference.

    Args:
        spec: Complete actual model definition, excluding runtime-only telemetry.
        reference_spec: Original v1 broad, compact or broad48 model definition.
        registered_spec: Exact immutable admitted model definition; any kernel or
            prior variation must appear here before its evaluation.
        allow_compact_rank1: Permit rank1 only for the five-input compact model.

    Returns:
        Registered-specification fingerprint and protected feature/window checks.
    """
    if spec != registered_spec:
        raise ValueError("Actual GP specification differs from its immutable admission record")
    required = {"inputs", "tickers", "rank", "maxiter", "rolling_months"}
    if not required.issubset(spec) or not required.issubset(reference_spec):
        raise ValueError("Model specifications must state exact features/tasks/rank/window/maxiter")
    _assets(spec["tickers"])
    if (spec["inputs"] != reference_spec["inputs"] or "t_index" not in spec["inputs"]
            or _EXCLUDED_INPUTS.intersection(spec["inputs"])):
        raise ValueError("Features and transformations cannot be redefined or raw HY inputs reintroduced")
    if spec["rolling_months"] != reference_spec["rolling_months"] or spec["rolling_months"] not in [None, 48]:
        raise ValueError("Original expanding/exact48 labeled-month windows must remain unchanged")
    compact = spec["inputs"] == ["t_index", "mom12m_skip1m", "mom6m", "cpi_chg_12p",
                                "hy_spread_z_12p", "term_spread"]
    if spec["maxiter"] != 15 or spec["rank"] not in ({1, 2} if compact and allow_compact_rank1 else {2}):
        raise ValueError("All maxiter remain15; only an admitted compact specification may use rank1")
    if spec.get("time_modulation_mode", "neither") != "neither":
        raise ValueError("Time modulation is unchanged; only registered kernel/prior hypotheses may differ")
    for key in ["target", "target_column", "input_transform", "outcome_transform", "mean",
                "feature_lags", "feature_transformations", "likelihood_family"]:
        if spec.get(key) != reference_spec.get(key):
            raise ValueError(f"Registered kernel/prior hypotheses cannot redefine {key}")
    encoded = json.dumps(registered_spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"passed": True, "registered_spec_sha256": hashlib.sha256(encoded).hexdigest(),
            "inputs": list(spec["inputs"]), "GP_tasks": ASSETS, "rank": spec["rank"],
            "maxiter": 15, "rolling_months": spec["rolling_months"], "compact_rank1_permitted": compact}


def validate_fit_record(record: dict, spec: dict, *, origin: str, seed: int) -> dict:
    """Check actual fitted settings and the exact pre-origin labeled window.

    Args:
        record: Actual fit telemetry containing date, seed, rank, maxiter,
            train_start, train_end and train_months.
        spec: Independently validated admitted model specification.
        origin: Historical or live business-month-end forecast origin.
        seed: One of the three approved initialization seeds.

    Returns:
        Actual window/settings/target-availability acceptance evidence.
    """
    origin_date = pd.Timestamp(origin)
    if (seed not in [27, 127, 227] or origin_date > pd.Timestamp("2026-10-30")
            or origin_date != origin_date + pd.offsets.BMonthEnd(0)):
        raise ValueError("Approved seed and completed business-month-end origin required")
    dates = pd.date_range("2017-05-31", origin_date, freq="BME")[:-1]
    if spec["rolling_months"] is not None:
        dates = dates[-spec["rolling_months"]:]
    if not len(dates) or record.get("status") != "fit":
        raise ValueError("A successful actual fit and a nonempty pre-origin training window are required")
    expected = {"date": origin, "seed": seed, "rank": spec["rank"], "maxiter": 15,
                "train_start": dates[0].strftime("%Y-%m-%d"),
                "train_end": dates[-1].strftime("%Y-%m-%d"), "train_months": len(dates)}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Actual fit settings, population or labeled training window differ from the admitted job")
    if pd.Timestamp(record["train_end"]) + pd.offsets.BMonthEnd(1) > origin_date:
        raise ValueError("An unavailable target was used for training")
    if "tickers" in record:
        _assets(record["tickers"])
    return {"passed": True, **expected, "all15_training_tasks_required": True,
            "live_November_target_used": False}
