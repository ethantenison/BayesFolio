"""Independent November portfolio and label-availability acceptance checks.

Experiment-only audit adapter. No fits, candidate search, fallback, repair,
or output writes. Forecasts/targets are decimal discounted excess returns,
covariance and variance ceilings squared decimals, and weights fractions.
The original one-portfolio conic allocator is an independent parity reference.
"""

from __future__ import annotations

import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ASSETS = ["SPY", "MGK", "VTV", "IWM", "VEA", "VWO", "VSS", "EWX",
          "BND", "IEF", "LQD", "HYG", "BNDX", "VWOB", "HYEM"]
_REFERENCE_PATH = (Path(__file__).resolve().parents[1]
                   / "2026-10-portfolio-evaluation-v1/monthly_extension_v1/extension_math.py")
_WEIGHT_PARITY_ATOL = 1e-6


@lru_cache(maxsize=1)
def _reference() -> ModuleType:
    specification = importlib.util.spec_from_file_location("november_independent_original_conic", _REFERENCE_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load the original frozen hard-rank allocator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _validate_assets(assets: list[str]) -> None:
    if list(assets) != ASSETS:
        raise ValueError("Exact approved fifteen-ETF ordering including IWM is required")


def _raw_checks(mean: np.ndarray, covariance: np.ndarray, budget: float, weights: np.ndarray) -> dict:
    if (mean.shape != (15,) or covariance.shape != (15, 15) or weights.shape != (15,)
            or not np.isfinite(budget) or budget <= 0
            or not all(np.isfinite(value).all() for value in [mean, covariance, weights])):
        raise ValueError("Finite single-portfolio moments, raw weights and positive variance ceiling required")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("Symmetric observed covariance required; no repair")
    np.linalg.cholesky(covariance)
    total = float(weights.sum())
    minimum = float(weights.min())
    squared = float(weights @ weights)
    variance = float(weights @ covariance @ weights)
    if minimum < -1e-8 or abs(total - 1) > 1e-7 or squared > 1 / 7 + 1e-7:
        raise ValueError("Raw weights violate long-only, full-investment or effective-assets constraints")
    if squared <= 0 or variance > budget * (1 + 1e-6):
        raise ValueError("Raw weights violate the unchanged variance ceiling")
    scores = rankdata(mean, method="average") / 15
    return {"weight_sum": total, "minimum_weight": minimum, "maximum_weight": float(weights.max()),
            "sum_squared_weights": squared, "effective_assets": 1 / squared,
            "predicted_variance": variance, "variance_ceiling": float(budget),
            "variance_to_ceiling": variance / budget,
            "rank_objective": float(scores @ weights - 1e-8 * squared),
            "score_ranks": (scores * 15).tolist(), "position_cap": None,
            "raw_weights_changed": False, "fallback_used": False}


def validate_portfolio(mean: np.ndarray, covariance: np.ndarray, budget: float,
                       weights: np.ndarray, assets: list[str]) -> dict:
    """Independently validate raw weights and original-conic allocation parity.

    Args:
        mean: Shape (15,) pooled decimal forecasts before allocation.
        covariance: Shape (15,15) pooled observed covariance, squared decimals.
        budget: Causal expanding historical-EW variance ceiling, squared decimals.
        weights: Shape (15,) delivered raw fractional weights, never repaired.
        assets: Exact approved fifteen-ETF order including all zero holdings.

    Returns:
        Complete feasibility, original strict-optimal solver and objective/weight
        parity evidence. No clipping, normalization or substitution is performed.

    Raises:
        ValueError: Shapes, units, constraints, strict solver status or parity fail.
    """
    _validate_assets(assets)
    mean, covariance, weights = (np.asarray(value, dtype=float) for value in [mean, covariance, weights])
    raw = _raw_checks(mean, covariance, budget, weights)
    reference = _reference().hard_rank(mean, covariance, float(budget))
    attempts = reference["numerical_attempts"]
    if not attempts or attempts[-1].get("status") != "optimal" or reference.get("fallback_used"):
        raise ValueError("Independent original conic solve did not return strict optimal without fallback")
    original_weights = np.asarray(reference["weights"], dtype=float)
    original = _raw_checks(mean, covariance, budget, original_weights)
    objective_gap = abs(raw["rank_objective"] - original["rank_objective"])
    weight_gap = float(np.max(np.abs(weights - original_weights)))
    if objective_gap > 1e-7 or weight_gap > _WEIGHT_PARITY_ATOL:
        raise ValueError(f"Original-conic parity failed: objective_gap={objective_gap}, max_weight_gap={weight_gap}")
    return {"passed": True, "assets": list(assets), "raw_checks": raw,
            "original_conic": {"solver_status": attempts[-1]["status"], "attempts": attempts,
                               "weights": original_weights.tolist(), "raw_checks": original,
                               "source": str(_REFERENCE_PATH),
                               "source_sha256": hashlib.sha256(_REFERENCE_PATH.read_bytes()).hexdigest()},
            "maximum_absolute_weight_difference": weight_gap,
            "weight_parity_absolute_tolerance": _WEIGHT_PARITY_ATOL,
            "absolute_objective_difference": objective_gap,
            "no_postsolve_repairs": True}


def validate_weights_csv(path: Path, mean: np.ndarray, covariance: np.ndarray, budget: float,
                         weights: np.ndarray, assets: list[str], *, asset_column: str = "asset_id",
                         weight_column: str = "weight", expected_metadata: dict[str, str] | None = None) -> dict:
    """Validate the serialized all-asset decimal weight table without correction.

    Args:
        path: Existing final CSV; this function never writes or modifies it.
        mean: Shape (15,) pooled decimal forecasts used for the delivered portfolio.
        covariance: Matching observed covariance in squared-decimal units.
        budget: Unchanged causal variance ceiling in squared-decimal units.
        weights: Exact expected raw weights before CSV serialization.
        assets: Approved order of all fifteen ETFs, including zero weights.
        asset_column: CSV column containing ticker identifiers.
        weight_column: CSV column containing decimal fractional weights.
        expected_metadata: Optional columns whose value must match in every row.

    Returns:
        CSV fingerprint, raw serialization parity and independent portfolio audit.
    """
    _validate_assets(assets)
    frame = pd.read_csv(path, dtype={asset_column: str})
    if asset_column not in frame or weight_column not in frame:
        raise ValueError("Weight CSV is missing required asset/weight columns")
    if frame[asset_column].tolist() != assets or frame[asset_column].duplicated().any():
        raise ValueError("CSV must contain exactly all fifteen assets once in the approved order")
    serialized = pd.to_numeric(frame[weight_column], errors="raise").to_numpy(dtype=float)
    expected = np.asarray(weights, dtype=float)
    if expected.shape != (15,) or not np.isfinite(serialized).all():
        raise ValueError("All serialized weights must be finite decimal fractions")
    difference = float(np.max(abs(serialized - expected)))
    if difference > 1e-12:
        raise ValueError("CSV weights differ from delivered raw optimizer weights; no rounding repair allowed")
    for column, expected_value in (expected_metadata or {}).items():
        if column not in frame or not frame[column].astype(str).eq(str(expected_value)).all():
            raise ValueError(f"CSV metadata does not match the frozen decision: {column}")
    result = validate_portfolio(mean, covariance, budget, serialized, assets)
    result.update(csv=str(path), csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  csv_rows=len(frame), zero_weight_assets=frame.loc[serialized == 0, asset_column].tolist(),
                  raw_serialized_maximum_difference=difference,
                  metadata_checked=dict(expected_metadata or {}))
    return result


def validate_training_availability(panel: pd.DataFrame, assets: list[str], input_columns: list[str] | None = None,
                                   *, origin: str = "2026-10-30", as_of: str = "2026-10-31") -> dict:
    """Validate final November origin, complete prior labels and unknown live labels.

    Args:
        panel: Frozen monthly panel with date, asset_id and y_excess_lead columns.
        assets: Approved fifteen-ETF order; every date must contain every task.
        input_columns: Optional exact model-input union, including t_index.
        origin: October30 market origin; a Saturday October31 bar is forbidden.
        as_of: October31 logical information/retrieval date.

    Returns:
        Date/target-availability and optional model-input checks. This is row-level
        timing validation, not a claim of point-in-time historical macro vintages.
    """
    _validate_assets(assets)
    origin_date, as_of_date = pd.Timestamp(origin), pd.Timestamp(as_of)
    if (origin != "2026-10-30" or as_of != "2026-10-31"
            or origin_date != as_of_date + pd.offsets.BMonthEnd(-1)):
        raise ValueError("October31 logical as-of must map to October30 business month-end origin")
    required = {"date", "asset_id", "y_excess_lead"}
    if not required.issubset(panel.columns):
        raise ValueError("Missing monthly panel keys/target")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any() or frame.duplicated(["date", "asset_id"]).any():
        raise ValueError("Missing dates or duplicated ETF-month observations")
    expected_dates = pd.date_range("2017-05-31", origin_date, freq="BME")
    expected = pd.MultiIndex.from_product([expected_dates, sorted(assets)], names=["date", "asset_id"])
    observed = pd.MultiIndex.from_frame(frame[["date", "asset_id"]]).sort_values()
    if not observed.equals(expected):
        raise ValueError("Complete 114-date fifteen-ETF chronology required; no omitted or future dates")
    training = frame.loc[frame["date"] < origin_date]
    live = frame.loc[frame["date"] == origin_date]
    if (training["date"].nunique() != 113 or training["date"].max() != pd.Timestamp("2026-09-30")
            or not np.isfinite(training["y_excess_lead"].to_numpy(dtype=float)).all()
            or not live["y_excess_lead"].isna().all()):
        raise ValueError("113 fully known training targets and fifteen unknown November targets required")
    target_availability = training["date"] + pd.offsets.BMonthEnd(1)
    if (target_availability > origin_date).any():
        raise ValueError("An unavailable target entered training")
    if input_columns is not None:
        excluded = {"hy_spread", "hy_spread_chg_1p", "hy_spread_chg_1m"}
        if "t_index" not in input_columns or excluded.intersection(input_columns):
            raise ValueError("Model inputs must retain t_index and exclude raw HY level/change")
        if not set(input_columns).issubset(frame.columns):
            raise ValueError("Missing registered model inputs")
        if not np.isfinite(frame[input_columns].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite model inputs cannot be filled or dropped")
    return {"passed": True, "as_of_date": as_of, "market_origin": origin,
            "target_month": "2026-11", "training_months_per_asset": 113,
            "first_training_origin": "2017-05-31", "last_training_origin": "2026-09-30",
            "last_training_target_available_on": "2026-10-30", "unavailable_live_targets": len(live),
            "live_target_is_unknown": True, "input_columns_checked": list(input_columns or []),
            "row_timing_only_not_macro_vintage_certification": True}
