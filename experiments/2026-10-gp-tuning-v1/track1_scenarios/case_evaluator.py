"""Evaluate immutable GP-tuning portfolio cases without fits or model selection.

This isolated research I/O adapter reads only the campaign's context/component
panels and frozen numerical helper modules. Means/targets/returns are decimals,
covariance/variance budgets are decimal squared, weights fractions and NAV dollars.
Every case has static blending coefficients, three seeds and 28 ordered months.
Failed allocation pairs remain explicit; partial paths never receive scores.

Usage:
    Import ``evaluate_case(case, output_root)`` from the parent campaign runner.
    The parent owns process parallelism; this module never starts another pool.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from cvxpy.error import SolverError

_HERE = Path(__file__).resolve().parent
_FRAMEWORK = _HERE.parents[1] / "2026-10-portfolio-evaluation-v1"
_HELPERS = {
    "hard": _FRAMEWORK / "monthly_extension_v1/extension_math.py",
    "sampling": _FRAMEWORK / "scenario_riskfolio_v1/scenario_allocator.py",
    "scenarios": _FRAMEWORK / "scenario_sharpe_v3/signed_sharpe.py",
    "metrics": _FRAMEWORK / "metrics.py",
    "wealth": _FRAMEWORK / "schwab_benchmark_v1/benchmark_math.py",
}
_ASSETS = [
    "SPY", "MGK", "VTV", "IWM", "VEA", "VWO", "VSS", "EWX",
    "BND", "IEF", "LQD", "HYG", "BNDX", "VWOB", "HYEM",
]
_PERIODS = {"tuning_first16": slice(0, 16), "validation_last12": slice(16, 28), "all28": slice(0, 28)}
_SHAPE = (3, 28, 15)
_BASE_ALPHA = np.array([0.5, 0.0, 0.375, 0.25])


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(path: Path, name: str) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__ or "").resolve() != path.resolve():
            raise ValueError(f"Module name collision: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def _dependency(kind: str) -> ModuleType:
    if kind == "scenarios":
        # The unmodified frozen signed solver imports this exact module name.
        # It exposes audit_weights; its obsolete CVaR allocator is never called.
        _load_module(_HELPERS["sampling"], "scenario_allocator")
    return _load_module(_HELPERS[kind], f"_bayesfolio_gp_tuning_frozen_{kind}")


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value) is None:
        raise ValueError("Case and component IDs must be safe single path segments")
    return value


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be one static numeric scalar, not a monthly/seed array")
    result = float(value)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _validate_case(case: dict[str, Any]) -> None:
    _identifier(case["id"])
    for key in ["broad_id", "compact_id"]:
        _identifier(case[key])
    if case["allocator"] not in {"hard_rank", "scenarios"}:
        raise ValueError("Only the two frozen allocator methods are permitted")
    if not isinstance(case["alpha"], list) or len(case["alpha"]) != 4:
        raise ValueError("alpha must be four static family fractions, not month-specific weights")
    for value in case["alpha"]:
        _fraction(value, "alpha")
    for key in ["cov_compact_fraction", "third_fraction", "third_cov_fraction"]:
        _fraction(case[key], key)
    if case["third_id"] is not None:
        _identifier(case["third_id"])
    elif case["third_fraction"] != 0 or case["third_cov_fraction"] != 0:
        raise ValueError("A nonzero third-model fraction requires third_id")
    if not isinstance(case["track"], (int, str)) or isinstance(case["track"], bool):
        raise ValueError("A track label is required")
    if not isinstance(case["stage"], str) or not case["stage"]:
        raise ValueError("A stage label is required")
    json.dumps(case, allow_nan=False)


def _read_panel(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _finite_array(panel: dict[str, np.ndarray], key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = panel[key]
    if value.shape != shape or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"Complete finite {key} with shape {shape} required")
    return value


def _axes(panel: dict[str, np.ndarray], context: dict[str, np.ndarray] | None = None) -> None:
    if panel["assets"].tolist() != _ASSETS or panel["seeds"].tolist() != [27, 127, 227]:
        raise ValueError("Exact 15-ETF order and the three approved seeds required")
    if panel["dates"].shape != (28,):
        raise ValueError("All 28 forecast origins required")
    dates = panel["dates"].astype("datetime64[D]")
    if (np.isnat(dates).any() or np.any(np.diff(dates) <= np.timedelta64(0, "D"))
            or str(dates[0]) != "2024-05-31" or str(dates[-1]) != "2026-08-31"):
        raise ValueError("The approved strictly ordered May2024-Aug2026 origins are required")
    if context is not None:
        for key in ["dates", "assets", "seeds"]:
            np.testing.assert_array_equal(panel[key], context[key], err_msg=f"Component {key} mismatch")


def _moments(mean: np.ndarray, covariance: np.ndarray) -> None:
    if mean.shape != _SHAPE or covariance.shape != (*_SHAPE, 15):
        raise ValueError("Complete three-seed/28-month/15-ETF moments required")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Missing forecasts cannot be dropped")
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-14):
        raise ValueError("Symmetric observed covariance required; no repair")
    np.linalg.cholesky(covariance)


def _inputs(case: dict[str, Any], root: Path) -> tuple[dict, dict, dict[str, str]]:
    context_path = root / "context.npz"
    paths = {"context": context_path}
    for key in ["broad_id", "compact_id", "third_id"]:
        if case[key] is not None:
            paths[key] = root / "components" / f"{case[key]}.npz"
    hashes = {str(path): _hash(path) for path in paths.values()}
    hashes.update({str(path): _hash(path) for path in [Path(__file__).resolve(), *_HELPERS.values()]})
    context = _read_panel(context_path)
    _axes(context)
    for key, shape in {
        "actual": (28, 15), "risk_free": (28,), "risk_budget": (28,),
        "baseline_mean": _SHAPE, "baseline_covariance": (*_SHAPE, 15),
        "baseline_hard_weights": _SHAPE, "baseline_scenario_weights": _SHAPE,
        "benchmark_schwab_total": (28,), "benchmark_momentum_total": (28,),
    }.items():
        _finite_array(context, key, shape)
    _moments(context["baseline_mean"], context["baseline_covariance"])
    if ((context["actual"] <= -1).any() or (context["risk_free"] <= -1).any()
            or (context["risk_budget"] <= 0).any()):
        raise ValueError("Invalid realized returns, RF or variance budget")
    components = {}
    for key in ["broad_id", "compact_id", "third_id"]:
        if key not in paths:
            continue
        panel = _read_panel(paths[key])
        _axes(panel, context)
        _moments(panel["mean"], panel["covariance"])
        components[key] = panel
    return context, components, hashes


def _blend(case: dict[str, Any], components: dict) -> tuple[np.ndarray, np.ndarray]:
    broad, compact = components["broad_id"], components["compact_id"]
    fractions = np.repeat(np.asarray(case["alpha"], dtype=float), [4, 4, 4, 3])
    mean = (1 - fractions) * broad["mean"] + fractions * compact["mean"]
    fraction = case["cov_compact_fraction"]
    covariance = (1 - fraction) * broad["covariance"] + fraction * compact["covariance"]
    if case["third_id"] is not None:
        third = components["third_id"]
        fraction = case["third_fraction"]
        mean = (1 - fraction) * mean + fraction * third["mean"]
        fraction = case["third_cov_fraction"]
        covariance = (1 - fraction) * covariance + fraction * third["covariance"]
    _moments(mean, covariance)
    return mean, covariance


def _is_baseline(case: dict[str, Any]) -> bool:
    return (case["broad_id"] == "broad_r2_i15_exp" and case["compact_id"] == "compact_r2_i15_exp"
            and np.array_equal(case["alpha"], _BASE_ALPHA) and case["cov_compact_fraction"] == 0.5
            and case["third_id"] is None and case["third_fraction"] == 0 and case["third_cov_fraction"] == 0)


def _audit(weights: np.ndarray, covariance: np.ndarray, budget: float, allocator: str) -> dict[str, float]:
    if (weights.shape != (15,) or not np.isfinite(weights).all() or weights.min() < -1e-8
            or abs(float(weights.sum()) - 1) > 1e-7):
        raise ValueError("Raw weights must be finite, long-only and fully invested; no repair")
    squared = float(weights @ weights)
    variance = float(weights @ covariance @ weights)
    if allocator == "scenarios":
        if squared > 0.1 + 1e-8 or weights.max() > 0.25 + 1e-8:
            raise ValueError("Raw scenario weights violate effective-count/cap tolerances")
    elif squared > 1 / 7 + 1e-7 or variance > budget * (1 + 1e-6):
        raise ValueError("Raw hard-rank weights violate effective-count/variance tolerances")
    return {"effective_count": 1 / squared, "equity_weight": float(weights[:8].sum()),
            "max_weight": float(weights.max()), "predicted_variance": variance,
            "variance_to_budget": variance / budget}


def _solve(allocator: str, mean: np.ndarray, covariance: np.ndarray,
           budget: float, seed: int, origin: str, assets: list[str]) -> dict:
    if allocator == "hard_rank":
        return _dependency("hard").hard_rank(mean, covariance, budget)
    scenarios = _dependency("sampling").sample_joint(mean, covariance, seed=seed, month=origin, count=5000)
    if scenarios.shape != (5000, 15):
        raise ValueError("All 5,000 joint scenarios and 15 ETF columns required")
    return _dependency("scenarios").allocate_scenarios(
        scenarios, assets, seed=20261024 + seed + int(origin.replace("-", "")))


def _failure_kind(error: Exception, allocator: str) -> str:
    if allocator == "hard_rank" and "infeasible" in str(error).lower():
        return "risk_budget_infeasible_under_frozen_constraints"
    return "numerical_or_weight_validation_failure"


def _allocations(case: dict, context: dict, mean: np.ndarray, covariance: np.ndarray) -> tuple[dict, list, list]:
    arrays = {"weights": np.full(_SHAPE, np.nan), "raw_weights": np.full(_SHAPE, np.nan),
              "allocated": np.zeros((3, 28), dtype=bool), "effective_count": np.full((3, 28), np.nan),
              "equity_weight": np.full((3, 28), np.nan), "predicted_variance": np.full((3, 28), np.nan)}
    failures, records = [], []
    baseline = _is_baseline(case)
    if baseline:
        np.testing.assert_allclose(mean, context["baseline_mean"], rtol=0, atol=1e-12,
                                   err_msg="Cached baseline mean identity failed")
        np.testing.assert_allclose(covariance, context["baseline_covariance"], rtol=0, atol=1e-12,
                                   err_msg="Cached baseline covariance identity failed")
    assets = context["assets"].tolist()
    for seed_index, seed in enumerate(context["seeds"]):
        for month_index, date in enumerate(context["dates"]):
            origin, budget = str(date), float(context["risk_budget"][month_index])
            identity = {"seed": int(seed), "origin": origin, "seed_index": seed_index, "month_index": month_index}
            try:
                if baseline:
                    key = "baseline_hard_weights" if case["allocator"] == "hard_rank" else "baseline_scenario_weights"
                    result = {"weights": context[key][seed_index, month_index], "solver_route": "exact_baseline_cache"}
                else:
                    result = _solve(case["allocator"], mean[seed_index, month_index],
                                    covariance[seed_index, month_index], budget, int(seed), origin, assets)
                weights = np.asarray(result["weights"], dtype=float)
                raw = np.asarray(result.get("raw_weights", weights), dtype=float)
                audit = _audit(weights, covariance[seed_index, month_index], budget, case["allocator"])
                _audit(raw, covariance[seed_index, month_index], budget, case["allocator"])
                if np.max(np.abs(weights - raw)) > 1e-8:
                    raise ValueError("Returned weights differ materially from raw solver weights")
                arrays["weights"][seed_index, month_index] = weights
                arrays["raw_weights"][seed_index, month_index] = raw
                arrays["allocated"][seed_index, month_index] = True
                for key in ["effective_count", "equity_weight", "predicted_variance"]:
                    arrays[key][seed_index, month_index] = audit[key]
                numerical = {key: value for key, value in result.items() if key not in {"weights", "raw_weights"}}
                if "solver_route" not in numerical:
                    attempts = result.get("numerical_attempts", [])
                    last = attempts[-1] if attempts else {}
                    numerical["solver_route"] = f"hard_rank_{last.get('solver', 'CLARABEL')}"
                records.append({**identity, "status": "allocated", "weight_audit": audit, **numerical})
            except (ValueError, RuntimeError, ArithmeticError, SolverError) as error:
                failure = {**identity, "stage": "allocation", "kind": _failure_kind(error, case["allocator"]),
                           "error": f"{type(error).__name__}: {error}"}
                failures.append(failure)
                records.append({**failure, "status": "failed"})
    return arrays, records, failures


def _scores(context: dict, mean: np.ndarray, covariance: np.ndarray, arrays: dict) -> tuple[dict, dict]:
    metrics = _dependency("metrics")
    actual, rf = context["actual"], context["risk_free"]
    weights = arrays["weights"]
    total = (1 + np.einsum("sta,ta->st", weights, actual)) * (1 + rf) - 1
    if not np.isfinite(total).all() or (total <= -1).any():
        raise ValueError("Complete valid monthly portfolio returns required")
    wealth = _dependency("wealth").wealth_path(total)
    ic = np.stack([metrics.rank_ic(actual, seed_mean) for seed_mean in mean])
    if not np.isfinite(ic).all():
        raise ValueError("Undefined monthly cross-sectional ranking IC; no partial ranking score")
    joint = [metrics.joint_gaussian_scores(actual, mean[s], covariance[s]) for s in range(3)]
    nll = np.stack([value["joint_nll"] for value in joint])
    mahalanobis = np.stack([value["mahalanobis_squared_per_asset"] for value in joint])
    marginal = np.stack([metrics.gaussian_scores(actual, mean[s], np.sqrt(
        np.diagonal(covariance[s], axis1=-2, axis2=-1)))["nll"] for s in range(3)])
    arrays.update(total_returns=total, wealth=wealth, rank_ic=ic, joint_nll=nll,
                  mahalanobis_squared_per_asset=mahalanobis, marginal_nll=marginal)
    periods = {}
    for name, section in _PERIODS.items():
        per_seed = []
        for seed_index, seed in enumerate(context["seeds"]):
            performance = metrics.portfolio_performance(total[seed_index, section], rf[section])
            sharpe = performance["conventional_sharpe"]
            if sharpe is None or not np.isfinite(sharpe):
                raise ValueError("Undefined Sharpe; this complete case cannot enter the Pareto ranking")
            per_seed.append({
                "seed": int(seed), "sharpe": float(sharpe),
                "total_return": performance["compounded_total_return"],
                "rank_ic": float(ic[seed_index, section].mean()),
                "max_drawdown": performance["max_drawdown_total_nav"],
                "ending_value": 10000 * (1 + performance["compounded_total_return"]),
                "annualized_excess_vol": performance["annualized_excess_vol"],
                "worst_month": performance["worst_total_return_month"],
                "effective_assets": float(arrays["effective_count"][seed_index, section].mean()),
                "equity_weight": float(arrays["equity_weight"][seed_index, section].mean()),
                "joint_nll": float(nll[seed_index, section].mean()),
                "joint_nll_per_asset": float(nll[seed_index, section].mean() / 15),
                "marginal_nll": float(marginal[seed_index, section].mean()),
                "mahalanobis_squared_per_asset": float(mahalanobis[seed_index, section].mean()),
            })
        keys = [key for key in per_seed[0] if key != "seed"]
        periods[name] = {key: float(np.mean([row[key] for row in per_seed])) for key in keys}
        periods[name].update(months=len(rf[section]), per_seed=per_seed,
                             sharpe_seed_sd=float(np.std([row["sharpe"] for row in per_seed], ddof=1)),
                             return_seed_sd=float(np.std([row["total_return"] for row in per_seed], ddof=1)))
    equal = np.full(15, 1 / 15)
    calibration = []
    for seed_index, seed in enumerate(context["seeds"]):
        std = np.sqrt(np.einsum("a,tab,b->t", equal, covariance[seed_index], equal))
        values = metrics.gaussian_scores(actual @ equal, mean[seed_index] @ equal, std)
        calibration.append({"seed": int(seed), **metrics.calibration_summary(values)})
    return periods, {"fixed_equal_weight_portfolio": calibration, "calibration_parameters_fitted": False}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _immutable_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to replace existing case evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"Another worker wrote different immutable evidence: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _cached(case: dict, root: Path, destination: Path) -> dict | None:
    if not destination.exists():
        return None
    record = json.loads(destination.read_text())
    if record["case"] != case or record["id"] != case["id"]:
        raise ValueError("Existing case ID belongs to different settings")
    for name, expected in record["input_sha256"].items():
        if _hash(Path(name)) != expected:
            raise ValueError(f"Immutable case input changed: {name}")
    artifact = root / "cases" / f"{case['id']}.npz"
    if _hash(artifact) != record["artifact_sha256"]:
        raise ValueError("Immutable case arrays changed")
    return record


def evaluate_case(case: dict, output_root: str) -> dict:
    """Evaluate one fixed blending/allocator recipe and seal its complete evidence.

    Args:
        case: JSON-compatible ID/track/stage, allocator, broad_id/compact_id,
            four static alpha fractions, scalar cov_compact_fraction, nullable
            third_id, scalar third_fraction and third_cov_fraction. Fractions
            cannot vary over seeds or months; all must lie in [0, 1].
        output_root: Campaign outputs directory containing context.npz and
            components/{id}.npz. Only cases/{case_id}.json and .npz are written.

    Returns:
        Immutable complete/failed record, with all28, tuning_first16 and
        validation_last12 metrics only when every seed-month allocation and
        primary metric is valid. Metrics average three separately scored seeds.
        Failed cases retain errors/masks/partial arrays, never partial scores.

    Raises:
        ValueError: Malformed case IDs/schema or immutable evidence/input changes.
            Numerical and missing-panel failures instead produce failed records.
    """
    _validate_case(case)
    root = Path(output_root).resolve()
    destination = root / "cases" / f"{case['id']}.json"
    previous = _cached(case, root, destination)
    if previous is not None:
        return previous
    artifact = destination.with_suffix(".npz")
    if artifact.exists():
        raise ValueError(f"Unsealed case arrays exist without their manifest: {artifact}")
    started = time.monotonic()
    record: dict[str, Any] = {"schema": "bayesfolio.gp_tuning.evaluated_case.v1", "id": case["id"],
                              "case": case, "status": "failed", "periods": {}, "failures": [],
                              "input_sha256": {}, "baseline_weights_reused": False,
                              "gross_of_explicit_costs_and_taxes": True,
                              "all_months_share_static_blend": True, "calibration": {}}
    arrays = {"mean": np.full(_SHAPE, np.nan), "covariance": np.full((*_SHAPE, 15), np.nan),
              "weights": np.full(_SHAPE, np.nan), "raw_weights": np.full(_SHAPE, np.nan),
              "allocated": np.zeros((3, 28), dtype=bool), "total_returns": np.full((3, 28), np.nan),
              "wealth": np.full((3, 29), np.nan)}
    numerical = []
    try:
        context, components, hashes = _inputs(case, root)
        record["input_sha256"] = hashes
        arrays.update({key: context[key] for key in ["dates", "assets", "seeds", "actual", "risk_free"]})
        mean, covariance = _blend(case, components)
        arrays.update(mean=mean, covariance=covariance)
        allocated, numerical, failures = _allocations(case, context, mean, covariance)
        arrays.update(allocated)
        record["failures"].extend(failures)
        record["baseline_weights_reused"] = _is_baseline(case)
        if arrays["allocated"].all():
            periods, calibration = _scores(context, mean, covariance, arrays)
            record.update(status="complete", periods=periods, calibration=calibration)
        else:
            # Valid monthly returns remain diagnostic only. No partial wealth or
            # period metric is constructed for an incomplete case.
            arrays["total_returns"] = (1 + np.einsum(
                "sta,ta->st", arrays["weights"], context["actual"])) * (1 + context["risk_free"]) - 1
    except (OSError, KeyError, ValueError, RuntimeError, ArithmeticError, AssertionError, SolverError) as error:
        record["failures"].append({"stage": "inputs_or_complete_case_scoring", "kind": "input_or_metric_failure",
                                   "error": f"{type(error).__name__}: {error}"})
    routes = Counter(row["solver_route"] for row in numerical if row["status"] == "allocated")
    record.update(numerical_route_counts=dict(routes), numerical_records=numerical,
                  successful_allocations=int(arrays["allocated"].sum()), expected_allocations=84,
                  failed_allocation_pairs=84 - int(arrays["allocated"].sum()),
                  failure_kind_counts=dict(Counter(row["kind"] for row in record["failures"])),
                  wall_seconds=time.monotonic() - started)
    if record["status"] == "complete" and record["successful_allocations"] != 84:
        raise ValueError("A partial case cannot be marked complete")
    arrays["status"] = np.array(record["status"])
    # Include fingerprints of existing inputs even when a missing/bad panel
    # prevented full loading, so the failed evidence remains immutable too.
    if not record["input_sha256"]:
        sources = [Path(__file__).resolve(), root / "context.npz", *_HELPERS.values()]
        sources += [root / "components" / f"{case[key]}.npz"
                    for key in ["broad_id", "compact_id", "third_id"] if case[key] is not None]
        record["input_sha256"] = {str(path): _hash(path) for path in sources if path.exists()}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    content = buffer.getvalue()
    _immutable_bytes(artifact, content)
    record["artifact_sha256"] = hashlib.sha256(content).hexdigest()
    record = _jsonable(record)
    _immutable_bytes(destination, (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return record
