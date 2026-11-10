"""Compare continuously risk-sensitive allocations on sealed cached GP forecasts.

Usage: .venv/bin/python <this_file> --execute; then --seal or --verify.
Research-only orchestration reuses the previous scorer and strict solver. Only
this experiment's outputs are writable. Returns are decimals; covariance and
unchanged variance budgets are decimal squared. No fitting or recalibration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_key] = "1"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
FRAMEWORK = HERE.parent
PRIOR = FRAMEWORK / "covariance_matching_v1/outputs"
sys.path.extend([str(FRAMEWORK / "blend_sleeves_v1"), str(FRAMEWORK / "blend_interactions_v2")])

import numpy as np  # noqa: E402
import run_blends as base  # noqa: E402
from rank_risk_math import allocate_rank_risk_checked  # noqa: E402
from run_interactions import _audit  # noqa: E402

RECIPES = {"sharpe": [0.5, 0, 0.375, 0.25], "compromise": [0.375, 0, 0.25, 0.0625]}
METHODS = ["half", "broad", "historical", "diagonal_half"]
RISK_AVERSIONS = [0, 0.05, 0.1, 0.25, 0.5, 1, 2]


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: object) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Only new rank-risk outputs are writable")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Conflicting immutable evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def _case(prediction: str, method: str, risk_aversion: float) -> dict:
    label = f"{risk_aversion:g}".replace(".", "p")
    return {
        "id": f"{prediction}__{method}__lambda_{label}",
        "prediction": prediction,
        "alpha": RECIPES[prediction],
        "covariance": method,
        "decision": "mean",
        "risk_aversion": risk_aversion,
    }


def _cases() -> list[dict]:
    return [_case(name, method, value) for name in RECIPES for method in METHODS for value in RISK_AVERSIONS]


def _covariance(source: dict, method: str) -> np.ndarray:
    if method == "broad":
        return source["broad_covariance"].copy()
    if method == "historical":
        return np.broadcast_to(source["historical_covariance"], source["broad_covariance"].shape).copy()
    half = (source["broad_covariance"] + source["compact_covariance"]) * 0.5
    if method == "half":
        return half
    if method == "diagonal_half":
        result = np.zeros_like(half)
        indices = np.arange(15)
        result[..., indices, indices] = np.diagonal(half, axis1=-2, axis2=-1)
        return result
    raise ValueError(f"Unknown covariance source: {method}")


def _panel(case: dict, context: dict) -> dict:
    with np.load(OUTPUT / "input_panel.npz", allow_pickle=False) as source:
        if source["assets"].tolist() != context["assets"] or source["dates"].tolist() != context["dates"]:
            raise ValueError("Frozen ETF/month axes changed")
        return {
            "mean": source[f"mean_{case['prediction']}"].copy(),
            "scores": source[f"scores_{case['prediction']}"].copy(),
            "covariance": _covariance(source, case["covariance"]),
        }


def _worker(case: dict) -> dict:
    # Adapters are local to an isolated worker process, never edits to sealed code.
    base.OUTPUT = OUTPUT
    base._panel = _panel
    base.allocate_checked = partial(allocate_rank_risk_checked, risk_aversion=case["risk_aversion"])
    return base._job(case)


def _prepare() -> dict:
    seal_path = PRIOR / "COMPLETED.json"
    seal = _read(seal_path)
    if seal["passed"] is not True:
        raise ValueError("Previous covariance study incomplete")
    inputs = {**seal["input_sha256"], **seal["artifact_sha256"], str(seal_path): base._hash(seal_path)}
    for path in [*sorted(HERE.glob("*.py")), HERE / "README.md"]:
        inputs[str(path)] = base._hash(path)
    base._verify(inputs)
    context = base._context()
    with np.load(PRIOR / "input_panel.npz", allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    arrays["historical_covariance"] = context["historical_covariance"].copy()
    for key in ["actual", "risk_budget", "risk_free"]:
        np.testing.assert_array_equal(arrays[key], context[key])
    _write(
        OUTPUT / "contract.json",
        {
            "schema": "bayesfolio.rank_risk.v1",
            "input_sha256": inputs,
            "cases": _cases(),
            "families_in_alpha_order": base.FAMILIES,
            "prediction_recipes": RECIPES,
            "covariance_sources": METHODS,
            "risk_aversion_grid": RISK_AVERSIONS,
            "objective": "rank_score - lambda * (portfolio_variance / original_budget) - 1e-8 * sum_weights_squared",
            "variance_reference": "same_training_only_historical_equal_weight_variance_across_all_candidates",
            "variance_ceiling": "original_monthly_absolute_budget_unchanged",
            "minimum_effective": 7,
            "long_only": True,
            "fully_invested": True,
            "ETF_position_cap": None,
            "CVaR": False,
            "fallback_weights": False,
            "covariance_repair": False,
            "realized_error_calibration": False,
            "new_GP_fits": 0,
            "rank": 2,
            "maxiter": 15,
            "seeds": base.SEEDS,
            "forecast_months": 24,
            "initial_training_months": 84,
            "selection": "report_fixed_grid_tradeoffs_and_descriptive_Sharpe_leaders; not_production_promotion",
            "limitation": "same_24_repeatedly_examined_market_months; three_seeds_are_not_independent_histories",
            "verification": "replay_both_half_zero_controls_before_grid; independently_audit_all_raw_weights",
        },
    )
    base._snapshot(OUTPUT / "input_panel.npz", arrays)
    return context


def _control_check(prediction: str, context: dict) -> dict:
    case = _case(prediction, "half", 0)
    path = OUTPUT / "cases" / f"{case['id']}.json"
    _audit(path, context)
    with (
        np.load(path.with_suffix(".npz"), allow_pickle=False) as result,
        np.load(OUTPUT / "input_panel.npz", allow_pickle=False) as source,
    ):
        difference = float(np.max(np.abs(result["weights"] - source[f"weights_{prediction}"])))
        np.testing.assert_allclose(result["weights"], source[f"weights_{prediction}"], rtol=0, atol=1e-7)
    old = _read(PRIOR / "cases" / f"{prediction}__half.json")["summary"]
    row = _read(path)["summary"]
    for metric in ["conventional_sharpe", "compounded_total_return", "pooled_IC", "joint_nll"]:
        np.testing.assert_allclose(row[metric], old[metric], rtol=0, atol=1e-8)
    return {"prediction": prediction, "passed": True, "maximum_raw_weight_difference": difference}


def _diagnostics(case: dict, context: dict) -> dict:
    path = OUTPUT / "cases" / f"{case['id']}.json"
    _audit(path, context)
    result = _read(path)
    control_case = _case(case["prediction"], case["covariance"], 0)
    with (
        np.load(path.with_suffix(".npz"), allow_pickle=False) as panel,
        np.load(OUTPUT / "cases" / f"{control_case['id']}.npz", allow_pickle=False) as control,
        np.load(OUTPUT / "input_panel.npz", allow_pickle=False) as source,
    ):
        weights, scores, matrix = (panel[key] for key in ["weights", "scores", "covariance"])
        np.testing.assert_array_equal(panel["mean"], source[f"mean_{case['prediction']}"])
        np.testing.assert_array_equal(scores, source[f"scores_{case['prediction']}"])
        np.testing.assert_array_equal(matrix, _covariance(source, case["covariance"]))
        rank_score = np.sum(weights * scores, axis=2)
        zero_score = np.sum(control["weights"] * scores, axis=2)
        equal_score = scores.mean(axis=2)
        gap = zero_score - equal_score
        if not (gap > 1e-12).all():
            raise ValueError("Rank advantage retention undefined; cannot omit these months")
        variance = np.einsum("sta,stab,stb->st", weights, matrix, weights)
        zero_variance = np.einsum("sta,stab,stb->st", control["weights"], matrix, control["weights"])
        difference = np.max(np.abs(weights - control["weights"]), axis=2)
        frozen = source[f"weights_{case['prediction']}"]
        utility = rank_score - case["risk_aversion"] * variance / source["risk_budget"]
        utility -= 1e-8 * np.sum(weights**2, axis=2)
        for offset, seed in enumerate(result["seeds"]):
            for index, month in enumerate(seed["months"]):
                assert month["risk_aversion"] == case["risk_aversion"]
                np.testing.assert_allclose(month["rank_risk_utility"], utility[offset, index], rtol=0, atol=1e-12)
        diagnostics = {
            "weighted_rank_score": float(rank_score.mean()),
            "monthly_weighted_rank_score": rank_score.tolist(),
            "rank_advantage_retained_vs_own_zero": float(np.mean((rank_score - equal_score) / gap)),
            "normalized_predicted_variance": float(np.mean(variance / source["risk_budget"])),
            "monthly_normalized_predicted_variance": (variance / source["risk_budget"]).tolist(),
            "predicted_variance_ratio_vs_own_zero": float(np.mean(variance / zero_variance)),
            "materially_changed_months_vs_own_zero": int((difference > 1e-5).sum()),
            "maximum_weight_change_vs_own_zero": float(difference.max()),
            "materially_changed_months_vs_prior_half": int((np.max(np.abs(weights - frozen), axis=2) > 1e-5).sum()),
            "mean_one_way_weight_difference_vs_own_zero": float(
                np.abs(weights - control["weights"]).sum(axis=2).mean() / 2
            ),
            "mean_rank_risk_utility": float(utility.mean()),
        }
        if case["risk_aversion"] == 0 and case["covariance"] in {"half", "broad"}:
            with np.load(PRIOR / "cases" / f"{case['prediction']}__{case['covariance']}.npz") as old:
                np.testing.assert_allclose(weights, old["weights"], rtol=0, atol=1e-7)
    return {**result["summary"], "diagnostics": diagnostics}


def _monotonic_check(rows: list[dict]) -> dict:
    largest_risk_increase, largest_score_increase = 0.0, 0.0
    for prediction in RECIPES:
        for method in METHODS:
            local = sorted(
                (r for r in rows if r["case"]["prediction"] == prediction and r["case"]["covariance"] == method),
                key=lambda r: r["case"]["risk_aversion"],
            )
            risk = np.array([r["diagnostics"]["monthly_normalized_predicted_variance"] for r in local])
            score = np.array([r["diagnostics"]["monthly_weighted_rank_score"] for r in local])
            largest_risk_increase = max(largest_risk_increase, float(np.diff(risk, axis=0).max()))
            largest_score_increase = max(largest_score_increase, float(np.diff(score, axis=0).max()))
            if np.diff(risk, axis=0).max() > 1e-6 or np.diff(score, axis=0).max() > 1e-6:
                raise ValueError("Increasing risk aversion violated the expected monthly efficient frontier")
            for metric in ["pooled_IC", "rmse", "joint_nll"]:
                np.testing.assert_allclose([r[metric] for r in local], local[0][metric], rtol=0, atol=1e-12)
    return {
        "passed": True,
        "maximum_normalized_risk_increase": largest_risk_increase,
        "maximum_rank_score_increase": largest_score_increase,
        "forecast_metrics_constant_across_lambda": True,
    }


def _aggregate_fixed(rows: list[dict]) -> list[dict]:
    aggregated = []
    for row in rows:
        if row["case"]["risk_aversion"] != 0:
            continue  # Fixed forecast distributions do not depend on lambda.
        result = _read(OUTPUT / "cases" / f"{row['id']}.json")
        labels = sorted({entry["portfolio"] for seed in result["seeds"] for entry in seed["fixed_reference_scores"]})
        for label in labels:
            records = [
                entry
                for seed in result["seeds"]
                for entry in seed["fixed_reference_scores"]
                if entry["portfolio"] == label
            ]
            assert len(records) == 3
            fields = [
                key
                for key, value in records[0].items()
                if isinstance(value, (int, float)) and key not in {"seed", "months"}
            ]
            aggregated.append(
                {
                    "prediction": row["case"]["prediction"],
                    "covariance": row["case"]["covariance"],
                    "portfolio": label,
                    **{key: float(np.mean([record[key] for record in records])) for key in fields},
                }
            )
    return aggregated


def _table(rows: list[dict]) -> None:
    lines = [
        "# Rank and risk allocation comparisons",
        "",
        "All returns are gross over the same 24 reused months. Sharpe and return average three seeds. "
        "Rank retention is the fraction of each covariance source's zero-penalty advantage over equal weight. "
        "It is an allocation score, not IC or expected return. GP forecast IC is fixed within each recipe.",
        "",
        "| Prediction | Covariance | Lambda | Sharpe | Return | Volatility | Drawdown | "
        "Rank retained | Effective assets | Changed months |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        case, diagnostic = row["case"], row["diagnostics"]
        lines.append(
            f"| {case['prediction']} | {case['covariance']} | {case['risk_aversion']:g} | "
            f"{row['conventional_sharpe']:.6f} | {row['compounded_total_return']:.2%} | "
            f"{row['annualized_excess_vol']:.2%} | {row['max_drawdown_total_nav']:.2%} | "
            f"{diagnostic['rank_advantage_retained_vs_own_zero']:.2%} | {row['mean_effective_count']:.3f} | "
            f"{diagnostic['materially_changed_months_vs_own_zero']}/72 |"
        )
    payload = "\n".join(lines) + "\n"
    path = OUTPUT / "all_configurations.md"
    if path.exists() and path.read_text() != payload:
        raise ValueError("Conflicting immutable comparison table")
    path.write_text(payload)


def _execute() -> None:
    started = time.monotonic()
    context = _prepare()
    cases = _cases()
    workers = min(15, max(1, (os.cpu_count() or 3) - 2))
    controls = [_case(name, "half", 0) for name in RECIPES]
    print(json.dumps({"cases": len(cases), "workers": workers, "phase": "replay_controls_before_grid"}), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_worker, controls):
            print(f"Replayed control: {result['id']}", flush=True)
        checks = [_control_check(name, context) for name in RECIPES]
        _write(OUTPUT / "control_replay.json", {"passed": True, "checks": checks})
        remaining = [case for case in cases if case not in controls]
        futures = [pool.submit(_worker, case) for case in remaining]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"Completed {index + 2}/{len(cases)}: {result['id']}", flush=True)
    rows = [_diagnostics(case, context) for case in cases]
    monotonic = _monotonic_check(rows)
    for name in RECIPES:
        local = [r for r in rows if r["case"]["prediction"] == name]
        for metric in ["pooled_IC", "rmse"]:
            np.testing.assert_allclose([r[metric] for r in local], local[0][metric], rtol=0, atol=1e-12)
    _write(OUTPUT / "fixed_reference_summary.json", {"summaries": _aggregate_fixed(rows)})
    _write(
        OUTPUT / "comparison.json",
        {
            "summaries": rows,
            "best_by_Sharpe": max(rows, key=lambda r: r["conventional_sharpe"]),
            "best_by_recipe_and_covariance": {
                f"{name}__{method}": max(
                    (r for r in rows if r["case"]["prediction"] == name and r["case"]["covariance"] == method),
                    key=lambda r: r["conventional_sharpe"],
                )
                for name in RECIPES
                for method in METHODS
            },
        },
    )
    _write(
        OUTPUT / "audit.json",
        {
            "passed": True,
            "portfolio_count": len(cases) * 72,
            "replayed_prior_controls": checks,
            "efficient_frontier": monotonic,
            "fixed_forecasts_verified": True,
            "raw_weights_and_all_original_constraints_verified": True,
            "new_GP_fits": 0,
            "fallback_weights": False,
            "constraint_relaxation": False,
        },
    )
    _table(rows)
    inputs = _read(OUTPUT / "contract.json")["input_sha256"]
    base._verify(inputs)
    _write(
        OUTPUT / "EVALUATED.json",
        {
            "passed": True,
            "input_sha256": inputs,
            "artifact_sha256": {
                str(p): base._hash(p)
                for p in sorted(OUTPUT.rglob("*"))
                if p.is_file() and p.name not in {"EVALUATED.json", "COMPLETED.json"}
            },
        },
    )
    print(
        json.dumps(
            {
                "seconds": time.monotonic() - started,
                "portfolio_count": len(cases) * 72,
                "best": max(rows, key=lambda r: r["conventional_sharpe"])["id"],
            }
        ),
        flush=True,
    )


def _seal() -> None:
    evidence = _read(OUTPUT / "EVALUATED.json")
    base._verify(evidence["input_sha256"])
    base._verify(evidence["artifact_sha256"])
    paths = [p for p in sorted(OUTPUT.rglob("*")) if p.is_file() and p.name != "COMPLETED.json"]
    paths.append(HERE / "FINDINGS.md")
    _write(
        OUTPUT / "COMPLETED.json",
        {
            "passed": True,
            "input_sha256": evidence["input_sha256"],
            "artifact_sha256": {str(p): base._hash(p) for p in paths},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true")
    group.add_argument("--seal", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.execute:
        _execute()
    elif args.seal:
        _seal()
    else:
        completed = _read(OUTPUT / "COMPLETED.json")
        base._verify(completed["input_sha256"])
        base._verify(completed["artifact_sha256"])
        print("Verified rank-risk allocations, audits and all protected prior evidence")
