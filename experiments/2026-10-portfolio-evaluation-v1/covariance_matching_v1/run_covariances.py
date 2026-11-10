"""Compare covariance hypotheses while freezing both leading prediction recipes.

Usage: .venv/bin/python <this_file> --execute; then --seal or --verify.
Research-only orchestration reuses the sealed allocator/scorer. Worker-local
adapters replace only its panel reader and output directory. Means are decimal
discounted-excess returns; covariance and unchanged risk budgets are decimal
squared. No GP refits, realized-error calibration, new mean search or 80/20 policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_key] = "1"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
FRAMEWORK = HERE.parent
PRIOR = FRAMEWORK / "blend_interactions_v2/outputs"
sys.path.extend([str(FRAMEWORK / "blend_sleeves_v1"), str(FRAMEWORK / "blend_interactions_v2")])

import numpy as np  # noqa: E402
import run_blends as base  # noqa: E402
from covariance_math import METHODS, matched_covariance  # noqa: E402
from run_interactions import _audit  # noqa: E402

RECIPES = {"sharpe": [0.5, 0, 0.375, 0.25], "compromise": [0.375, 0, 0.25, 0.0625]}


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: object) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Only new covariance outputs are writable")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Conflicting immutable evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def _role(method: str) -> str:
    if method == "root_independent":
        return "independence_diagnostic_not_winner_eligible"
    return "dependence_sensitivity" if method in {"root_shared_50", "root_shared_90"} else "primary"


def _cases() -> list[dict]:
    return [
        {
            "id": f"{prediction}__{method}",
            "prediction": prediction,
            "alpha": alpha,
            "covariance": method,
            "decision": "mean",
            "role": _role(method),
        }
        for prediction, alpha in RECIPES.items()
        for method in METHODS
    ]


def _frozen_panel(case: dict, context: dict) -> dict:
    with np.load(OUTPUT / "input_panel.npz", allow_pickle=False) as source:
        if source["assets"].tolist() != context["assets"] or source["dates"].tolist() != context["dates"]:
            raise ValueError("Frozen ETF/month axes changed")
        alpha = base._alpha(case, context)
        covariance = matched_covariance(
            source["broad_covariance"], source["compact_covariance"], alpha, case["covariance"]
        )
        return {
            "mean": source[f"mean_{case['prediction']}"].copy(),
            "scores": source[f"scores_{case['prediction']}"].copy(),
            "covariance": covariance,
        }


def _worker(case: dict) -> dict:
    # Adapters are process-local. Frozen previous functions and files stay unchanged.
    base.OUTPUT = OUTPUT
    base._panel = _frozen_panel
    return base._job(case)


def _prepare() -> dict:
    seal = _read(PRIOR / "COMPLETED.json")
    if seal["passed"] is not True:
        raise ValueError("Prior study incomplete")
    inputs = {
        **seal["input_sha256"],
        **seal["artifact_sha256"],
        str(PRIOR / "COMPLETED.json"): base._hash(PRIOR / "COMPLETED.json"),
    }
    for path in sorted(HERE.glob("*.py")):
        inputs[str(path)] = base._hash(path)
    base._verify(inputs)
    context, arrays = base._context(), {}
    with np.load(PRIOR / "source_panel.npz", allow_pickle=False) as source:
        for key in [
            "broad_covariance",
            "compact_covariance",
            "actual",
            "risk_free",
            "risk_budget",
            "assets",
            "dates",
            "seeds",
        ]:
            arrays[key] = source[key].copy()
    rows = _read(PRIOR / "comparison.json")["grid_summaries"]
    references = {}
    for name, alpha in RECIPES.items():
        row = next(row for row in rows if row["case"]["alpha"] == alpha)
        references[name] = row
        with np.load(PRIOR / "cases" / f"{row['id']}.npz", allow_pickle=False) as source:
            for key in ["mean", "scores", "weights"]:
                arrays[f"{key}_{name}"] = source[key].copy()
    _write(
        OUTPUT / "contract.json",
        {
            "schema": "bayesfolio.covariance_matching.v1",
            "input_sha256": inputs,
            "frozen_references": references,
            "cases": _cases(),
            "GP_refits": 0,
            "rank": 2,
            "maxiter": 15,
            "seeds": base.SEEDS,
            "forecast_months": 24,
            "families_in_alpha_order": base.FAMILIES,
            "minimum_effective": 7,
            "absolute_variance_budget": "unchanged_expanding_training_historical_equal_weight_15_ETF_variance",
            "objective": "rank",
            "ETF_position_cap": None,
            "CVaR": False,
            "capital_split": "unrestricted_full_15_ETF_optimizer",
            "fallback_weights": False,
            "covariance_repair": False,
            "realized_error_calibration": False,
            "dependence_parameters": "hypothesized_0_0.5_0.9_1_not_estimated_from_realized_errors",
            "selection": "primary_methods_only; compare_Sharpe_and_fixed_portfolio_density_losses; IC_frozen",
            "limitation": "same_24_reused_market_months; exploratory_not_independent_validation",
        },
    )
    base._snapshot(OUTPUT / "input_panel.npz", arrays)
    return context


def _correlation(matrix: np.ndarray) -> np.ndarray:
    standard = np.sqrt(np.diagonal(matrix, axis1=-2, axis2=-1))
    return matrix / standard[..., :, None] / standard[..., None, :]


def _diagnostics(case: dict, context: dict) -> dict:
    path = OUTPUT / "cases" / f"{case['id']}.json"
    _audit(path, context)
    result = _read(path)
    with (
        np.load(OUTPUT / "input_panel.npz", allow_pickle=False) as source,
        np.load(path.with_suffix(".npz"), allow_pickle=False) as panel,
    ):
        reference = 0.5 * (source["broad_covariance"] + source["compact_covariance"])
        covariance, weights = panel["covariance"], panel["weights"]
        frozen = source[f"weights_{case['prediction']}"]
        np.testing.assert_array_equal(panel["mean"], source[f"mean_{case['prediction']}"])
        np.testing.assert_array_equal(panel["scores"], source[f"scores_{case['prediction']}"])
        expected = matched_covariance(
            source["broad_covariance"], source["compact_covariance"], base._alpha(case, context), case["covariance"]
        )
        np.testing.assert_allclose(covariance, expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(source["actual"], context["actual"])
        np.testing.assert_array_equal(source["risk_budget"], context["risk_budget"])
        fixed_risk = np.einsum("sta,stab,stb->st", frozen, covariance, frozen)
        old_risk = np.einsum("sta,stab,stb->st", frozen, reference, frozen)
        ratio = fixed_risk / old_risk
        diagonal_ratio = np.diagonal(covariance, axis1=-2, axis2=-1) / np.diagonal(reference, axis1=-2, axis2=-1)
        traces = np.trace(covariance, axis1=-2, axis2=-1) / np.trace(reference, axis1=-2, axis2=-1)
        equal = np.full(15, 1 / 15)
        ew_ratio = np.einsum("a,stab,b->st", equal, covariance, equal) / np.einsum(
            "a,stab,b->st", equal, reference, equal
        )
        difference = np.max(np.abs(weights - frozen), axis=2)
        changes = _correlation(covariance) - _correlation(reference)
        summary = {
            "role": case["role"],
            "same_fixed_means_and_scores": True,
            "fixed_control_weights_variance_ratio_mean": float(ratio.mean()),
            "fixed_control_weights_variance_ratio_min": float(ratio.min()),
            "fixed_control_weights_variance_ratio_max": float(ratio.max()),
            "equal_weight_variance_ratio_mean": float(ew_ratio.mean()),
            "marginal_variance_ratio_mean": float(diagonal_ratio.mean()),
            "marginal_variance_ratio_min": float(diagonal_ratio.min()),
            "marginal_variance_ratio_max": float(diagonal_ratio.max()),
            "trace_ratio_mean": float(traces.mean()),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
            "correlation_change_rms": float(np.sqrt(np.mean(changes**2))),
            "materially_changed_weight_months": int((difference > 1e-5).sum()),
            "maximum_weight_change": float(difference.max()),
            "control_weights_would_exceed_budget_months": int((fixed_risk > context["risk_budget"] * (1 + 1e-6)).sum()),
            "monthly_fixed_control_weights_variance_ratio": ratio.tolist(),
        }
        if case["covariance"] == "half":
            np.testing.assert_allclose(weights, frozen, rtol=0, atol=1e-7)
    return {**result["summary"], "diagnostics": summary}


def _execute() -> None:
    started = time.monotonic()
    context = _prepare()
    cases = _cases()
    workers = min(13, max(1, (os.cpu_count() or 3) - 2))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, case) for case in cases]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            print(f"Completed {index}/{len(cases)}: {row['id']}", flush=True)
    rows = [_diagnostics(case, context) for case in cases]
    references = _read(OUTPUT / "contract.json")["frozen_references"]
    for name in RECIPES:
        for row in rows:
            if row["case"]["prediction"] != name:
                continue
            for metric in ["pooled_IC", "rmse"]:
                np.testing.assert_allclose(row[metric], references[name][metric], rtol=0, atol=1e-12)
            if row["case"]["covariance"] == "half":
                for metric in ["conventional_sharpe", "compounded_total_return"]:
                    np.testing.assert_allclose(row[metric], references[name][metric], rtol=0, atol=1e-8)
    _write(
        OUTPUT / "comparison.json",
        {
            "summaries": rows,
            "previous_references": references,
            "best_primary_by_Sharpe": {
                name: max(
                    (r for r in rows if r["case"]["prediction"] == name and r["diagnostics"]["role"] == "primary"),
                    key=lambda r: r["conventional_sharpe"],
                )
                for name in RECIPES
            },
        },
    )
    _write(
        OUTPUT / "audit.json",
        {
            "passed": True,
            "portfolio_count": len(cases) * 72,
            "fixed_mean_and_IC_verified": True,
            "replayed_half_controls_verified": True,
            "GP_refits": 0,
            "absolute_budgets_unchanged": True,
            "no_fallbacks": True,
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
                "results": [
                    {
                        "id": r["id"],
                        "Sharpe": r["conventional_sharpe"],
                        "NLL": r["joint_nll"],
                        "risk_ratio": r["diagnostics"]["fixed_control_weights_variance_ratio_mean"],
                        "binding": r["risk_binding_months"],
                    }
                    for r in rows
                ],
            }
        ),
        flush=True,
    )


def _table(rows: list[dict]) -> None:
    lines = [
        "# Fixed prediction covariance comparisons",
        "",
        "All rows retain their recipe's mean predictions and original absolute risk budgets. "
        "Risk ratio evaluates each covariance on the SAME saved 50/50-control portfolio weights. "
        "Root independence is a diagnostic, not an eligible winner. All results are exploratory and gross.",
        "",
        "| Prediction | Covariance | Role | Sharpe | Return | Joint NLL | Fixed-weight risk ratio | Binding months |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']['prediction']} | [{row['case']['covariance']}](cases/{row['id']}.json) | "
            f"{row['diagnostics']['role']} | {row['conventional_sharpe']:.6f} | "
            f"{row['compounded_total_return']:.2%} | {row['joint_nll']:.4f} | "
            f"{row['diagnostics']['fixed_control_weights_variance_ratio_mean']:.4f} | "
            f"{row['risk_binding_months']}/72 |"
        )
    path = OUTPUT / "all_configurations.md"
    payload = "\n".join(lines) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError("Changed immutable comparison table")
    path.write_text(payload)


def _seal() -> None:
    evidence = _read(OUTPUT / "EVALUATED.json")
    base._verify(evidence["input_sha256"])
    base._verify(evidence["artifact_sha256"])
    files = sorted(p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "COMPLETED.json")
    files.append(HERE / "FINDINGS.md")
    _write(
        OUTPUT / "COMPLETED.json",
        {
            "passed": True,
            "input_sha256": evidence["input_sha256"],
            "artifact_sha256": {str(p): base._hash(p) for p in files},
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
        print("Verified fixed-mean covariance comparison, audits and all protected previous evidence")
