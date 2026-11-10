"""Compare expected-rank scores using cached forecasts and an unchanged allocator.

Usage: .venv/bin/python <this_file> --execute; then --seal or --verify.
Research orchestration writes only this folder's outputs. All means and realized
returns are decimals; covariance and budgets are decimal squared. No GP fitting,
recalibration, continuous risk penalty or allocation-covariance search occurs.
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
PRIOR = FRAMEWORK / "rank_risk_v1/outputs"
sys.path.extend([str(FRAMEWORK / "blend_sleeves_v1"), str(FRAMEWORK / "blend_interactions_v2")])

import numpy as np  # noqa: E402
import run_blends as base  # noqa: E402
from expected_rank_math import METHODS, allocation_scores, pairwise_diagnostics  # noqa: E402
from metrics import rank_ic  # noqa: E402
from run_interactions import _audit  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

RECIPES = {"sharpe": [0.5, 0, 0.375, 0.25], "compromise": [0.375, 0, 0.25, 0.0625]}


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: object) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Only new uncertainty-rank outputs are writable")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Conflicting immutable evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def _case(prediction: str, method: str) -> dict:
    return {
        "id": f"{prediction}__{method}",
        "prediction": prediction,
        "alpha": RECIPES[prediction],
        "covariance": "half",
        "decision": method,
        "role": "primary" if method in {"hard_rank", "expected_full"} else "diagnostic",
    }


def _cases() -> list[dict]:
    return [_case(name, method) for name in RECIPES for method in METHODS]


def _panel(case: dict, context: dict) -> dict:
    with np.load(OUTPUT / "input_panel.npz", allow_pickle=False) as source:
        if source["assets"].tolist() != context["assets"] or source["dates"].tolist() != context["dates"]:
            raise ValueError("Frozen ETF/month axes changed")
        mean = source[f"mean_{case['prediction']}"].copy()
        covariance = (source["broad_covariance"] + source["compact_covariance"]) * 0.5
    scoring = allocation_scores(mean, covariance, case["decision"])
    return {"mean": mean, "covariance": covariance, **scoring}


def _worker(case: dict) -> dict:
    # Only the score panel and output path change inside this isolated worker.
    # The original allocator, its covariance and all portfolio constraints stay fixed.
    base.OUTPUT = OUTPUT
    base._panel = _panel
    return base._job(case)


def _prepare() -> dict:
    seal_path = PRIOR / "COMPLETED.json"
    seal = _read(seal_path)
    if seal["passed"] is not True:
        raise ValueError("Previous rank-risk study incomplete")
    inputs = {**seal["input_sha256"], **seal["artifact_sha256"], str(seal_path): base._hash(seal_path)}
    for path in [*sorted(HERE.glob("*.py")), HERE / "README.md"]:
        inputs[str(path)] = base._hash(path)
    base._verify(inputs)
    context = base._context()
    with np.load(PRIOR / "input_panel.npz", allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    for key in ["actual", "risk_budget", "risk_free"]:
        np.testing.assert_array_equal(arrays[key], context[key])
    _write(
        OUTPUT / "contract.json",
        {
            "schema": "bayesfolio.uncertainty_rank.v1",
            "input_sha256": inputs,
            "cases": _cases(),
            "recipes": RECIPES,
            "families_in_alpha_order": base.FAMILIES,
            "score": "(1 + sum_j_ne_i Phi((mu_i-mu_j)/sqrt(var_i+var_j-2cov_ij))) / 15",
            "score_variants": METHODS,
            "probability_interpretation": "observed_next_month_outperformance; not_latent_expected_return_confidence",
            "hypothesis": "frozen_blended_means_plus_half_GP_covariance; not_a_refitted_single_GP_posterior",
            "allocator": "unchanged_StableScoreAllocator_via_allocate_checked",
            "allocation_covariance": "same_arithmetic_half_GP_matrix_for_every_case_including_diagonal_score_ablation",
            "objective": "score @ weights - 1e-8 * sum(weights_squared); no_continuous_risk_penalty",
            "absolute_variance_ceiling": "unchanged_training_only_historical_equal_weight_variance",
            "minimum_effective": 7,
            "long_only": True,
            "fully_invested": True,
            "ETF_position_cap": None,
            "CVaR": False,
            "fallback_weights": False,
            "constraint_relaxation": False,
            "covariance_repair": False,
            "realized_error_calibration": False,
            "new_GP_fits": 0,
            "rank": 2,
            "maxiter": 15,
            "seeds": base.SEEDS,
            "forecast_months": 24,
            "initial_training_months": 84,
            "primary_comparison": "expected_full_vs_hard_rank_for_each_frozen_mean_recipe",
            "diagnostics": "diagonal_uncertainty_ablation_and_hard_reranking_of_expected_scores; not_primary_winners",
            "record": "decision_score_IC_separate_from_raw_mean_IC; forecast_density_unchanged; "
            "pairwise_Brier_and_logloss",
            "limitation": "same_24_reused_market_months; correlated_ETF_pairs_and_seeds_are_not_independent_histories",
            "verification": "hard_rank_controls_replay_before_launch; "
            "audit_raw_weights_scores_and_all_pairwise_identities",
        },
    )
    base._snapshot(OUTPUT / "input_panel.npz", arrays)
    return context


def _control_check(prediction: str, context: dict) -> dict:
    path = OUTPUT / "cases" / f"{prediction}__hard_rank.json"
    _audit(path, context)
    old_path = PRIOR / "cases" / f"{prediction}__half__lambda_0.json"
    with np.load(path.with_suffix(".npz")) as result, np.load(old_path.with_suffix(".npz")) as old:
        difference = float(np.max(np.abs(result["weights"] - old["weights"])))
        np.testing.assert_array_equal(result["scores"], old["scores"])
        np.testing.assert_array_equal(result["covariance"], old["covariance"])
        np.testing.assert_allclose(result["weights"], old["weights"], rtol=0, atol=1e-7)
    for metric in ["conventional_sharpe", "compounded_total_return", "decision_IC", "joint_nll"]:
        np.testing.assert_allclose(
            _read(path)["summary"][metric], _read(old_path)["summary"][metric], rtol=0, atol=1e-8
        )
    return {"prediction": prediction, "passed": True, "maximum_weight_difference": difference}


def _diagnostics(case: dict, context: dict) -> dict:
    path = OUTPUT / "cases" / f"{case['id']}.json"
    _audit(path, context)
    artifact = _read(path)
    with (
        np.load(path.with_suffix(".npz"), allow_pickle=False) as panel,
        np.load(OUTPUT / "cases" / f"{case['prediction']}__hard_rank.npz", allow_pickle=False) as control,
    ):
        mean, covariance, scores, weights = (panel[key] for key in ["mean", "covariance", "scores", "weights"])
        np.testing.assert_array_equal(mean, control["mean"])
        np.testing.assert_array_equal(covariance, control["covariance"])
        expected = allocation_scores(mean, covariance, case["decision"])
        for key, value in expected.items():
            np.testing.assert_array_equal(panel[key], value)
        probability = panel["probability"]
        np.testing.assert_allclose(probability + probability.swapaxes(-1, -2), 1, rtol=0, atol=2e-15)
        np.testing.assert_allclose(panel["expected_scores"].sum(axis=-1), 8, rtol=0, atol=3e-14)
        if not ((scores >= 1 / 15 - 1e-14) & (scores <= 1 + 1e-14)).all():
            raise ValueError("Scores changed the registered normalized-rank range")
        ranks = rankdata(scores, axis=-1)
        old_ranks = rankdata(control["scores"], axis=-1)
        changed_order = ranks != old_ranks
        weight_difference = np.max(np.abs(weights - control["weights"]), axis=-1)
        actual = np.broadcast_to(context["actual"], mean.shape)
        decision_ic = np.array([rank_ic(context["actual"], score) for score in scores])
        scopes = {
            "pooled": list(range(15)),
            **{name: [context["assets"].index(a) for a in assets] for name, assets in context["families"].items()},
        }
        scoped_ic = {}
        for name, indices in scopes.items():
            values = np.array([rank_ic(context["actual"][:, indices], score[:, indices]) for score in scores])
            if not np.isfinite(values).all():
                raise ValueError("Undefined score IC; no months can be omitted")
            scoped_ic[name] = {"mean": float(values.mean()), "monthly_by_seed": values.tolist()}
        scoring_covariance = covariance
        if case["decision"] == "expected_diagonal":
            scoring_covariance = np.zeros_like(covariance)
            indices = np.arange(15)
            scoring_covariance[..., indices, indices] = np.diagonal(covariance, axis1=-2, axis2=-1)
        pairwise = pairwise_diagnostics(actual, mean, scoring_covariance)
        for offset, seed in enumerate(artifact["seeds"]):
            np.testing.assert_allclose(seed["decision_IC"], decision_ic[offset].mean(), rtol=0, atol=1e-12)
        diagnostics = {
            "decision_score_IC_by_scope": scoped_ic,
            "decision_score_IC_seed_values": decision_ic.mean(axis=1).tolist(),
            "rank_order_changed_months": int(changed_order.any(axis=-1).sum()),
            "mean_absolute_rank_position_change": float(np.abs(ranks - old_ranks).mean()),
            "maximum_absolute_rank_position_change": float(np.abs(ranks - old_ranks).max()),
            "score_spread_fraction_of_hard_rank": float(
                np.mean(np.ptp(scores, axis=-1) / np.ptp(control["scores"], axis=-1))
            ),
            "materially_changed_holding_months": int((weight_difference > 1e-5).sum()),
            "maximum_weight_change": float(weight_difference.max()),
            "mean_one_way_weight_difference": float(np.abs(weights - control["weights"]).sum(axis=-1).mean() / 2),
            "pairwise_gaussian_scores": {key: float(value.mean()) for key, value in pairwise.items()},
            "monthly_pairwise_gaussian_scores": {key: value.tolist() for key, value in pairwise.items()},
            "pairwise_scores_use": "full_Gaussian_except_declared_diagonal_score_ablation; "
            "no_probability_claim_for_hard_ranks",
            "mean_equity_weight": float(artifact["summary"]["mean_equity_weight"]),
        }
    return {**artifact["summary"], "diagnostics": diagnostics}


def _table(rows: list[dict]) -> None:
    lines = [
        "# Expected rank comparisons",
        "",
        "Fixed allocation covariance and original allocator for every row. Returns are gross and cumulative; "
        "volatility is annualized. Figures average three seeds on the same 24 reused months. "
        "Decision IC evaluates the actual allocation scores; raw-mean forecast IC and joint NLL stay fixed.",
        "",
        "| Recipe | Score | Role | Sharpe | Return | Volatility | Drawdown | Decision IC | Raw mean IC | "
        "Changed order months | Changed holding months |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        case, diagnostic = row["case"], row["diagnostics"]
        lines.append(
            f"| {case['prediction']} | {case['decision']} | {case['role']} | {row['conventional_sharpe']:.6f} | "
            f"{row['compounded_total_return']:.2%} | {row['annualized_excess_vol']:.2%} | "
            f"{row['max_drawdown_total_nav']:.2%} | {row['decision_IC']:.6f} | {row['pooled_IC']:.6f} | "
            f"{diagnostic['rank_order_changed_months']}/72 | {diagnostic['materially_changed_holding_months']}/72 |"
        )
    path = OUTPUT / "all_configurations.md"
    payload = "\n".join(lines) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError("Conflicting immutable comparison table")
    path.write_text(payload)


def _execute() -> None:
    if (OUTPUT / "COMPLETED.json").exists():
        _verify_completed()
        return
    started = time.monotonic()
    context = _prepare()
    cases = _cases()
    workers = min(15, max(1, (os.cpu_count() or 3) - 2), len(cases))
    controls = [_case(name, "hard_rank") for name in RECIPES]
    print(json.dumps({"cases": len(cases), "workers": workers, "phase": "replay_hard_rank_controls"}), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(_worker, controls):
            print(f"Replayed {row['id']}", flush=True)
        checks = [_control_check(name, context) for name in RECIPES]
        _write(OUTPUT / "control_replay.json", {"passed": True, "checks": checks})
        futures = [pool.submit(_worker, case) for case in cases if case not in controls]
        for future in as_completed(futures):
            print(f"Completed {future.result()['id']}", flush=True)
    rows = [_diagnostics(case, context) for case in cases]
    for name in RECIPES:
        local = [r for r in rows if r["case"]["prediction"] == name]
        for metric in ["pooled_IC", "rmse", "joint_nll"]:
            np.testing.assert_allclose([r[metric] for r in local], local[0][metric], rtol=0, atol=1e-12)
        direct = next(r for r in local if r["case"]["decision"] == "expected_full")
        reranked = next(r for r in local if r["case"]["decision"] == "reranked_expected_full")
        np.testing.assert_allclose(direct["decision_IC"], reranked["decision_IC"], rtol=0, atol=1e-12)
    _write(
        OUTPUT / "comparison.json",
        {
            "summaries": rows,
            "best_primary_by_Sharpe": max(
                (r for r in rows if r["case"]["role"] == "primary"), key=lambda r: r["conventional_sharpe"]
            ),
        },
    )
    _write(
        OUTPUT / "audit.json",
        {
            "passed": True,
            "portfolio_count": len(cases) * 72,
            "replayed_controls": checks,
            "allocation_covariance_and_means_exactly_unchanged": True,
            "original_allocator_unchanged": True,
            "raw_weights_and_original_constraints_verified": True,
            "pairwise_identities_and_IC_verified": True,
            "new_GP_fits": 0,
            "no_fallbacks_or_recalibration": True,
        },
    )
    _table(rows)
    inputs = _read(OUTPUT / "contract.json")["input_sha256"]
    base._verify(inputs)
    artifacts = [
        OUTPUT / "contract.json",
        OUTPUT / "input_panel.npz",
        OUTPUT / "control_replay.json",
        OUTPUT / "comparison.json",
        OUTPUT / "audit.json",
        OUTPUT / "all_configurations.md",
    ]
    artifacts.extend(sorted((OUTPUT / "cases").glob("*")))
    _write(
        OUTPUT / "EVALUATED.json",
        {
            "passed": True,
            "input_sha256": inputs,
            "artifact_sha256": {str(p): base._hash(p) for p in artifacts if p.is_file()},
        },
    )
    print(json.dumps({"seconds": time.monotonic() - started, "portfolio_count": len(cases) * 72}), flush=True)


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


def _verify_completed() -> None:
    evidence = _read(OUTPUT / "COMPLETED.json")
    base._verify(evidence["input_sha256"])
    base._verify(evidence["artifact_sha256"])
    print("Verified expected-rank comparisons, original allocator and all protected prior evidence")


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
        _verify_completed()
