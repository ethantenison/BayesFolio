"""Complete a bounded factorial search of missing family prediction blends.

Usage: .venv/bin/python <this_file> --execute; then --seal or --verify.
Research-only orchestration reuses the sealed blend_sleeves_v1 evaluator and its
decimal-return forecasts. Covariance/budgets are decimal squared. Old evidence
is read-only; isolated worker processes redirect only the frozen evaluator's
output destination to this new folder. No GP fits, new risk policy or 80/20 test.
"""

from __future__ import annotations

import argparse
import itertools
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
PREVIOUS = HERE.parent / "blend_sleeves_v1"
sys.path.insert(0, str(PREVIOUS))

import numpy as np  # noqa: E402
import run_blends as base  # noqa: E402

GRID = {
    "us_equity": [0.25, 0.375, 0.5, 0.625],
    "international_equity": [0, 0.0625, 0.125, 0.25],
    "us_fixed_income": [0, 0.0625, 0.125, 0.25, 0.375, 0.5],
    "international_fixed_income": [0, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1],
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: object) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Only new interaction outputs are writable")
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Conflicting immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def _cases() -> list[dict]:
    return [
        base._case(list(fractions), 0.5) for fractions in itertools.product(*(GRID[family] for family in base.FAMILIES))
    ]


def _worker(case: dict) -> dict:
    # This function executes only in an isolated process during experiments.
    # Reuse the frozen implementation rather than duplicate metric/solver logic.
    base.OUTPUT = OUTPUT
    return base._job(case)


def _verify_previous() -> dict:
    completed = _read(PREVIOUS / "outputs/COMPLETED.json")
    delivery = _read(PREVIOUS / "outputs/DELIVERY.json")
    base._verify(completed["input_sha256"])
    base._verify(completed["artifact_sha256"])
    base._verify(delivery["artifact_sha256"])
    inputs = {**completed["input_sha256"], **completed["artifact_sha256"], **delivery["artifact_sha256"]}
    for path in [PREVIOUS / "outputs/DELIVERY.json", PREVIOUS / "outputs/COMPLETED.json", *HERE.glob("*.py")]:
        inputs[str(path)] = base._hash(path)
    return inputs


def _audit(path: Path, context: dict) -> int:
    artifact = _read(path)
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as panel:
        weights = panel["weights"].copy()
        covariance = panel["covariance"].copy()
        assert panel["mean"].shape == (3, 24, 15)
    assert weights.shape == (3, 24, 15) and covariance.shape == (3, 24, 15, 15)
    assert artifact["mode"] == "blend" and artifact["summary"]["complete_seeds"] == 3
    np.linalg.cholesky(covariance)
    assert np.isfinite(weights).all() and weights.min() >= -1e-8
    np.testing.assert_allclose(weights.sum(axis=2), 1, rtol=0, atol=1e-7)
    assert (np.sum(weights**2, axis=2) <= 1 / 7 + 1e-7).all()
    variances = np.einsum("sta,stab,stb->st", weights, covariance, weights)
    assert (variances <= context["risk_budget"] * (1 + 1e-6)).all()
    for offset, seed in enumerate(artifact["seeds"]):
        assert seed["seed"] == base.SEEDS[offset] and seed["solved_months"] == 24
        for index, month in enumerate(seed["months"]):
            assert month["index"] == index and month["date"] == context["dates"][index]
            assert month["status"] == "solved" and month["numerical_attempts"][-1]["status"] == "optimal"
            assert month["ETF_position_cap"] is None and month["fallback_used"] is False
            assert month["constraint_relaxation"] is False
        np.testing.assert_allclose(weights[offset], [month["weights"] for month in seed["months"]], atol=1e-12)
        raw = np.einsum("ta,ta->t", weights[offset], context["actual"])
        total = (1 + raw) * (1 + context["risk_free"]) - 1
        excess = total - context["risk_free"]
        nav = np.r_[1, np.cumprod(1 + total)]
        metrics = {
            "conventional_sharpe": np.sqrt(12) * excess.mean() / excess.std(ddof=1),
            "compounded_total_return": nav[-1] - 1,
            "annualized_excess_vol": np.sqrt(12) * excess.std(ddof=1),
            "max_drawdown_total_nav": np.min(nav / np.maximum.accumulate(nav) - 1),
        }
        for key, value in metrics.items():
            np.testing.assert_allclose(seed["performance"][key], value, rtol=0, atol=1e-12)
    return 72


def _execute() -> None:
    started = time.monotonic()
    inputs = _verify_previous()
    previous = _read(PREVIOUS / "outputs/comparison.json")
    old = {row["id"]: row for row in previous["summaries"]}
    cases = _cases()
    reused = [old[case["id"]] for case in cases if case["id"] in old]
    missing = [case for case in cases if case["id"] not in old]
    _write(
        OUTPUT / "contract.json",
        {
            "input_sha256": inputs,
            "grid": GRID,
            "families_in_alpha_order": base.FAMILIES,
            "case_count": len(cases),
            "new_case_count": len(missing),
            "reused_case_count": len(reused),
            "policy": {
                **base.POLICY,
                "schema": "bayesfolio.blend_interactions.v2",
                "scope": "prediction_interactions_only; fixed_half_arithmetic_covariance; unconstrained_capital_split",
                "limitation": "additional_adaptive_search_on_same_24_exploratory_months; not_untouched_validation",
            },
        },
    )
    _write(
        OUTPUT / "plan.json", {"all_cases": cases, "new_cases": missing, "reused_ids": [row["id"] for row in reused]}
    )
    with np.load(PREVIOUS / "outputs/source_panel.npz", allow_pickle=False) as source:
        base._snapshot(OUTPUT / "source_panel.npz", {key: source[key].copy() for key in source.files})
    workers = min(13, max(1, (os.cpu_count() or 3) - 2))
    results = []
    print(json.dumps({"grid": len(cases), "new": len(missing), "reused": len(reused), "workers": workers}), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, case) for case in missing]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 100 == 0 or len(results) == len(missing):
                print(f"Completed {len(results)}/{len(missing)} new cases", flush=True)
    grid_rows = base._ordered([*reused, *results])
    cumulative = base._ordered([*previous["summaries"], *results])
    context = base._context()
    audited = 0
    for case in cases:
        parent = PREVIOUS / "outputs" if case["id"] in old else OUTPUT
        audited += _audit(parent / "cases" / f"{case['id']}.json", context)
    _write(
        OUTPUT / "audit.json",
        {
            "passed": True,
            "grid_portfolios_audited": audited,
            "new_monthly_portfolios": len(missing) * 72,
            "reused_portfolios": len(reused) * 72,
            "GP_fits": 0,
            "all_weights_raw_strictly_accepted": True,
        },
    )
    result = {
        "grid_summaries": grid_rows,
        "new_summaries": base._ordered(results),
        "cumulative_summaries": cumulative,
        "grid_pareto_ids": base._pareto(grid_rows),
        "cumulative_pareto_ids": base._pareto(cumulative),
        "previous_Sharpe_leader": previous["winner"],
        "previous_IC_leader": max(previous["summaries"], key=lambda row: row["decision_IC"]),
        "grid_Sharpe_leader": grid_rows[0],
        "grid_IC_leader": max(grid_rows, key=lambda row: row["decision_IC"]),
        "cumulative_Sharpe_leader": cumulative[0],
        "cumulative_IC_leader": max(cumulative, key=lambda row: row["decision_IC"]),
    }
    _write(OUTPUT / "comparison.json", result)
    _table(grid_rows)
    base._verify(inputs)
    _write(
        OUTPUT / "EVALUATED.json",
        {
            "passed": True,
            "input_sha256": inputs,
            "artifact_sha256": {
                str(p): base._hash(p)
                for p in sorted(OUTPUT.rglob("*"))
                if p.is_file()
                and p.suffix in {".json", ".npz", ".md"}
                and p.name not in {"EVALUATED.json", "COMPLETED.json"}
            },
        },
    )
    print(
        json.dumps(
            {
                "seconds": time.monotonic() - started,
                "audited_portfolios": audited,
                "Sharpe_leader": grid_rows[0],
                "IC_leader": result["grid_IC_leader"],
            }
        ),
        flush=True,
    )


def _table(rows: list[dict]) -> None:
    lines = [
        "# Complete family prediction interaction grid",
        "",
        "All fractions are compact-model prediction weights; covariance is fixed at 50/50. "
        "Gross return covers the same 24 forecast months; metrics average three seeds. "
        "Full per-case files retain every forecast, covariance, weight and evaluation metric.",
        "",
        "| Case | US equity | Intl equity | US FI | Intl FI | Sharpe | Return | Pooled IC | Intl EQ IC | Intl FI IC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    old_ids = {row["id"] for row in _read(PREVIOUS / "outputs/comparison.json")["summaries"]}
    for row in rows:
        prefix = "../../blend_sleeves_v1/outputs" if row["id"] in old_ids else "."
        weights = " | ".join(f"{a:g}" for a in row["case"]["alpha"])
        lines.append(
            f"| [{row['id']}]({prefix}/cases/{row['id']}.json) | {weights} | "
            f"{row['conventional_sharpe']:.6f} | {row['compounded_total_return']:.2%} | "
            f"{row['pooled_IC']:.6f} | {row['international_equity_IC']:.6f} | "
            f"{row['international_fixed_income_IC']:.6f} |"
        )
    path = OUTPUT / "all_configurations.md"
    payload = "\n".join(lines) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError("Changed immutable configuration table")
    path.write_text(payload)


def _seal() -> None:
    evidence = _read(OUTPUT / "EVALUATED.json")
    base._verify(evidence["input_sha256"])
    base._verify(evidence["artifact_sha256"])
    paths = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "COMPLETED.json"]
    paths.append(HERE / "FINDINGS.md")
    _write(
        OUTPUT / "COMPLETED.json",
        {
            "passed": True,
            "input_sha256": evidence["input_sha256"],
            "artifact_sha256": {str(p): base._hash(p) for p in sorted(paths)},
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
        seal = _read(OUTPUT / "COMPLETED.json")
        base._verify(seal["input_sha256"])
        base._verify(seal["artifact_sha256"])
        print("Verified all interaction evidence and protected previous study")
