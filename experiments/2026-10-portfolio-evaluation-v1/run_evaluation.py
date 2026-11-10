"""Offline reproducible evaluation of cached GP forecasts, with no new fits.

Writes only inside this new experiment folder. Existing data and studies are
read-only, inputs/code are SHA-256 sealed, and matching reruns verify output
bytes. Scores are exploratory; 24 months reused across many configurations are
not an untouched holdout. Three seeds are never counted as 72 market months.

Usage:
    .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/run_evaluation.py \
        --covariance-mode scenarios --output cached_scenarios_v1
    Default mode 'none' leaves GP joint/projection scores explicitly unavailable.
    Analytic mode requires --analytic-root with documented NPZ/JSON sidecars.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate import evaluate_model
from metrics import paired_block_bootstrap

from artifacts import Reader, load_context, load_covariance, load_run, verify_legacy_audit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEGACY = REPO / "experiments/2026-10-repaired-global-feature-selection"
SEEDS = [27, 127, 227]
MODEL = "global_e6c41f2020fd"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Conflicting output exists; choose a new output name: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _tables(results: list[dict]) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ["ranking", "monthly_ranking", "etfs", "portfolio_distributions", "portfolios"]:
        tables[name] = pd.DataFrame([{**row, "seed": result["seed"], "model": result["model"]}
                                    for result in results for row in result[name]])
    portfolios = tables["portfolios"]
    columns = ["conventional_sharpe", "annualized_excess_vol", "compounded_total_return", "max_drawdown_total_nav"]
    tables["portfolio_seed_summary"] = portfolios.groupby("portfolio")[columns].agg(["mean", "std"])
    tables["portfolio_seed_summary"].columns = ["_".join(column) for column in
                                               tables["portfolio_seed_summary"].columns]
    tables["portfolio_seed_summary"] = tables["portfolio_seed_summary"].reset_index()
    return tables


def _evaluate(reader: Reader, args: argparse.Namespace, context: dict) -> tuple[list[dict], dict]:
    results = []
    for seed in SEEDS:
        run = load_run(reader, args.legacy_root, args.model_id, seed, context)
        if args.model_id == MODEL and "hy_spread_chg_1p" in run["manifest"]["command"].split("--custom-macro-cols")[1]:
            raise ValueError("Accepted baseline must not include the experimental HY-change variable")
        covariance, meta = load_covariance(reader, run, context, args.covariance_mode, args.analytic_root, seed)
        result = evaluate_model(run, context, covariance, meta)
        result.update(seed=seed, model=args.model_id)
        # Independent parity with old arithmetic on preserved target units.
        saved = reader.read(run["run"] / "portfolio_returns.csv")
        saved = saved[saved.strategy.eq("gp_scenarios_riskfolio")].set_index("date")["return"]
        np.testing.assert_allclose(result["monthly_portfolio_paths"]["legacy_maxret"]["discounted_excess_legacy"],
                                   saved.reindex(context["dates"]).to_numpy(), rtol=0, atol=1e-10)
        results.append(result)
    comparisons = {}
    for policy in ["legacy_maxret", "linear_rank"]:
        left = np.array([item["monthly_portfolio_paths"][policy]["arithmetic_excess"] for item in results]).T
        for benchmark in ["equal_weight", "fixed_60_40"]:
            right = np.array([item["monthly_portfolio_paths"][benchmark]["arithmetic_excess"] for item in results]).T
            comparisons[f"{policy}_vs_{benchmark}"] = paired_block_bootstrap(left, right)
    return results, comparisons


def execute(args: argparse.Namespace) -> dict:
    """Run deterministic cached evaluation, seal provenance, and verify repeatability.

    Args:
        args: Parsed CLI arguments for source model, covariance precision and output.

    Returns:
        Verification summary with measured wall time; timings are not scientific
        results and are excluded from deterministic artifact fingerprints.
    """
    started = time.perf_counter()
    reader = Reader()
    for path in sorted(HERE.glob("*.py")):
        reader.read(path)
    reader.read(HERE / "specification.json")
    reader.read(REPO / "uv.lock")
    verified_original_sources = verify_legacy_audit(reader, args.legacy_root)
    context = load_context(reader, args.legacy_root)
    loaded = time.perf_counter()
    results, comparisons = _evaluate(reader, args, context)
    computed = time.perf_counter()
    output = HERE / "outputs" / args.output
    if output.resolve().parent != (HERE / "outputs").resolve():
        raise ValueError("Output must be one directory name inside this new experiment")
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "pandas", "scikit-learn"]}
    manifest = {"schema": "bayesfolio.research.portfolio_evaluation.v1", "model": args.model_id,
        "seeds": SEEDS, "months_not_multiplied_by_seeds": len(context["dates"]), "assets": context["assets"],
        "dates": context["dates"], "covariance_mode": args.covariance_mode,
        "return_convention": context["return_convention"], "return_conversion": "total=(1+target)*(1+RF)-1",
        "source_sha256": reader.hashes, "versions": versions, "python": platform.python_version(),
        "cost_basis": "gross_costs_not_supplied", "holdout_status": "reused_exploratory_evaluation_months",
        "new_gp_fits": 0, "source_studies_modified": False, "automatic_promotion": False,
        "verified_original_audit_sources": verified_original_sources,
        "covariance_jitter_or_diagonal_fallback": False,
        "precision": results[0]["joint_distribution"],
        "limitations": ["Latest source vintage, not a point-in-time data certification",
                        "Total returns reconstructed with frozen RF; original target labels unchanged",
                        "Bootstrap intervals conditional on explored candidates, not selection-adjusted",
                        "Legacy allocation is MaxRet without an explicit variance budget",
                        "Fixed probe scores are comparable; adaptive allocation scores change the evaluated portfolio"]}
    # Precision metadata only; per-month scores belong in results, not duplicated in provenance.
    manifest["precision"] = {key: value for key, value in results[0]["joint_distribution"].items()
                            if key in ["source", "exact_gp_joint_nll_available", "monte_carlo_approximation", "status"]}
    reader.verify()
    payloads = {"manifest.json": _json_bytes(manifest), "results.json": _json_bytes(results),
                "paired_comparisons.json": _json_bytes(comparisons)}
    for name, table in _tables(results).items():
        payloads[name + ".csv"] = table.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    for name, payload in payloads.items():
        _write_immutable(output / name, payload)
    verification = {"passed": True, "result_sha256": hashes, "source_files": len(reader.hashes),
                    "three_complete_seed_runs": True, "source_files_unchanged": True,
                    "legacy_return_parity_atol": 1e-10, "new_gp_fits": 0}
    _write_immutable(output / "VERIFIED.json", _json_bytes(verification))
    # Each input is fingerprinted again after calculation and publishing.
    reader.verify()
    timings = {"context_seconds": loaded - started, "evaluation_seconds": computed - loaded,
               "total_wall_seconds": time.perf_counter() - started,
               "pid": os.getpid(), "thread_controls": {key: os.environ.get(key) for key in
                   ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]}}
    # Runtime telemetry is append-only and deliberately not part of score reproducibility.
    with (output / "timings.jsonl").open("a") as handle:
        handle.write(json.dumps(timings, sort_keys=True) + "\n")
    return {**verification, "output": str(output), "timing": timings}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=LEGACY)
    parser.add_argument("--model-id", default=MODEL)
    parser.add_argument("--covariance-mode", choices=["none", "scenarios", "analytic"], default="none")
    parser.add_argument("--analytic-root", type=Path)
    parser.add_argument("--output", default="cached_marginals_v1")
    args = parser.parse_args()
    print(json.dumps(execute(args), indent=2))


if __name__ == "__main__":
    _main()
