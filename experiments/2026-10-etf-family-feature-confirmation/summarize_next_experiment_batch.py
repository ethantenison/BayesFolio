"""Validate and summarize the preregistered next U.S. ETF experiment batch.

The candidate definitions are read from the validated planning artifact. Each
candidate must have all three deterministic seeds and the fixed 24-date panel.
Promotion is based only on Pareto nondominance and positive fixed-reference
hypervolume contribution relative to the Step-3 incumbent set.

Usage:
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/summarize_next_experiment_batch.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from summarize_us_confirmation import aggregate_configurations, hypervolume_3d
from summarize_us_pilots import date_hash, gaussian_nll, is_dominated, read_scored_dates

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_CANDIDATES = EXPERIMENT_DIR / "planning" / "next_experiment_candidates.json"
DEFAULT_EVIDENCE = EXPERIMENT_DIR / "planning" / "step4_evidence.json"
INCUMBENT_AGGREGATES = EXPERIMENT_DIR / "results" / "us_confirmation_configuration_aggregates.csv"
RESULTS_DIR = EXPERIMENT_DIR / "results"
RUN_ROOT = EXPERIMENT_DIR / "runs"
EXPECTED_TICKERS = ["SPY", "MGK", "VTV", "IWM"]
EXPECTED_SEEDS = [27, 127, 227]


def parse_args() -> argparse.Namespace:
    """Parse optional planning-artifact overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def argument_value(arguments: list[str], flag: str) -> str | None:
    """Return the value following a configured runner flag."""
    if flag not in arguments:
        return None
    index = arguments.index(flag)
    if index + 1 >= len(arguments):
        raise ValueError(f"Configured flag {flag} is missing its value.")
    return arguments[index + 1]


def expected_custom_columns(arguments: list[str], flag: str) -> list[str] | None:
    """Return a configured comma-separated column list, when present."""
    value = argument_value(arguments, flag)
    return None if value is None else [column for column in value.split(",") if column]


def validate_manifest(
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    expected_dates: list[str],
    *,
    seed: int,
) -> list[str]:
    """Validate fixed controls plus the candidate's exact inputs and model switches."""
    arguments = candidate["exact_runner_args"]
    modeling = manifest["modeling"]
    data = manifest["data"]
    wrapper = manifest.get("us_equity_first_round_wrapper", {})
    checks = {
        "training_universe": data["training_universe"] == EXPECTED_TICKERS,
        "final_universe": data["final_portfolio_universe"] == EXPECTED_TICKERS,
        "dates": manifest["rebalance_dates"] == expected_dates,
        "seed": modeling["seed"] == seed,
        "rank": modeling["rank"] == 2,
        "maxiter": modeling["maxiter"] == 15,
        "scenarios": modeling["posterior_scenarios_per_rebalance"] == 256,
        "experiment": modeling["experiment"] == "signed_lkj_eta_2",
        "time_index": wrapper.get("time_index_included") is True,
        "input_columns": wrapper.get("input_columns") == candidate["cached_columns_required"],
        "feature_set": wrapper.get("feature_set") == argument_value(arguments, "--feature-set"),
        "custom_etf_columns": wrapper.get("custom_etf_columns")
        == expected_custom_columns(arguments, "--custom-etf-cols"),
        "custom_macro_columns": wrapper.get("custom_macro_columns")
        == expected_custom_columns(arguments, "--custom-macro-cols"),
        "time_modulation": modeling["time_modulation_mode"] == argument_value(arguments, "--time-modulation-mode"),
        "interactions_only": wrapper.get("interactions_only") is ("--interactions-only" in arguments),
        "three_way_interaction": wrapper.get("three_way_interaction_included")
        is ("--include-three-way-interaction" in arguments),
        "date_count": manifest["portfolio"]["n_realized_rebalances"] == len(expected_dates),
    }
    optional_manifest_checks = {
        "--covar-grouping-proposal": (wrapper, "covar_grouping_proposal"),
        "--macro-component-proposal": (wrapper, "macro_component_proposal"),
        "--kernel-composition-proposal": (modeling["kernel_composition_proposal"], "name"),
    }
    for flag, (container, key) in optional_manifest_checks.items():
        expected = argument_value(arguments, flag)
        if expected is not None:
            checks[flag] = container.get(key) == expected
    return [f"{candidate['id']} seed {seed}: {name}" for name, passed in checks.items() if not passed]


def summarize_run(
    candidate: dict[str, Any], expected_dates: list[str], *, seed: int
) -> tuple[dict[str, Any], list[str]]:
    """Validate one candidate run and calculate objectives and diagnostics."""
    run_id = f"{candidate['id']}_s{seed}"
    run_dir = RUN_ROOT / run_id
    required = [
        "manifest.json",
        "gp_window_predictions.csv",
        "gp_window_ic.csv",
        "portfolio_summary.csv",
        "portfolio_returns.csv",
        "REPORT.md",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{run_id} is missing required outputs: {missing}")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    failures = validate_manifest(candidate, manifest, expected_dates, seed=seed)
    predictions = pd.read_csv(run_dir / "gp_window_predictions.csv")
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.date.astype(str)
    predictions["gaussian_nll"] = gaussian_nll(predictions)
    actual_dates = predictions["date"].drop_duplicates().tolist()
    actual_assets = sorted(predictions["asset_id"].drop_duplicates().tolist())
    if actual_dates != expected_dates:
        failures.append(f"{run_id}: prediction_dates")
    if actual_assets != sorted(EXPECTED_TICKERS):
        failures.append(f"{run_id}: prediction_assets")
    if len(predictions) != len(expected_dates) * len(EXPECTED_TICKERS):
        failures.append(f"{run_id}: prediction_row_count")
    numeric_columns = predictions[["y_true", "y_pred", "y_std", "gaussian_nll"]]
    if not np.isfinite(numeric_columns).all().all():
        failures.append(f"{run_id}: nonfinite_predictions")

    by_date_asset = predictions.loc[:, ["date", "asset_id", "gaussian_nll"]].copy()
    by_date_asset.insert(2, "n", 1)
    by_date = predictions.groupby("date", as_index=False, observed=True).agg(
        n=("gaussian_nll", "size"), mean_gaussian_nll=("gaussian_nll", "mean")
    )
    by_date_asset.to_csv(run_dir / "forecast_nll_by_date_asset.csv", index=False)
    by_date.to_csv(run_dir / "forecast_nll_by_date.csv", index=False)

    ic = pd.read_csv(run_dir / "gp_window_ic.csv")
    returns = pd.read_csv(run_dir / "portfolio_returns.csv")
    gp_returns = returns.loc[returns["strategy"].eq("gp_scenarios_riskfolio"), ["date", "return"]].copy()
    gp_returns["date"] = pd.to_datetime(gp_returns["date"]).dt.date.astype(str)
    if gp_returns["date"].tolist() != expected_dates:
        failures.append(f"{run_id}: portfolio_return_dates")
    values = gp_returns["return"].to_numpy(dtype=float)
    return_std = float(np.std(values, ddof=1))
    sharpe = float(np.mean(values) / return_std * math.sqrt(12.0)) if return_std > 0 else math.nan
    return (
        {
            "configuration": candidate["id"],
            "mode": candidate["mode"],
            "priority": candidate["priority"],
            "seed": seed,
            "n_dates": len(expected_dates),
            "date_hash_sha256": date_hash(expected_dates),
            "mean_ic": float(ic["ic"].mean()),
            "mean_gaussian_nll": float(predictions["gaussian_nll"].mean()),
            "sharpe": sharpe,
            "cumulative_return_diagnostic_only": float(np.prod(1.0 + values) - 1.0),
            "rmse_diagnostic_only": float(np.sqrt(np.mean(np.square(predictions["y_true"] - predictions["y_pred"])))),
            "diagnostic_metrics_affect_pareto": False,
            "validation_passed": not failures,
        },
        failures,
    )


def objective_hypervolume(frame: pd.DataFrame, reference: dict[str, float]) -> float:
    """Calculate three-objective hypervolume after converting NLL to maximization space."""
    points = [
        (float(row.mean_ic), -float(row.mean_gaussian_nll), float(row.sharpe)) for row in frame.itertuples(index=False)
    ]
    raw_nll = float(reference["mean_gaussian_nll"])
    transformed_reference = (
        float(reference["mean_ic"]),
        0.0 if raw_nll == 0.0 else -raw_nll,
        float(reference["sharpe"]),
    )
    return hypervolume_3d(points, transformed_reference)


def add_promotion_results(
    candidate_aggregates: pd.DataFrame,
    incumbents: pd.DataFrame,
    *,
    baseline_hypervolume: float,
    reference: dict[str, float],
) -> pd.DataFrame:
    """Add individual Pareto and hypervolume promotion decisions."""
    result = candidate_aggregates.copy()
    nondominated: list[bool] = []
    hypervolumes: list[float] = []
    increments: list[float] = []
    for row in result.itertuples(index=False):
        candidate_frame = pd.DataFrame(
            [
                {
                    "configuration": row.configuration,
                    "mean_ic": row.mean_ic,
                    "mean_gaussian_nll": row.mean_gaussian_nll,
                    "sharpe": row.sharpe,
                }
            ]
        )
        comparison = pd.concat(
            [
                incumbents.loc[:, ["configuration", "mean_ic", "mean_gaussian_nll", "sharpe"]],
                candidate_frame,
            ],
            ignore_index=True,
        )
        candidate_row = comparison.iloc[-1]
        is_nondominated = not is_dominated(candidate_row, comparison)
        candidate_hypervolume = objective_hypervolume(comparison, reference)
        nondominated.append(is_nondominated)
        hypervolumes.append(candidate_hypervolume)
        increments.append(candidate_hypervolume - baseline_hypervolume)
    result["nondominated_vs_incumbents"] = nondominated
    result["hypervolume_with_incumbents"] = hypervolumes
    result["incremental_hypervolume"] = increments
    result["promoted"] = result["nondominated_vs_incumbents"] & (result["incremental_hypervolume"] > 0.0)
    return result


def write_markdown(
    aggregates: pd.DataFrame,
    combined_pareto: pd.DataFrame,
    *,
    baseline_hypervolume: float,
    combined_hypervolume: float,
    validation: dict[str, Any],
) -> None:
    """Write the human-readable next-batch decision report."""
    display = aggregates.drop(columns=["validation_passed"]).copy()
    lines = [
        "# Next U.S. ETF Experiment Batch Results",
        "",
        "All candidates use the frozen 24-window panel and seeds `27,127,227`.",
        "The Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher conventional annualized",
        "Sharpe. Cumulative return and RMSE are recorded only as diagnostics.",
        "",
        "## Candidate aggregates and individual promotion decisions",
        "",
        display.to_markdown(index=False, floatfmt=".9f"),
        "",
        "A candidate is promoted only when it is nondominated against the incumbents and its individual insertion",
        "strictly increases fixed-reference hypervolume.",
        "",
        "## Combined Pareto front",
        "",
        combined_pareto.to_markdown(index=False, floatfmt=".9f"),
        "",
        "## Hypervolume",
        "",
        f"- Step-3 incumbent hypervolume: `{baseline_hypervolume:.12f}`",
        f"- Incumbents plus the complete new batch: `{combined_hypervolume:.12f}`",
        f"- Combined improvement: `{combined_hypervolume - baseline_hypervolume:.12f}`",
        "- Reference point remains IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.",
        "",
        "## Validation",
        "",
        f"- All 12 requested runs passed: `{validation['all_passed']}`",
        f"- Common date hash: `{validation['date_hash_sha256']}`",
        f"- Promoted candidates: `{validation['promoted_candidates']}`",
        "- The task universe, `t_index`, rank 2, maxiter 15, and 256 scenarios were verified from every manifest.",
    ]
    (RESULTS_DIR / "NEXT_EXPERIMENT_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Validate all candidate runs, aggregate objectives, and write promotion results."""
    args = parse_args()
    candidates_path = args.candidates.resolve()
    evidence_path = args.evidence.resolve()
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if payload["based_on"]["evidence_sha256"] != sha256_file(evidence_path):
        raise ValueError("Candidate plan evidence hash mismatch.")
    if payload["based_on"]["seeds"] != EXPECTED_SEEDS:
        raise ValueError("Candidate seed panel changed.")

    expected_dates = read_scored_dates()
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for candidate in sorted(payload["candidates"], key=lambda item: item["priority"]):
        for seed in EXPECTED_SEEDS:
            row, run_failures = summarize_run(candidate, expected_dates, seed=seed)
            rows.append(row)
            failures.extend(run_failures)

    run_objectives = pd.DataFrame(rows)
    candidate_aggregates = aggregate_configurations(run_objectives)
    metadata = pd.DataFrame(
        [
            {"configuration": item["id"], "mode": item["mode"], "priority": item["priority"]}
            for item in payload["candidates"]
        ]
    )
    candidate_aggregates = metadata.merge(candidate_aggregates, on="configuration", how="left")
    incumbents = pd.read_csv(INCUMBENT_AGGREGATES)
    reference = evidence["hypervolume"]["reference_point_raw_objectives"]
    baseline_hypervolume = float(evidence["hypervolume"]["value"])
    candidate_aggregates = add_promotion_results(
        candidate_aggregates,
        incumbents,
        baseline_hypervolume=baseline_hypervolume,
        reference=reference,
    )

    combined = pd.concat(
        [
            incumbents.loc[:, ["configuration", "mean_ic", "mean_gaussian_nll", "sharpe"]],
            candidate_aggregates.loc[:, ["configuration", "mean_ic", "mean_gaussian_nll", "sharpe"]],
        ],
        ignore_index=True,
    )
    combined["pareto_nondominated"] = [not is_dominated(row, combined) for _, row in combined.iterrows()]
    combined_pareto = combined.loc[combined["pareto_nondominated"]].copy()
    combined_hypervolume = objective_hypervolume(combined, reference)
    promoted = candidate_aggregates.loc[candidate_aggregates["promoted"], "configuration"].tolist()
    validation = {
        "schema": "bayesfolio.etf_family_next_batch_validation.v1",
        "all_passed": not failures and bool(run_objectives["validation_passed"].all()),
        "failures": failures,
        "seeds": EXPECTED_SEEDS,
        "run_count": len(run_objectives),
        "candidate_count": len(candidate_aggregates),
        "n_dates": len(expected_dates),
        "date_min": expected_dates[0],
        "date_max": expected_dates[-1],
        "date_hash_sha256": date_hash(expected_dates),
        "candidate_plan_sha256": sha256_file(candidates_path),
        "evidence_sha256": sha256_file(evidence_path),
        "pareto_objectives": {
            "maximize": ["mean_ic", "sharpe"],
            "minimize": ["mean_gaussian_nll"],
        },
        "diagnostic_only": ["cumulative_return", "rmse"],
        "turnover_excluded": True,
        "promoted_candidates": promoted,
    }
    hypervolume = {
        "schema": "bayesfolio.etf_family_next_batch_hypervolume.v1",
        "baseline_value": baseline_hypervolume,
        "combined_value": combined_hypervolume,
        "combined_increment": combined_hypervolume - baseline_hypervolume,
        "reference_point_raw_objectives": reference,
        "candidate_individual_increments": {
            row.configuration: row.incremental_hypervolume for row in candidate_aggregates.itertuples(index=False)
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_objectives.to_csv(RESULTS_DIR / "next_experiment_run_objectives.csv", index=False)
    candidate_aggregates.to_csv(RESULTS_DIR / "next_experiment_candidate_aggregates.csv", index=False)
    combined_pareto.to_csv(RESULTS_DIR / "next_experiment_combined_pareto.csv", index=False)
    (RESULTS_DIR / "next_experiment_hypervolume.json").write_text(
        json.dumps(hypervolume, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RESULTS_DIR / "next_experiment_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(
        candidate_aggregates,
        combined_pareto,
        baseline_hypervolume=baseline_hypervolume,
        combined_hypervolume=combined_hypervolume,
        validation=validation,
    )
    if failures:
        raise ValueError(f"Next-batch validation failed: {failures}")
    print(candidate_aggregates.to_string(index=False))
    print(f"combined_hypervolume={combined_hypervolume:.12f}")


if __name__ == "__main__":
    main()
