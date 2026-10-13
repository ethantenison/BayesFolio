"""Aggregate validated U.S. confirmation runs across deterministic seeds.

This script reuses the per-run manifest and metric checks from the seed-27
pilot summarizer, then computes the Pareto frontier from mean IC, mean Gaussian
NLL, and conventional annualized Sharpe across the requested seed panel.
Return and RMSE remain diagnostic-only, and turnover is excluded.

Usage:
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/summarize_us_confirmation.py \
      --seeds 27,127,227
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from summarize_us_pilots import date_hash, is_dominated, read_scored_dates, summarize_run

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "us_confirmation_config.json"
RESULTS_DIR = EXPERIMENT_DIR / "results"


def hypervolume_2d(points: list[tuple[float, float]], reference: tuple[float, float]) -> float:
    """Return the union area dominated by maximization points above a reference."""
    reference_y, reference_z = reference
    eligible = [(y, z) for y, z in points if y > reference_y and z > reference_z]
    boundaries = [reference_y, *sorted({y for y, _ in eligible})]
    area = 0.0
    for lower, upper in zip(boundaries, boundaries[1:]):
        active_z = [z for y, z in eligible if y >= upper]
        if active_z:
            area += (upper - lower) * (max(active_z) - reference_z)
    return area


def hypervolume_3d(points: list[tuple[float, float, float]], reference: tuple[float, float, float]) -> float:
    """Return exact three-objective dominated hypervolume for maximization points."""
    if not all(math.isfinite(value) for point in points for value in point):
        raise ValueError("Hypervolume points must be finite.")
    if not all(math.isfinite(value) for value in reference):
        raise ValueError("The hypervolume reference point must be finite.")
    reference_x, reference_y, reference_z = reference
    eligible = [
        (x, y, z)
        for x, y, z in points
        if x > reference_x and y > reference_y and z > reference_z
    ]
    boundaries = [reference_x, *sorted({x for x, _, _ in eligible})]
    volume = 0.0
    for lower, upper in zip(boundaries, boundaries[1:]):
        cross_section = [(y, z) for x, y, z in eligible if x >= upper]
        volume += (upper - lower) * hypervolume_2d(cross_section, (reference_y, reference_z))
    return volume


def calculate_hypervolume(aggregates: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Transform the three objectives and calculate hypervolume from a frozen reference."""
    raw_reference = config["hypervolume"]["reference_point"]
    nll_reference = float(raw_reference["mean_gaussian_nll"])
    reference = (
        float(raw_reference["mean_ic"]),
        0.0 if nll_reference == 0.0 else -nll_reference,
        float(raw_reference["sharpe"]),
    )
    points = [
        (float(row.mean_ic), -float(row.mean_gaussian_nll), float(row.sharpe))
        for row in aggregates.itertuples(index=False)
    ]
    return {
        "schema": "bayesfolio.etf_family_confirmation_hypervolume.v1",
        "value": hypervolume_3d(points, reference),
        "reference_point_raw_objectives": raw_reference,
        "reference_point_maximization_space": {
            "mean_ic": reference[0],
            "negative_mean_gaussian_nll": reference[1],
            "sharpe": reference[2],
        },
        "objective_transform": config["hypervolume"]["objective_transform"],
        "aggregation": "configuration means across the requested seed panel",
        "configuration_count": len(aggregates),
        "pareto_configuration_count": int(aggregates["pareto_nondominated"].sum()),
    }


def parse_seed_list(raw: str) -> list[int]:
    """Parse a unique, ordered comma-separated seed list."""
    seeds: list[int] = []
    for value in raw.split(","):
        seed = int(value.strip())
        if seed < 0:
            raise ValueError("Seeds must be non-negative integers.")
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the confirmation aggregator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", required=True, help="Comma-separated deterministic seeds to aggregate.")
    return parser.parse_args()


def aggregate_configurations(run_objectives: pd.DataFrame) -> pd.DataFrame:
    """Aggregate objectives and diagnostic-only metrics over seeds."""
    grouped = run_objectives.groupby("configuration", sort=False, observed=True)
    aggregates = grouped.agg(
        seed_count=("seed", "nunique"),
        mean_ic=("mean_ic", "mean"),
        std_ic=("mean_ic", "std"),
        mean_gaussian_nll=("mean_gaussian_nll", "mean"),
        std_gaussian_nll=("mean_gaussian_nll", "std"),
        sharpe=("sharpe", "mean"),
        std_sharpe=("sharpe", "std"),
        cumulative_return_diagnostic_only=("cumulative_return_diagnostic_only", "mean"),
        rmse_diagnostic_only=("rmse_diagnostic_only", "mean"),
        validation_passed=("validation_passed", "all"),
    ).reset_index()
    seed_values = grouped["seed"].apply(lambda values: ",".join(str(seed) for seed in sorted(values.unique())))
    aggregates.insert(2, "seeds", aggregates["configuration"].map(seed_values))
    objective_view = aggregates.loc[:, ["configuration", "mean_ic", "mean_gaussian_nll", "sharpe"]]
    aggregates["pareto_nondominated"] = [
        not is_dominated(row, objective_view) for _, row in objective_view.iterrows()
    ]
    return aggregates


def write_markdown(
    aggregates: pd.DataFrame,
    pareto: pd.DataFrame,
    *,
    seeds: list[int],
    validation: dict[str, Any],
    hypervolume: dict[str, Any],
) -> None:
    """Write a concise multi-seed U.S. confirmation report."""
    lines = [
        "# U.S. Feature Confirmation Results",
        "",
        f"Seeds: `{','.join(str(seed) for seed in seeds)}`.",
        "",
        "Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional annualized",
        "Sharpe across seeds. The monthly return series is already risk-free adjusted. Return and RMSE are",
        "diagnostic-only; turnover is excluded.",
        "",
        "## Configuration aggregates",
        "",
        aggregates.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Nondominated configurations",
        "",
        ", ".join(pareto["configuration"].tolist()),
        "",
        "## Hypervolume",
        "",
        f"- Observed dominated hypervolume: `{hypervolume['value']:.9f}`",
        "- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.",
        "- NLL is negated for the maximization-space calculation; the reference is frozen for subsequent rounds.",
        "",
        "## Validation",
        "",
        f"- Common date hash: `{validation['date_hash_sha256']}`",
        f"- All requested runs passed: `{validation['all_passed']}`",
        f"- Failures: `{validation['failures']}`",
        "- IJR is excluded and t_index is present in every model.",
        "- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.",
    ]
    (RESULTS_DIR / "US_CONFIRMATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Validate requested runs and write run-level and aggregate reports."""
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_dates = read_scored_dates()
    run_root = EXPERIMENT_DIR.parents[1] / config["output_root"]

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    expected_runs: list[str] = []
    for seed in seeds:
        for configuration in config["configurations"]:
            label = str(configuration["label"])
            run_id = f"{configuration['id']}_s{seed}"
            expected_runs.append(run_id)
            row, run_failures = summarize_run(
                label,
                run_root / run_id,
                expected_dates,
                expected_seed=seed,
            )
            rows.append(row)
            failures.extend(run_failures)

    run_objectives = pd.DataFrame(rows)
    aggregates = aggregate_configurations(run_objectives)
    pareto = aggregates.loc[aggregates["pareto_nondominated"]].copy()
    hypervolume = calculate_hypervolume(aggregates, config)
    output_columns = [*run_objectives.columns, *aggregates.columns]
    if any("turnover" in column.lower() for column in output_columns):
        raise AssertionError("Turnover must not appear in confirmation evaluation outputs.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_objectives.to_csv(RESULTS_DIR / "us_confirmation_run_objectives.csv", index=False)
    aggregates.to_csv(RESULTS_DIR / "us_confirmation_configuration_aggregates.csv", index=False)
    pareto.to_csv(RESULTS_DIR / "us_confirmation_pareto.csv", index=False)
    (RESULTS_DIR / "us_confirmation_hypervolume.json").write_text(
        json.dumps(hypervolume, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = {
        "schema": "bayesfolio.etf_family_confirmation_validation.v2",
        "seeds": seeds,
        "expected_runs": expected_runs,
        "n_dates": len(expected_dates),
        "date_min": expected_dates[0],
        "date_max": expected_dates[-1],
        "date_hash_sha256": date_hash(expected_dates),
        "all_passed": not failures and bool(run_objectives["validation_passed"].all()),
        "failures": failures,
        "pareto_objectives": {
            "maximize": ["mean_ic", "sharpe"],
            "minimize": ["mean_gaussian_nll"],
        },
        "sharpe_definition": {
            "formula": "mean(monthly_excess_return) / sample_std(monthly_excess_return) * sqrt(12)",
            "return_series": "y_excess_lead portfolio return, already adjusted for the integrated risk-free return",
            "standard_deviation_ddof": 1,
            "periods_per_year": 12.0,
        },
        "diagnostic_only": ["cumulative_return", "rmse"],
        "turnover_excluded": True,
    }
    (RESULTS_DIR / "us_confirmation_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(aggregates, pareto, seeds=seeds, validation=validation, hypervolume=hypervolume)
    if failures:
        raise ValueError(f"Confirmation validation failed: {failures}")
    print(aggregates.to_string(index=False))


if __name__ == "__main__":
    main()
