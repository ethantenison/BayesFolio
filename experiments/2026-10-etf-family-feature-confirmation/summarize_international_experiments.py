"""Validate and summarize international confirmation or ablation runs.

Usage:
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/summarize_international_experiments.py \
      --config experiments/2026-10-etf-family-feature-confirmation/international_confirmation_config.json \
      --seeds 27,127,227
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from summarize_us_confirmation import aggregate_configurations, calculate_hypervolume, parse_seed_list
from summarize_us_pilots import date_hash, gaussian_nll

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
PERIODS_PER_YEAR = 12.0


def resolve_repo_path(raw: str | Path) -> Path:
    """Resolve a configured path against the repository root."""
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def read_dates(path: Path) -> list[str]:
    """Read an ordered scored-date file while ignoring comments and blanks."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def argument_value(arguments: list[str], flag: str, default: str | None = None) -> str | None:
    """Return the value following a configured flag or a supplied default."""
    if flag not in arguments:
        return default
    index = arguments.index(flag)
    if index + 1 >= len(arguments):
        raise ValueError(f"Configured flag {flag} is missing its value.")
    return arguments[index + 1]


def validate_run_manifest(
    label: str,
    configuration: dict[str, Any],
    manifest: dict[str, Any],
    expected_dates: list[str],
    expected_tickers: list[str],
    *,
    expected_seed: int,
) -> list[str]:
    """Validate one international run against its exact frozen configuration."""
    failures: list[str] = []
    data = manifest["data"]
    modeling = manifest["modeling"]
    wrapper = manifest["us_equity_first_round_wrapper"]
    arguments = [str(value) for value in configuration["args"]]
    expected_kernel = argument_value(arguments, "--kernel-composition-proposal", "none")
    checks = {
        "training_universe": data["training_universe"] == expected_tickers,
        "final_universe": data["final_portfolio_universe"] == expected_tickers,
        "dates": manifest["rebalance_dates"] == expected_dates,
        "seed": modeling["seed"] == expected_seed,
        "rank": modeling["rank"] == 2,
        "maxiter": modeling["maxiter"] == 15,
        "scenarios": modeling["posterior_scenarios_per_rebalance"] == 256,
        "signed_lkj": modeling["experiment"] == "signed_lkj_eta_2",
        "date_count": manifest["portfolio"]["n_realized_rebalances"] == len(expected_dates),
        "time_index": wrapper["time_index_included"],
        "input_columns": wrapper["input_columns"] == configuration["expected_input_columns"],
        "feature_set": wrapper["feature_set"] == argument_value(arguments, "--feature-set"),
        "grouping": wrapper["covar_grouping_proposal"]
        == argument_value(arguments, "--covar-grouping-proposal", "default"),
        "macro_component": wrapper["macro_component_proposal"]
        == argument_value(arguments, "--macro-component-proposal", "default"),
        "time_modulation": modeling["time_modulation_mode"]
        == argument_value(arguments, "--time-modulation-mode", "neither"),
        "kernel_composition": modeling["kernel_composition_proposal"]["name"] == expected_kernel,
    }
    failures.extend(f"{label}: {name}" for name, passed in checks.items() if not passed)
    return failures


def summarize_run(
    label: str,
    configuration: dict[str, Any],
    run_dir: Path,
    expected_dates: list[str],
    expected_tickers: list[str],
    *,
    expected_seed: int,
) -> tuple[dict[str, Any], list[str]]:
    """Validate one run and calculate its three objectives and diagnostics."""
    required = [
        "manifest.json",
        "gp_window_predictions.csv",
        "gp_window_ic.csv",
        "portfolio_returns.csv",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{label} is missing required outputs: {missing}")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    failures = validate_run_manifest(
        label,
        configuration,
        manifest,
        expected_dates,
        expected_tickers,
        expected_seed=expected_seed,
    )
    predictions = pd.read_csv(run_dir / "gp_window_predictions.csv")
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.date.astype(str)
    predictions["gaussian_nll"] = gaussian_nll(predictions)
    if predictions["date"].drop_duplicates().tolist() != expected_dates:
        failures.append(f"{label}: prediction_dates")
    if sorted(predictions["asset_id"].drop_duplicates().tolist()) != sorted(expected_tickers):
        failures.append(f"{label}: prediction_assets")
    if len(predictions) != len(expected_dates) * len(expected_tickers):
        failures.append(f"{label}: prediction_row_count")
    if not np.isfinite(predictions[["y_true", "y_pred", "y_std", "gaussian_nll"]]).all().all():
        failures.append(f"{label}: nonfinite_predictions")

    predictions.loc[:, ["date", "asset_id", "gaussian_nll"]].to_csv(
        run_dir / "forecast_nll_by_date_asset.csv", index=False
    )
    predictions.groupby("date", as_index=False, observed=True).agg(
        n=("gaussian_nll", "size"), mean_gaussian_nll=("gaussian_nll", "mean")
    ).to_csv(run_dir / "forecast_nll_by_date.csv", index=False)

    ic = pd.read_csv(run_dir / "gp_window_ic.csv")
    portfolio_returns = pd.read_csv(run_dir / "portfolio_returns.csv")
    gp_returns = portfolio_returns.loc[
        portfolio_returns["strategy"].eq("gp_scenarios_riskfolio"), ["date", "return"]
    ].copy()
    gp_returns["date"] = pd.to_datetime(gp_returns["date"]).dt.date.astype(str)
    if gp_returns["date"].tolist() != expected_dates:
        failures.append(f"{label}: portfolio_return_dates")
    returns = gp_returns["return"].to_numpy(dtype=float)
    return_std = float(np.std(returns, ddof=1))
    sharpe = float(np.mean(returns) / return_std * math.sqrt(PERIODS_PER_YEAR)) if return_std > 0 else math.nan
    return {
        "configuration": label,
        "seed": expected_seed,
        "n_dates": len(expected_dates),
        "date_hash_sha256": date_hash(expected_dates),
        "mean_ic": float(ic["ic"].mean()),
        "mean_gaussian_nll": float(predictions["gaussian_nll"].mean()),
        "sharpe": sharpe,
        "cumulative_return_diagnostic_only": float(np.prod(1.0 + returns) - 1.0),
        "rmse_diagnostic_only": float(np.sqrt(np.mean(np.square(predictions["y_true"] - predictions["y_pred"])))),
        "diagnostic_metrics_affect_pareto": False,
        "validation_passed": not failures,
    }, failures


def parse_args() -> argparse.Namespace:
    """Parse a frozen phase configuration and deterministic seed panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", required=True)
    return parser.parse_args()


def write_markdown(
    path: Path,
    config: dict[str, Any],
    aggregates: pd.DataFrame,
    pareto: pd.DataFrame,
    validation: dict[str, Any],
    hypervolume: dict[str, Any],
) -> None:
    """Write the human-readable phase report."""
    phase_name = str(config["phase"]).replace("_", " ").title()
    lines = [
        f"# International {phase_name} Results",
        "",
        f"Seeds: `{','.join(str(seed) for seed in validation['seeds'])}`.",
        "",
        "Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional",
        "annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.",
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
        "",
        "## Validation",
        "",
        f"- Common date hash: `{validation['date_hash_sha256']}`",
        f"- All requested runs passed: `{validation['all_passed']}`",
        f"- Failures: `{validation['failures']}`",
        "- VEA, VWO, VSS, and EWX form the exact task universe; t_index is present throughout.",
        "- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Validate a phase and write run, aggregate, frontier, and hypervolume artifacts."""
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_dates = read_dates(resolve_repo_path(config["scored_date_file"]))
    run_root = resolve_repo_path(config["output_root"])
    configurations = [*config.get("comparison_configurations", []), *config["configurations"]]

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    expected_runs: list[str] = []
    for seed in seeds:
        for configuration in configurations:
            label = str(configuration["label"])
            run_id = f"{configuration['id']}_s{seed}"
            expected_runs.append(run_id)
            row, run_failures = summarize_run(
                label,
                configuration,
                run_root / run_id,
                expected_dates,
                config["tickers"],
                expected_seed=seed,
            )
            row["executed_in_phase"] = configuration in config["configurations"]
            rows.append(row)
            failures.extend(run_failures)

    run_objectives = pd.DataFrame(rows)
    aggregates = aggregate_configurations(run_objectives)
    pareto = aggregates.loc[aggregates["pareto_nondominated"]].copy()
    hypervolume = calculate_hypervolume(aggregates, config)
    if any("turnover" in column.lower() for column in [*run_objectives.columns, *aggregates.columns]):
        raise AssertionError("Turnover must not appear in international evaluation outputs.")

    prefix = str(config["results_prefix"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_objectives.to_csv(RESULTS_DIR / f"{prefix}_run_objectives.csv", index=False)
    aggregates.to_csv(RESULTS_DIR / f"{prefix}_configuration_aggregates.csv", index=False)
    pareto.to_csv(RESULTS_DIR / f"{prefix}_pareto.csv", index=False)
    (RESULTS_DIR / f"{prefix}_hypervolume.json").write_text(
        json.dumps(hypervolume, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validation = {
        "schema": "bayesfolio.international_equity_experiment_validation.v1",
        "phase": config["phase"],
        "seeds": seeds,
        "expected_runs": expected_runs,
        "n_dates": len(expected_dates),
        "date_min": expected_dates[0],
        "date_max": expected_dates[-1],
        "date_hash_sha256": date_hash(expected_dates),
        "all_passed": not failures and bool(run_objectives["validation_passed"].all()),
        "failures": failures,
        "pareto_objectives": {"maximize": ["mean_ic", "sharpe"], "minimize": ["mean_gaussian_nll"]},
        "sharpe_definition": {
            "formula": "mean(monthly_excess_return) / sample_std(monthly_excess_return) * sqrt(12)",
            "standard_deviation_ddof": 1,
            "periods_per_year": 12.0,
        },
        "diagnostic_only": ["cumulative_return", "rmse"],
        "turnover_excluded": True,
    }
    (RESULTS_DIR / f"{prefix}_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_name = f"{prefix.upper()}_RESULTS.md"
    write_markdown(RESULTS_DIR / summary_name, config, aggregates, pareto, validation, hypervolume)
    if failures:
        raise ValueError(f"International experiment validation failed: {failures}")
    print(aggregates.to_string(index=False))


if __name__ == "__main__":
    main()
