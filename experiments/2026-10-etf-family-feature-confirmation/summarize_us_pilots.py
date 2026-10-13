"""Validate and summarize the seed-27 U.S. feature-confirmation pilots.

The report uses IC, Gaussian NLL, and Sharpe as the only Pareto objectives.
Cumulative return and RMSE are retained as explicitly diagnostic-only fields;
turnover is intentionally absent from every generated evaluation table.

Usage:
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/summarize_us_pilots.py
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
RUNS_DIR = EXPERIMENT_DIR / "runs"
RESULTS_DIR = EXPERIMENT_DIR / "results"
DATE_FILE = EXPERIMENT_DIR / "artifacts/us_scored_dates_24.txt"
ARTIFACT_MANIFEST = EXPERIMENT_DIR / "artifacts/us_confirmation_cached_inputs_20261012.manifest.json"

RUNS = {
    "U-AC": RUNS_DIR / "us_u005_ac_s27",
    "U-C005": RUNS_DIR / "us_c005_s27",
    "U-C009": RUNS_DIR / "us_c009_s27",
    "U-ALL-MATCH": RUNS_DIR / "us_all_match_s27",
}
EXPECTED_TICKERS = ["SPY", "MGK", "VTV", "IWM"]
EXPECTED_SEED = 27
EXPECTED_RANK = 2
EXPECTED_MAXITER = 15
EXPECTED_SCENARIOS = 256
VARIANCE_FLOOR = 1e-12
PERIODS_PER_YEAR = 12.0


def read_scored_dates() -> list[str]:
    """Read the frozen scored-date panel, ignoring comments and blank lines."""
    return [
        line.strip()
        for line in DATE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def date_hash(dates: list[str]) -> str:
    """Return a stable digest for an ordered scored-date panel."""
    payload = "\n".join(dates).encode()
    return hashlib.sha256(payload).hexdigest()


def gaussian_nll(predictions: pd.DataFrame) -> pd.Series:
    """Compute pointwise Gaussian negative log likelihood from forecast rows."""
    y_true = predictions["y_true"].to_numpy(dtype=float)
    y_pred = predictions["y_pred"].to_numpy(dtype=float)
    variance = np.maximum(np.square(predictions["y_std"].to_numpy(dtype=float)), VARIANCE_FLOOR)
    values = 0.5 * (np.log(2.0 * math.pi * variance) + np.square(y_true - y_pred) / variance)
    return pd.Series(values, index=predictions.index, name="gaussian_nll")


def is_dominated(row: pd.Series, objectives: pd.DataFrame) -> bool:
    """Return whether another row Pareto-dominates the supplied objective row."""
    no_worse = (
        (objectives["mean_ic"] >= row["mean_ic"])
        & (objectives["mean_gaussian_nll"] <= row["mean_gaussian_nll"])
        & (objectives["sharpe"] >= row["sharpe"])
    )
    strictly_better = (
        (objectives["mean_ic"] > row["mean_ic"])
        | (objectives["mean_gaussian_nll"] < row["mean_gaussian_nll"])
        | (objectives["sharpe"] > row["sharpe"])
    )
    return bool((no_worse & strictly_better).any())


def validate_manifest(
    label: str,
    manifest: dict[str, Any],
    expected_dates: list[str],
    *,
    expected_seed: int = EXPECTED_SEED,
) -> list[str]:
    """Validate the common task, date, budget, and time-input contract for one pilot."""
    failures: list[str] = []
    data = manifest["data"]
    modeling = manifest["modeling"]
    wrapper = manifest["us_equity_first_round_wrapper"]
    checks = {
        "training_universe": data["training_universe"] == EXPECTED_TICKERS,
        "final_universe": data["final_portfolio_universe"] == EXPECTED_TICKERS,
        "ijr_excluded": "IJR" not in data["training_universe"],
        "dates": manifest["rebalance_dates"] == expected_dates,
        "seed": modeling["seed"] == expected_seed,
        "rank": modeling["rank"] == EXPECTED_RANK,
        "maxiter": modeling["maxiter"] == EXPECTED_MAXITER,
        "scenarios": modeling["posterior_scenarios_per_rebalance"] == EXPECTED_SCENARIOS,
        "signed_lkj": modeling["experiment"] == "signed_lkj_eta_2",
        "time_index": wrapper["time_index_included"] and wrapper["input_columns"][0] == "t_index",
        "date_count": manifest["portfolio"]["n_realized_rebalances"] == len(expected_dates),
    }
    failures.extend(f"{label}: {name}" for name, passed in checks.items() if not passed)

    if label == "U-AC":
        ac_checks = {
            "grouping": wrapper["covar_grouping_proposal"] == "macro_split_custom_pending",
            "market_kernel": wrapper["macro_market_kernel_id"] == "rq_lin_ard_on",
            "credit_kernel": wrapper["macro_credit_inflation_kernel_id"] == "m05_ard_on",
            "market_credit_pair": wrapper["macro_market_credit_interaction_included"],
            "time_etf_market_three_way": wrapper["time_etf_market_three_way_included"],
            "market_credit_etf_three_way": wrapper["macro_market_credit_etf_three_way_included"],
            "time_modulation": modeling["time_modulation_mode"] == "neither",
        }
        failures.extend(f"{label}: {name}" for name, passed in ac_checks.items() if not passed)
    elif label in {"U-C005", "U-C009"}:
        compact_checks = {
            "grouping": wrapper["covar_grouping_proposal"] == "default",
            "macro_component": wrapper["macro_component_proposal"] == "matern_linear",
            "time_modulation": modeling["time_modulation_mode"] == "neither",
            "interaction_mode": wrapper["interactions_only"] is (label == "U-C009"),
        }
        failures.extend(f"{label}: {name}" for name, passed in compact_checks.items() if not passed)
    else:
        all_checks = {
            "kernel_composition": (
                modeling["kernel_composition_proposal"]["name"] == "e_plus_m_plus_t_plus_et_plus_mt_plus_emt"
            ),
            "time_modulation": modeling["time_modulation_mode"] == "lengthscale_only",
            "etf_feature_count": len(wrapper["etf_columns"]) == 11,
            "macro_feature_count": len(wrapper["macro_columns"]) == 16,
        }
        failures.extend(f"{label}: {name}" for name, passed in all_checks.items() if not passed)
    return failures


def summarize_run(
    label: str,
    run_dir: Path,
    expected_dates: list[str],
    *,
    expected_seed: int = EXPECTED_SEED,
) -> tuple[dict[str, Any], list[str]]:
    """Validate one run and write its date-level Gaussian NLL artifacts."""
    required = [
        "manifest.json",
        "gp_window_predictions.csv",
        "gp_window_ic.csv",
        "portfolio_summary.csv",
        "portfolio_returns.csv",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{label} is missing required outputs: {missing}")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    failures = validate_manifest(label, manifest, expected_dates, expected_seed=expected_seed)
    predictions = pd.read_csv(run_dir / "gp_window_predictions.csv")
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.date.astype(str)
    predictions["gaussian_nll"] = gaussian_nll(predictions)

    actual_dates = predictions["date"].drop_duplicates().tolist()
    actual_assets = sorted(predictions["asset_id"].drop_duplicates().tolist())
    if actual_dates != expected_dates:
        failures.append(f"{label}: prediction_dates")
    if actual_assets != sorted(EXPECTED_TICKERS):
        failures.append(f"{label}: prediction_assets")
    if len(predictions) != len(expected_dates) * len(EXPECTED_TICKERS):
        failures.append(f"{label}: prediction_row_count")
    if not np.isfinite(predictions[["y_true", "y_pred", "y_std", "gaussian_nll"]]).all().all():
        failures.append(f"{label}: nonfinite_predictions")

    by_date_asset = predictions.loc[:, ["date", "asset_id", "gaussian_nll"]].copy()
    by_date_asset.insert(2, "n", 1)
    by_date = (
        predictions.groupby("date", as_index=False, observed=True)
        .agg(n=("gaussian_nll", "size"), mean_gaussian_nll=("gaussian_nll", "mean"))
    )
    by_date_asset.to_csv(run_dir / "forecast_nll_by_date_asset.csv", index=False)
    by_date.to_csv(run_dir / "forecast_nll_by_date.csv", index=False)

    ic = pd.read_csv(run_dir / "gp_window_ic.csv")
    portfolio_returns = pd.read_csv(run_dir / "portfolio_returns.csv")
    gp_returns = portfolio_returns.loc[
        portfolio_returns["strategy"].eq("gp_scenarios_riskfolio"), ["date", "return"]
    ].copy()
    gp_returns["date"] = pd.to_datetime(gp_returns["date"]).dt.date.astype(str)
    if gp_returns["date"].tolist() != expected_dates:
        failures.append(f"{label}: portfolio_return_dates")
    return_values = gp_returns["return"].to_numpy(dtype=float)
    return_std = float(np.std(return_values, ddof=1))
    conventional_sharpe = (
        float(np.mean(return_values) / return_std * math.sqrt(PERIODS_PER_YEAR)) if return_std > 0 else math.nan
    )
    cumulative_return = float(np.prod(1.0 + return_values) - 1.0)
    result = {
        "configuration": label,
        "seed": expected_seed,
        "n_dates": len(expected_dates),
        "date_hash_sha256": date_hash(expected_dates),
        "mean_ic": float(ic["ic"].mean()),
        "mean_gaussian_nll": float(predictions["gaussian_nll"].mean()),
        "sharpe": conventional_sharpe,
        "cumulative_return_diagnostic_only": cumulative_return,
        "rmse_diagnostic_only": float(np.sqrt(np.mean(np.square(predictions["y_true"] - predictions["y_pred"])))),
        "diagnostic_metrics_affect_pareto": False,
        "validation_passed": not failures,
    }
    return result, failures


def write_markdown(objectives: pd.DataFrame, pareto: pd.DataFrame, validation: dict[str, Any]) -> None:
    """Write a concise human-readable seed-27 pilot report."""
    objective_display = objectives.drop(columns=["date_hash_sha256"]).copy()
    lines = [
        "# U.S. Seed-27 Pilot Results",
        "",
        "These are pilot results from one seed, not promotion decisions. The Pareto objectives are higher IC, lower",
        "Gaussian NLL, and higher conventional annualized Sharpe. The return series is already risk-free adjusted;",
        "Sharpe uses its sample standard deviation. Cumulative return and RMSE are diagnostic-only.",
        "Turnover is excluded.",
        "",
        "## Objective table",
        "",
        objective_display.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Seed-27 nondominated configurations",
        "",
        ", ".join(pareto["configuration"].tolist()),
        "",
        "## Validation",
        "",
        f"- Common date hash: `{validation['date_hash_sha256']}`",
        f"- Dates: `{validation['date_min']}` through `{validation['date_max']}` ({validation['n_dates']} windows)",
        f"- All run validations passed: `{validation['all_passed']}`",
        f"- Validation failures: `{validation['failures']}`",
        "- IJR excluded and t_index included in every pilot.",
        "- Rank 2, maxiter 15, 256 scenarios, and seed 27 were held constant.",
        "",
        "## Gate",
        "",
        "Do not promote a configuration from this table. Seeds 127 and 227 require separate approval.",
    ]
    (RESULTS_DIR / "PILOT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Build validated seed-27 objective, Pareto, and diagnostic-only reports."""
    expected_dates = read_scored_dates()
    artifact_manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    if artifact_manifest["approved_scored_dates"] != expected_dates:
        raise ValueError("Frozen date file does not match the cached artifact manifest.")

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for label, run_dir in RUNS.items():
        row, run_failures = summarize_run(label, run_dir, expected_dates)
        rows.append(row)
        failures.extend(run_failures)

    objectives = pd.DataFrame(rows)
    objective_view = objectives.loc[:, ["configuration", "mean_ic", "mean_gaussian_nll", "sharpe"]]
    objectives["pareto_nondominated_seed27"] = [
        not is_dominated(row, objective_view) for _, row in objective_view.iterrows()
    ]
    pareto = objectives.loc[objectives["pareto_nondominated_seed27"]].copy()

    if any("turnover" in column.lower() for column in objectives.columns):
        raise AssertionError("Turnover must not appear in the confirmation evaluation table.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    objectives.to_csv(RESULTS_DIR / "seed27_objectives.csv", index=False)
    pareto.to_csv(RESULTS_DIR / "seed27_pareto.csv", index=False)
    validation = {
        "schema": "bayesfolio.etf_family_confirmation_validation.v1",
        "seed": EXPECTED_SEED,
        "n_dates": len(expected_dates),
        "date_min": expected_dates[0],
        "date_max": expected_dates[-1],
        "date_hash_sha256": date_hash(expected_dates),
        "configurations": list(RUNS),
        "all_passed": not failures,
        "failures": failures,
        "pareto_objectives": {
            "maximize": ["mean_ic", "sharpe"],
            "minimize": ["mean_gaussian_nll"],
        },
        "sharpe_definition": {
            "formula": "mean(monthly_return) / sample_std(monthly_return) * sqrt(12)",
            "risk_free_return_per_period": 0.0,
            "return_series": "y_excess_lead portfolio return, already adjusted for the integrated risk-free return",
            "standard_deviation_ddof": 1,
            "periods_per_year": PERIODS_PER_YEAR,
        },
        "diagnostic_only": ["cumulative_return", "rmse"],
        "turnover_excluded": True,
    }
    (RESULTS_DIR / "seed27_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(objectives, pareto, validation)
    if failures:
        raise ValueError(f"Pilot validation failed: {failures}")
    print(objectives.to_string(index=False))


if __name__ == "__main__":
    main()
