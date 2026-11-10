"""Independently reconstruct joint-study objectives from saved decimal forecasts.

This research-only verifier does not import the experiment metric summarizers.
It recomputes monthly Spearman IC with scipy ranks, marginal Gaussian NLL, and
arithmetic annualized sample-SD Sharpe from actual weighted excess returns.
Published cumulative rows are comparison targets, not calculation inputs.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_joint_objectives.py

Exit zero requires every registered joint seed and its current report to pass.
Incomplete/live runs are recorded and return a nonzero exit; never launch fits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import repair_dataset as data
from scipy.stats import rankdata

OBJECTIVES = ["mean_ic", "mean_gaussian_nll", "sharpe"]


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Calculate Spearman correlation with average tied ranks and explicit undefined values.

    Args:
        left: One monthly cross-section of predictions or returns.
        right: The corresponding actual returns or predictions.

    Returns:
        Pearson correlation of ranked inputs, or NaN for a constant cross-section.
    """
    left_rank, right_rank = rankdata(left), rankdata(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def reconstruct(run_dir: Path, expected_tickers: list[str], expected_dates: list[str]) -> dict[str, Any]:
    """Compute objectives and uncertainty independently from predictions and allocations.

    Args:
        run_dir: Saved joint-run directory containing decimal forecasts and weights.
        expected_tickers: Exact approved joint universe.
        expected_dates: Exact common scored decision dates.

    Returns:
        Objectives, diagnostic return/RMSE, monthly path and within-family evidence.

    Raises:
        ValueError: Coverage, portfolio arithmetic, finite values, or Sharpe is invalid.
    """
    predictions = pd.read_csv(run_dir / "gp_window_predictions.csv")
    expected = pd.MultiIndex.from_product([expected_dates, sorted(expected_tickers)], names=["date", "asset_id"])
    indexed = predictions.set_index(["date", "asset_id"]).sort_index()
    if indexed.index.has_duplicates or not indexed.index.equals(expected):
        raise ValueError("Forecast dates/tasks are incomplete or not exactly the approved joint panel")
    if not np.isfinite(indexed[["y_pred", "y_true", "y_std"]].to_numpy()).all() or not indexed.y_std.gt(0).all():
        raise ValueError("Nonfinite forecasts/targets or nonpositive predictive spread")
    predictions["nll"] = (0.5 * np.log(2 * np.pi * np.square(predictions.y_std))
                          + 0.5 * np.square((predictions.y_true - predictions.y_pred) / predictions.y_std))
    weights = pd.read_csv(run_dir / "portfolio_weights.csv")
    weights = weights[weights.strategy.eq("gp_scenarios_riskfolio")]
    joined = predictions.merge(weights[["date", "asset_id", "weight"]], on=["date", "asset_id"],
                               validate="one_to_one", how="left")
    if joined.weight.isna().any() or not np.isfinite(joined.weight).all():
        raise ValueError("Missing or invalid portfolio allocations")
    if not np.allclose(joined.groupby("date").weight.sum(), 1, atol=1e-7, rtol=0):
        raise ValueError("Saved allocations do not sum to one")
    joined["contribution"] = joined.weight * joined.y_true
    path = joined.groupby("date").contribution.sum().reindex(expected_dates)
    saved_returns = pd.read_csv(run_dir / "portfolio_returns.csv")
    saved_returns = saved_returns[saved_returns.strategy.eq("gp_scenarios_riskfolio")].set_index("date")["return"]
    if saved_returns.index.has_duplicates or set(saved_returns.index) != set(expected_dates):
        raise ValueError("Portfolio path omits or duplicates scored dates")
    np.testing.assert_allclose(path, saved_returns.reindex(expected_dates), rtol=0, atol=1e-10)
    deviation = float(path.std(ddof=1))
    if len(path) != 24 or not np.isfinite(path).all() or deviation <= 0:
        raise ValueError("Conventional Sharpe is not defined for the full monthly path")
    by_date = predictions.groupby("date")
    ic = by_date.apply(lambda x: spearman(x.y_pred.to_numpy(), x.y_true.to_numpy()), include_groups=False)
    if len(ic) != 24 or not np.isfinite(ic).all():
        raise ValueError("IC is undefined in at least one scored joint cross-section")
    families = {}
    for family, tickers in data.FAMILIES.items():
        frame = predictions[predictions.asset_id.isin(tickers)]
        family_ic = frame.groupby("date").apply(
            lambda x: spearman(x.y_pred.to_numpy(), x.y_true.to_numpy()), include_groups=False)
        row = {"within_family_ic": float(family_ic.mean()),
               "ic_defined_months": int(family_ic.notna().sum()), "mean_gaussian_nll": float(frame.nll.mean()),
               "mean_predictive_std": float(frame.y_std.mean())}
        error = (frame.y_true - frame.y_pred).abs()
        for level, z in [(50, 0.6744897501960817), (80, 1.2815515655446004), (95, 1.959963984540054)]:
            row[f"coverage_{level}"] = float((error <= z * frame.y_std).mean())
        families[family] = row
    return {"mean_ic": float(ic.mean()), "mean_gaussian_nll": float(predictions.nll.mean()),
            "sharpe": float(path.mean() / deviation * math.sqrt(12)),
            "cumulative_return_diagnostic_only": float((1 + path).prod() - 1),
            "rmse_diagnostic_only": float(np.sqrt(np.square(predictions.y_true - predictions.y_pred).mean())),
            "portfolio_returns": {str(key): float(value) for key, value in path.items()}, "families": families}


def audit() -> dict[str, Any]:
    """Compare every registered joint seed to independently reconstructed published metrics.

    Returns:
        Audit evidence with explicit pending/failing seeds and source hashes.
    """
    # Import execution fingerprints only here, not in the independent arithmetic.
    import run_study as study

    root = data.HERE / "results/cumulative/global"
    publications = [root / "objectives_by_seed.csv", root / "family_diagnostics_by_seed.csv"]
    publication_hashes = {str(path.relative_to(data.ROOT)): data.digest(path) for path in publications}
    published = pd.read_csv(root / "objectives_by_seed.csv").set_index(["configuration", "seed"])
    family_rows = pd.read_csv(root / "family_diagnostics_by_seed.csv").set_index(["configuration", "seed", "family"])
    if published.index.has_duplicates or family_rows.index.has_duplicates:
        raise ValueError("Published metrics contain duplicate configuration/seed/family keys")
    dates = (data.ARTIFACTS / "scored_dates_24.txt").read_text().splitlines()
    hashes = study.code_hashes()
    checked, unresolved = [], []
    source_paths = [Path(__file__).resolve(), *publications]
    for plan in sorted((data.HERE / "plans").glob("*.json")):
        config = json.loads(plan.read_text())
        if config["family"] != "global":
            continue
        source_paths.append(plan)
        for item in config["configurations"]:
            for seed in data.STUDY["seeds"]:
                identifier = item["id"]
                run_dir = data.HERE / "runs" / f"{identifier}_s{seed}"
                try:
                    stamp = json.loads((run_dir / "repaired_execution_fingerprint.json").read_text())
                    if stamp.get("state") != "completed" or stamp.get("signature") != study.signature(
                        config, item, seed, hashes
                    ):
                        raise ValueError("Run is not complete under matching fingerprints")
                    if (identifier, seed) not in published.index or any(
                        (identifier, seed, family) not in family_rows.index for family in data.FAMILIES
                    ):
                        raise ValueError("Completed run is absent from cumulative tables; refresh analyze_results.py")
                    rebuilt = reconstruct(run_dir, config["tickers"], dates)
                    row = published.loc[(identifier, seed)]
                    for metric in [*OBJECTIVES, "cumulative_return_diagnostic_only", "rmse_diagnostic_only"]:
                        np.testing.assert_allclose(rebuilt[metric], row[metric], atol=1e-9, rtol=0)
                    for family, values in rebuilt["families"].items():
                        for metric, value in values.items():
                            np.testing.assert_allclose(value, family_rows.loc[(identifier, seed, family), metric],
                                                       atol=1e-9, rtol=0, equal_nan=True)
                    checked.append({"configuration": identifier, "seed": seed, **rebuilt})
                    source_paths.extend(run_dir / name for name in ["gp_window_predictions.csv",
                        "portfolio_weights.csv", "portfolio_returns.csv", "repaired_execution_fingerprint.json"])
                except (FileNotFoundError, ValueError, KeyError, AssertionError) as error:
                    unresolved.append({"configuration": identifier, "seed": seed, "reason": str(error)})
    report = {"passed": not unresolved, "checked_seed_runs": len(checked), "checked": checked,
              "unresolved": unresolved, "source_sha256": {
                  str(path.relative_to(data.ROOT)): data.digest(path) for path in source_paths},
              "arithmetic_independent_of_experiment_summarizers": True,
              "sharpe_definition": "mean(monthly excess returns)/sample_SD(ddof=1)*sqrt(12)",
              "scope": "All registered actual joint models; complete source-matched three-seed evidence required."}
    if hashes != study.code_hashes() or any(
        data.digest(data.ROOT / relative) != digest for relative, digest in publication_hashes.items()
    ):
        raise ValueError("Execution sources or published evidence changed during the audit")
    data.write_json(data.HERE / "results/joint_objective_audit.json", report)
    return report


if __name__ == "__main__":
    result = audit()
    print(json.dumps({key: result[key] for key in ["passed", "checked_seed_runs", "unresolved"]}))
    raise SystemExit(0 if result["passed"] else 1)
