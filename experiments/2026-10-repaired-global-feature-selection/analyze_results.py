"""Build cumulative Pareto, uncertainty, and conditional-ablation evidence.

Research reporting consumes only fingerprinted complete three-seed runs and
preserves missing/incomplete entries explicitly. Family component Sharpe from a
joint portfolio is labelled separately from the actual joint portfolio Sharpe.
Returns and predictive spreads are decimal units; turnover is never consumed.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/analyze_results.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import repair_dataset as data
import run_study as study
from summarize_us_pilots import gaussian_nll


def sharpe(returns: pd.Series) -> float | None:
    """Return annualized arithmetic excess-return Sharpe if a full path is defined."""
    if len(returns) != 24 or not np.isfinite(returns).all():
        return None
    deviation = float(returns.std(ddof=1))
    return float(returns.mean() / deviation * math.sqrt(12)) if deviation > 0 else None


def family_diagnostics(run_dir: Path, identifier: str, seed: int) -> list[dict[str, Any]]:
    """Calculate family forecast quality and explicitly labelled allocation attribution."""
    predictions = pd.read_csv(run_dir / "gp_window_predictions.csv")
    predictions["gaussian_nll"] = gaussian_nll(predictions)
    weights = pd.read_csv(run_dir / "portfolio_weights.csv")
    weights = weights[weights.strategy.eq("gp_scenarios_riskfolio")]
    merged = predictions.merge(weights[["date", "asset_id", "weight"]], on=["date", "asset_id"],
                               how="left", validate="one_to_one")
    merged["component_return"] = merged.weight * merged.y_true
    result = []
    for family, tickers in data.FAMILIES.items():
        frame = merged[merged.asset_id.isin(tickers)]
        if frame.empty:
            continue
        ic = frame.groupby("date").apply(lambda x: x.y_pred.corr(x.y_true, method="spearman"), include_groups=False)
        errors = (frame.y_true - frame.y_pred).abs()
        attribution = frame.groupby("date").agg(capital_weight=("weight", "sum"),
                                                 contribution=("component_return", "sum"))
        active = attribution.capital_weight.gt(1e-10)
        normalized = attribution.loc[active, "contribution"] / attribution.loc[active, "capital_weight"]
        row = {"configuration": identifier, "seed": seed, "family": family,
               "within_family_ic": float(ic.mean()), "ic_defined_months": int(ic.notna().sum()),
               "mean_gaussian_nll": float(frame.gaussian_nll.mean()),
               "rmse_diagnostic_only": float(np.sqrt(np.square(frame.y_true - frame.y_pred).mean())),
               "mean_predictive_std": float(frame.y_std.mean()), "mean_family_capital_weight": float(
                   attribution.capital_weight.mean()), "active_allocation_months": int(active.sum()),
               "allocated_sleeve_sharpe_diagnostic_only": sharpe(normalized),
               "mean_contribution_to_global_return": float(attribution.contribution.mean())}
        for level, z in [(50, 0.6744897501960817), (80, 1.2815515655446004), (95, 1.959963984540054)]:
            row[f"coverage_{level}"] = float((errors <= z * frame.y_std).mean())
        result.append(row)
    return result


def paired_evidence(items: dict[str, dict[str, Any]], runs: pd.DataFrame) -> pd.DataFrame:
    """Compute exact one-feature conditional effects; never infer importance from frequency."""
    rows = []
    for without_id, without in items.items():
        without_set = set(without["expected_input_columns"])
        for with_id, with_item in items.items():
            extra = set(with_item["expected_input_columns"]) - without_set
            if len(extra) != 1 or not without_set < set(with_item["expected_input_columns"]):
                continue
            left = runs[runs.configuration.eq(with_id)].set_index("seed")
            right = runs[runs.configuration.eq(without_id)].set_index("seed")
            difference = left[["mean_ic", "mean_gaussian_nll", "sharpe"]] - right[
                ["mean_ic", "mean_gaussian_nll", "sharpe"]]
            means = difference.mean()
            favorable = np.array([means.mean_ic, -means.mean_gaussian_nll, means.sharpe])
            if (favorable >= -1e-10).all() and (favorable > 1e-10).any():
                finding = "helpful_in_this_context"
            elif (favorable <= 1e-10).all() and (favorable < -1e-10).any():
                finding = "harmful_in_this_context"
            elif np.abs(favorable).max() <= 1e-10:
                finding = "no_detectable_effect_in_this_context"
            else:
                finding = "objective_tradeoff_or_interaction_dependent"
            rows.append({"feature": next(iter(extra)), "with_configuration": with_id,
                         "without_configuration": without_id, "conditional_finding": finding,
                         "delta_ic_with_minus_without": float(means.mean_ic),
                         "delta_nll_with_minus_without": float(means.mean_gaussian_nll),
                         "delta_sharpe_with_minus_without": float(means.sharpe),
                         "seed_ic_wins": int(difference.mean_ic.gt(0).sum()),
                         "seed_nll_wins": int(difference.mean_gaussian_nll.lt(0).sum()),
                         "seed_sharpe_wins": int(difference.sharpe.gt(0).sum())})
    return pd.DataFrame(rows)


def collect() -> None:
    """Write cumulative complete evidence while retaining explicit pending/failed rows."""
    groups: dict[str, dict[str, Any]] = {}
    incomplete = []
    hashes_at_start = study.code_hashes()
    for path in sorted((data.HERE / "plans").glob("*.json")):
        config = json.loads(path.read_text())
        family = config["family"]
        group = groups.setdefault(family, {"config": config, "items": {}, "runs": [], "diagnostics": []})
        for item in config["configurations"]:
            rows, diagnostic_rows = [], []
            for seed in data.STUDY["seeds"]:
                run_dir = data.HERE / "runs" / f"{item['id']}_s{seed}"
                try:
                    stamp = json.loads((run_dir / "repaired_execution_fingerprint.json").read_text())
                    expected = study.signature(config, item, seed, hashes_at_start)
                    if stamp.get("state") != "completed" or stamp.get("signature") != expected:
                        raise ValueError("Fingerprint is incomplete or no longer matches execution sources")
                    row, failures = study.previous.summarize_run(item["label"], item, run_dir,
                        study.previous.read_dates(data.ARTIFACTS / "scored_dates_24.txt"), config["tickers"],
                        expected_seed=seed)
                    if failures or not np.isfinite([row["mean_ic"], row["mean_gaussian_nll"], row["sharpe"]]).all():
                        raise ValueError(f"Invalid objective evidence: {failures}")
                    study.verify_training(run_dir, config["tickers"])
                    rows.append(row)
                    diagnostic_rows.extend(family_diagnostics(run_dir, item["id"], seed))
                except (FileNotFoundError, ValueError, KeyError) as error:
                    incomplete.append({"family": family, "configuration": item["id"], "seed": seed,
                                       "reason": str(error)})
            if len(rows) == len(data.STUDY["seeds"]):
                group["items"][item["id"]] = item
                group["runs"].extend(rows)
                group["diagnostics"].extend(diagnostic_rows)
    root = data.HERE / "results/cumulative"
    root.mkdir(parents=True, exist_ok=True)
    data.write_json(root / "incomplete_runs.json", incomplete)
    summary = {}
    for family, group in groups.items():
        if not group["runs"]:
            summary[family] = {"complete_configurations": 0, "state": "awaiting_three_seed_evidence"}
            continue
        out = root / family
        out.mkdir(exist_ok=True)
        runs = pd.DataFrame(group["runs"])
        aggregate = study.aggregate_configurations(runs)
        aggregate["input_count_excluding_time"] = aggregate.configuration.map(
            {identifier: len(item["expected_input_columns"]) - 1 for identifier, item in group["items"].items()})
        runs.to_csv(out / "objectives_by_seed.csv", index=False)
        aggregate.to_csv(out / "configuration_aggregates.csv", index=False)
        pareto = aggregate[aggregate.pareto_nondominated]
        pareto.to_csv(out / "pareto.csv", index=False)
        pd.DataFrame(group["diagnostics"]).to_csv(out / "family_diagnostics_by_seed.csv", index=False)
        paired = paired_evidence(group["items"], runs)
        paired.to_csv(out / "conditional_feature_pairs.csv", index=False)
        data.write_json(out / "exact_inputs.json", {identifier: item["expected_input_columns"]
                                                   for identifier, item in group["items"].items()})
        hypervolume = study.calculate_hypervolume(aggregate, group["config"])
        data.write_json(out / "hypervolume.json", hypervolume)
        lines = [f"# {family.replace('_', ' ').title()} cumulative evidence", "",
                 "Only complete, fingerprint-validated three-seed configurations appear below.", "",
                 "## Pareto configurations", "", pareto.to_markdown(index=False), "",
                 f"Fixed-reference hypervolume: {hypervolume['value']:.9f}.", "",
                 f"Conditional one-feature paired contexts: {len(paired)}.", "",
                 "Exact inputs, seed objectives, family uncertainty, and allocation attribution are adjacent files.",
                 "No untouched holdout is claimed; the dates are reused for adaptive research.", ""]
        (out / "RESULTS.md").write_text("\n".join(lines))
        summary[family] = {"complete_configurations": len(aggregate), "pareto_configurations": len(pareto),
                           "hypervolume": hypervolume["value"], "conditional_pairs": len(paired)}
    if hashes_at_start != study.code_hashes():
        raise RuntimeError("Execution-critical sources changed while evidence was being aggregated")
    data.write_json(root / "status.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    collect()
