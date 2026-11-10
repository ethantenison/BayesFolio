"""Summarize conditional component evidence without selecting inputs by frequency.

This research-only reporting layer consumes validated cumulative three-seed
tables. It preserves every matched context, conflicting effect, and untested
feature/family combination. Forecast spreads and returns are decimal units.
It does not fit models, rank hypotheses, or treat width/count as objectives.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/synthesize_components.py
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import repair_dataset as data
from analyze_results import gaussian_nll


def context_summary(rows: pd.DataFrame) -> dict[str, Any]:
    """Describe matched feature contexts without voting to decide importance.

    Args:
        rows: Exact one-feature comparisons, with effects defined as with-minus-without.

    Returns:
        Descriptive context counts, conflicts, and optimization-seed stability.
    """
    if rows.empty:
        return {"evidence_state": "no_matched_test", "matched_contexts": 0,
                "helpful_contexts": 0, "harmful_contexts": 0, "tradeoff_contexts": 0,
                "all_seed_all_objective_helpful_contexts": 0,
                "all_seed_all_objective_harmful_contexts": 0}
    helpful = rows.conditional_finding.eq("helpful_in_this_context")
    harmful = rows.conditional_finding.eq("harmful_in_this_context")
    tradeoff = rows.conditional_finding.eq("objective_tradeoff_or_interaction_dependent")
    if helpful.any() and harmful.any():
        state = "conflicting_contexts"
    elif tradeoff.any():
        state = "tradeoffs_or_context_dependence"
    elif helpful.any():
        state = "helpful_in_tested_contexts_only"
    elif harmful.any():
        state = "harmful_in_tested_contexts_only"
    else:
        state = "no_detectable_effect_in_tested_contexts"
    wins = rows[["seed_ic_wins", "seed_nll_wins", "seed_sharpe_wins"]]
    return {"evidence_state": state, "matched_contexts": len(rows),
            "helpful_contexts": int(helpful.sum()), "harmful_contexts": int(harmful.sum()),
            "tradeoff_contexts": int(tradeoff.sum()),
            "all_seed_all_objective_helpful_contexts": int((helpful & wins.eq(3).all(axis=1)).sum()),
            "all_seed_all_objective_harmful_contexts": int((harmful & wins.eq(0).all(axis=1)).sum())}


def window_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return monthly pooled/family calibration and sharpness, never a selection score.

    Args:
        predictions: Forecast mean, standard deviation, and true decimal returns.

    Returns:
        Per-date pooled and within-family forecast diagnostics.
    """
    pieces = []
    for family, tickers in {"pooled": list(data.TICKERS), **data.FAMILIES}.items():
        subset = predictions[predictions.asset_id.isin(tickers)].copy()
        if subset.empty:
            continue
        subset["gaussian_nll"] = gaussian_nll(subset)
        subset["squared_error"] = np.square(subset.y_true - subset.y_pred)
        subset["error"] = (subset.y_true - subset.y_pred).abs()
        for date, frame in subset.groupby("date", sort=True):
            row = {"family": family, "date": date, "tasks": len(frame),
                   "ic": float(frame.y_pred.corr(frame.y_true, method="spearman")),
                   "mean_gaussian_nll": float(frame.gaussian_nll.mean()),
                   "rmse_diagnostic_only": float(np.sqrt(frame.squared_error.mean())),
                   "mean_predictive_std": float(frame.y_std.mean()),
                   "mean_interval_width_95": float(2 * 1.959963984540054 * frame.y_std.mean())}
            for level, z in [(50, 0.6744897501960817), (80, 1.2815515655446004), (95, 1.959963984540054)]:
                row[f"coverage_{level}"] = float((frame.error <= z * frame.y_std).mean())
            pieces.append(row)
    return pd.DataFrame(pieces)


def synthesize() -> None:
    """Write auditable component tables and frontier monthly uncertainty summaries.

    Raises:
        ValueError: A registered run is incomplete or cumulative evidence is stale.
    """
    root = data.HERE / "results/cumulative"
    if json.loads((root / "incomplete_runs.json").read_text()):
        raise ValueError("Refresh cumulative evidence after all registered runs complete")
    expected: dict[str, set[str]] = {}
    for plan in (data.HERE / "plans").glob("*.json"):
        config = json.loads(plan.read_text())
        expected.setdefault(config["family"], set()).update(item["id"] for item in config["configurations"])
    source_files = [path for path in root.rglob("*") if path.is_file()]
    source_files.extend(data.HERE / name for name in [
        "synthesize_components.py", "analyze_results.py", "compare_round.py"])
    source_hashes = {str(path.relative_to(data.ROOT)): data.digest(path) for path in source_files}
    pairs, fronts, windows, matrix = [], [], [], []
    features: set[str] = set()
    inputs = {}
    for family in [*data.FAMILIES, "global"]:
        directory = root / family
        inputs[family] = json.loads((directory / "exact_inputs.json").read_text())
        if set(inputs[family]) != expected[family]:
            raise ValueError(f"Stale cumulative evidence for {family}")
        features.update(column for cols in inputs[family].values() for column in cols if column != "t_index")
        pair_path = directory / "conditional_feature_pairs.csv"
        frame = pd.read_csv(pair_path) if pair_path.stat().st_size > 1 else pd.DataFrame()
        if not frame.empty:
            frame.insert(0, "family", family)
            pairs.append(frame)
        front = pd.read_csv(directory / "pareto.csv")
        front.insert(0, "family", family)
        fronts.append(front)
        for identifier in front.configuration:
            for seed in data.STUDY["seeds"]:
                run_path = data.HERE / "runs" / f"{identifier}_s{seed}" / "gp_window_predictions.csv"
                source_hashes[str(run_path.relative_to(data.ROOT))] = data.digest(run_path)
                monthly = window_diagnostics(pd.read_csv(run_path))
                monthly.insert(0, "seed", seed)
                monthly.insert(0, "configuration", identifier)
                monthly.insert(0, "model_universe", family)
                windows.append(monthly)
    all_pairs = pd.concat(pairs, ignore_index=True)
    for family in [*data.FAMILIES, "global"]:
        for feature in sorted(features):
            matched = all_pairs[all_pairs.family.eq(family) & all_pairs.feature.eq(feature)]
            matrix.append({"family": family, "feature": feature, **context_summary(matched)})
    effects = []
    for path in sorted((data.HERE / "results/rounds").glob("*_round*.json")):
        report = json.loads(path.read_text())
        source_hashes[str(path.relative_to(data.ROOT))] = data.digest(path)
        effects.extend({"family": report["family"], "round_index": report["round_index"], **effect}
                       for effect in report["hypothesis_comparisons"])
    out = data.HERE / "results/component_synthesis"
    out.mkdir(exist_ok=True)
    pd.DataFrame(matrix).to_csv(out / "feature_family_matrix.csv", index=False)
    all_pairs.to_csv(out / "conditional_feature_evidence.csv", index=False)
    pd.concat(fronts, ignore_index=True).to_csv(out / "frontier_by_family.csv", index=False)
    pd.DataFrame(effects).to_csv(out / "round_hypothesis_effects.csv", index=False)
    pd.concat(windows, ignore_index=True).to_csv(out / "frontier_monthly_diagnostics.csv", index=False)
    data.write_json(out / "component_exact_inputs.json", inputs)
    for relative, digest in source_hashes.items():
        if data.digest(data.ROOT / relative) != digest:
            raise ValueError(f"Evidence changed during synthesis: {relative}")
    data.write_json(out / "manifest.json", {
        "source_sha256": source_hashes, "registered_configurations": {k: len(v) for k, v in expected.items()},
        "feature_family_rows": len(matrix), "matched_contexts": len(all_pairs),
        "importance_not_inferred_from_frequency": True,
        "boundary": "Matched contexts establish conditional evidence, not universal feature importance. "
                    "Optimization-seed repeats are not independent economic samples; reused score dates are adaptive."})
    print(json.dumps({"output": str(out), "matched_contexts": len(all_pairs), "feature_family_rows": len(matrix)}))


if __name__ == "__main__":
    synthesize()
