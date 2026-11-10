"""Measure one preregistered round against its frozen pre-round evidence.

This deterministic research report never chooses or launches experiments. It
checks complete three-seed outputs, computes cumulative Pareto/hypervolume
changes and exact paired parent effects, and records the two-stalled-round rule.
Later rounds cannot retroactively change the frozen before/after comparison.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/compare_round.py \
      --plan experiments/2026-10-repaired-global-feature-selection/plans/10_international_equity_round1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import repair_dataset as data
import run_study as study
import workflow

OBJECTIVES = ["mean_ic", "mean_gaussian_nll", "sharpe"]


def frontier_progress(before: pd.DataFrame, after: pd.DataFrame, new_ids: set[str]) -> dict[str, Any]:
    """Distinguish new objective regions from equal-objective input simplifications."""
    old_front = before[before.pareto_nondominated]
    new_front = after[after.pareto_nondominated & after.configuration.isin(new_ids)]
    old_points = old_front[OBJECTIVES].to_numpy(dtype=float)
    novel, simpler = [], []
    for row in new_front.itertuples(index=False):
        point = np.array([getattr(row, name) for name in OBJECTIVES], dtype=float)
        equal = np.isclose(old_points, point, rtol=0, atol=1e-12).all(axis=1)
        if not equal.any():
            novel.append(row.configuration)
        elif int(row.input_count_excluding_time) < int(old_front.loc[equal, "input_count_excluding_time"].min()):
            simpler.append(row.configuration)
    return {"new_frontier_points": novel, "equal_objective_simplifications": simpler,
            "frontier_improved": bool(novel), "competitive_simplification": bool(simpler),
            "stalled": not (novel or simpler)}


def compare(plan: Path) -> dict[str, Any]:
    """Validate a full round and write objective deltas without using later results."""
    config = json.loads(plan.read_text())
    workflow.validate_plan(config, plan, workflow.validate_frozen_data())
    bundle = {Path(path).name: data.ROOT / path for path in config["evidence_reports"]}
    before_raw = pd.read_csv(bundle["objectives_by_seed.csv"])
    before_inputs = json.loads(bundle["exact_inputs.json"].read_text())
    if before_raw.duplicated(["configuration", "seed"]).any():
        raise ValueError("Frozen before-round evidence contains duplicate seeds")
    if not before_raw.groupby("configuration").seed.apply(lambda x: set(x) == set(data.STUDY["seeds"])).all():
        raise ValueError("Frozen before-round evidence omits a required seed")
    before = study.aggregate_configurations(before_raw)
    before["input_count_excluding_time"] = before.configuration.map({key: len(cols) - 1
                                                                    for key, cols in before_inputs.items()})
    new_rows = []
    hashes = study.code_hashes()
    for item in config["configurations"]:
        for seed in data.STUDY["seeds"]:
            directory = data.HERE / "runs" / f"{item['id']}_s{seed}"
            stamp = json.loads((directory / "repaired_execution_fingerprint.json").read_text())
            if stamp.get("state") != "completed" or stamp["signature"] != study.signature(config, item, seed, hashes):
                raise ValueError("Round is not fully complete under matching execution fingerprints")
            row, failures = study.previous.summarize_run(
                item["label"], item, directory, study.previous.read_dates(data.ARTIFACTS / "scored_dates_24.txt"),
                config["tickers"], expected_seed=seed)
            if failures or not np.isfinite([row[name] for name in OBJECTIVES]).all():
                raise ValueError(f"Invalid round outputs: {failures}")
            study.verify_training(directory, config["tickers"])
            new_rows.append(row)
    new_raw = pd.DataFrame(new_rows)
    after = study.aggregate_configurations(pd.concat([before_raw, new_raw], ignore_index=True))
    sizes = {key: len(cols) - 1 for key, cols in before_inputs.items()}
    sizes.update({item["id"]: len(item["expected_input_columns"]) - 1 for item in config["configurations"]})
    after["input_count_excluding_time"] = after.configuration.map(sizes)
    progress = frontier_progress(before, after, {item["id"] for item in config["configurations"]})
    before_hv = study.calculate_hypervolume(before, config)["value"]
    after_hv = study.calculate_hypervolume(after, config)["value"]
    changes = []
    for item in config["configurations"]:
        parent = item["base_configuration"]
        new_seed = new_raw[new_raw.configuration.eq(item["id"])].set_index("seed")
        old_seed = before_raw[before_raw.configuration.eq(parent)].set_index("seed")
        delta = new_seed[OBJECTIVES] - old_seed[OBJECTIVES]
        changes.append({"configuration": item["id"], "hypothesis_name": item["hypothesis_name"],
                        "base_configuration": parent, "category": item["category"],
                        "delta_ic": float(delta.mean_ic.mean()), "delta_nll": float(delta.mean_gaussian_nll.mean()),
                        "delta_sharpe": float(delta.sharpe.mean()),
                        "input_count_change": sizes[item["id"]] - sizes[parent],
                        "seed_ic_wins": int(delta.mean_ic.gt(0).sum()),
                        "seed_nll_wins": int(delta.mean_gaussian_nll.lt(0).sum()),
                        "seed_sharpe_wins": int(delta.sharpe.gt(0).sum()),
                        "on_cumulative_front": bool(after.set_index("configuration").loc[
                            item["id"], "pareto_nondominated"])})
    family, index = config["family"], config["round_index"]
    previous_stalled = False
    previous_path = data.HERE / "results/rounds" / f"{family}_round{index - 1}.json"
    if index > 1 and previous_path.is_file():
        previous_stalled = bool(json.loads(previous_path.read_text())["stalled"])
    result = {"family": family, "round_index": index, "study_stage": config["study_stage"],
              "before_configurations": len(before), "new_configurations": len(config["configurations"]),
              "after_configurations": len(after), "seeds": data.STUDY["seeds"],
              "hypervolume_before": before_hv, "hypervolume_after": after_hv,
              "hypervolume_delta": after_hv - before_hv, **progress,
              "two_consecutive_stalled_rounds": bool(progress["stalled"] and previous_stalled),
              "plan_sha256": data.digest(plan), "hypothesis_comparisons": changes,
              "interpretation_boundary": "Seed repeats measure optimization variability on the same reused dates, "
              "not independent economic samples or an untouched holdout."}
    out = data.HERE / "results/rounds"
    out.mkdir(parents=True, exist_ok=True)
    data.write_json(out / f"{family}_round{index}.json", result)
    after.to_csv(out / f"{family}_round{index}_cumulative.csv", index=False)
    table = pd.DataFrame(changes)
    lines = [f"# {family.replace('_', ' ').title()} round {index}", "",
             f"Cumulative configurations: {len(before)} -> {len(after)}.", "",
             f"Fixed-reference hypervolume: {before_hv:.9f} -> {after_hv:.9f}.", "",
             f"Stalled: {progress['stalled']}; two consecutive stalled rounds: "
             f"{result['two_consecutive_stalled_rounds']}.", "", table.to_markdown(index=False), "",
             "Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.",
             "Count, return, RMSE, and interval width are not hidden selection objectives.", "",
             result["interpretation_boundary"], ""]
    (out / f"{family}_round{index}.md").write_text("\n".join(lines))
    return result


def main() -> None:
    """Compare a completed preregistered round against its immutable before snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.plan.resolve())
    print(json.dumps({key: value for key, value in result.items() if key != "hypothesis_comparisons"}))


if __name__ == "__main__":
    main()
