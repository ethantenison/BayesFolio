"""Single-command guard and dispatcher for the repaired-data research workflow.

This experiment entrypoint validates immutable datasets, fixed model controls,
family/global round budgets, feature masks, and reasoning artifacts before
delegating to the existing deterministic runner infrastructure. Hosted Codex
writes evidence-based next-round plans; no local LLM or API key is needed.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/workflow.py --check
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/workflow.py --execute --workers 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import repair_dataset as data
import run_study as study
from candidate_masks import research_candidate


def approved_dataset_paths() -> list[Path]:
    """List frozen study versions, without admitting arbitrary replacement feature artifacts."""
    paths = [data.ARTIFACTS]
    extension = data.HERE / "artifacts/v2"
    if (extension / "dataset.manifest.json").is_file():
        paths.append(extension)
    return paths


def audited_baseline_ids(family: str) -> set[str]:
    """Limit baseline exemptions to audited historical masks and the actual all-input control."""
    result = set()
    inventory = json.loads((data.HERE / "historical_inventory.json").read_text())
    result.update(item["id"] for item in inventory["configurations"].get(family, []))
    for name in ["historical_scope_audit.json", "historical_pool_scope_audit.json", "historical_macro_only_audit.json"]:
        path = data.HERE / "results" / name
        if path.is_file():
            records = json.loads(path.read_text())["records"]
            result.update(record["repaired_mask_id"] for record in records
                          if record.get("family") == family and record.get("repaired_mask_id"))
    if family == "global":
        result.add(study.candidate("global", study.BASE_ETF, study.BASE_MACRO, ["baseline"])["id"])
    return result


def plan_stage(config: dict[str, Any], path: Path) -> str:
    """Identify historical, initial-global, and bounded adaptive planning stages."""
    if config.get("study_stage") == "global_initial":
        return "global_initial"
    if path.name.startswith("00_"):
        return "historical_rebaseline"
    return "global_refinement" if config["family"] == "global" else "family_refinement"


def validate_plan(config: dict[str, Any], path: Path, available: set[str]) -> None:
    """Reject altered model controls, unsupported inputs, or unbounded adaptive plans."""
    family = config["family"]
    if family not in {*data.FAMILIES, "global"}:
        raise ValueError("Unsupported task family")
    items = config["configurations"]
    if not items:
        raise ValueError("Empty experiment plan")
    feature = study.previous.resolve_repo_path(config["feature_path"]).resolve()
    allowed = {directory / f"{family}.parquet" for directory in approved_dataset_paths()}
    if feature not in allowed:
        raise ValueError("Feature artifact is not an approved immutable study version")
    feature_meta = json.loads(feature.with_suffix(".manifest.json").read_text())
    actual_columns = set(feature_meta["required_columns"])
    if not actual_columns.issubset(available):
        raise ValueError("Dataset feature pool was not validated")
    extra_etf = []
    if feature.parent.name == "v2":
        extension_meta = json.loads((feature.parent / "dataset.manifest.json").read_text())
        extra_etf = extension_meta["extra_etf_columns"]
    allowed_etf = {*data.ETF_COLUMNS, "lag_y_excess_lead", "lag2_y_excess_lead", *extra_etf}
    if config.get("default_seeds") != data.STUDY["seeds"]:
        raise ValueError("Every batch must specify all three seeds")
    if config["common_args"] != study.phase_config(family, items, path.stem)["common_args"]:
        raise ValueError("Fitting/portfolio controls cannot vary with the feature mask")
    stage = plan_stage(config, path)
    if stage != "historical_rebaseline":
        round_index = config.get("round_index")
        max_rounds = data.STUDY["global_rounds" if family == "global" else "family_rounds"]
        cap = data.STUDY["max_new_global_configurations_per_round" if family == "global"
                         else "max_new_family_configurations_per_round"]
        valid_round = isinstance(round_index, int) and 1 <= round_index <= max_rounds
        if stage == "global_initial":
            valid_round = family == "global" and round_index == 0
        if not valid_round or len(items) > cap:
            raise ValueError("Adaptive plan violates round/configuration caps")
        if not config.get("evidence_reports") or not config.get("round_interpretation"):
            raise ValueError("Adaptive planning must reference interpreted deterministic evidence")
        for report in config["evidence_reports"]:
            evidence = (data.ROOT / report).resolve()
            if not evidence.is_relative_to(data.HERE) or not evidence.is_file():
                raise ValueError(f"Missing planning evidence: {report}")
            if "evidence_sha256" in config and config["evidence_sha256"].get(report) != data.digest(evidence):
                raise ValueError("Frozen planning evidence changed after registration")
        if "proposal_sha256" in config:
            proposal = (data.ROOT / config["proposal_path"]).resolve()
            if not proposal.is_relative_to(data.HERE) or data.digest(proposal) != config["proposal_sha256"]:
                raise ValueError("Registered reasoning proposal changed")
        if any(len(str(item.get("hypothesis", ""))) < 20 for item in items):
            raise ValueError("Every new candidate needs a falsifiable research hypothesis")
    for item in items:
        expected = research_candidate(family, item["etf_columns"], item["macro_columns"], item["source_evidence"])
        expected["expected_feature_sha256"] = feature_meta["artifact_sha256"]
        for key in ["id", "label", "args", "expected_input_columns", "expected_feature_sha256",
                    "expected_training_start", "kernel_policy"]:
            if item[key] != expected[key]:
                raise ValueError(f"Candidate changed a fixed/default-kernel constraint: {key}")
        if set(item["expected_input_columns"]) - actual_columns:
            raise ValueError("Candidate references an unvalidated input")
        if set(item["etf_columns"]) - allowed_etf:
            raise ValueError("ETF/macro block membership changed")
        if set(item["etf_columns"]) & set(item["macro_columns"]):
            raise ValueError("Feature occurs in multiple blocks")
        if set(item["macro_columns"]) & (allowed_etf | {"t_index", "y_excess_lead"}):
            raise ValueError("Identifier/target/local lag cannot be injected as a macro predictor")
    if stage == "historical_rebaseline":
        unknown = {item["id"] for item in items} - audited_baseline_ids(family)
        if unknown:
            raise ValueError("An unaudited new mask cannot bypass adaptive limits by using a baseline filename")
    study.previous.validate_config(config, path)


def validate_frozen_data() -> set[str]:
    """Check every frozen artifact and exact full-panel coverage before scheduling fits."""
    manifest = json.loads((data.ARTIFACTS / "dataset.manifest.json").read_text())
    for output in manifest["outputs"]:
        if data.digest(data.ROOT / output["path"]) != output["sha256"]:
            raise ValueError(f"Frozen dataset output changed: {output['path']}")
    if data.digest(data.HERE / "study.json") != manifest["study_sha256"]:
        raise ValueError("Study controls changed after dataset freeze")
    if data.digest(data.HERE / "repair_dataset.py") != manifest["builder_sha256"]:
        raise ValueError("Dataset builder changed after dataset freeze")
    global_frame = pd.read_parquet(data.ARTIFACTS / "global.parquet")
    data.validate_panel(global_frame)
    for family, tickers in data.FAMILIES.items():
        subset = pd.read_parquet(data.ARTIFACTS / f"{family}.parquet")
        expected = global_frame[global_frame.asset_id.isin(tickers)].reset_index(drop=True)
        pd.testing.assert_frame_equal(subset, expected)
    available = set(global_frame.columns)
    for extension in approved_dataset_paths()[1:]:
        audit = json.loads((extension / "dataset.manifest.json").read_text())
        if audit["base_global_sha256"] != data.digest(data.ARTIFACTS / "global.parquet"):
            raise ValueError("Feature-pool extension references a different base dataset")
        for output in audit["outputs"]:
            if data.digest(data.ROOT / output["path"]) != output["sha256"]:
                raise ValueError("Frozen extended feature artifact changed")
        expanded = pd.read_parquet(extension / "global.parquet")
        data.validate_panel(expanded)
        pd.testing.assert_frame_equal(expanded[global_frame.columns], global_frame, check_exact=True)
        for family, tickers in data.FAMILIES.items():
            subset = pd.read_parquet(extension / f"{family}.parquet")
            pd.testing.assert_frame_equal(subset, expanded[expanded.asset_id.isin(tickers)].reset_index(drop=True))
        available.update(expanded.columns)
    return available


def completed_batch_is_valid(path: Path, config: dict[str, Any]) -> bool:
    """Revalidate complete fingerprinted batches without repeating their preflight processes."""
    state_path = data.HERE / "executions" / path.stem / "status.json"
    if not state_path.is_file() or json.loads(state_path.read_text()).get("state") != "completed":
        return False
    hashes = study.code_hashes()
    for item in config["configurations"]:
        for seed in data.STUDY["seeds"]:
            run_dir = data.HERE / "runs" / f"{item['id']}_s{seed}"
            stamp = run_dir / "repaired_execution_fingerprint.json"
            if not stamp.is_file():
                return False
            fingerprint = json.loads(stamp.read_text())
            if fingerprint.get("state") != "completed" or fingerprint.get("signature") != study.signature(
                config, item, seed, hashes
            ):
                return False
            paths = study.previous.validate_config(config, path)
            if not study.previous.existing_run_is_valid(config, paths, item, seed):
                return False
            study.verify_training(run_dir, config["tickers"])
    return True


def main() -> None:
    """Validate all plans and run complete batches under one global worker limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--family", choices=[*data.FAMILIES, "global"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    study.previous.validate_worker_count(args.workers)
    available = validate_frozen_data()
    paths = sorted((data.HERE / "plans").glob("*.json"))
    registered: dict[str, set[str]] = {}
    rounds: set[tuple[str, int]] = set()
    for path in paths:
        config = json.loads(path.read_text())
        validate_plan(config, path, available)
        family = config["family"]
        seen = registered.setdefault(family, set())
        ids = {item["id"] for item in config["configurations"]}
        if seen & ids:
            raise ValueError(f"Duplicated configuration across study batches: {sorted(seen & ids)}")
        seen.update(ids)
        if plan_stage(config, path) == "global_initial" and config.get("round_index") != 0:
            raise ValueError("Initial global comparison is round zero, not an extra adaptive round")
        if "round_index" in config:
            key = (family, config["round_index"])
            if key in rounds:
                raise ValueError("Round budget cannot be expanded by splitting a round into multiple files")
            rounds.add(key)
    print(json.dumps({"validated_plans": len(paths), "candidate_counts": {
        family: len(ids) for family, ids in registered.items()}, "workers_total_max": args.workers}), flush=True)
    if args.execute:
        for path in paths:
            config = json.loads(path.read_text())
            if args.family and config["family"] != args.family:
                continue
            if not args.smoke and completed_batch_is_valid(path, config):
                study.summarize(config, config["configurations"], data.STUDY["seeds"],
                                data.HERE / "results" / path.stem)
                print(f"Verified complete batch without refitting: {path.stem}", flush=True)
                continue
            study.execute_batch(path, 1 if args.smoke else args.workers, args.smoke)
            if args.smoke:
                break


if __name__ == "__main__":
    main()
