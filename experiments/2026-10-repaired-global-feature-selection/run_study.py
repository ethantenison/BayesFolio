"""Inventory and execute immutable repaired-data feature-selection batches.

Research orchestration reuses the existing walk-forward command builder,
preflight contracts, run validator, objectives, Pareto, and hypervolume code.
Resolved masks use one fixed default kernel-construction rule, not historical
kernel tuning. Outputs and provenance remain under this new study directory.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/run_study.py --prepare
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/run_study.py --execute --smoke
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/run_study.py --execute --workers 15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import repair_dataset as data

ROOT = data.ROOT
HERE = data.HERE
OLD = ROOT / "experiments/2026-10-etf-family-feature-confirmation"
if str(OLD) not in sys.path:
    sys.path.insert(0, str(OLD))
import orchestrate_international_experiments as previous  # noqa: E402
from summarize_us_confirmation import aggregate_configurations, calculate_hypervolume  # noqa: E402

previous.EXPECTED_TICKERS_BY_FAMILY.update(data.FAMILIES)
previous.EXPECTED_TICKERS_BY_FAMILY["global"] = data.TICKERS
RUNNER = ROOT / "experiments/2026-07-international-equity-etf-features/run_first_round_walkforward.py"
TEMPLATE = json.loads((OLD / "international_confirmation_config.json").read_text())
BASE_ETF = ["lag_y_excess_lead", "baspread", "ret_kurt", "chmom", "mom12m", "mom36m", "cs_mom_rank",
            "max_dd_6m", "ma_signal", "ret_autocorr", "vol_z"]
BASE_MACRO = ["hy_spread", "hy_spread_chg_1p", "hy_spread_z_12p", "vix_slope", "vix_ts_z_12p", "vix",
              "spy_flow_z_12p", "spy_ret", "erp", "cpi_chg_12p", "cpi_chg_1p", "copper_ret", "oil_ret",
              "gold_crude_ratio", "pct_above_50dma", "em_fx_ret"]
KERNEL_POLICY = {"grouping": "default", "macro_components": "default", "main_effects": True,
                 "pairwise_interactions": True, "higher_order_interactions": False, "time_modulation": "neither"}


def canonical(columns: list[str]) -> list[str]:
    """Normalize verified monthly naming aliases and order a unique feature mask."""
    return sorted({data.ALIASES.get(column, column) for column in columns})


def candidate(family: str, etf: list[str], macro: list[str], evidence: list[str]) -> dict[str, Any]:
    """Construct a deterministic mask identifier and exact default-kernel arguments."""
    etf, macro = canonical(etf), canonical(macro)
    if not etf:
        raise ValueError("A candidate must retain an ETF-local block")
    key = hashlib.sha256(json.dumps([family, etf, macro, KERNEL_POLICY], sort_keys=True).encode()).hexdigest()[:12]
    identifier = f"{family}_{key}"
    return {"id": identifier, "label": identifier, "expected_input_columns": ["t_index", *etf, *macro],
            "expected_feature_sha256": data.digest(data.ARTIFACTS / f"{family}.parquet"),
            "expected_training_start": data.STUDY["training_start"],
            "args": ["--feature-set", "momentum_plus_macro_market", "--custom-etf-cols", ",".join(etf),
                     "--custom-macro-cols", ",".join(macro), "--covar-grouping-proposal", "default",
                     "--macro-component-proposal", "default", "--time-modulation-mode", "neither"],
            "source_evidence": evidence, "etf_columns": etf, "macro_columns": macro,
            "kernel_policy": KERNEL_POLICY}


def phase_config(family: str, configurations: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Generate a compatible frozen batch config without mutating prior specifications."""
    result = {key: value for key, value in TEMPLATE.items()
              if key not in {"configurations", "comparison_configurations"}}
    result.update({"schema": "bayesfolio.etf_family_experiment.v1", "phase": "refinement", "family": family,
                   "tickers": data.FAMILIES.get(family, data.TICKERS), "runner": str(RUNNER.relative_to(ROOT)),
                   "feature_path": str((data.ARTIFACTS / f"{family}.parquet").relative_to(ROOT)),
                   "feature_manifest": str((data.ARTIFACTS / f"{family}.manifest.json").relative_to(ROOT)),
                   "scored_date_file": str((data.ARTIFACTS / "scored_dates_24.txt").relative_to(ROOT)),
                   "output_root": str((HERE / "runs").relative_to(ROOT)),
                   "snapshot_root": str((HERE / "executions").relative_to(ROOT)),
                   "results_prefix": name, "configurations": configurations, "comparison_configurations": [],
                   "default_seeds": data.STUDY["seeds"], "kernel_policy": KERNEL_POLICY})
    result["common_args"] = list(TEMPLATE["common_args"]) + ["--scenario-workers", "1", "--torch-num-threads", "1"]
    return result


def prepare() -> None:
    """Deduplicate historical masks, record exclusions, and preregister baseline batches."""
    panel = pd.read_parquet(data.ARTIFACTS / "global.parquet")
    inventory: dict[str, dict[str, dict[str, Any]]] = {family: {} for family in data.FAMILIES}
    exclusions = []
    manifests = sorted((OLD / "runs").glob("*/manifest.json"))
    for path in manifests:
        manifest = json.loads(path.read_text())
        universe = manifest.get("data", {}).get("training_universe", [])
        family = next((key for key, tickers in data.FAMILIES.items() if universe == tickers), None)
        wrapper = manifest.get("us_equity_first_round_wrapper", {})
        reason = None
        if family is None:
            reason = "not an approved exact family universe"
        elif not wrapper.get("time_index_included"):
            reason = "excluded time index"
        elif len(manifest.get("rebalance_dates", [])) != 24:
            reason = "not a complete 24-window October study"
        elif not (path.parent / "gp_window_predictions.csv").exists():
            reason = "missing completed predictions"
        if reason:
            exclusions.append({"manifest": str(path.relative_to(ROOT)), "reason": reason})
            continue
        etf, macro = canonical(wrapper["etf_columns"]), canonical(wrapper["macro_columns"])
        missing = sorted(set(etf + macro) - set(panel.columns))
        if missing:
            exclusions.append({"manifest": str(path.relative_to(ROOT)), "reason": f"unvalidated features: {missing}"})
            continue
        item = candidate(family, etf, macro, [str(path.relative_to(ROOT))])
        if item["id"] in inventory[family]:
            inventory[family][item["id"]]["source_evidence"].extend(item["source_evidence"])
        else:
            inventory[family][item["id"]] = item
    for family in data.FAMILIES:
        for etf, macro, description in [
            (BASE_ETF, BASE_MACRO, "Compute-matched current all-input comparator"),
            (["chmom", "mom12m", "ma_signal", "ret_autocorr"], [], "Simple local-only no-lag comparator"),
            (["lag_y_excess_lead", "chmom", "mom12m", "ma_signal", "ret_autocorr"], [],
             "Exact with-lag counterpart of simple local baseline")]:
            item = candidate(family, etf, macro, [description])
            inventory[family].setdefault(item["id"], item)
        items = list(inventory[family].values())
        config = phase_config(family, items, f"{family}_rebaseline")
        path = HERE / "plans" / f"00_{family}_rebaseline.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError(f"Refusing to change preregistered batch: {path}")
        data.write_json(path, config)
    data.write_json(HERE / "historical_inventory.json", {
        "source_manifest_count": len(manifests), "exclusions": exclusions,
        "deduplicated_counts": {family: len(items) for family, items in inventory.items()},
        "kernel_policy": KERNEL_POLICY,
        "interpretation": "Rebaseline historical variable masks with a fixed kernel rule; not exact old-kernel replay.",
        "planned_seed_runs": sum(len(items) for items in inventory.values()) * len(data.STUDY["seeds"]),
        "configurations": {family: list(items.values()) for family, items in inventory.items()}})
    print(json.dumps({family: len(items) for family, items in inventory.items()}), flush=True)


def code_hashes() -> dict[str, str]:
    """Fingerprint execution-critical sources and the environment lockfile."""
    files = [Path(__file__), HERE / "study.json", RUNNER, ROOT / "uv.lock",
             ROOT / "experiments/2026-06-portfolio-optimization/run_monthly_optimization_walkforward.py",
             OLD / "orchestrate_international_experiments.py", OLD / "summarize_international_experiments.py",
             OLD / "summarize_us_confirmation.py", OLD / "summarize_us_pilots.py"]
    files += list((ROOT / "bayesfolio/engine/forecast/gp").rglob("*.py"))
    files += list((ROOT / "experiments/2026-06-task-covariance").glob("*.py"))
    return {str(path.relative_to(ROOT)): data.digest(path) for path in sorted(set(files))}


def signature(config: dict[str, Any], item: dict[str, Any], seed: int, hashes: dict[str, str]) -> str:
    """Bind resume eligibility to exact code, data, seed, controls, and date bytes."""
    payload = {"code": hashes, "feature_sha256": item["expected_feature_sha256"],
               "score_sha256": data.digest(data.ARTIFACTS / "scored_dates_24.txt"),
               "candidate": item, "seed": seed, "common_args": config["common_args"],
               "tickers": config["tickers"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_command(command: list[str], log: Path) -> int:
    """Execute one existing deterministic runner with isolated logs and thread limits."""
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1", "MPLCONFIGDIR": str(HERE / "artifacts/matplotlib"),
                "MPLBACKEND": "Agg", "XDG_CACHE_HOME": str(HERE / "artifacts/runtime_cache")})
    with log.open("w") as handle:
        if "--preflight-only" in command:
            with log.with_suffix(".stderr.log").open("w") as errors:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=errors, check=False)
        else:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return process.returncode


def verify_training(run_dir: Path, tickers: list[str]) -> None:
    """Verify each expanding window has exactly 84+k fully labelled training dates."""
    frame = pd.read_csv(run_dir / "window_training_history.csv")
    if len(frame) != 24 or frame.unique_train_dates.tolist() != list(range(84, 108)):
        raise ValueError("Actual expanding training counts differ from the 84-month contract")
    if frame.iloc[0].train_rows != 84 * len(tickers):
        raise ValueError("First training task-row count differs from the complete initial panel")


def execute_batch(path: Path, workers: int, smoke: bool) -> None:
    """Preflight, fit/resume, validate, and checkpoint a single immutable batch."""
    config = json.loads(path.read_text())
    paths = previous.validate_config(config, path)
    hashes = code_hashes()
    phase = path.stem
    execution = HERE / "executions" / phase
    execution.mkdir(parents=True, exist_ok=True)
    items = config["configurations"]
    seeds = data.STUDY["seeds"]
    if smoke:
        # A complete 24-window fit, not a misleading one-window success.
        items = [max(items, key=lambda item: len(item["expected_input_columns"]))]
        seeds = [27]
    state_path = execution / ("smoke_status.json" if smoke else "status.json")
    state = {"phase": phase, "state": "preflighting", "runs": {}, "code_hashes": hashes,
             "config_sha256": data.digest(path), "workers": workers, "started_at": datetime.now(UTC).isoformat()}
    data.write_json(state_path, state)
    for item in items:
        command = previous.build_command(config, paths, item, seed=27, preflight=True)
        log = execution / "logs" / f"preflight_{item['id']}.log"
        if run_command(command, log):
            raise RuntimeError(f"Preflight failed: {log}")
        preflight = json.loads(log.read_text())
        if preflight["scored_dates"] != previous.read_dates(paths["dates"]):
            raise ValueError("Preflight scored dates differ")
    pending = []
    for item in items:
        for seed in seeds:
            run_id = f"{item['id']}_s{seed}"
            run_dir = paths["output_root"] / run_id
            stamp = run_dir / "repaired_execution_fingerprint.json"
            expected = signature(config, item, seed, hashes)
            if stamp.exists() and json.loads(stamp.read_text()).get("signature") == expected:
                if previous.existing_run_is_valid(config, paths, item, seed):
                    verify_training(run_dir, config["tickers"])
                    state["runs"][run_id] = {"state": "completed_existing", "signature": expected}
                    continue
            if run_dir.exists() and (not stamp.exists() or json.loads(stamp.read_text()).get("signature") != expected):
                # The delegated runner's resume is not an authority to reuse a different fingerprint.
                raise ValueError(f"Existing run lacks a valid matching fingerprint: {run_dir}")
            command = previous.build_command(config, paths, item, seed=seed, preflight=False)
            data.write_json(stamp, {"signature": expected, "state": "pending_or_running", "code_hashes": hashes})
            pending.append((item, seed, run_id, expected, command))
            state["runs"][run_id] = {"state": "queued", "signature": expected, "command": command}
    state["state"] = "running"
    data.write_json(state_path, state)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for item, seed, run_id, expected, command in pending:
            future = pool.submit(run_command, command, execution / "logs" / f"{run_id}.log")
            futures[future] = (item, seed, run_id, expected)
        for future in as_completed(futures):
            item, seed, run_id, expected = futures[future]
            returncode = future.result()
            entry = state["runs"][run_id]
            run_dir = paths["output_root"] / run_id
            if hashes != code_hashes() or data.digest(path) != state["config_sha256"]:
                raise RuntimeError("Execution-critical code or configuration changed during the batch")
            try:
                if returncode or not previous.existing_run_is_valid(config, paths, item, seed):
                    raise RuntimeError(f"Runner exit {returncode} or output validation failure")
                verify_training(run_dir, config["tickers"])
                data.write_json(run_dir / "repaired_execution_fingerprint.json",
                                {"signature": expected, "state": "completed", "code_hashes": hashes})
                entry.update({"state": "completed", "returncode": returncode})
                print(f"Validated {run_id}", flush=True)
            except (ValueError, RuntimeError, FileNotFoundError) as error:
                entry.update({"state": "failed", "returncode": returncode, "error": str(error)})
                print(f"FAILED {run_id}: {error}", flush=True)
            data.write_json(state_path, state)
    failed = [run_id for run_id, entry in state["runs"].items() if entry["state"] == "failed"]
    state["state"] = "failed" if failed else "completed"
    state["finished_at"] = datetime.now(UTC).isoformat()
    data.write_json(state_path, state)
    if failed:
        raise RuntimeError(f"Batch contains failed runs: {failed}")
    if not smoke:
        summarize(config, items, seeds, HERE / "results" / phase)


def summarize(config: dict[str, Any], items: list[dict[str, Any]], seeds: list[int], out: Path) -> None:
    """Write validated objectives, Pareto, uncertainty diagnostics, and hypervolume."""
    out.mkdir(parents=True, exist_ok=True)
    rows, diagnostic_rows = [], []
    for item in items:
        for seed in seeds:
            run_dir = HERE / "runs" / f"{item['id']}_s{seed}"
            row, failures = previous.summarize_run(item["label"], item, run_dir,
                                                  previous.read_dates(data.ARTIFACTS / "scored_dates_24.txt"),
                                                  config["tickers"], expected_seed=seed)
            if failures:
                raise ValueError(f"Unverified summary: {failures}")
            rows.append(row)
            predictions = pd.read_csv(run_dir / "gp_window_predictions.csv")
            residual = (predictions.y_true - predictions.y_pred).abs()
            for family, tickers in data.FAMILIES.items():
                subset = predictions[predictions.asset_id.isin(tickers)]
                if subset.empty:
                    continue
                ic = subset.groupby("date").apply(lambda frame: frame.y_pred.corr(frame.y_true, method="spearman"),
                                                   include_groups=False)
                mask = predictions.asset_id.isin(tickers)
                diagnostic_rows.append({"configuration": item["label"], "seed": seed, "family": family,
                                        "within_family_ic": float(ic.mean()),
                                        "coverage_95": float((residual[mask] <= 1.95996398454 * subset.y_std).mean()),
                                        "mean_predictive_std": float(subset.y_std.mean())})
    runs = pd.DataFrame(rows)
    aggregates = aggregate_configurations(runs)
    runs.to_csv(out / "objectives_by_seed.csv", index=False)
    aggregates.to_csv(out / "configuration_aggregates.csv", index=False)
    aggregates[aggregates.pareto_nondominated].to_csv(out / "pareto.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(out / "family_uncertainty_diagnostics.csv", index=False)
    data.write_json(out / "hypervolume.json", calculate_hypervolume(aggregates, config))


def main() -> None:
    """Prepare the immutable registry or execute selected pending study batches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--family", choices=[*data.FAMILIES, "global"])
    args = parser.parse_args()
    previous.validate_worker_count(args.workers)
    if args.prepare:
        prepare()
    if args.execute:
        paths = sorted((HERE / "plans").glob("*.json"))
        if args.family:
            paths = [path for path in paths if json.loads(path.read_text())["family"] == args.family]
        for path in paths:
            execute_batch(path, 1 if args.smoke else args.workers, args.smoke)
            if args.smoke:
                break


if __name__ == "__main__":
    main()
