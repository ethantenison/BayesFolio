"""Run a frozen international ETF confirmation or targeted-ablation phase.

Usage:
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/orchestrate_international_experiments.py \
      --config experiments/2026-10-etf-family-feature-confirmation/international_confirmation_config.json \
      --execute --seeds 27,127,227
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from summarize_international_experiments import read_dates, summarize_run
from summarize_us_confirmation import parse_seed_list

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "international_confirmation_config.json"
EXPECTED_TICKERS = ["VEA", "VWO", "VSS", "EWX"]


def parse_args() -> argparse.Namespace:
    """Parse execution mode, configuration, and seed panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execution-id", default=None)
    return parser.parse_args()


def resolve_repo_path(raw: str | Path) -> Path:
    """Resolve a configured path against the repository root."""
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write formatted checkpoint JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def argument_value(arguments: list[str], flag: str) -> str | None:
    """Return the configured value following a flag."""
    if flag not in arguments:
        return None
    index = arguments.index(flag)
    if index + 1 >= len(arguments):
        raise ValueError(f"Configured flag {flag} is missing its value.")
    return arguments[index + 1]


def validate_config(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    """Validate immutable experiment decisions and resolve configured paths."""
    if config.get("schema") != "bayesfolio.international_equity_experiment.v1":
        raise ValueError("Unsupported international experiment schema.")
    if config.get("phase") not in {"confirmation", "targeted_ablation"}:
        raise ValueError("Phase must be confirmation or targeted_ablation.")
    if config.get("tickers") != EXPECTED_TICKERS:
        raise ValueError(f"Task universe must be exactly {EXPECTED_TICKERS}.")
    configurations = config.get("configurations", [])
    comparisons = config.get("comparison_configurations", [])
    identifiers = [item.get("id") for item in [*configurations, *comparisons]]
    if not configurations or len(identifiers) != len(set(identifiers)) or any(not value for value in identifiers):
        raise ValueError("Configuration IDs must be unique non-empty strings.")
    if config.get("fixed_decisions") != {
        "include_t_index": True,
        "cached_data_only": True,
        "production_budget_reference_excluded": True,
    }:
        raise ValueError("Fixed international decisions changed.")
    selection = config.get("selection", {})
    if selection != {
        "maximize": ["mean_ic", "sharpe"],
        "minimize": ["mean_gaussian_nll"],
        "sharpe_definition": "mean(monthly excess return) / sample_std(monthly excess return) * sqrt(12)",
        "diagnostic_only": ["cumulative_return", "rmse"],
        "turnover_excluded": True,
    }:
        raise ValueError("The three-objective selection contract changed.")
    expected_hypervolume = {
        "reference_point": {"mean_ic": 0.0, "mean_gaussian_nll": 0.0, "sharpe": 0.0},
        "objective_transform": {
            "mean_ic": "maximize",
            "mean_gaussian_nll": "negate_then_maximize",
            "sharpe": "maximize",
        },
        "use_configuration_means_across_seeds": True,
        "fixed_for_subsequent_rounds": True,
    }
    if config.get("hypervolume") != expected_hypervolume:
        raise ValueError("The fixed hypervolume contract changed.")

    common = [str(value) for value in config.get("common_args", [])]
    fixed_values = {
        "--mtgp-rank": "2",
        "--maxiter": "15",
        "--posterior-scenarios": "256",
        "--gp-experiment": "signed_lkj_eta_2",
        "--upperlng": "0.35",
        "--nea": "3",
        "--gp-riskfolio-rm": "CVaR",
        "--gp-riskfolio-obj": "MaxRet",
    }
    for flag, expected in fixed_values.items():
        if argument_value(common, flag) != expected:
            raise ValueError(f"{flag} must remain fixed at {expected}.")
    if "--resume-scenarios" not in common:
        raise ValueError("--resume-scenarios is required.")
    forbidden = {"--feature-path", "--output-root", "--run-id", "--seed", "--scored-date-file", "--tickers"}
    for configuration in [*configurations, *comparisons]:
        columns = configuration.get("expected_input_columns", [])
        if not columns or columns[0] != "t_index" or any("turnover" in str(value).lower() for value in columns):
            raise ValueError(f"{configuration.get('label')} violates the fixed input contract.")
        overlap = forbidden.intersection(str(value) for value in configuration.get("args", []))
        if overlap:
            raise ValueError(f"Orchestrator-owned arguments found: {sorted(overlap)}")

    try:
        config_path.resolve().relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError("Configuration must be stored inside the repository.") from error
    paths = {
        "config": config_path.resolve(),
        "runner": resolve_repo_path(config["runner"]),
        "feature": resolve_repo_path(config["feature_path"]),
        "feature_manifest": resolve_repo_path(config["feature_manifest"]),
        "dates": resolve_repo_path(config["scored_date_file"]),
        "output_root": resolve_repo_path(config["output_root"]),
        "snapshot_root": resolve_repo_path(config["snapshot_root"]),
    }
    missing = [
        str(path) for name, path in paths.items() if name not in {"output_root", "snapshot_root"} and not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Configured inputs are missing: {missing}")
    feature_manifest = json.loads(paths["feature_manifest"].read_text(encoding="utf-8"))
    expected_dates = read_dates(paths["dates"])
    if feature_manifest["artifact_sha256"] != sha256_file(paths["feature"]):
        raise ValueError("Feature artifact hash does not match its manifest.")
    if feature_manifest["approved_scored_dates"] != expected_dates:
        raise ValueError("Scored dates do not match the feature manifest.")
    if feature_manifest["tickers"] != EXPECTED_TICKERS or len(expected_dates) != 24:
        raise ValueError("Feature manifest does not match the frozen universe and 24-date panel.")
    return paths


def build_command(
    config: dict[str, Any],
    paths: dict[str, Path],
    configuration: dict[str, Any],
    *,
    seed: int,
    preflight: bool,
) -> list[str]:
    """Build an exact runner command from frozen configuration data."""
    command = [
        sys.executable,
        str(paths["runner"]),
        "--feature-path",
        str(paths["feature"]),
        "--output-root",
        str(paths["output_root"]),
        "--run-id",
        f"{configuration['id']}_s{seed}",
        "--tickers",
        ",".join(config["tickers"]),
        "--scored-date-file",
        str(paths["dates"]),
        "--seed",
        str(seed),
        *[str(value) for value in config["common_args"]],
        *[str(value) for value in configuration["args"]],
    ]
    if preflight:
        command.append("--preflight-only")
    return command


def snapshot_execution(
    execution_dir: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    seeds: list[int],
    mode: str,
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    """Snapshot execution-critical code, config, hashes, and resolved commands."""
    critical = [
        Path(__file__).resolve(),
        EXPERIMENT_DIR / "summarize_international_experiments.py",
        paths["runner"],
        REPO_ROOT / "experiments/2026-06-portfolio-optimization/run_monthly_optimization_walkforward.py",
        REPO_ROOT / "experiments/2026-06-task-covariance/run_task_covariance_rollforward.py",
        REPO_ROOT / "bayesfolio/engine/forecast/gp/multitask_builder.py",
        paths["config"],
    ]
    source_hashes: dict[str, str] = {}
    for source in critical:
        relative = source.relative_to(REPO_ROOT)
        destination = execution_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = sha256_file(source)
    manifest = {
        "schema": "bayesfolio.international_equity_execution.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "phase": config["phase"],
        "seeds": seeds,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "source_sha256": source_hashes,
        "feature_sha256": sha256_file(paths["feature"]),
        "feature_manifest_sha256": sha256_file(paths["feature_manifest"]),
        "scored_dates_sha256": sha256_file(paths["dates"]),
        "resolved_commands": commands,
        "cached_data_only": True,
        "selection": config["selection"],
    }
    write_json_atomic(execution_dir / "execution_manifest.json", manifest)
    return manifest


def verify_execution_inputs(manifest: dict[str, Any], paths: dict[str, Path]) -> None:
    """Fail if cached inputs or execution-critical sources changed after snapshotting."""
    expected = {
        **{str(REPO_ROOT / path): digest for path, digest in manifest["source_sha256"].items()},
        str(paths["feature"]): manifest["feature_sha256"],
        str(paths["feature_manifest"]): manifest["feature_manifest_sha256"],
        str(paths["dates"]): manifest["scored_dates_sha256"],
    }
    changed = [path for path, digest in expected.items() if sha256_file(Path(path)) != digest]
    if changed:
        raise RuntimeError(f"Execution inputs changed after snapshotting: {changed}")


def run_captured(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    """Run a short command and save its combined output."""
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result


def run_streamed(command: list[str], log_path: Path) -> int:
    """Run a long command while teeing output to the console and a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        if process.stdout is None:
            raise RuntimeError("Unable to capture subprocess output.")
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return process.wait()


def preflight_all(config: dict[str, Any], paths: dict[str, Path], seed: int, execution_dir: Path) -> dict[str, Any]:
    """Run one deterministic preflight for every configuration executed in this phase."""
    expected_dates = read_dates(paths["dates"])
    results: dict[str, Any] = {}
    for configuration in config["configurations"]:
        result = run_captured(
            build_command(config, paths, configuration, seed=seed, preflight=True),
            execution_dir / "logs" / f"preflight_{configuration['id']}.log",
        )
        if result.returncode != 0:
            raise RuntimeError(f"Preflight failed for {configuration['label']}.")
        payload = json.loads(result.stdout)
        if payload["scored_dates"] != expected_dates or payload["n_scored_dates"] != len(expected_dates):
            raise ValueError(f"Preflight date mismatch for {configuration['label']}.")
        results[str(configuration["label"])] = payload
    write_json_atomic(execution_dir / "preflight_results.json", results)
    return results


def existing_run_is_valid(
    config: dict[str, Any], paths: dict[str, Path], configuration: dict[str, Any], seed: int
) -> bool:
    """Return whether an existing run satisfies its full frozen contract."""
    try:
        _, failures = summarize_run(
            str(configuration["label"]),
            configuration,
            paths["output_root"] / f"{configuration['id']}_s{seed}",
            read_dates(paths["dates"]),
            config["tickers"],
            expected_seed=seed,
        )
    except (FileNotFoundError, KeyError, ValueError):
        return False
    return not failures


def execute_runs(
    config: dict[str, Any],
    paths: dict[str, Path],
    seeds: list[int],
    execution_dir: Path,
    manifest: dict[str, Any],
    status: dict[str, Any],
) -> None:
    """Execute or reuse every configured run and checkpoint each transition."""
    status_path = execution_dir / "status.json"
    for configuration in config["configurations"]:
        for seed in seeds:
            verify_execution_inputs(manifest, paths)
            run_id = f"{configuration['id']}_s{seed}"
            run_status = status["runs"][run_id]
            if existing_run_is_valid(config, paths, configuration, seed):
                run_status.update({"state": "completed_existing", "finished_at": datetime.now(UTC).isoformat()})
                write_json_atomic(status_path, status)
                print(f"Reusing validated run {run_id}")
                continue
            run_status.update({"state": "running", "started_at": datetime.now(UTC).isoformat()})
            write_json_atomic(status_path, status)
            returncode = run_streamed(
                build_command(config, paths, configuration, seed=seed, preflight=False),
                execution_dir / "logs" / f"run_{run_id}.log",
            )
            if returncode != 0:
                run_status.update({"state": "failed", "returncode": returncode})
                write_json_atomic(status_path, status)
                raise RuntimeError(f"Run {run_id} failed with exit code {returncode}.")
            if not existing_run_is_valid(config, paths, configuration, seed):
                run_status["state"] = "validation_failed"
                write_json_atomic(status_path, status)
                raise ValueError(f"Run {run_id} failed validation.")
            run_status.update({"state": "completed", "returncode": 0, "finished_at": datetime.now(UTC).isoformat()})
            write_json_atomic(status_path, status)


def main() -> None:
    """Validate, preflight, execute, checkpoint, and summarize one phase."""
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = validate_config(config, config_path)
    seeds = parse_seed_list(args.seeds) if args.seeds else [int(seed) for seed in config["default_seeds"]]
    mode = "execute" if args.execute else ("preflight_only" if args.preflight_only else "dry_run")
    commands = [
        {
            "configuration": configuration["label"],
            "seed": seed,
            "run_id": f"{configuration['id']}_s{seed}",
            "command": build_command(config, paths, configuration, seed=seed, preflight=False),
        }
        for configuration in config["configurations"]
        for seed in seeds
    ]
    if mode == "dry_run":
        print(
            json.dumps(
                {
                    "mode": mode,
                    "phase": config["phase"],
                    "seeds": seeds,
                    "run_count": len(commands),
                    "commands": [{**item, "command_display": shlex.join(item["command"])} for item in commands],
                },
                indent=2,
            )
        )
        return

    execution_id = args.execution_id or f"international_{config['phase']}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    execution_dir = paths["snapshot_root"] / execution_id
    execution_dir.mkdir(parents=True, exist_ok=False)
    manifest = snapshot_execution(execution_dir, config, paths, seeds, mode, commands)
    (paths["snapshot_root"] / f"LATEST_{config['phase'].upper()}").write_text(execution_id + "\n", encoding="utf-8")
    status = {
        "schema": "bayesfolio.international_equity_status.v1",
        "execution_id": execution_id,
        "phase": config["phase"],
        "state": "preflighting",
        "mode": mode,
        "seeds": seeds,
        "runs": {
            item["run_id"]: {
                "configuration": item["configuration"],
                "seed": item["seed"],
                "state": "pending",
            }
            for item in commands
        },
    }
    status_path = execution_dir / "status.json"
    write_json_atomic(status_path, status)
    try:
        verify_execution_inputs(manifest, paths)
        preflight_all(config, paths, seeds[0], execution_dir)
        if mode == "preflight_only":
            status.update({"state": "preflight_complete", "input_integrity": "passed"})
            write_json_atomic(status_path, status)
            print(f"Preflight complete: {execution_dir}")
            return
        status["state"] = "running"
        write_json_atomic(status_path, status)
        execute_runs(config, paths, seeds, execution_dir, manifest, status)
        verify_execution_inputs(manifest, paths)
        summary_command = [
            sys.executable,
            str(EXPERIMENT_DIR / "summarize_international_experiments.py"),
            "--config",
            str(config_path),
            "--seeds",
            ",".join(str(seed) for seed in seeds),
        ]
        if run_streamed(summary_command, execution_dir / "logs" / "summary.log") != 0:
            raise RuntimeError("International aggregation failed.")
        status.update({"state": "completed", "input_integrity": "passed", "finished_at": datetime.now(UTC).isoformat()})
        write_json_atomic(status_path, status)
        print(f"International workflow complete: {execution_dir}")
    except Exception as error:
        status.update({"state": "failed", "error": str(error), "finished_at": datetime.now(UTC).isoformat()})
        write_json_atomic(status_path, status)
        raise


if __name__ == "__main__":
    main()
