"""Run the reproducible U.S. ETF feature-confirmation workflow.

The orchestrator validates a frozen JSON configuration, snapshots the exact
execution-critical sources, runs all four configurations for requested seeds,
checkpoints after every subprocess, validates each run, and writes the
multi-seed Pareto report. It uses cached artifacts only.

Usage:
    # Print the exact plan without writing or fitting.
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/orchestrate_us_confirmation.py \
      --seeds 27,127,227

    # Run only schema/date preflights.
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/orchestrate_us_confirmation.py \
      --preflight-only --seeds 27,127,227

    # Execute, resume safely, validate, and summarize in one command.
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/orchestrate_us_confirmation.py \
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

from summarize_us_confirmation import parse_seed_list
from summarize_us_pilots import read_scored_dates, summarize_run

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "us_confirmation_config.json"
EXPECTED_TICKERS = ["SPY", "MGK", "VTV", "IWM"]
EXPECTED_LABELS = ["U-AC", "U-C005", "U-C009", "U-ALL-MATCH"]
CRITICAL_SOURCE_PATHS = [
    "experiments/2026-10-etf-family-feature-confirmation/orchestrate_us_confirmation.py",
    "experiments/2026-10-etf-family-feature-confirmation/summarize_us_confirmation.py",
    "experiments/2026-10-etf-family-feature-confirmation/summarize_us_pilots.py",
    "experiments/2026-07-us-equity-etf-features/run_first_round_walkforward.py",
    "experiments/2026-06-portfolio-optimization/run_monthly_optimization_walkforward.py",
    "experiments/2026-06-task-covariance/run_task_covariance_rollforward.py",
    "bayesfolio/engine/forecast/gp/multitask_builder.py",
]


def parse_args() -> argparse.Namespace:
    """Parse orchestrator mode, configuration, and deterministic seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds; defaults to config.default_seeds.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run fits, validation, and aggregation.")
    mode.add_argument("--preflight-only", action="store_true", help="Run all preflights but no GP fits.")
    parser.add_argument(
        "--execution-id",
        default=None,
        help="Optional identifier for the execution snapshot directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(raw: str | Path) -> Path:
    """Resolve a configured path against the repository root."""
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write formatted JSON checkpoint data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_output(*args: str) -> str:
    """Return stripped output from a read-only Git command."""
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def argument_value(arguments: list[str], flag: str) -> str | None:
    """Return the value following a configured flag, if present."""
    if flag not in arguments:
        return None
    index = arguments.index(flag)
    if index + 1 >= len(arguments):
        raise ValueError(f"Configured flag {flag} is missing its value.")
    return arguments[index + 1]


def validate_config(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    """Validate fixed experiment decisions and return resolved configured paths."""
    if config.get("schema") != "bayesfolio.etf_family_confirmation_orchestrator.v1":
        raise ValueError("Unsupported orchestrator configuration schema.")
    if config.get("tickers") != EXPECTED_TICKERS or "IJR" in config.get("tickers", []):
        raise ValueError(f"The U.S. task universe must be exactly {EXPECTED_TICKERS}.")
    if [item.get("label") for item in config.get("configurations", [])] != EXPECTED_LABELS:
        raise ValueError(f"Configurations must be ordered as {EXPECTED_LABELS}.")
    identifiers = [item.get("id") for item in config["configurations"]]
    if len(set(identifiers)) != len(identifiers) or any(not identifier for identifier in identifiers):
        raise ValueError("Configuration IDs must be unique non-empty strings.")

    fixed = config.get("fixed_decisions", {})
    expected_fixed = {
        "exclude_ijr": True,
        "include_t_index": True,
        "cached_data_only": True,
        "production_budget_reference_excluded": True,
    }
    if fixed != expected_fixed:
        raise ValueError(f"Fixed decisions must equal {expected_fixed}.")
    selection = config.get("selection", {})
    if selection.get("maximize") != ["mean_ic", "sharpe"]:
        raise ValueError("Selection must maximize mean_ic and sharpe only.")
    if selection.get("minimize") != ["mean_gaussian_nll"]:
        raise ValueError("Selection must minimize mean_gaussian_nll only.")
    if selection.get("diagnostic_only") != ["cumulative_return", "rmse"]:
        raise ValueError("Only cumulative_return and rmse may be diagnostic-only metrics.")
    if selection.get("turnover_excluded") is not True:
        raise ValueError("Turnover must be excluded.")
    expected_sharpe = "mean(monthly excess return) / sample_std(monthly excess return) * sqrt(12)"
    if selection.get("sharpe_definition") != expected_sharpe:
        raise ValueError(f"Sharpe definition must be {expected_sharpe!r}.")
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
        raise ValueError(f"Hypervolume settings must equal {expected_hypervolume}.")

    common_arguments = [str(value) for value in config.get("common_args", [])]
    expected_common_values = {
        "--mtgp-rank": "2",
        "--maxiter": "15",
        "--posterior-scenarios": "256",
        "--gp-experiment": "signed_lkj_eta_2",
        "--upperlng": "0.35",
        "--nea": "3",
        "--gp-riskfolio-rm": "CVaR",
        "--gp-riskfolio-obj": "MaxRet",
    }
    for flag, expected in expected_common_values.items():
        if argument_value(common_arguments, flag) != expected:
            raise ValueError(f"{flag} must remain fixed at {expected}.")
    if "--resume-scenarios" not in common_arguments:
        raise ValueError("--resume-scenarios is required for checkpointed execution.")

    all_arguments = [*common_arguments]
    for configuration in config["configurations"]:
        all_arguments.extend(configuration.get("args", []))
    if "--exclude-time-index" in all_arguments:
        raise ValueError("The fixed t_index input cannot be excluded.")
    forbidden = {"--feature-path", "--output-root", "--run-id", "--seed", "--scored-date-file", "--tickers"}
    overlap = sorted(forbidden.intersection(all_arguments))
    if overlap:
        raise ValueError(f"Orchestrator-owned arguments cannot appear in config argument lists: {overlap}")
    if any("turnover" in str(value).lower() for value in all_arguments):
        raise ValueError("Turnover cannot appear in confirmation features or execution arguments.")

    try:
        config_path.resolve().relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError("The orchestrator config must be stored inside the repository.") from error

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
        str(path)
        for name, path in paths.items()
        if name not in {"output_root", "snapshot_root"} and not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Configured inputs are missing: {missing}")

    feature_manifest = json.loads(paths["feature_manifest"].read_text(encoding="utf-8"))
    if feature_manifest["artifact_sha256"] != sha256_file(paths["feature"]):
        raise ValueError("Feature artifact hash does not match its manifest.")
    if feature_manifest["approved_scored_dates"] != read_scored_dates():
        raise ValueError("Frozen scored dates do not match the feature manifest.")
    if feature_manifest["tickers"] != EXPECTED_TICKERS:
        raise ValueError("Feature manifest task universe does not match the approved U.S. universe.")
    return paths


def build_command(
    config: dict[str, Any],
    paths: dict[str, Path],
    configuration: dict[str, Any],
    *,
    seed: int,
    preflight: bool,
) -> list[str]:
    """Build one exact wrapper command from frozen configuration data."""
    run_id = f"{configuration['id']}_s{seed}"
    command = [
        sys.executable,
        str(paths["runner"]),
        "--feature-path",
        str(paths["feature"]),
        "--output-root",
        str(paths["output_root"]),
        "--run-id",
        run_id,
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


def command_display(command: list[str]) -> str:
    """Return a shell-readable command without executing a shell."""
    return shlex.join(command)


def snapshot_execution(
    execution_dir: Path,
    *,
    config: dict[str, Any],
    paths: dict[str, Path],
    seeds: list[int],
    mode: str,
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    """Save exact critical sources, hashes, Git state, and resolved commands."""
    source_root = execution_dir / "source_snapshot"
    source_hashes: dict[str, str] = {}
    for relative in CRITICAL_SOURCE_PATHS:
        source = REPO_ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"Critical execution source is missing: {source}")
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[relative] = sha256_file(source)

    config_destination = source_root / paths["config"].relative_to(REPO_ROOT)
    config_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths["config"], config_destination)
    source_hashes[str(paths["config"].relative_to(REPO_ROOT))] = sha256_file(paths["config"])

    critical_diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", *CRITICAL_SOURCE_PATHS],
        cwd=REPO_ROOT,
    )
    (execution_dir / "critical_sources.diff").write_bytes(critical_diff)
    manifest = {
        "schema": "bayesfolio.etf_family_confirmation_execution.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "seeds": seeds,
        "configurations": [item["label"] for item in config["configurations"]],
        "git_sha": git_output("rev-parse", "HEAD"),
        "git_status_at_start": git_output("status", "--short"),
        "critical_diff_sha256": hashlib.sha256(critical_diff).hexdigest(),
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
    """Fail if frozen code, configuration, or cached inputs changed after snapshotting."""
    expected_hashes = {
        **{str(REPO_ROOT / relative): digest for relative, digest in manifest["source_sha256"].items()},
        str(paths["feature"]): manifest["feature_sha256"],
        str(paths["feature_manifest"]): manifest["feature_manifest_sha256"],
        str(paths["dates"]): manifest["scored_dates_sha256"],
    }
    changed = [path for path, expected in expected_hashes.items() if sha256_file(Path(path)) != expected]
    if changed:
        raise RuntimeError(f"Execution inputs changed after snapshotting: {changed}")


def run_captured(command: list[str], *, log_path: Path) -> subprocess.CompletedProcess[str]:
    """Run a short command and save its stdout and stderr."""
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result


def run_streamed(command: list[str], *, log_path: Path) -> int:
    """Run a long command while teeing combined output to console and a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Unable to capture subprocess output.")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


def preflight_all(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    seed: int,
    execution_dir: Path,
) -> dict[str, Any]:
    """Run and validate one preflight per distinct configuration."""
    expected_dates = read_scored_dates()
    results: dict[str, Any] = {}
    for configuration in config["configurations"]:
        command = build_command(config, paths, configuration, seed=seed, preflight=True)
        label = str(configuration["label"])
        result = run_captured(command, log_path=execution_dir / "logs" / f"preflight_{configuration['id']}.log")
        if result.returncode != 0:
            raise RuntimeError(f"Preflight failed for {label}; see execution logs.")
        payload = json.loads(result.stdout)
        if payload["scored_dates"] != expected_dates or payload["n_scored_dates"] != len(expected_dates):
            raise ValueError(f"Preflight date mismatch for {label}.")
        results[label] = payload
    write_json_atomic(execution_dir / "preflight_results.json", results)
    return results


def existing_run_is_valid(label: str, run_dir: Path, *, seed: int) -> bool:
    """Return whether an existing run passes the full per-run validation contract."""
    if not run_dir.exists():
        return False
    try:
        _, failures = summarize_run(label, run_dir, read_scored_dates(), expected_seed=seed)
    except (FileNotFoundError, KeyError, ValueError):
        return False
    return not failures


def execute_runs(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    seeds: list[int],
    execution_dir: Path,
    execution_manifest: dict[str, Any],
    status: dict[str, Any],
) -> None:
    """Execute or reuse every requested run with an atomic checkpoint per transition."""
    status_path = execution_dir / "status.json"
    for seed in seeds:
        for configuration in config["configurations"]:
            verify_execution_inputs(execution_manifest, paths)
            label = str(configuration["label"])
            run_id = f"{configuration['id']}_s{seed}"
            run_dir = paths["output_root"] / run_id
            run_status = status["runs"][run_id]
            if existing_run_is_valid(label, run_dir, seed=seed):
                run_status.update({"state": "completed_existing", "finished_at": datetime.now(UTC).isoformat()})
                write_json_atomic(status_path, status)
                print(f"Reusing validated run {run_id}")
                continue

            command = build_command(config, paths, configuration, seed=seed, preflight=False)
            run_status.update({"state": "running", "started_at": datetime.now(UTC).isoformat()})
            write_json_atomic(status_path, status)
            returncode = run_streamed(command, log_path=execution_dir / "logs" / f"run_{run_id}.log")
            if returncode != 0:
                run_status.update(
                    {"state": "failed", "returncode": returncode, "finished_at": datetime.now(UTC).isoformat()}
                )
                status["state"] = "failed"
                write_json_atomic(status_path, status)
                raise RuntimeError(f"Run {run_id} failed with exit code {returncode}.")
            _, failures = summarize_run(label, run_dir, read_scored_dates(), expected_seed=seed)
            if failures:
                run_status.update(
                    {"state": "validation_failed", "failures": failures, "finished_at": datetime.now(UTC).isoformat()}
                )
                status["state"] = "failed"
                write_json_atomic(status_path, status)
                raise ValueError(f"Run {run_id} failed validation: {failures}")
            run_status.update({"state": "completed", "returncode": 0, "finished_at": datetime.now(UTC).isoformat()})
            write_json_atomic(status_path, status)


def main() -> None:
    """Validate, snapshot, preflight, execute, checkpoint, and summarize the workflow."""
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = validate_config(config, config_path)
    seeds = parse_seed_list(args.seeds) if args.seeds else [int(seed) for seed in config["default_seeds"]]
    mode = "execute" if args.execute else ("preflight_only" if args.preflight_only else "dry_run")
    resolved_commands = [
        {
            "configuration": configuration["label"],
            "seed": seed,
            "run_id": f"{configuration['id']}_s{seed}",
            "command": build_command(config, paths, configuration, seed=seed, preflight=False),
        }
        for seed in seeds
        for configuration in config["configurations"]
    ]
    if mode == "dry_run":
        payload = {
            "mode": mode,
            "seeds": seeds,
            "run_count": len(resolved_commands),
            "commands": [
                {**item, "command_display": command_display(item["command"])} for item in resolved_commands
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    execution_id = args.execution_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    execution_dir = paths["snapshot_root"] / execution_id
    execution_dir.mkdir(parents=True, exist_ok=False)
    execution_manifest = snapshot_execution(
        execution_dir,
        config=config,
        paths=paths,
        seeds=seeds,
        mode=mode,
        commands=resolved_commands,
    )
    (paths["snapshot_root"] / "LATEST").write_text(execution_id + "\n", encoding="utf-8")
    status = {
        "schema": "bayesfolio.etf_family_confirmation_status.v1",
        "execution_id": execution_id,
        "state": "preflighting",
        "mode": mode,
        "seeds": seeds,
        "runs": {
            item["run_id"]: {
                "configuration": item["configuration"],
                "seed": item["seed"],
                "state": "pending",
            }
            for item in resolved_commands
        },
    }
    status_path = execution_dir / "status.json"
    write_json_atomic(status_path, status)
    try:
        verify_execution_inputs(execution_manifest, paths)
        preflight_all(config, paths, seed=seeds[0], execution_dir=execution_dir)
        if mode == "preflight_only":
            for run_status in status["runs"].values():
                run_status["state"] = "preflight_passed"
            status["state"] = "preflight_complete"
            status["input_integrity"] = "passed"
            status["finished_at"] = datetime.now(UTC).isoformat()
            write_json_atomic(status_path, status)
            print(f"Preflight complete: {execution_dir}")
            return

        status["state"] = "running"
        write_json_atomic(status_path, status)
        execute_runs(
            config,
            paths,
            seeds=seeds,
            execution_dir=execution_dir,
            execution_manifest=execution_manifest,
            status=status,
        )
        verify_execution_inputs(execution_manifest, paths)
        summary_command = [
            sys.executable,
            str(EXPERIMENT_DIR / "summarize_us_confirmation.py"),
            "--config",
            str(config_path),
            "--seeds",
            ",".join(str(seed) for seed in seeds),
        ]
        returncode = run_streamed(summary_command, log_path=execution_dir / "logs" / "summary.log")
        if returncode != 0:
            raise RuntimeError(f"Confirmation aggregation failed with exit code {returncode}.")
        status["state"] = "completed"
        status["input_integrity"] = "passed"
        status["finished_at"] = datetime.now(UTC).isoformat()
        write_json_atomic(status_path, status)
        print(f"Confirmation workflow complete: {execution_dir}")
    except Exception as error:
        status["state"] = "failed"
        status["error"] = str(error)
        status["finished_at"] = datetime.now(UTC).isoformat()
        write_json_atomic(status_path, status)
        raise


if __name__ == "__main__":
    main()
