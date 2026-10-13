"""Run deterministic confirmation followed by one xhigh Qwen planning turn.

Stage 3 remains fully deterministic: execute or resume the 12-run U.S. seed
panel, validate it, and calculate the Pareto frontier and hypervolume. Only
after that gate passes does the workflow give Qwen one bounded xhigh turn to
research and preregister the next experiment batch. The proposed batch is
validated but never executed by this workflow.

Usage:
    # Print the two-stage workflow without changing anything.
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/run_qwen_confirmation_workflow.py

    # Execute/resume stage 3, then run and validate Qwen stage 4.
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/run_qwen_confirmation_workflow.py \
      --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
PLANNING_DIR = EXPERIMENT_DIR / "planning"
ORCHESTRATOR = EXPERIMENT_DIR / "orchestrate_us_confirmation.py"
PREPARE_TURN = EXPERIMENT_DIR / "prepare_next_experiment_turn.py"
VALIDATE_PLAN = EXPERIMENT_DIR / "validate_next_experiment_plan.py"
QWEN_LAUNCHER = Path("/Users/et/local-llm/qwen-codex.sh")
QWEN_SERVER = Path("/Users/et/local-llm/start-server.sh")
QWEN_ENDPOINT = "http://127.0.0.1:8080/v1/models"
SEEDS = "27,127,227"
QWEN_PROTECTED_PATHS = [
    ORCHESTRATOR,
    EXPERIMENT_DIR / "summarize_us_confirmation.py",
    PREPARE_TURN,
    VALIDATE_PLAN,
    EXPERIMENT_DIR / "run_qwen_confirmation_workflow.py",
    EXPERIMENT_DIR / "us_confirmation_config.json",
    PLANNING_DIR / "step4_evidence.json",
    PLANNING_DIR / "next_experiment_candidates.schema.json",
]


def parse_args() -> argparse.Namespace:
    """Parse dry-run, execution, and safe resume options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run stage 3 and the one-turn xhigh planning stage.")
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Reuse already-complete three-seed results; the step-3 validator still runs.",
    )
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a workflow status checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a protected workflow file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_protected_hashes() -> dict[str, str]:
    """Capture files that the Qwen planning turn is not authorized to modify."""
    missing = [str(path) for path in QWEN_PROTECTED_PATHS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Protected Qwen workflow inputs are missing: {missing}")
    return {str(path): sha256_file(path) for path in QWEN_PROTECTED_PATHS}


def verify_protected_hashes(expected_hashes: dict[str, str]) -> None:
    """Fail if Qwen modified its evidence, validator, schema, or execution code."""
    changed = [path for path, expected in expected_hashes.items() if sha256_file(Path(path)) != expected]
    if changed:
        raise RuntimeError(f"Qwen modified protected workflow inputs: {changed}")


def run_streamed(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run a command while streaming output and optionally preserving a log."""
    log_handle: IO[str] | None = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Unable to capture subprocess output.")
        for line in process.stdout:
            print(line, end="", flush=True)
            if log_handle is not None:
                log_handle.write(line)
                log_handle.flush()
        return process.wait()
    finally:
        if log_handle is not None:
            log_handle.close()


def qwen_server_ready() -> bool:
    """Return whether the local Qwen model endpoint is responding."""
    try:
        with urllib.request.urlopen(QWEN_ENDPOINT, timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def start_qwen_server(log_path: Path) -> tuple[subprocess.Popen[str], IO[str]]:
    """Start the local server and wait until its model endpoint is ready."""
    if not QWEN_SERVER.exists():
        raise FileNotFoundError(f"Qwen server launcher is missing: {QWEN_SERVER}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(QWEN_SERVER)],
        cwd=QWEN_SERVER.parent,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if qwen_server_ready():
            return process, log_handle
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(f"Qwen server exited with code {process.returncode}; see {log_path}.")
        time.sleep(2)
    process.terminate()
    log_handle.close()
    raise TimeoutError(f"Qwen server did not become ready within 300 seconds; see {log_path}.")


def stop_qwen_server(process: subprocess.Popen[str], log_handle: IO[str]) -> None:
    """Stop a server started by this workflow without touching a pre-existing server."""
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    finally:
        log_handle.close()


def main() -> None:
    """Run deterministic stage 3, then exactly one xhigh Qwen planning turn."""
    args = parse_args()
    confirmation_command = [sys.executable, str(ORCHESTRATOR), "--execute", "--seeds", SEEDS]
    prepare_command = [sys.executable, str(PREPARE_TURN), "--seeds", SEEDS]
    validate_command = [sys.executable, str(VALIDATE_PLAN)]
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "stage_3": confirmation_command,
                    "stage_4_prepare": prepare_command,
                    "stage_4_model": str(QWEN_LAUNCHER),
                    "stage_4_reasoning": "xhigh",
                    "stage_4_turn_count": 1,
                    "stage_4_validation": validate_command,
                    "new_gp_runs_from_stage_4": 0,
                },
                indent=2,
            )
        )
        return

    PLANNING_DIR.mkdir(parents=True, exist_ok=True)
    status_path = PLANNING_DIR / "qwen_workflow_status.json"
    status: dict[str, Any] = {
        "schema": "bayesfolio.etf_family_qwen_workflow_status.v1",
        "started_at": datetime.now(UTC).isoformat(),
        "state": "starting",
        "seeds": [27, 127, 227],
        "qwen_reasoning": "xhigh",
        "qwen_planning_turns": 1,
    }
    write_json_atomic(status_path, status)
    server_process: subprocess.Popen[str] | None = None
    server_log_handle: IO[str] | None = None
    try:
        if not args.skip_confirmation:
            status["state"] = "running_deterministic_stage_3"
            write_json_atomic(status_path, status)
            if run_streamed(confirmation_command, cwd=REPO_ROOT) != 0:
                raise RuntimeError("Deterministic stage 3 failed.")
        status["state"] = "preparing_qwen_stage_4"
        write_json_atomic(status_path, status)
        if run_streamed(prepare_command, cwd=REPO_ROOT) != 0:
            raise RuntimeError("Step-3 evidence did not pass the Qwen planning gate.")

        prompt_path = PLANNING_DIR / "QWEN_STEP4_PROMPT.md"
        if not QWEN_LAUNCHER.exists():
            raise FileNotFoundError(f"Qwen Codex launcher is missing: {QWEN_LAUNCHER}")
        if not qwen_server_ready():
            status["state"] = "starting_qwen_server"
            write_json_atomic(status_path, status)
            server_process, server_log_handle = start_qwen_server(PLANNING_DIR / "qwen_server.log")
            status["qwen_server_started_by_workflow"] = True
        else:
            status["qwen_server_started_by_workflow"] = False

        prompt = prompt_path.read_text(encoding="utf-8")
        protected_hashes = capture_protected_hashes()
        qwen_command = [
            str(QWEN_LAUNCHER),
            "-C",
            str(EXPERIMENT_DIR),
            "-a",
            "never",
            "exec",
            "--skip-git-repo-check",
            "-o",
            str(PLANNING_DIR / "qwen_step4_final.txt"),
            prompt,
        ]
        qwen_env = os.environ.copy()
        qwen_env["QWEN_REASONING"] = "xhigh"
        status["state"] = "running_qwen_stage_4"
        status["qwen_turn_started_at"] = datetime.now(UTC).isoformat()
        write_json_atomic(status_path, status)
        returncode = run_streamed(
            qwen_command,
            cwd=EXPERIMENT_DIR,
            log_path=PLANNING_DIR / "qwen_step4.log",
            env=qwen_env,
        )
        if returncode != 0:
            raise RuntimeError(f"Qwen planning turn failed with exit code {returncode}.")
        verify_protected_hashes(protected_hashes)
        status["state"] = "validating_qwen_stage_4"
        write_json_atomic(status_path, status)
        if run_streamed(validate_command, cwd=REPO_ROOT) != 0:
            raise RuntimeError("Qwen's next-experiment plan failed deterministic validation.")
        status["state"] = "completed"
        status["finished_at"] = datetime.now(UTC).isoformat()
        status["new_gp_runs_from_stage_4"] = 0
        write_json_atomic(status_path, status)
        print(f"Qwen confirmation workflow complete: {status_path}")
    except Exception as error:
        status["state"] = "failed"
        status["error"] = str(error)
        status["finished_at"] = datetime.now(UTC).isoformat()
        write_json_atomic(status_path, status)
        raise
    finally:
        if server_process is not None and server_log_handle is not None:
            stop_qwen_server(server_process, server_log_handle)


if __name__ == "__main__":
    main()
