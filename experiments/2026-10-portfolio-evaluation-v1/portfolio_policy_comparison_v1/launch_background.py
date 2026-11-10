"""Launch approved one-shot fitting/scoring jobs independently of chat sessions.

Usage: .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/portfolio_policy_comparison_v1/launch_background.py
Use --fits-only while preparing the scorer. Detached sessions have explicit
logs, PIDs and single-writer locks; no launch agents, cron or recurring jobs are
installed. Fits resume immutable checkpoints; model/portfolio policies are not
changed by this process-lifetime helper.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parent
REPO = HERE.parents[2]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _locked(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def _start(script: Path, flag: str, log: Path) -> dict:
    command = ["/usr/bin/caffeinate", "-i", str(REPO / ".venv/bin/python"), str(script), flag]
    with log.open("ab", buffering=0) as output:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return {
        "pid": process.pid,
        "command": command,
        "log": str(log),
        "started_unix": time.time(),
        "detached_session": True,
        "one_shot_job": True,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits-only", action="store_true")
    args = parser.parse_args()
    record_path = HERE / "background_jobs.json"
    with (HERE / "background_launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = json.loads(record_path.read_text()) if record_path.exists() else {}
        jobs = [("fits", FRAMEWORK / "full_rerun_v1/full_rerun.py", "--execute", FRAMEWORK / "full_rerun_v1")]
        if not args.fits_only:
            jobs.append(("policy_comparison", HERE / "run_comparison.py", "--follow", HERE))
        for name, script, flag, folder in jobs:
            if (folder / "outputs/COMPLETED.json").exists():
                records.setdefault(name, {})["already_completed"] = True
                continue
            if _locked(folder / "outputs/run.lock"):
                continue
            previous = records.get(name)
            if previous and previous.get("pid") and _alive(previous["pid"]):
                continue
            records[name] = _start(script, flag, folder / "background.log")
        temporary = record_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        temporary.replace(record_path)
    print(json.dumps(records, indent=2), flush=True)


if __name__ == "__main__":
    _main()
