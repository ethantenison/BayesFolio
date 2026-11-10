"""Launch this approved one-shot study independently of chat tool sessions.

Usage: .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/approved_scope_v1/launch_background.py
Detached session, explicit logs/PID, single writer, no cron or recurring jobs.
Only starts this study's --execute; it never restarts previous completed studies.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


def _main() -> None:
    here = Path(__file__).resolve().parent
    root = here / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = root / "background_job.json"
        if (root / "COMPLETED.json").exists():
            print("Study already completed; no process started")
            return
        with (root / "run.lock").open("a") as writer:
            try:
                fcntl.flock(writer, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Study already running; no duplicate started")
                return
        if record.exists():
            old = json.loads(record.read_text())
            try:
                os.kill(old["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                print("Recorded process is alive; no duplicate started")
                return
        repo = here.parents[2]
        command = ["/usr/bin/caffeinate", "-i", str(repo / ".venv/bin/python"), str(here / "run_study.py"), "--execute"]
        with (root / "background.log").open("ab", buffering=0) as output:
            process = subprocess.Popen(
                command,
                cwd=repo,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        value = {
            "pid": process.pid,
            "command": command,
            "started_unix": time.time(),
            "detached_session": True,
            "one_shot_job": True,
            "log": str(root / "background.log"),
        }
        record.write_text(json.dumps(value, indent=2) + "\n")
        print(json.dumps(value, indent=2))


if __name__ == "__main__":
    _main()
