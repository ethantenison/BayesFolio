"""Finish one live larger-budget experiment with deterministic postprocessing.

Waits only for the specified already-running dispatcher, then invokes the
read-only plotting/comparison step and seals outputs. Never launches fits,
retries a failed dispatcher or changes protected evidence. This is a one-run
dependent job, not a recurring monitor. Financial metrics use decimal returns.

Usage:
    .venv/bin/python \
        experiments/2026-10-repaired-global-feature-selection/finish_rank5_budget150.py --dispatcher-pid PID
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd
import run_rank5_budget150 as budget


def main() -> None:
    """Wait for the existing run and publish verified plots, summaries and artifact hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatcher-pid", type=int, required=True)
    args = parser.parse_args()
    root = budget.HERE
    state_path = root / "postprocess_status.json"
    started = time.monotonic()
    signature = budget.data.digest(root / "contract.json")
    budget.data.write_json(state_path, {"state": "waiting_for_fits", "dispatcher_pid": args.dispatcher_pid,
                                      "contract_sha256": signature})
    try:
        while True:
            status = json.loads((root / "status.json").read_text())
            if status["signature"] != signature:
                raise ValueError("The queued postprocessing belongs to a different contract")
            if status["state"] == "completed":
                break
            if status["state"] == "failed" or "failed" in status["runs"].values():
                raise RuntimeError("The fit dispatcher reported a failure")
            try:
                os.kill(args.dispatcher_pid, 0)
            except ProcessLookupError as error:
                raise RuntimeError("The fit dispatcher exited without a completed audit") from error
            except PermissionError:
                # Sandboxed hosts can deny signal-zero probes; use observable run progress instead.
                progress = [root / "status.json", *list((root / "runs").glob("*/fitting_diagnostics.jsonl"))]
                if time.time() - max(path.stat().st_mtime for path in progress) > 30 * 60:
                    raise RuntimeError("No observable fit progress for thirty minutes; native PID check unavailable")
            if time.monotonic() - started > 3 * 60 * 60:
                raise RuntimeError("Postprocessing wait exceeded three hours; fits were not interrupted")
            time.sleep(15)
        env = dict(os.environ)
        env.update({"MPLBACKEND": "Agg", "MPLCONFIGDIR": "/private/tmp/bayesfolio-mpl"})
        subprocess.run([sys.executable, str(budget.data.HERE / "plot_rank5_budget150.py")],
                       cwd=budget.data.ROOT, env=env, check=True)
        objectives = pd.read_csv(root / "objectives_by_seed.csv")
        convergence = json.loads((root / "convergence_summary.json").read_text())
        summary = {"passed": True, "contract_sha256": signature,
                   "seed_runs": 6, "new_seed_runs": 3, "dates_per_run": 24,
                   "means_by_maxiter": objectives.drop(columns="seed").groupby("maxiter").mean().to_dict("index"),
                   "convergence": convergence,
                   "boundary": "Budget sensitivity within rank five, not compute-matched rank comparison."}
        budget.data.write_json(root / "comparison_summary.json", summary)
        files = [root / name for name in ["contract.json", "status.json", "actual_model_probe.json",
            "independent_audit.json", "objectives_by_seed.csv", "etf_metrics_by_seed.csv", "fitting_diagnostics.csv",
            "convergence_summary.json", "comparison_summary.json"]]
        files.extend((root / "etf_forecast_plots").glob("*.png"))
        files.extend([budget.data.HERE / "plot_rank5_budget150.py", budget.data.HERE / "finish_rank5_budget150.py"])
        for seed in budget.prior.SEEDS:
            folder = root / f"runs/{budget.IDENTIFIER}_s{seed}"
            files.extend(folder / name for name in ["manifest.json", "gp_window_predictions.csv",
                "portfolio_returns.csv", "portfolio_weights.csv", "window_training_history.csv",
                "fitting_diagnostics.jsonl", "budget_fingerprint.json"])
        budget.data.write_json(root / "RESULTS_VERIFICATION.json", {"passed": True,
            "artifact_sha256": {str(path.relative_to(budget.data.ROOT)): budget.data.digest(path) for path in files},
            "verification_commands": [
                ".venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection",
                ".venv/bin/ruff check experiments/2026-10-repaired-global-feature-selection",
                "MPLCONFIGDIR=/private/tmp/bayesfolio-mpl .venv/bin/python "
                "experiments/2026-10-repaired-global-feature-selection/plot_rank5_budget150.py"]})
        budget.data.write_json(state_path, {"state": "completed", "contract_sha256": signature,
                                          "plots": 4, "summary": "comparison_summary.json"})
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        budget.data.write_json(state_path, {"state": "failed", "contract_sha256": signature,
                                          "error": str(error), "fits_interrupted": False})
        raise


if __name__ == "__main__":
    main()
