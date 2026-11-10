"""Append-only monthly GP fit adapter for the September 2026 extension.

Research orchestration only. Reuses the sealed GP builder, window fitter, dense
observed-covariance readout and original window seeds. The only fitter change is
extending its training-panel assertion from indices 0..23 to 24..27. Means are
decimal discounted excess returns; covariance is decimal squared.
Usage: initialized by run_fits.py in spawned, single-thread processes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

STATE: dict[str, Any] = {}


def initialize(configs: dict, framework: str, legacy: str) -> None:
    """Initialize the existing exact fitter without running legacy allocation.

    Args:
        configs: Two registered models with the full 28-origin date axis.
        framework: Read-only evaluation framework directory.
        legacy: Read-only generic GP worker directory.
    """
    root = Path(framework)
    sys.path.extend([str(root), str(root / "selection_refresh_v1"), str(root / "parallel_pipeline_v1"), legacy])
    import refresh_workers

    refresh_workers.initialize_fit(configs, legacy, str(root / "parallel_pipeline_v1"))
    STATE.update(refresh=refresh_workers, configs=configs)


def validate_job(job: dict, config: dict) -> None:
    """Reject refits of old origins, renumbered seeds or unscored September.

    Args:
        job: Component, seed, global month index, date and isolated run directory.
        config: Registered full date axis and exact ETF/input lists.
    """
    if job["seed"] not in [27, 127, 227] or job["index"] not in range(24, 28):
        raise ValueError("Only the 24 additional model/seed/month fits are authorized")
    if config["dates"][job["index"]] != job["date"]:
        raise ValueError("Original global window index must not be renumbered")


def fit_job(job: dict) -> dict:
    """Fit one new origin using the frozen model and dense covariance exporter.

    Args:
        job: Registered new-origin fit; old origins are rejected before fitting.

    Returns:
        Original fit artifacts, optimizer telemetry and exact prediction readout.
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.optim.fit import fit_gpytorch_mll_scipy

    refresh = STATE["refresh"]
    generic = refresh.STATE["generic"]
    config = STATE["configs"][job["component"]]
    validate_job(job, config)
    if generic.STATE["component"] != job["component"]:
        generic._configure(job["component"])
    context = generic.STATE["context"]

    def fit(mll: Any, **kwargs: Any) -> Any:
        if mll.model.train_inputs[0].shape[-1] != len(config["inputs"]) + 1:
            raise ValueError("Actual input width changed")
        attempts = 0

        def optimize(model: Any, **options: Any) -> Any:
            nonlocal attempts
            rows = model.model.train_inputs[0].shape[0]
            tasks = len(config["tickers"])
            index = rows // tasks - 84
            if rows % tasks or index != job["index"] or index not in range(24, 28):
                raise ValueError("Unexpected appended training panel")
            if list(model.model.covar_module.kernels[1].covar_factor.shape) != [tasks, 2]:
                raise ValueError("Task covariance rank changed")
            if options.get("options", {}).get("maxiter") != 15:
                raise ValueError("Fitting budget changed")
            attempts += 1
            result = fit_gpytorch_mll_scipy(model, **options)
            payload = {
                "date": job["date"], "seed": context["seed"], "window_index": index,
                "attempt": attempts, "train_rows": rows, "tasks": tasks, "rank": 2, "maxiter": 15,
                "iterations": result.step, "status": result.status.name,
                "message": result.message, "runtime_seconds": result.runtime,
            }
            with context["log"].open("a") as handle:
                handle.write(json.dumps(payload) + "\n")
            return result

        return fit_gpytorch_mll(mll, optimizer=optimize, **kwargs)

    generic.STATE["runner"].fit_gpytorch_mll = fit
    return refresh.fit_job(job)
