"""Log unchanged SciPy GP fitting results in an isolated experiment subprocess.

Wraps BoTorch's default L-BFGS-B optimizer without changing its options, return
values, retries or RNG. Writes iteration progress and per-attempt stopping
diagnostics beside the saved decimal-return forecasts. Protected runner files
are loaded normally, never edited. Used only by the rank-five budget experiment.

Usage:
    python log_fit_budget.py /absolute/path/to/runner.py <normal runner arguments>
"""

from __future__ import annotations

import json
import math
import runpy
import sys
from pathlib import Path
from typing import Any

import botorch.fit
from botorch.optim.fit import fit_gpytorch_mll_scipy


def observed_optimizer(log: Path, dates: list[str], seed: int) -> Any:
    """Wrap the default optimizer with observational progress/stopping diagnostics.

    Args:
        log: JSONL output path; financial forecasts are not modified.
        dates: Frozen scored dates corresponding to 84+k initial training months.
        seed: Existing fixed experiment seed.

    Returns:
        A compatible optimizer calling the unmodified BoTorch SciPy implementation.
    """
    attempts: dict[str, int] = {}

    def append(payload: dict[str, Any]) -> None:
        payload = {key: None if isinstance(value, float) and not math.isfinite(value) else value
                   for key, value in payload.items()}
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as handle:
            handle.write(json.dumps(payload, allow_nan=False) + "\n")

    def optimize(mll: Any, **kwargs: Any) -> Any:
        rows = mll.model.train_inputs[0].shape[0]
        window = rows // 15 - 84
        if rows % 15 or not 0 <= window < len(dates):
            raise ValueError("Unexpected expanding joint training panel")
        if list(mll.model.covar_module.kernels[1].covar_factor.shape) != [15, 5]:
            raise ValueError("Actual fitted task factor is not rank five")
        if kwargs.get("options", {}).get("maxiter") != 150 or kwargs.get("callback") is not None:
            raise ValueError("Unexpected fitting options/callback")
        date = dates[window]
        attempts[date] = attempts.get(date, 0) + 1
        common = {"date": date, "seed": seed, "attempt": attempts[date], "train_rows": rows, "maxiter": 150}

        def progress(parameters: Any, result: Any) -> None:
            if result.step == 1 or result.step % 10 == 0:
                append({**common, "event": "iteration", "iterations": result.step,
                        "negative_mll": float(result.fval)})

        result = fit_gpytorch_mll_scipy(mll, **{**kwargs, "callback": progress})
        append({**common, "event": "optimizer_return", "iterations": result.step,
                "negative_mll": float(result.fval), "status": result.status.name,
                "message": result.message, "runtime_seconds": result.runtime})
        return result

    return optimize


def main() -> None:
    """Install subprocess-local observation and execute the unchanged real runner."""
    runner, arguments = Path(sys.argv[1]), sys.argv[2:]
    def value(flag: str) -> str:
        return arguments[arguments.index(flag) + 1]
    output = Path(value("--output-root")) / value("--run-id")
    dates = Path(value("--scored-date-file")).read_text().splitlines()
    optimizer = observed_optimizer(output / "fitting_diagnostics.jsonl", dates, int(value("--seed")))
    original = botorch.fit.fit_gpytorch_mll

    def fit(mll: Any, **kwargs: Any) -> Any:
        if "optimizer" in kwargs:
            raise ValueError("Expected the default SciPy fitting path")
        return original(mll, optimizer=optimizer, **kwargs)

    botorch.fit.fit_gpytorch_mll = fit
    sys.argv = [str(runner), *arguments]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
