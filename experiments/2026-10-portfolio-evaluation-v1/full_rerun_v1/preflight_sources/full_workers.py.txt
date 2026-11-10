"""Isolated, bounded-pool workers for all-configuration GP refits and scoring.

Usage: initialized by full_rerun.py. Existing validated generic fit/export code
is imported read-only. No source modifications, covariance repair, or fallback.
All means/SDs are decimal discounted excess; covariance is decimal squared.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from full_math import NoCapAllocator, _aggregate_results, _score_seed

from artifacts import aligned_matrix

STATE = {}


def initialize_evaluation(context: dict, anchor: np.ndarray) -> None:
    """Cache common deterministic risk problems after the full fitting phase.

    Args:
        context: Frozen 24-month, 15-ETF targets, RF, risk budget and covariance.
        anchor: Frozen seed/month/asset forecasts for conditional family tests.
    """
    STATE.clear()
    STATE.update(
        context=context,
        anchor=anchor,
        allocator=NoCapAllocator(context["historical_covariance"], context["risk_budget"]),
    )


def evaluate_job(job: dict) -> dict:
    """Score every saved seed/month for one configuration without further fits.

    Args:
        job: Candidate identity, raw task/input registration and new output root.

    Returns:
        One complete summary row and hashes of complete seed-array/score outputs.
    """
    root = Path(job["root"])
    candidate, context = job["candidate"], STATE["context"]
    name, assets = candidate["configuration"], candidate["tickers"]
    results, paths = [], []
    for offset, seed in enumerate([27, 127, 227]):
        months = []
        for date in context["dates"]:
            folder = root / f"runs/{name}_s{seed}/analytic_months"
            with np.load(folder / f"covariance_{date}.npz", allow_pickle=False) as values:
                if values["date"].item() != date or values["assets"].tolist() != assets:
                    raise ValueError("Exact covariance date/task registration mismatch")
                months.append(
                    {key: values[key].copy() for key in ["mean", "std", "covariance", "likelihood_noise_diagonal"]}
                )
            predictions = pd.read_csv(folder / f"predictions_{date}.csv")
            indices = [context["assets"].index(a) for a in assets]
            actual = aligned_matrix(predictions, "y_true", [date], assets)[0]
            if not np.allclose(actual, context["actual"][context["dates"].index(date), indices], rtol=0, atol=1e-12):
                raise ValueError("New forecasts differ from frozen target values")
        values = {key: np.stack([row[key] for row in months]) for key in months[0]}
        if not np.allclose(
            np.diagonal(values["covariance"], axis1=-2, axis2=-1), values["std"] ** 2, rtol=1e-8, atol=1e-12
        ):
            raise ValueError("Raw observed covariance diagonal differs from predictive variance")
        if not (values["likelihood_noise_diagonal"] > 0).all():
            raise ValueError("Missing observed predictive likelihood noise")
        folder = root / "analytic" / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"s{seed}.npz"
        np.savez(
            path,
            dates=np.array(context["dates"]),
            assets=np.array(assets),
            observation_noise=np.array(True),
            units=np.array("raw_discounted_excess"),
            **values,
        )
        paths.append(path)
        results.append(
            {"seed": seed, **_score_seed(values, candidate, context, STATE["anchor"][offset], STATE["allocator"])}
        )
    summary = _aggregate_results(candidate, results)
    folder = root / "evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(
        json.dumps(
            {"candidate": candidate, "summary": summary, "seeds": results}, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )
    paths.append(path)
    return {
        "configuration": name,
        "summary": summary,
        "artifact_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
