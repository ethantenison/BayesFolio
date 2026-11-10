"""September/October forecast adapter around the frozen GP fit implementation.

Only registered new dates and the isolated input/output paths differ. Original
kernel, normalization, likelihood, rank2/maxiter15 and seed convention remain.
Targets are decimals; unknown November evaluation labels remain NaN.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from common import TUNING


def training_panel(frame: pd.DataFrame, date: str, index: int, tickers: list[str],
                   rolling_months: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select only available pre-origin target months for the two registered dates.

    Args:
        frame: Frozen114-month dataset, including unknown October live target.
        date: September30 or October30 forecast origin.
        index: Original May2024-relative index28 or29.
        tickers: Exact approved fifteen-ETF order.
        rolling_months: None for expanding or48 for the third expert.

    Returns:
        Complete causal train and evaluation panels with live labels untouched.
    """
    if {28: "2026-09-30", 29: "2026-10-30"}.get(index) != date or rolling_months not in {None, 48}:
        raise ValueError("Only registered September/October windows permitted")
    train = frame[frame.date.lt(pd.Timestamp(date)) & frame.y_excess_lead.notna()].copy()
    evaluation = frame[frame.date.eq(pd.Timestamp(date))].copy()
    if train.date.nunique() != 84 + index:
        raise ValueError("Available-label chronology changed")
    count = rolling_months or 84 + index
    if rolling_months:
        retained = sorted(train.date.unique())[-48:]
        train = train[train.date.isin(retained)].copy()
    counts = train.groupby("asset_id", observed=True).size()
    if (len(train) != count * 15 or set(counts.index) != set(tickers) or not counts.eq(count).all()
            or not np.isfinite(train.y_excess_lead.to_numpy()).all()
            or len(evaluation) != 15 or set(evaluation.asset_id) != set(tickers)):
        raise ValueError("Incomplete causal fit panel")
    if date == "2026-10-30" and not evaluation.y_excess_lead.isna().all():
        raise ValueError("November outcomes cannot be known in October")
    return train, evaluation


def initialize(configs: dict, framework: str) -> None:
    """Initialize a single-threaded frozen model worker with two date indexes.

    Args:
        configs: Immutable three-expert registration with September/October axes.
        framework: Existing read-only dense-covariance helper directory.
    """
    sys.path.insert(0, str(TUNING))
    import tuning_worker

    mapped = {name: {**config, "dates": {28: "2026-09-30", 29: "2026-10-30"}}
              for name, config in configs.items()}
    tuning_worker.initialize(mapped, framework)
    tuning_worker.training_panel = training_panel


def fit_job(job: dict) -> dict:
    """Fit one registered seed/month without editing the original fitter.

    Args:
        job: Exact component, seed, original index, date and isolated attempt path.

    Returns:
        Original fitted parameters, analytic moments, timing and optimizer records.
    """
    import tuning_worker

    return tuning_worker.fit_job(job)
