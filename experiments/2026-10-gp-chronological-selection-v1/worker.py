"""Causal extended-origin adapter around the unchanged frozen monthly GP worker.

Research orchestration only. Reuses construction, exact covariance export,
optimizer telemetry and fit-seed rules. The sole runtime adapter replaces the
old hard-coded28-date training slice with registered52-date coverage. No old
source is edited. Returns are decimals and covariance squared decimals.
"""

from __future__ import annotations

import sys
from pathlib import Path

from selection_math import training_panel


def initialize(configs: dict, framework: str) -> None:
    """Initialize one single-thread worker with immutable expanded-origin specs.

    Args:
        configs: JSON model specs with the exact chronological date axis.
        framework: Frozen evaluation helper directory.
    """
    campaign = Path(__file__).resolve().parents[1] / "2026-10-gp-tuning-v1"
    sys.path.insert(0, str(campaign))
    import tuning_worker

    actual = {key: {**value, "dates": dict(enumerate(value["dates"], start=-24))}
              for key, value in configs.items()}
    tuning_worker.initialize(actual, framework)
    tuning_worker.training_panel = training_panel


def fit_job(job: dict) -> dict:
    """Fit one isolated registered month without changing original model code.

    Args:
        job: Component, original seed/index/date and new attempt output path.

    Returns:
        Original worker telemetry/artifacts with unchanged fit-seed convention.
    """
    import tuning_worker

    return tuning_worker.fit_job(job)
