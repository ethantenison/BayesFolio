"""Synthetic guards against incomplete joint-front comparisons.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/ -k joint_followup
"""

from __future__ import annotations

import pandas as pd
import pytest
from compare_joint_followup import METRICS, validate_seed_coverage


def valid_rows() -> pd.DataFrame:
    """Return one complete three-seed synthetic decimal-metric configuration."""
    return pd.DataFrame([{"configuration": "a", "seed": seed, "validation_passed": True,
                          **dict.fromkeys(METRICS, 0.1)} for seed in [27, 127, 227]])


def test_complete_three_seed_panel_passes() -> None:
    """A complete finite panel has the expected registry coverage."""
    validate_seed_coverage(valid_rows(), {"a"})


def test_missing_seed_fails() -> None:
    """Two completed seeds cannot enter the aggregate as a three-seed point."""
    with pytest.raises(ValueError, match="three"):
        validate_seed_coverage(valid_rows().iloc[:2], {"a"})


def test_duplicate_seed_fails() -> None:
    """Repeated rows cannot stand in for independent optimizer seeds."""
    rows = valid_rows()
    with pytest.raises(ValueError, match="duplicated"):
        validate_seed_coverage(pd.concat([rows, rows.iloc[:1]]), {"a"})


def test_missing_configuration_fails() -> None:
    """A new report cannot omit a registered losing or incomplete candidate."""
    with pytest.raises(ValueError, match="Missing"):
        validate_seed_coverage(valid_rows(), {"a", "b"})


def test_unvalidated_objective_fails() -> None:
    """Finite metrics are insufficient without successful run validation."""
    rows = valid_rows()
    rows.loc[0, "validation_passed"] = False
    with pytest.raises(ValueError, match="unvalidated"):
        validate_seed_coverage(rows, {"a"})
