"""Synthetic tests of honest multi-objective progress and plateau detection.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_round_progress.py
"""

from __future__ import annotations

import compare_round as comparison
import pandas as pd


def point(name: str, ic: float, nll: float, ratio: float, count: int, frontier: bool = True) -> dict:
    """Build a minimal objective-space fixture with explicit frontier membership."""
    return {"configuration": name, "mean_ic": ic, "mean_gaussian_nll": nll, "sharpe": ratio,
            "input_count_excluding_time": count, "pareto_nondominated": frontier}


def test_new_objective_region_is_progress() -> None:
    """A genuine Pareto-region improvement is not reduced to an arbitrary weighted score."""
    before = pd.DataFrame([point("old", 0.1, -2.0, 1.0, 8)])
    after = pd.DataFrame([point("old", 0.1, -2.0, 1.0, 8, False), point("new", 0.2, -2.1, 1.1, 9)])
    result = comparison.frontier_progress(before, after, {"new"})
    assert result["frontier_improved"]
    assert not result["stalled"]


def test_equal_objectives_with_fewer_inputs_is_simplification() -> None:
    """An exact objective tie can still establish a smaller competitive representation."""
    before = pd.DataFrame([point("old", 0.1, -2.0, 1.0, 8)])
    after = pd.DataFrame([point("old", 0.1, -2.0, 1.0, 8), point("new", 0.1, -2.0, 1.0, 6)])
    result = comparison.frontier_progress(before, after, {"new"})
    assert not result["frontier_improved"]
    assert result["competitive_simplification"]
    assert not result["stalled"]


def test_dominated_new_mask_is_stalled() -> None:
    """Merely fitting another configuration does not imply frontier improvement."""
    before = pd.DataFrame([point("old", 0.1, -2.0, 1.0, 8)])
    after = pd.DataFrame([point("old", 0.1, -2.0, 1.0, 8), point("new", 0.0, -1.9, 0.9, 5, False)])
    assert comparison.frontier_progress(before, after, {"new"})["stalled"]


def test_negative_ic_tradeoff_is_not_hidden_by_zero_reference() -> None:
    """The full Pareto frontier remains distinct from hypervolume above the fixed zero reference."""
    before = pd.DataFrame([point("old", 0.1, -2.0, 1.0, 8)])
    after = pd.DataFrame([point("old", 0.1, -2.0, 1.0, 8), point("new", -0.1, -2.4, 1.2, 7)])
    assert comparison.frontier_progress(before, after, {"new"})["frontier_improved"]
