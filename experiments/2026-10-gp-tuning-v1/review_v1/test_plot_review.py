"""Regression checks for visibility and exact-data scientific GP review plots."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_review import BASELINE, BLUE, _figure  # noqa: E402


def test_baseline_drawn_above_overlapping_curves() -> None:
    """A later near-identical candidate must not conceal the blue reference."""
    dates = np.arange("2024-05", "2026-09", dtype="datetime64[M]").astype("datetime64[D]")
    reference = np.linspace(10000., 15810.7, 29)
    candidate = reference + np.linspace(0., 2.46, 29)
    context = {"dates": dates.astype(str), "benchmark_schwab_total": np.full(28, .01),
               "benchmark_momentum_total": np.full(28, .009)}
    figure, differences = _figure(context, {BASELINE: reference, "candidate": candidate},
                                  [{"id": "candidate", "label": "Nearly identical", "difference_panel": True}],
                                  "Visibility regression")
    top, bottom = figure.axes
    baseline = next(line for line in top.lines if line.get_gid() == "hard_rank_baseline")
    assert baseline.get_color() == BLUE
    assert baseline.get_zorder() > max(line.get_zorder() for line in top.lines if line is not baseline)
    np.testing.assert_array_equal(baseline.get_ydata(), reference)
    np.testing.assert_array_equal(bottom.lines[0].get_ydata(), candidate - reference)
    assert abs(differences["candidate"]["ending_dollars_minus_baseline"] - 2.46) < 1e-8
    plt.close(figure)
