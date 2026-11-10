"""Synthetic checks that component synthesis preserves conflicting evidence.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_component_synthesis.py
"""

from __future__ import annotations

import pandas as pd
import pytest
import synthesize_components as synthesis


def test_no_pair_is_not_evidence_of_irrelevance() -> None:
    """An untested feature must remain untested, not become an automatic deletion."""
    assert synthesis.context_summary(pd.DataFrame())["evidence_state"] == "no_matched_test"


def test_conflicting_contexts_are_not_majority_voted() -> None:
    """Many helpful contexts do not erase an opposing conditional result."""
    rows = pd.DataFrame([
        {"conditional_finding": "helpful_in_this_context", "seed_ic_wins": 3,
         "seed_nll_wins": 3, "seed_sharpe_wins": 3}] * 5 + [
        {"conditional_finding": "harmful_in_this_context", "seed_ic_wins": 0,
         "seed_nll_wins": 0, "seed_sharpe_wins": 0}])
    result = synthesis.context_summary(rows)
    assert result["evidence_state"] == "conflicting_contexts"
    assert result["all_seed_all_objective_helpful_contexts"] == 5
    assert result["all_seed_all_objective_harmful_contexts"] == 1


def test_window_diagnostics_separate_coverage_from_width() -> None:
    """A narrower under-covering forecast cannot masquerade as calibrated uncertainty."""
    rows = pd.DataFrame({"date": ["2025-01-31"] * 4,
                         "asset_id": ["SPY", "MGK", "VTV", "IWM"],
                         "y_true": [0.01, 0.02, 0.03, 0.04],
                         "y_pred": [0.0, 0.01, 0.02, 0.03], "y_std": [0.001] * 4})
    result = synthesis.window_diagnostics(rows).set_index("family")
    assert result.loc["us_equity", "coverage_95"] == 0
    assert result.loc["us_equity", "ic"] == pytest.approx(1)
    assert result.loc["us_equity", "mean_interval_width_95"] == pytest.approx(0.003919927969080108)


def test_pooled_ic_is_not_an_average_of_component_ics() -> None:
    """Strong between-family ordering can coexist with failed within-family ordering."""
    rows = pd.DataFrame({"date": ["2025-01-31"] * 8,
                         "asset_id": ["SPY", "MGK", "VTV", "IWM", "VEA", "VWO", "VSS", "EWX"],
                         "y_true": [0.04, 0.03, 0.02, 0.01, 0.08, 0.07, 0.06, 0.05],
                         "y_pred": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08], "y_std": [0.02] * 8})
    result = synthesis.window_diagnostics(rows).set_index("family")
    assert result.loc["us_equity", "ic"] == pytest.approx(-1)
    assert result.loc["international_equity", "ic"] == pytest.approx(-1)
    assert result.loc["pooled", "ic"] > 0
