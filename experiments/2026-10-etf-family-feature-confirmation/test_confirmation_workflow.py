"""Focused tests for hypervolume and the gated Qwen planning workflow."""

from __future__ import annotations

import pandas as pd
import pytest
from prepare_next_experiment_turn import validate_step3_results
from summarize_us_confirmation import hypervolume_3d
from validate_next_experiment_plan import validate_candidate


def test_hypervolume_3d_handles_overlapping_boxes() -> None:
    """The sweep should count overlapping dominated volume exactly once."""
    assert hypervolume_3d([(1.0, 1.0, 1.0)], (0.0, 0.0, 0.0)) == 1.0
    assert hypervolume_3d([(1.0, 1.0, 1.0), (2.0, 0.5, 1.0)], (0.0, 0.0, 0.0)) == 1.5
    assert hypervolume_3d([(1.0, 2.0, 1.0), (2.0, 1.0, 1.0)], (0.0, 0.0, 0.0)) == 3.0


def test_step3_gate_requires_complete_three_seed_panel() -> None:
    """Qwen planning must not start from partial confirmation results."""
    run_objectives = pd.DataFrame({"seed": [27] * 4})
    aggregates = pd.DataFrame({"seed_count": [1] * 4, "validation_passed": [True] * 4})
    with pytest.raises(ValueError, match="not ready"):
        validate_step3_results(
            seeds=[27, 127, 227],
            run_objectives=run_objectives,
            aggregates=aggregates,
            validation={"seeds": [27], "all_passed": True, "failures": []},
            hypervolume={"schema": "bayesfolio.etf_family_confirmation_hypervolume.v1", "value": 0.5},
        )


def test_step3_gate_accepts_complete_three_seed_panel() -> None:
    """A valid 12-run aggregate should pass the model-turn gate."""
    run_objectives = pd.DataFrame({"seed": [27, 127, 227] * 4})
    aggregates = pd.DataFrame({"seed_count": [3] * 4, "validation_passed": [True] * 4})
    validate_step3_results(
        seeds=[27, 127, 227],
        run_objectives=run_objectives,
        aggregates=aggregates,
        validation={"seeds": [27, 127, 227], "all_passed": True, "failures": []},
        hypervolume={"schema": "bayesfolio.etf_family_confirmation_hypervolume.v1", "value": 0.5},
    )


def test_candidate_validator_accepts_fixed_controls() -> None:
    """A bounded candidate with exact controls and allowlisted evidence should pass."""
    evidence_path = "experiments/2026-10-etf-family-feature-confirmation/PLAN.md"
    candidate = {
        "id": "test_candidate",
        "mode": "explore",
        "priority": 1,
        "parent_configurations": ["U-AC"],
        "hypothesis": "A coherent cached block may improve calibration.",
        "frontier_gap_targeted": ["mean_gaussian_nll"],
        "exact_feature_changes": {"add": ["cached_feature"], "remove": []},
        "exact_model_changes": [],
        "exact_runner_args": [
            "--mtgp-rank",
            "2",
            "--maxiter",
            "15",
            "--posterior-scenarios",
            "256",
        ],
        "cached_columns_required": ["cached_feature"],
        "evidence": [{"path": evidence_path, "finding": "The plan permits coherent cached feature blocks."}],
        "expected_direction": {
            "mean_ic": "uncertain",
            "mean_gaussian_nll": "decrease",
            "sharpe": "uncertain",
        },
        "falsification_condition": "The configuration is dominated on all three objectives.",
        "risk": "The added feature may increase noise.",
        "compute": {"seeds": [27, 127, 227], "windows": 24, "run_count": 3},
    }
    assert validate_candidate(candidate, allowlist={evidence_path}) == []
