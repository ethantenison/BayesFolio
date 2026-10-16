"""Focused tests for hypervolume and the gated Qwen planning workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from orchestrate_international_experiments import validate_config as validate_international_config
from orchestrate_international_experiments import validate_worker_count
from prepare_next_experiment_turn import validate_step3_results
from summarize_international_experiments import validate_run_manifest
from summarize_next_experiment_batch import add_promotion_results, objective_hypervolume
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


def test_next_batch_promotion_requires_frontier_and_hypervolume_gain() -> None:
    """Only a nondominated candidate with positive incumbent-relative gain is promoted."""
    reference = {"mean_ic": 0.0, "mean_gaussian_nll": 0.0, "sharpe": 0.0}
    incumbents = pd.DataFrame(
        [{"configuration": "incumbent", "mean_ic": 0.2, "mean_gaussian_nll": -1.0, "sharpe": 1.0}]
    )
    candidates = pd.DataFrame(
        [
            {"configuration": "improver", "mean_ic": 0.2, "mean_gaussian_nll": -1.1, "sharpe": 1.1},
            {"configuration": "dominated", "mean_ic": 0.1, "mean_gaussian_nll": -0.9, "sharpe": 0.9},
        ]
    )
    baseline = objective_hypervolume(incumbents, reference)

    result = add_promotion_results(
        candidates,
        incumbents,
        baseline_hypervolume=baseline,
        reference=reference,
    ).set_index("configuration")

    assert bool(result.loc["improver", "promoted"])
    assert result.loc["improver", "incremental_hypervolume"] > 0.0
    assert not bool(result.loc["dominated", "nondominated_vs_incumbents"])
    assert not bool(result.loc["dominated", "promoted"])


def test_international_configs_preserve_fixed_contract() -> None:
    """Both phases must retain the exact universe, objectives, and time input."""
    experiment_dir = Path(__file__).resolve().parent
    configs = {
        "international_confirmation_config.json": 4,
        "international_targeted_ablation_config.json": 4,
        "international_oil_only_config.json": 1,
        "international_ratio_only_config.json": 1,
        "international_oil_backward_selection_config.json": 7,
    }
    for name, expected_count in configs.items():
        path = experiment_dir / name
        config = json.loads(path.read_text(encoding="utf-8"))
        paths = validate_international_config(config, path)
        assert len(config["configurations"]) == expected_count
        assert paths["feature"].exists()
        assert all(item["expected_input_columns"][0] == "t_index" for item in config["configurations"])


def test_international_worker_count_is_bounded() -> None:
    """Fit-level parallelism must stay within the approved Mac Studio cap."""
    assert validate_worker_count(15) == 15
    with pytest.raises(ValueError, match="between 1 and 15"):
        validate_worker_count(0)
    with pytest.raises(ValueError, match="between 1 and 15"):
        validate_worker_count(16)


def test_international_manifest_validator_checks_exact_inputs() -> None:
    """A changed input list must fail deterministic run validation."""
    configuration = {
        "label": "I-MOM",
        "expected_input_columns": ["t_index", "mom6m"],
        "args": [
            "--feature-set",
            "momentum_trend",
            "--covar-grouping-proposal",
            "default",
            "--macro-component-proposal",
            "default",
            "--time-modulation-mode",
            "neither",
        ],
    }
    manifest = {
        "data": {
            "training_universe": ["VEA", "VWO", "VSS", "EWX"],
            "final_portfolio_universe": ["VEA", "VWO", "VSS", "EWX"],
        },
        "modeling": {
            "seed": 27,
            "rank": 2,
            "maxiter": 15,
            "posterior_scenarios_per_rebalance": 256,
            "experiment": "signed_lkj_eta_2",
            "time_modulation_mode": "neither",
            "kernel_composition_proposal": {"name": "none"},
        },
        "portfolio": {"n_realized_rebalances": 1},
        "rebalance_dates": ["2026-05-29"],
        "us_equity_first_round_wrapper": {
            "time_index_included": True,
            "input_columns": ["t_index", "trend_slope"],
            "feature_set": "momentum_trend",
            "covar_grouping_proposal": "default",
            "macro_component_proposal": "default",
        },
    }
    failures = validate_run_manifest(
        "I-MOM", configuration, manifest, ["2026-05-29"], ["VEA", "VWO", "VSS", "EWX"], expected_seed=27
    )
    assert failures == ["I-MOM: input_columns"]
