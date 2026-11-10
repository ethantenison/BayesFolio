"""Offline tests for experiment budgets, kernel invariants, and historical masks.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import workflow  # noqa: E402


@pytest.fixture
def baseline() -> tuple[dict, Path, set[str]]:
    """Load the immutable global control and its approved column names."""
    path = HERE / "plans/00_global_all_input_control.json"
    config = json.loads(path.read_text())
    columns = set(json.loads((workflow.data.ARTIFACTS / "global.manifest.json").read_text())["required_columns"])
    return config, path, columns


def test_registered_baseline_passes(baseline: tuple[dict, Path, set[str]]) -> None:
    """The registered all-input control follows the current comparison contract."""
    workflow.validate_plan(*baseline)


def test_kernel_changes_cannot_hide_in_features(baseline: tuple[dict, Path, set[str]]) -> None:
    """A candidate cannot smuggle interaction tuning into a feature-selection batch."""
    config, path, columns = copy.deepcopy(baseline)
    config["configurations"][0]["args"].append("--interactions-only")
    with pytest.raises(ValueError, match="constraint"):
        workflow.validate_plan(config, path, columns)


def test_all_three_seeds_required(baseline: tuple[dict, Path, set[str]]) -> None:
    """A one-seed smoke result cannot count as completed three-seed evidence."""
    config, path, columns = copy.deepcopy(baseline)
    config["default_seeds"] = [27]
    with pytest.raises(ValueError, match="three seeds"):
        workflow.validate_plan(config, path, columns)


def test_global_round_cap_stays_six(baseline: tuple[dict, Path, set[str]]) -> None:
    """The family-only amendment does not authorize 12 global candidates."""
    config, path, columns = copy.deepcopy(baseline)
    config["round_index"] = 1
    config["configurations"] *= 7
    with pytest.raises(ValueError, match="caps"):
        workflow.validate_plan(config, path.with_name("10_global_round1.json"), columns)


def test_adaptive_round_needs_evidence(baseline: tuple[dict, Path, set[str]]) -> None:
    """A later batch requires an evidence trail before its fits can launch."""
    config, path, columns = copy.deepcopy(baseline)
    config["round_index"] = 1
    with pytest.raises(ValueError, match="interpreted deterministic evidence"):
        workflow.validate_plan(config, path.with_name("10_global_round1.json"), columns)


def test_verified_aliases_deduplicate() -> None:
    """Equivalent monthly names do not become two predictors or two experiments."""
    assert workflow.study.canonical(["hy_spread_z_12m", "hy_spread_z_12p"]) == ["hy_spread_z_12p"]


def test_all_input_baseline_matches_current_production_feature_lists() -> None:
    """The comparator preserves actual production inputs, not a remembered historical label."""
    from bayesfolio.contracts.commands.monthly_portfolio import MonthlyForecastConfig

    current = MonthlyForecastConfig()
    assert current.time_feature_columns == ["t_index"]
    assert current.etf_feature_columns == workflow.study.BASE_ETF
    assert current.macro_feature_columns == workflow.study.BASE_MACRO


def test_new_mask_cannot_evade_round_cap_with_baseline_name() -> None:
    """Only previously audited masks receive the historical-rebaseline budget exemption."""
    from register_pool_history import pool_candidate, pool_config

    item = pool_candidate("us_equity", ["mom6m", "vol_accel"], ["oil"], ["New synthetic mask"])
    config = pool_config("us_equity", [item], "test")
    with pytest.raises(ValueError, match="cannot bypass adaptive limits"):
        workflow.validate_plan(config, HERE / "plans/00_fake_baseline.json", workflow.validate_frozen_data())


def test_initial_global_comparison_has_separate_bounded_round_zero(baseline: tuple[dict, Path, set[str]]) -> None:
    """The initial joint hypotheses do not consume or silently expand the three adaptive rounds."""
    config, path, columns = copy.deepcopy(baseline)
    config["study_stage"] = "global_initial"
    config["round_index"] = 0
    config["evidence_reports"] = [str((HERE / "results/REBASELINE_INTERIM_INTERPRETATION.md").relative_to(
        workflow.data.ROOT))]
    config["round_interpretation"] = "Synthetic validation of the explicit initial-global stage"
    config["configurations"][0]["hypothesis"] = "A preregistered shared-input hypothesis is evaluated in the joint GP."
    workflow.validate_plan(config, path, columns)
