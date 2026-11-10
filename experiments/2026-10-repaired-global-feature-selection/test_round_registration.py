"""Offline checks that hosted reasoning resolves to novel, frozen, bounded plans.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_round_registration.py
"""

from __future__ import annotations

import copy
import json

import pytest
import register_round as registrar
import workflow


def test_operation_cannot_silently_drop_absent_feature() -> None:
    """Misspelled removals must not become undocumented no-op experiments."""
    with pytest.raises(ValueError, match="absent features"):
        registrar.apply_operations(["mom6m"], [], ["mom12m"], None)


def test_replacement_cannot_hide_extra_operations() -> None:
    """A representation transfer is explicit and cannot also silently add variables."""
    with pytest.raises(ValueError, match="replacement block"):
        registrar.apply_operations(["mom6m"], ["vol_1m"], [], ["chmom", "mom12m"])


def test_registered_round_has_twelve_new_three_seed_configurations() -> None:
    """The accepted family cap and saved evidence both hold for the actual first round."""
    path = registrar.data.HERE / "plans/10_international_equity_round1.json"
    config = json.loads(path.read_text())
    assert config["round_index"] == 1
    assert len(config["configurations"]) == 12
    assert config["default_seeds"] == [27, 127, 227]
    assert len({item["id"] for item in config["configurations"]}) == 12
    workflow.validate_plan(config, path, workflow.validate_frozen_data())


def test_changed_evidence_hash_is_rejected() -> None:
    """A planner cannot reinterpret later results as the original preregistration evidence."""
    path = registrar.data.HERE / "plans/10_international_equity_round1.json"
    config = copy.deepcopy(json.loads(path.read_text()))
    first = config["evidence_reports"][0]
    config["evidence_sha256"][first] = "0" * 64
    with pytest.raises(ValueError, match="evidence changed"):
        workflow.validate_plan(config, path, workflow.validate_frozen_data())


def test_actual_global_initial_union_matches_disclosed_family_sources() -> None:
    """The joint union must be literal and auditable, not a mislabeled hand-picked mask."""
    proposal = json.loads((registrar.data.HERE / "planning/global_round0.json").read_text())
    plan_path = registrar.data.HERE / "plans/05_global_round0.json"
    config = json.loads(plan_path.read_text())
    source = next(registrar.data.ROOT / path for path in config["evidence_reports"]
                  if path.endswith("component_exact_inputs.json"))
    inputs = json.loads(source.read_text())
    union = set().union(*(inputs[family][identifier] for family, identifier in
                          proposal["family_representatives_for_literal_union"].items()))
    literal = next(item for item in config["configurations"] if item["hypothesis_name"] ==
                   "literal_selected_family_union_29")
    assert set(literal["expected_input_columns"]) == union
    assert len(union) == 30  # Twenty-nine non-time axes plus the always-retained time index.
    assert len(config["configurations"]) == 6
    assert config["study_stage"] == "global_initial" and config["round_index"] == 0
    workflow.validate_plan(config, plan_path, workflow.validate_frozen_data())


def test_actual_global_initial_lag_pair_differs_only_in_lag() -> None:
    """The announced global lag contrast may not hide other representation changes."""
    config = json.loads((registrar.data.HERE / "plans/05_global_round0.json").read_text())
    items = {item["hypothesis_name"]: set(item["expected_input_columns"]) for item in config["configurations"]}
    without = items["shared_core_plus_fixed_income_exceptions_14"]
    with_lag = items["shared_core_plus_fixed_income_exceptions_and_lag_15"]
    assert with_lag - without == {"lag_y_excess_lead"}
    assert without - with_lag == set()
    assert "t_index" in without
