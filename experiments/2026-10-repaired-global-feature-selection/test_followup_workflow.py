"""Check isolated follow-up execution, immutable evidence and authorized masks.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_followup_workflow.py
"""

from __future__ import annotations

import json

import followup_workflow as followup
import pytest
import run_study as study


def test_execution_scope_restores_original_globals() -> None:
    """The output adapter cannot change the completed campaign outside its scope."""
    before = study.HERE, study.code_hashes
    with followup.execution_scope():
        assert study.HERE == followup.HERE
        assert followup.data.HERE == followup.BASE
        hashes = study.code_hashes()
        assert str(followup.CONTRACT.relative_to(followup.data.ROOT)) in hashes
        config = study.phase_config("global", [], "scope_test")
        assert config["output_root"] == str((followup.HERE / "runs").relative_to(followup.data.ROOT))
    assert (study.HERE, study.code_hashes) == before


def test_execution_scope_restores_after_error() -> None:
    """An interrupted adapter cannot leave legacy module paths redirected."""
    before = study.HERE, study.code_hashes
    with pytest.raises(RuntimeError), followup.execution_scope():
        raise RuntimeError("synthetic interruption")
    assert (study.HERE, study.code_hashes) == before


def test_authorized_masks_are_new_and_keep_time() -> None:
    """All twelve approved masks are distinct and novel against all original joint trials."""
    exact = json.loads((followup.BASE / "results/cumulative/global/exact_inputs.json").read_text())
    contract = json.loads(followup.CONTRACT.read_text())
    parent = exact[contract["base_configuration"]]
    seen = {frozenset(columns) for columns in exact.values()}
    counts = []
    for batch in contract["batches"]:
        batch_counts = []
        for hypothesis in batch:
            item = followup.candidate_for(hypothesis, parent, [])
            mask = frozenset(item["expected_input_columns"])
            assert "t_index" in mask and mask not in seen
            seen.add(mask)
            batch_counts.append(len(mask) - 1)
        counts.append(batch_counts)
    assert counts == [[14] * 6, [15, 15, 12, 13, 13, 12]]


def test_immutable_registration_rejects_changed_content(tmp_path) -> None:
    """Resume cannot quietly replace a frozen plan or evidence snapshot."""
    path = tmp_path / "registration.json"
    followup.immutable_json(path, {"value": 1})
    followup.immutable_json(path, {"value": 1})
    with pytest.raises(ValueError, match="Refusing"):
        followup.immutable_json(path, {"value": 2})


def test_snapshot_rejects_changed_source(tmp_path) -> None:
    """Changed upstream evidence needs a new registration, not an overwrite."""
    source, target = tmp_path / "source.json", tmp_path / "frozen.json"
    followup.data.write_json(source, {"version": 1})
    followup.snapshot(source, target)
    followup.data.write_json(source, {"version": 2})
    with pytest.raises(ValueError, match="Evidence changed"):
        followup.snapshot(source, target)
