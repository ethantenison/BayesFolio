"""Regression checks for legitimate empty local blocks and unchanged old masks.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_candidate_masks.py
"""

from __future__ import annotations

import json

import pytest
import run_study as study
import workflow
from candidate_masks import research_candidate
from register_pool_history import pool_candidate, pool_config


def test_normal_mask_contract_is_byte_equivalent() -> None:
    """Admitting macro-only baselines must not relabel or invalidate completed local masks."""
    args = ("global", ["mom6m", "mom12m_skip1m"], ["oil_ret", "hy_spread"], ["test"])
    assert research_candidate(*args) == study.candidate(*args)


def test_macro_only_mask_contains_no_placeholder_local_variable() -> None:
    """An omitted block is not simulated by a zero feature or a dummy local input."""
    item = research_candidate("us_equity", [], ["oil_ret"], ["test"])
    assert item["etf_columns"] == []
    assert item["expected_input_columns"] == ["t_index", "oil_ret"]
    index = item["args"].index("--custom-etf-cols")
    assert item["args"][index + 1] == ""


def test_unaudited_macro_only_baseline_cannot_bypass_adaptive_cap() -> None:
    """Empty local blocks do not create a new baseline-name budget loophole."""
    item = pool_candidate("us_equity", [], ["oil_ret"], ["new synthetic mask"])
    config = pool_config("us_equity", [item], "test")
    with pytest.raises(ValueError, match="cannot bypass adaptive limits"):
        workflow.validate_plan(config, study.HERE / "plans/00_fake_macro_only.json", workflow.validate_frozen_data())


def test_restored_historical_masks_are_deduplicated_and_capped_outside_adaptive_rounds() -> None:
    """Exactly the reviewed two old masks are restored, without adding another family round."""
    audit = json.loads((study.HERE / "results/historical_macro_only_audit.json").read_text())
    assert len(audit["records"]) == 5
    assert audit["new_historical_masks"] == {"us_equity": 1, "international_equity": 1}
    for family in audit["new_historical_masks"]:
        path = study.HERE / "plans" / f"00_{family}_macro_only_historical_extension.json"
        config = json.loads(path.read_text())
        assert len(config["configurations"]) == 1
        assert config["default_seeds"] == [27, 127, 227]
        assert config["configurations"][0]["etf_columns"] == []
        assert workflow.plan_stage(config, path) == "historical_rebaseline"
        workflow.validate_plan(config, path, workflow.validate_frozen_data())
