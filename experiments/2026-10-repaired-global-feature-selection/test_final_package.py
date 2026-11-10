"""Synthetic guards for explicit final research decisions and source freshness.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_final_package.py
"""

from __future__ import annotations

import pytest
from package_final_results import validate_decisions, validate_sources


def test_off_front_recommendation_is_rejected() -> None:
    """A named favorite cannot bypass the actual final Pareto evidence."""
    with pytest.raises(ValueError, match="outside"):
        validate_decisions({"production_promotion": False, "selected_configurations": {"global": {"lead": "x"}},
                            "shared_research_lead": "x"}, {"global": {"y"}})


def test_report_cannot_implicitly_promote_production() -> None:
    """The research scope does not authorize deployment."""
    with pytest.raises(ValueError, match="production"):
        validate_decisions({"production_promotion": True}, {})


def test_explicit_frontier_research_choice_is_allowed() -> None:
    """A disclosed axis preference is a recommendation, not a new weighted objective."""
    validate_decisions({"production_promotion": False, "selected_configurations": {"global": {"lead": "x"}},
                        "shared_research_lead": "x"}, {"global": {"x"}})


def test_source_hashes_are_required(tmp_path) -> None:
    """A prose success flag without auditable sources is insufficient."""
    with pytest.raises(ValueError, match="source-hashed"):
        validate_sources({}, tmp_path)
