"""Synthetic checks that execution audit cannot close an unfinished adaptive sequence.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_execution_coverage.py
"""

from __future__ import annotations

from audit_execution_coverage import adaptive_closure


def test_one_stalled_round_is_not_completion() -> None:
    """A single plateau cannot silently replace the requested loop."""
    closed, reason = adaptive_closure({1: {"stalled": True}}, 3)
    assert not closed and "2" in reason


def test_two_consecutive_stalled_rounds_close_early() -> None:
    """The explicit early-stop rule does not require an unnecessary third round."""
    assert adaptive_closure({1: {"stalled": True}, 2: {"stalled": True}}, 3)[0]


def test_missing_middle_round_is_not_hidden_by_a_final_report() -> None:
    """A round-three artifact cannot substitute for actually completing round two."""
    assert not adaptive_closure({1: {"stalled": False}, 3: {"stalled": False}}, 3)[0]


def test_round_zero_cannot_count_as_adaptive_work() -> None:
    """The initial comparison is outside the three-round adaptive budget."""
    assert not adaptive_closure({0: {"stalled": False}, 1: {"stalled": False}}, 3)[0]


def test_work_after_early_stop_is_detected() -> None:
    """The cap is not permission to ignore the two-plateau stopping condition."""
    assert not adaptive_closure({1: {"stalled": True}, 2: {"stalled": True}, 3: {"stalled": False}}, 3)[0]


def test_three_improving_rounds_close_at_budget_not_convergence() -> None:
    """A positive final round still ends at the accepted cap."""
    closed, reason = adaptive_closure({index: {"stalled": False} for index in [1, 2, 3]}, 3)
    assert closed and "no convergence" in reason
