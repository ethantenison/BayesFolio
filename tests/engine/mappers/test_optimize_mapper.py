"""Tests for optimization contract mappings."""

from __future__ import annotations

from bayesfolio.contracts.commands.optimize import OptimizeCommand
from bayesfolio.engine.mappers.optimize import command_to_engine_inputs


def test_command_to_engine_inputs_includes_alpha() -> None:
    command = OptimizeCommand(
        objective="Sharpe",
        risk_measure="CVaR",
        alpha=0.5,
        min_weight=0.01,
        max_weight=0.25,
        nea=10,
        hist=False,
    )

    assert command_to_engine_inputs(command) == {
        "objective": "Sharpe",
        "risk_measure": "CVaR",
        "alpha": 0.5,
        "min_weight": 0.01,
        "max_weight": 0.25,
        "nea": 10,
        "hist": False,
    }
