"""Tests for the portfolio optimization command contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bayesfolio.contracts.base import SchemaName
from bayesfolio.contracts.commands.optimize import OptimizeCommand


def _command(**overrides: object) -> OptimizeCommand:
    values = {"objective": "Sharpe", "risk_measure": "CVaR", **overrides}
    return OptimizeCommand.model_validate(values)


def test_optimize_command_identity_defaults_and_extra_forbid() -> None:
    command = _command()

    assert command.schema is SchemaName.OPTIMIZE_COMMAND
    assert command.schema_version == "0.1.0"
    assert command.alpha == 0.05

    with pytest.raises(ValidationError):
        _command(unknown_setting=True)


def test_optimize_command_alpha_round_trip() -> None:
    command = _command(alpha=0.5)

    restored = OptimizeCommand.model_validate(command.model_dump(mode="json"))

    assert restored == command
    assert restored.alpha == 0.5


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.1])
def test_optimize_command_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValidationError):
        _command(alpha=alpha)
