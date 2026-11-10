"""Guard the separately authorized rank-five command's single-factor change."""

from __future__ import annotations

import pytest
import run_rank5_comparison as rank5


def _command() -> list[str]:
    return ["python", "runner.py", "--mtgp-rank", "2", "--output-root", "/old",
            "--run-id", f"{rank5.BASE_ID}_s27", "--maxiter", "15", "--seed", "27"]


def test_rank_only_scientific_change() -> None:
    """Only rank and output identity may change; the original command stays intact."""
    original = _command()
    result = rank5.rank5_command(original)
    assert result[3] == "5" and original[3] == "2"
    for index in range(len(original)):
        if index not in [3, 5, 7]:
            assert result[index] == original[index]
    assert result[7] == f"{rank5.IDENTIFIER}_s27"


def test_wrong_rank_rejected() -> None:
    """Reject attempts to rebase the comparison onto a different starting rank."""
    command = _command()
    command[3] = "3"
    with pytest.raises(ValueError, match="rank-two baseline"):
        rank5.rank5_command(command)


def test_duplicate_rank_rejected() -> None:
    """Duplicate flags must not permit hidden last-argument overrides."""
    with pytest.raises(ValueError, match="duplicated"):
        rank5.rank5_command([*_command(), "--mtgp-rank", "2"])


def test_other_mask_or_seed_rejected() -> None:
    """The rank change is restricted to the approved mask and three existing seeds."""
    command = _command()
    command[7] = "other_mask_s999"
    with pytest.raises(ValueError, match="Unexpected baseline"):
        rank5.rank5_command(command)
