"""Synthetic development-only checks for frozen November robustness criteria.

Fixtures have a neutral initial five-month partial block and three complete
annual blocks. No real candidate or confirmation results are read. Run with:
    uv run --locked --no-sync pytest -q experiments/2026-11-portfolio-multiobjective-v2/test_v2_robustness.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from v2_robustness import assess

ANCHORS = {"aspiration_CAGR": .20, "aspiration_drawdown_loss": .10, "CAGR_scale": .02, "drawdown_scale": .02}


def _record(tmp_path, name: str, returns: np.ndarray) -> dict:
    path = tmp_path / (name + ".npz")
    dates = pd.date_range("2022-05-31", periods=len(returns), freq="BME").strftime("%Y-%m-%d").to_numpy(dtype="U10")
    np.savez_compressed(path, total_returns=returns, rf=np.zeros(len(returns)), rank_ic=np.full(len(returns), .1),
                        cost_10_total_returns=returns - .0001, dates=dates)
    return {"id": name, "complete": True, "feasibility_complete": True,
            "numerical": [{"passed": True} for _ in range(len(returns))], "artifact_path": str(path)}


def test_fixed_superior_candidate_retains_neutral_prefix_and_passes(tmp_path) -> None:
    """Forty-one-month distance and all checks use a fixed paired reference."""
    candidate_returns = np.r_[np.zeros(5), np.tile([.015, .005, .01], 12)]
    reference_returns = np.r_[np.zeros(5), np.tile([-.02, .005, .01], 12)]
    candidate = _record(tmp_path, "candidate", candidate_returns)
    reference = _record(tmp_path, "reference", reference_returns)
    result = assess(candidate, reference, ANCHORS, draws=64, seed=17)
    expected_cagr = np.prod(1 + candidate_returns) ** (12 / 41) - 1
    assert result["candidate_distance"] == pytest.approx((.20 - expected_cagr) / .02, abs=1e-12)
    assert result["eligible"] is True
    assert result["months"] == 41
    assert result["confirmation_used"] is False
    assert result["fixed_reference_id"] == "reference"
    assert result["annual_blocks_within_tolerance"] == 3
    assert result["bootstrap"]["3"]["distance_gain_fraction"] >= .60
    assert len(result["leave_one_month_out"]["distance_gains"]) == 41
    assert result == assess(candidate, reference, ANCHORS, draws=64, seed=17)


def test_identical_policy_does_not_manufacture_promotion(tmp_path) -> None:
    """A neutral tie has no positive gain and descriptive stress frequencies only."""
    returns = np.r_[np.zeros(5), np.tile([-.02, .005, .01], 12)]
    candidate = _record(tmp_path, "candidate", returns)
    reference = _record(tmp_path, "reference", returns)
    result = assess(candidate, reference, ANCHORS, draws=64, seed=17)
    assert result["distance_gain"] == pytest.approx(0)
    assert result["bootstrap"]["3"]["distance_gain_fraction"] == pytest.approx(1)
    assert result["bootstrap"]["3"]["positive_distance_gain_fraction"] == pytest.approx(0)
    assert result["leave_one_month_out"]["nonnegative_gain_fraction"] == pytest.approx(1)
    assert result["leave_one_month_out"]["positive_gain_fraction"] == pytest.approx(0)
    assert result["eligible"] is False
    assert "not future crash probabilities" in result["interpretation"]
    assert set(result["bootstrap"]["6"]["candidate_drawdown_exceedance_fractions"]) == {"0.1", "0.15", "0.2"}


def test_incomplete_or_confirmation_paths_cannot_be_scored(tmp_path) -> None:
    """Infeasibility stops before reading and a full53 artifact is rejected."""
    returns = np.r_[np.zeros(5), np.tile([-.02, .005, .01], 12)]
    reference = _record(tmp_path, "reference", returns)
    incomplete = {"id": "missing", "complete": False, "feasibility_complete": False}
    result = assess(incomplete, reference, ANCHORS, draws=16, seed=3)
    assert result["gates"] == {"feasibility_complete": False}
    full = _record(tmp_path, "full53", np.r_[returns, np.zeros(12)])
    full["numerical"] = full["numerical"][:41]
    with pytest.raises(ValueError, match="exactly41"):
        assess(full, reference, ANCHORS, draws=16, seed=3)
