"""Frozen primary-objective and confirmation acceptance regression tests."""

from __future__ import annotations

import v2_driver
from v2_finish import acceptance


def _fixture(monkeypatch):
    monkeypatch.setattr(v2_driver, "read", lambda _: {
        "aspiration_CAGR": .10, "aspiration_drawdown_loss": .10})
    summary = {"cagr": .15, "max_drawdown": -.08, "rank_ic": -.50}
    candidate = {"complete": True, "summary": summary, "latest12": summary}
    reference = {"complete": True, "summary": {"cagr": .08, "max_drawdown": -.12},
                 "latest12": {"cagr": .08, "max_drawdown": -.12}}
    return candidate, reference


def test_IC_is_not_an_automatic_portfolio_veto(monkeypatch) -> None:
    """A strong return/drawdown improvement can qualify despite a lower rank IC."""
    candidate, control = _fixture(monkeypatch)
    result = acceptance(candidate, {"eligible": True}, best_control_full=control,
                        best_control_latest=control, live_feasible=True)
    assert result["eligible"]


def test_confirmation_cannot_rescue_failed_development(monkeypatch) -> None:
    """The locked block cannot promote a policy rejected before it was opened."""
    candidate, control = _fixture(monkeypatch)
    result = acceptance(candidate, {"eligible": False}, best_control_full=control,
                        best_control_latest=control, live_feasible=True)
    assert not result["eligible"]
    assert not result["gates"]["development_eligible_before_unlock"]


def test_live_infeasibility_cannot_be_replaced_by_old_weights(monkeypatch) -> None:
    """A historical winner with no valid November solve is not eligible."""
    candidate, control = _fixture(monkeypatch)
    result = acceptance(candidate, {"eligible": True}, best_control_full=control,
                        best_control_latest=control, live_feasible=False)
    assert not result["eligible"]


def test_large_confirmation_drawdown_fails_even_with_high_growth(monkeypatch) -> None:
    """The fixed latest-block loss tolerance cannot be traded away after scoring."""
    candidate, control = _fixture(monkeypatch)
    candidate["latest12"] = {"cagr": .50, "max_drawdown": -.30}
    result = acceptance(candidate, {"eligible": True}, best_control_full=control,
                        best_control_latest=control, live_feasible=True)
    assert not result["eligible"]
    assert not result["gates"]["latest12_drawdown_loss_within_best_control_plus3pp"]
