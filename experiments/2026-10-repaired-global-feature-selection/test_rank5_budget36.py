"""Guard rank-five budget-only commands and observational optimizer logging."""

from __future__ import annotations

import json
from types import SimpleNamespace

import log_fit_budget36 as hook
import numpy as np
import pytest
import run_rank5_budget36 as budget


def _command() -> list[str]:
    return ["python", "runner", "--mtgp-rank", "5", "--maxiter", "15", "--output-root", "/baseline",
            "--run-id", f"{budget.prior.IDENTIFIER}_s27", "--seed", "27"]


def test_only_budget_and_output_identity_change() -> None:
    """Rank, seed and all other scientific arguments must remain untouched."""
    old = _command()
    new = budget.larger_budget_command(old)
    assert old[5] == "15" and new[5] == "36" and new[3] == "5"
    assert new[9] == f"{budget.IDENTIFIER}_s27"
    assert all(old[i] == new[i] for i in range(len(old)) if i not in [5, 7, 9])


def test_wrong_baseline_budget_rejected() -> None:
    """The budget experiment cannot silently start from another fitting cap."""
    old = _command()
    old[5] = "50"
    with pytest.raises(ValueError, match="maxiter fifteen"):
        budget.larger_budget_command(old)


def test_duplicate_rank_rejected() -> None:
    """Repeated flags must not conceal a last-argument rank override."""
    with pytest.raises(ValueError, match="duplicate"):
        budget.larger_budget_command([*_command(), "--mtgp-rank", "2"])


def test_logging_preserves_optimizer_inputs_and_return(tmp_path, monkeypatch) -> None:
    """Observation must delegate the same options/closure and return the identical result."""
    log = tmp_path / "fitting.jsonl"
    factor = SimpleNamespace(covar_factor=np.empty((15, 5)))
    model = SimpleNamespace(train_inputs=[np.empty((1260, 16))],
                            covar_module=SimpleNamespace(kernels=[None, factor]))
    mll = SimpleNamespace(model=model)
    result = SimpleNamespace(step=14, fval=0.42, status=SimpleNamespace(name="SUCCESS"),
                             message="converged", runtime=1.0)
    closure = object()

    def optimizer(actual, **kwargs):
        assert actual is mll and kwargs["closure"] is closure and kwargs["options"] == {"maxiter": 36}
        return result

    monkeypatch.setattr(hook, "fit_gpytorch_mll_scipy", optimizer)
    assert hook.observed_optimizer(log, ["2024-05-31"], 27)(
        mll, closure=closure, options={"maxiter": 36}) is result
    row = json.loads(log.read_text())
    assert row["status"] == "SUCCESS" and row["iterations"] == 14 and row["date"] == "2024-05-31"


def test_observer_rejects_wrong_rank(tmp_path) -> None:
    """Actual task factor dimensions are checked before any optimization."""
    model = SimpleNamespace(train_inputs=[np.empty((1260, 16))],
        covar_module=SimpleNamespace(kernels=[None, SimpleNamespace(covar_factor=np.empty((15, 2)))]))
    with pytest.raises(ValueError, match="rank five"):
        hook.observed_optimizer(tmp_path / "bad.jsonl", ["2024-05-31"], 27)(
            SimpleNamespace(model=model), options={"maxiter": 36})
