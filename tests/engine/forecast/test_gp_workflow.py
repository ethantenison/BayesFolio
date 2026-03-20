from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from bayesfolio.engine.forecast.gp.workflow import run_planned_multitask_gp_workflow


def _make_workflow_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for asset_idx, asset_id in enumerate(["AAA", "BBB"]):
        for step in range(8):
            rows.append(
                {
                    "macro_1": 0.1 * step,
                    "macro_2": 0.2 * step + 0.05 * asset_idx,
                    "etf_1": 0.15 * step + 0.02 * asset_idx,
                    "y_excess_lead": 0.01 + 0.002 * step + 0.001 * asset_idx,
                    "asset_id": asset_id,
                }
            )
    return pd.DataFrame(rows)


def test_run_planned_multitask_gp_workflow_smoke(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    artifacts = run_planned_multitask_gp_workflow(
        df=_make_workflow_df(),
        target_column="y_excess_lead",
        task_column="asset_id",
        feature_groups={"macro": ["macro_1", "macro_2"], "etf": ["etf_1"]},
        instruction_text="Use a matern kernel with ard for all input variables",
    )

    assert artifacts.result.final_status == "ok"
    assert artifacts.result.fit_validation.fit_success is True
    assert artifacts.result.fit_validation.prediction_success is True
    assert artifacts.model is not None


def test_run_planned_multitask_gp_workflow_records_repair_attempt(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    call_count = {"value": 0}

    class _FakeLikelihood:
        def eval(self) -> None:
            return None

    class _FakeModel:
        likelihood = _FakeLikelihood()

        def train(self) -> None:
            return None

        def eval(self) -> None:
            return None

        def posterior(self, train_x: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(mean=torch.zeros((train_x.shape[0], 1), dtype=train_x.dtype))

    def _flaky_build(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise RuntimeError("synthetic build failure")
        return _FakeModel()

    monkeypatch.setattr("bayesfolio.engine.forecast.gp.workflow.build_multitask_gp", _flaky_build)
    monkeypatch.setattr(
        "bayesfolio.engine.forecast.gp.workflow.ExactMarginalLogLikelihood",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr("bayesfolio.engine.forecast.gp.workflow.fit_gpytorch_mll", lambda *args, **kwargs: None)

    artifacts = run_planned_multitask_gp_workflow(
        df=_make_workflow_df(),
        target_column="y_excess_lead",
        task_column="asset_id",
        feature_groups={"macro": ["macro_1", "macro_2"], "etf": ["etf_1"]},
        instruction_text="Use a matern kernel with ard for all input variables",
    )

    assert artifacts.result.final_status == "ok"
    assert artifacts.result.repair_attempts
    assert artifacts.result.repair_attempts[0].action == "increase_noise_floor"
    assert artifacts.result.repair_attempts[0].status == "applied"


def test_run_planned_multitask_gp_workflow_rejects_null_task_values() -> None:
    df = _make_workflow_df()
    df.loc[0, "asset_id"] = None

    with pytest.raises(ValueError, match="contains null values"):
        run_planned_multitask_gp_workflow(
            df=df,
            target_column="y_excess_lead",
            task_column="asset_id",
            feature_groups={"macro": ["macro_1", "macro_2"], "etf": ["etf_1"]},
            instruction_text="Use a matern kernel with ard for all input variables",
        )
