from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.models import MultiTaskGP, SingleTaskGP
from gpytorch.kernels import Kernel, ProductKernel
from gpytorch.kernels.index_kernel import IndexKernel

from bayesfolio.engine.forecast import (
    build_gp_interpretation_report,
    display_gp_interpretation_report,
    render_gp_interpretation_report,
)
from bayesfolio.engine.forecast.gp.reporting import (
    GPInterpretationReport,
    RenderedGPInterpretationReport,
    _describe_kernel_component,
)


class _PositiveIndexKernelLike:
    def __init__(self) -> None:
        self.active_dims = torch.tensor([58])
        self.num_tasks = 3
        self.raw_covar_factor = torch.ones((3, 2), dtype=torch.double)


def test_build_gp_interpretation_report_reports_single_task_lengthscales_in_model_units() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [10.0, 20.0, 30.0, 40.0],
            "feature_b": [5.0, 10.0, 15.0, 20.0],
            "target": [0.1, 0.2, 0.15, 0.3],
        }
    )
    train_x = torch.tensor(
        [
            [0.0, 1.0],
            [0.5, 2.0],
            [1.0, 3.0],
            [1.5, 4.0],
        ],
        dtype=torch.double,
    )
    train_y = torch.tensor([[0.1], [0.2], [0.15], [0.3]], dtype=torch.double)

    model = SingleTaskGP(train_X=train_x, train_Y=train_y)
    model.covar_module.lengthscale = torch.tensor([[0.25, 0.50]], dtype=torch.double)

    report: GPInterpretationReport = build_gp_interpretation_report(
        df=df,
        model=model,
        target_column="target",
        feature_columns=["feature_a", "feature_b"],
    )

    feature_summary = report["feature_summary"]
    by_feature = feature_summary.set_index("feature")

    assert by_feature.loc["feature_a", "lengthscale_model_units"] == pytest.approx(0.25)
    assert by_feature.loc["feature_b", "lengthscale_model_units"] == pytest.approx(0.50)
    assert "scaling_note" not in feature_summary.columns


def test_build_gp_interpretation_report_maps_multitask_outputs() -> None:
    torch.manual_seed(7)

    df = pd.DataFrame(
        {
            "feature_a": [10.0, 20.0, 10.0, 20.0],
            "asset_id": ["SPY", "SPY", "QQQ", "QQQ"],
            "target": [0.05, 0.06, 0.04, 0.05],
        }
    )
    train_x = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_y = torch.tensor([[0.05], [0.06], [0.04], [0.05]], dtype=torch.double)

    model = MultiTaskGP(train_X=train_x, train_Y=train_y, task_feature=1, rank=1)
    product_kernel = cast(ProductKernel, model.covar_module)
    feature_kernel = cast(Kernel, product_kernel.kernels[0])
    task_kernel = cast(IndexKernel, product_kernel.kernels[1])
    feature_kernel.lengthscale = torch.tensor([[0.4]], dtype=torch.double)
    task_kernel.covar_factor.data = torch.tensor([[0.8], [0.2]], dtype=torch.double)
    raw_var = torch.tensor([0.3, 0.7], dtype=torch.double)
    task_kernel.initialize(raw_var=raw_var)

    report: GPInterpretationReport = build_gp_interpretation_report(
        df=df,
        model=model,
        target_column="target",
        task_column="asset_id",
        feature_columns=["feature_a"],
    )

    feature_summary = report["feature_summary"]
    task_summary = report["task_summary"]
    task_covariance = report["task_covariance"]
    task_correlation = report["task_correlation"]

    task_ids = torch.arange(int(cast(Any, task_kernel).num_tasks), dtype=torch.long)
    task_inputs = torch.zeros((len(task_ids), train_x.shape[-1]), dtype=torch.long)
    task_inputs[:, int(cast(Any, model)._task_feature)] = task_ids
    expected_covariance = task_kernel(task_inputs, task_inputs).to_dense().detach().cpu().numpy()
    expected_scale = expected_covariance.diagonal().clip(min=1e-12) ** 0.5
    expected_correlation = expected_covariance / np.outer(expected_scale, expected_scale)

    assert feature_summary.loc[0, "lengthscale_model_units"] == pytest.approx(0.4)
    assert task_summary is not None
    assert task_covariance is not None
    assert list(task_summary["task_label"]) == ["SPY", "QQQ"]
    assert task_correlation is not None
    np.testing.assert_allclose(task_covariance.to_numpy(), expected_covariance)
    np.testing.assert_allclose(task_correlation.to_numpy(), expected_correlation)
    assert list(task_correlation.index) == ["SPY", "QQQ"]
    assert list(task_correlation.columns) == ["SPY", "QQQ"]


def test_describe_kernel_component_extracts_rank_for_positive_index_kernel() -> None:
    kernel = _PositiveIndexKernelLike()

    component = _describe_kernel_component(
        kernel=kernel,
        path="covar_module.kernels[1]",
        all_feature_dims=list(range(58)),
        task_feature_index=58,
    )

    assert component["kernel_type"] == "_PositiveIndexKernelLike"
    assert component["num_tasks"] == 3
    assert component["rank"] == 2


def test_render_gp_interpretation_report_returns_tables_and_heatmap() -> None:
    torch.manual_seed(7)

    df = pd.DataFrame(
        {
            "feature_a": [10.0, 20.0, 10.0, 20.0],
            "asset_id": ["SPY", "SPY", "QQQ", "QQQ"],
            "target": [0.05, 0.06, 0.04, 0.05],
        }
    )
    train_x = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_y = torch.tensor([[0.05], [0.06], [0.04], [0.05]], dtype=torch.double)

    model = MultiTaskGP(train_X=train_x, train_Y=train_y, task_feature=1, rank=1)
    report: GPInterpretationReport = build_gp_interpretation_report(
        df=df,
        model=model,
        target_column="target",
        task_column="asset_id",
        feature_columns=["feature_a"],
    )

    rendered: RenderedGPInterpretationReport = render_gp_interpretation_report(report)

    assert "GP Interpretation Summary" in rendered["summary_markdown"]
    assert hasattr(rendered["summary_display"], "_repr_markdown_") or isinstance(rendered["summary_display"], str)
    assert rendered["feature_summary"].shape[0] == 1
    assert list(rendered["feature_summary"].columns) == ["feature", "kernel_type", "lengthscale_model_units"]
    assert rendered["task_summary"] is not None
    assert rendered["task_covariance"] is not None
    assert rendered["task_correlation_figure"] is not None
    figure = cast(Any, rendered["task_correlation_figure"])
    assert figure.layout.to_plotly_json()["title"]["text"] == "Task Correlation"


def test_display_gp_interpretation_report_calls_ipython_display(monkeypatch: pytest.MonkeyPatch) -> None:
    displayed: list[object] = []

    report: RenderedGPInterpretationReport = {
        "summary_markdown": "## Summary",
        "summary_display": "summary-object",
        "notes_markdown": "## Notes",
        "notes_display": "notes-object",
        "feature_summary": pd.DataFrame({"feature": ["x1"]}),
        "kernel_summary": pd.DataFrame({"kernel_type": ["RBFKernel"]}),
        "task_summary": pd.DataFrame({"task_label": ["SPY"]}),
        "task_covariance": pd.DataFrame([[1.0]], index=["SPY"], columns=["SPY"]),
        "noise_summary": pd.DataFrame({"noise_variance": [0.1]}),
        "task_correlation": pd.DataFrame([[1.0]], index=["SPY"], columns=["SPY"]),
        "task_correlation_figure": None,
    }

    import bayesfolio.engine.forecast.gp.reporting as reporting_module

    display_module = type("DisplayModule", (), {"display": lambda obj: displayed.append(obj)})
    import_module = getattr(reporting_module, "__import__", __import__)

    def _fake_import(name: str, globals_=None, locals_=None, fromlist=(), level: int = 0):
        if name == "IPython.display":
            return display_module
        return import_module(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    display_gp_interpretation_report(report)

    assert displayed[0] == "summary-object"
    assert displayed[1] == "notes-object"
    assert isinstance(displayed[2], pd.DataFrame)
    assert isinstance(displayed[4], pd.DataFrame)
