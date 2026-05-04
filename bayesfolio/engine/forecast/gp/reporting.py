"""GP hyperparameter interpretation utilities for forecast models.

This module belongs to the forecast layer and provides pure inspection helpers
for fitted GPyTorch and BoTorch Gaussian Process models. It converts learned
kernel and task hyperparameters into notebook-friendly pandas tables that map
input columns and output tasks to interpretable quantities such as feature
lengthscales, output scales, task covariance, and noise.

Inputs:
        - A pandas DataFrame aligned row-for-row with the model training data.
        - A fitted GPyTorch or BoTorch model with accessible training inputs.

Outputs:
        - A dictionary of pandas tables and metadata summaries suitable for direct
            inspection in notebooks. Feature lengthscales are reported in model input
            units so the report reflects the fitted kernel parameters directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

_EPSILON = 1e-12


class GPInterpretationReport(TypedDict):
    """Structured return type for GP interpretation details."""

    model_summary: dict[str, object]
    feature_summary: pd.DataFrame
    kernel_summary: pd.DataFrame
    task_summary: pd.DataFrame | None
    task_covariance: pd.DataFrame | None
    task_correlation: pd.DataFrame | None
    noise_summary: pd.DataFrame
    interpretation_notes: list[str]


class RenderedGPInterpretationReport(TypedDict):
    """Structured return type for notebook-ready GP interpretation artifacts."""

    summary_markdown: str
    summary_display: object
    notes_markdown: str
    notes_display: object
    feature_summary: pd.DataFrame
    kernel_summary: pd.DataFrame
    task_summary: pd.DataFrame | None
    task_covariance: pd.DataFrame | None
    noise_summary: pd.DataFrame
    task_correlation: pd.DataFrame | None
    task_correlation_figure: go.Figure | None


def build_gp_interpretation_report(
    df: pd.DataFrame,
    model: object,
    target_column: str,
    task_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
    feature_blocks: Mapping[str, Sequence[str]] | None = None,
) -> GPInterpretationReport:
    """Build a notebook-friendly interpretation report for a fitted GP model.

    The function assumes ``df`` is aligned row-for-row with the data used to fit
    ``model``. Feature lengthscales are reported directly in model input units,
    which makes the report reflect the fitted kernel parameters without trying to
    invert any preprocessing transforms.

    Args:
        df: Training DataFrame in original units. The DataFrame must contain the
            target column and any feature/task columns referenced below.
        model: Fitted GPyTorch or BoTorch model. The object must expose
            ``train_inputs`` in the standard ExactGP/BoTorch form.
        target_column: Name of the response column in ``df``. Target values are
            assumed to already be in the desired reporting units, for example
            decimal returns where ``0.02 = 2%``.
        task_column: Optional column holding raw task/output labels for multitask
            models, for example ETF tickers.
        feature_columns: Optional ordered feature columns. When omitted, the
            function uses all numeric columns except ``target_column`` and
            ``task_column``.
        feature_blocks: Optional mapping from human-readable block names to the
            feature columns belonging to each block, for example ``{"time":
            ["t_index"], "etf": [...], "macro": [...]}``. When provided,
            rendered kernel scopes can describe additive terms and interaction
            products in block-level terms such as ``time`` or ``product of
            time and macro``.

    Returns:
        dict[str, object]: Dictionary with notebook-friendly report sections:
            ``model_summary`` (dict), ``feature_summary`` (pd.DataFrame),
            ``kernel_summary`` (pd.DataFrame), ``task_summary``
            (pd.DataFrame | None), ``task_correlation`` (pd.DataFrame | None),
            ``noise_summary`` (pd.DataFrame), and ``interpretation_notes``
            (list[str]). Feature lengthscales are reported in model input units.

    Raises:
        ValueError: If required columns are missing, if the DataFrame does not
            align with the model's training inputs, or if task labels are
            inconsistent with encoded task indices.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    if task_column is not None and task_column not in df.columns:
        raise ValueError(f"Task column '{task_column}' not found in DataFrame.")

    train_x = _extract_train_inputs(model)
    task_feature_index = _resolve_task_feature_index(model, train_x.shape[-1])
    resolved_feature_columns = _resolve_feature_columns(df, target_column, task_column, feature_columns)
    model_feature_dims = [index for index in range(train_x.shape[-1]) if index != task_feature_index]

    if len(resolved_feature_columns) != len(model_feature_dims):
        raise ValueError(
            "Feature column count does not match the model training input dimensions after removing the task "
            f"feature. Got {len(resolved_feature_columns)} feature columns for {len(model_feature_dims)} "
            "model dimensions. Pass feature_columns explicitly in model-input order."
        )
    if len(df) != train_x.shape[0]:
        raise ValueError(
            "The DataFrame must have the same number of rows as the model training inputs. "
            f"Got {len(df)} rows and {train_x.shape[0]} training rows."
        )

    feature_name_by_dim = {
        model_dim: feature_name
        for model_dim, feature_name in zip(model_feature_dims, resolved_feature_columns, strict=True)
    }

    covar_module = getattr(model, "covar_module", None)
    if covar_module is None:
        raise ValueError("Model does not expose covar_module; cannot build a GP interpretation report.")

    kernel_components = _collect_kernel_components(
        kernel=covar_module,
        path="covar_module",
        all_feature_dims=model_feature_dims,
        task_feature_index=task_feature_index,
    )
    kernel_summary = pd.DataFrame(kernel_components)
    feature_summary = _build_feature_summary(
        kernel_components=kernel_components,
        feature_name_by_dim=feature_name_by_dim,
        feature_blocks=feature_blocks,
    )

    task_kernel = _resolve_task_kernel(model)
    task_label_map = _build_task_label_map(df, train_x, task_column, task_feature_index)
    task_summary, task_covariance, task_correlation = _build_task_report(
        model,
        task_kernel,
        task_label_map,
        target_column,
        train_x.shape[-1],
        task_feature_index,
    )
    noise_summary = _build_noise_summary(model, task_label_map, target_column)

    model_summary = {
        "model_type": model.__class__.__name__,
        "mean_module_type": getattr(getattr(model, "mean_module", None), "__class__", type(None)).__name__,
        "covar_module_type": getattr(getattr(model, "covar_module", None), "__class__", type(None)).__name__,
        "likelihood_type": getattr(getattr(model, "likelihood", None), "__class__", type(None)).__name__,
        "outcome_transform_type": getattr(
            getattr(model, "outcome_transform", None),
            "__class__",
            type(None),
        ).__name__,
        "target_column": target_column,
        "task_column": task_column,
        "task_feature_index": task_feature_index,
        "n_rows": int(len(df)),
        "n_features": int(len(resolved_feature_columns)),
    }

    return {
        "model_summary": model_summary,
        "feature_summary": feature_summary,
        "kernel_summary": kernel_summary,
        "task_summary": task_summary,
        "task_covariance": task_covariance,
        "task_correlation": task_correlation,
        "noise_summary": noise_summary,
        "interpretation_notes": _build_interpretation_notes(feature_summary, task_summary),
    }


def render_gp_interpretation_report(
    report: GPInterpretationReport,
    max_feature_rows: int = 20,
) -> RenderedGPInterpretationReport:
    """Prepare notebook-friendly tables and figures from a GP interpretation report.

    This helper keeps the core reporting logic separate from notebook display
    concerns. It returns sorted pandas tables, concise markdown summaries, and a
    Plotly heatmap for task correlations when the model is multitask.

    Args:
        report: Output from ``build_gp_interpretation_report``.
        max_feature_rows: Maximum number of rows to keep in the rendered feature
            summary table.

    Returns:
        dict[str, object]: Dictionary with markdown strings, display tables, and
            an optional Plotly figure under ``task_correlation_figure``.
    """
    feature_summary = _ensure_dataframe(report.get("feature_summary"))
    kernel_summary = _ensure_dataframe(report.get("kernel_summary"))
    task_summary = _ensure_optional_dataframe(report.get("task_summary"))
    task_covariance = _ensure_optional_dataframe(report.get("task_covariance"))
    noise_summary = _ensure_dataframe(report.get("noise_summary"))
    task_correlation = _ensure_optional_dataframe(report.get("task_correlation"))
    model_summary = cast(dict[str, object], report.get("model_summary", {}))
    interpretation_notes = cast(list[str], report.get("interpretation_notes", []))

    rendered_feature_summary = _prepare_feature_display_table(feature_summary, max_feature_rows=max_feature_rows)
    rendered_kernel_summary = _prepare_kernel_display_table(kernel_summary)
    rendered_task_summary = _prepare_task_display_table(task_summary)
    rendered_noise_summary = noise_summary.copy()

    summary_markdown = _build_summary_markdown(model_summary, rendered_feature_summary, rendered_task_summary)
    notes_markdown = _build_notes_markdown(interpretation_notes)
    task_correlation_figure = _build_task_correlation_figure(task_correlation)
    summary_display = _build_markdown_display(summary_markdown)
    notes_display = _build_markdown_display(notes_markdown)

    return {
        "summary_markdown": summary_markdown,
        "summary_display": summary_display,
        "notes_markdown": notes_markdown,
        "notes_display": notes_display,
        "feature_summary": rendered_feature_summary,
        "kernel_summary": rendered_kernel_summary,
        "task_summary": rendered_task_summary,
        "task_covariance": task_covariance,
        "noise_summary": rendered_noise_summary,
        "task_correlation": task_correlation,
        "task_correlation_figure": task_correlation_figure,
    }


def display_gp_interpretation_report(rendered_report: RenderedGPInterpretationReport) -> None:
    """Display a rendered GP interpretation report inside a notebook.

    This helper is intended for IPython and Jupyter environments. It renders
    markdown sections as markdown rather than raw strings, followed by the main
    tables and an optional task-correlation heatmap.

    Args:
        rendered_report: Output from ``render_gp_interpretation_report``.
    """
    try:
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover - notebook-only fallback
        raise RuntimeError("display_gp_interpretation_report requires IPython/Jupyter display support.") from exc

    display(rendered_report["summary_display"])
    if rendered_report["notes_markdown"]:
        display(rendered_report["notes_display"])
    display(rendered_report["feature_summary"])
    display(rendered_report["kernel_summary"])

    task_summary = rendered_report["task_summary"]
    if task_summary is not None and not task_summary.empty:
        display(task_summary)

    task_covariance = rendered_report["task_covariance"]
    if task_covariance is not None and not task_covariance.empty:
        display(task_covariance)

    task_correlation = rendered_report["task_correlation"]
    if task_correlation is not None and not task_correlation.empty:
        display(task_correlation)

    display(rendered_report["noise_summary"])

    task_correlation_figure = rendered_report["task_correlation_figure"]
    if task_correlation_figure is not None:
        display(task_correlation_figure)


def _extract_train_inputs(model: object) -> torch.Tensor:
    train_inputs = getattr(model, "train_inputs", None)
    if not train_inputs:
        raise ValueError("Model does not expose train_inputs; cannot align features to learned hyperparameters.")

    train_x = train_inputs[0]
    if not isinstance(train_x, torch.Tensor):
        raise ValueError("Expected model.train_inputs[0] to be a torch.Tensor.")

    return train_x.detach().cpu()


def _ensure_dataframe(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame()


def _ensure_optional_dataframe(value: object) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return None


def _build_markdown_display(markdown_text: str) -> object:
    if not markdown_text:
        return markdown_text

    try:
        from IPython.display import Markdown
    except ImportError:
        return markdown_text

    return Markdown(markdown_text)


def _resolve_task_feature_index(model: object, input_dim: int) -> int | None:
    task_feature = getattr(model, "task_feature", None)
    if task_feature is None:
        task_feature = getattr(model, "_task_feature", None)
    if task_feature is None:
        return None

    task_feature_int = int(task_feature)
    if task_feature_int < 0:
        task_feature_int += input_dim
    return task_feature_int


def _resolve_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    task_column: str | None,
    feature_columns: Sequence[str] | None,
) -> list[str]:
    if feature_columns is not None:
        missing_columns = [column for column in feature_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Feature columns not found in DataFrame: {missing_columns}")
        return list(feature_columns)

    excluded_columns = {target_column}
    if task_column is not None:
        excluded_columns.add(task_column)

    resolved_columns: list[str] = []
    for column in df.columns:
        if column in excluded_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            resolved_columns.append(column)
    return resolved_columns


def _collect_kernel_components(
    kernel: Any,
    path: str,
    all_feature_dims: Sequence[int],
    task_feature_index: int | None,
) -> list[dict[str, object]]:
    component = _describe_kernel_component(kernel, path, all_feature_dims, task_feature_index)
    components = [component]

    if hasattr(kernel, "base_kernel"):
        components.extend(
            _collect_kernel_components(
                kernel=kernel.base_kernel,
                path=f"{path}.base_kernel",
                all_feature_dims=all_feature_dims,
                task_feature_index=task_feature_index,
            )
        )
    if hasattr(kernel, "kernels"):
        for index, sub_kernel in enumerate(kernel.kernels):
            components.extend(
                _collect_kernel_components(
                    kernel=sub_kernel,
                    path=f"{path}.kernels[{index}]",
                    all_feature_dims=all_feature_dims,
                    task_feature_index=task_feature_index,
                )
            )
    if hasattr(kernel, "data_covar_module"):
        components.extend(
            _collect_kernel_components(
                kernel=kernel.data_covar_module,
                path=f"{path}.data_covar_module",
                all_feature_dims=all_feature_dims,
                task_feature_index=task_feature_index,
            )
        )
    if hasattr(kernel, "task_covar_module"):
        components.extend(
            _collect_kernel_components(
                kernel=kernel.task_covar_module,
                path=f"{path}.task_covar_module",
                all_feature_dims=all_feature_dims,
                task_feature_index=task_feature_index,
            )
        )
    if hasattr(kernel, "covar_module_list"):
        for index, sub_kernel in enumerate(kernel.covar_module_list):
            components.extend(
                _collect_kernel_components(
                    kernel=sub_kernel,
                    path=f"{path}.covar_module_list[{index}]",
                    all_feature_dims=all_feature_dims,
                    task_feature_index=task_feature_index,
                )
            )

    return components


def _describe_kernel_component(
    kernel: Any,
    path: str,
    all_feature_dims: Sequence[int],
    task_feature_index: int | None,
) -> dict[str, object]:
    active_dims = _normalize_dims(getattr(kernel, "active_dims", None))
    feature_dims = list(all_feature_dims)
    if active_dims is not None:
        feature_dims = [dim for dim in active_dims if dim != task_feature_index]

    return {
        "kernel_path": path,
        "kernel_type": kernel.__class__.__name__,
        "active_dims": active_dims,
        "feature_dims": feature_dims,
        "ard_num_dims": getattr(kernel, "ard_num_dims", None),
        "lengthscale_model_units": _tensor_to_flat_list(getattr(kernel, "lengthscale", None)),
        "outputscale": _tensor_to_scalar(getattr(kernel, "outputscale", None)),
        "variance": _tensor_to_scalar_or_list(getattr(kernel, "variance", None)),
        "alpha": _tensor_to_scalar(getattr(kernel, "alpha", None)),
        "period_length": _tensor_to_scalar(getattr(kernel, "period_length", None)),
        "num_tasks": getattr(kernel, "num_tasks", None),
        "rank": _extract_kernel_rank(kernel),
    }


def _extract_kernel_rank(kernel: Any) -> int | None:
    rank = getattr(kernel, "rank", None)
    if rank is not None:
        try:
            return int(rank)
        except (TypeError, ValueError):
            return None

    for attr_name in ["raw_covar_factor", "covar_factor"]:
        factor = getattr(kernel, attr_name, None)
        if isinstance(factor, torch.Tensor) and factor.ndim >= 2:
            return int(factor.shape[-1])

    return None


def _normalize_dims(value: object) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    try:
        return [int(cast(Any, value))]
    except (TypeError, ValueError):
        return None


def _build_feature_summary(
    kernel_components: Sequence[dict[str, object]],
    feature_name_by_dim: dict[int, str],
    feature_blocks: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scope_by_path = _build_kernel_scope_by_path(
        kernel_components=kernel_components,
        feature_name_by_dim=feature_name_by_dim,
        feature_blocks=feature_blocks,
    )

    for component in kernel_components:
        lengthscale_values = cast(list[float] | None, component["lengthscale_model_units"])
        if lengthscale_values is None:
            continue

        feature_dims = cast(list[int], component["feature_dims"])
        kernel_scope = scope_by_path.get(
            cast(str, component["kernel_path"]),
            _describe_kernel_scope(feature_dims, feature_name_by_dim),
        )
        mapped_values = _expand_lengthscales(lengthscale_values, feature_dims)
        for feature_dim, model_lengthscale in mapped_values:
            if feature_dim not in feature_name_by_dim:
                continue

            rows.append(
                {
                    "feature": feature_name_by_dim[feature_dim],
                    "model_input_dim": feature_dim,
                    "kernel_path": component["kernel_path"],
                    "kernel_scope": kernel_scope,
                    "kernel_type": component["kernel_type"],
                    "lengthscale_model_units": float(model_lengthscale),
                    "interpretation": (
                        "Shorter lengthscales imply the posterior can vary more quickly along this input."
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "model_input_dim",
                "kernel_path",
                "kernel_scope",
                "kernel_type",
                "lengthscale_model_units",
                "interpretation",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=["feature", "kernel_path"],
        kind="stable",
        ignore_index=True,
    )


def _expand_lengthscales(lengthscale_values: Sequence[float], feature_dims: Sequence[int]) -> list[tuple[int, float]]:
    if not feature_dims:
        return []
    if len(lengthscale_values) == 1:
        return [(feature_dim, float(lengthscale_values[0])) for feature_dim in feature_dims]
    if len(lengthscale_values) == len(feature_dims):
        return [
            (feature_dim, float(value)) for feature_dim, value in zip(feature_dims, lengthscale_values, strict=True)
        ]
    return []


def _describe_kernel_scope(
    feature_dims: Sequence[int],
    feature_name_by_dim: dict[int, str],
    *,
    preview_count: int = 3,
) -> str:
    feature_names = [
        feature_name_by_dim[feature_dim] for feature_dim in feature_dims if feature_dim in feature_name_by_dim
    ]
    if not feature_names:
        return "all features"
    if len(feature_names) == 1:
        return feature_names[0]
    if len(feature_names) <= preview_count:
        return ", ".join(feature_names)
    preview = ", ".join(feature_names[:preview_count])
    return f"{len(feature_names)} features incl. {preview}"


def _build_kernel_scope_by_path(
    kernel_components: Sequence[dict[str, object]],
    feature_name_by_dim: dict[int, str],
    feature_blocks: Mapping[str, Sequence[str]] | None,
) -> dict[str, str]:
    component_by_path = {cast(str, component["kernel_path"]): component for component in kernel_components}
    scope_by_path: dict[str, str] = {}
    normalized_feature_blocks = _normalize_feature_blocks(feature_name_by_dim, feature_blocks)

    for component in kernel_components:
        component_path = cast(str, component["kernel_path"])
        feature_dims = cast(list[int], component["feature_dims"])
        base_scope = _resolve_block_label(
            feature_dims=feature_dims,
            feature_name_by_dim=feature_name_by_dim,
            feature_blocks=normalized_feature_blocks,
        ) or _describe_kernel_scope(feature_dims, feature_name_by_dim)

        product_path = _find_nearest_kernel_ancestor(component_path, component_by_path, kernel_type="ProductKernel")
        if product_path is None:
            scope_by_path[component_path] = base_scope
            continue

        product_parts = _collect_product_scope_parts(
            kernel_components=kernel_components,
            product_path=product_path,
            feature_name_by_dim=feature_name_by_dim,
            feature_blocks=normalized_feature_blocks,
        )
        if len(product_parts) >= 2:
            scope_by_path[component_path] = f"product of {' and '.join(product_parts)}"
            continue

        scope_by_path[component_path] = base_scope

    return scope_by_path


def _normalize_feature_blocks(
    feature_name_by_dim: dict[int, str],
    feature_blocks: Mapping[str, Sequence[str]] | None,
) -> dict[str, set[str]]:
    if feature_blocks is None:
        return {}

    feature_names_in_model = set(feature_name_by_dim.values())
    normalized_blocks: dict[str, set[str]] = {}
    for block_name, block_columns in feature_blocks.items():
        block_feature_names = {column for column in block_columns if column in feature_names_in_model}
        if block_feature_names:
            normalized_blocks[block_name] = block_feature_names
    return normalized_blocks


def _resolve_block_label(
    feature_dims: Sequence[int],
    feature_name_by_dim: dict[int, str],
    feature_blocks: Mapping[str, set[str]],
) -> str | None:
    if not feature_blocks:
        return None

    feature_names = {
        feature_name_by_dim[feature_dim] for feature_dim in feature_dims if feature_dim in feature_name_by_dim
    }
    if not feature_names:
        return None

    for block_name, block_feature_names in feature_blocks.items():
        if feature_names == block_feature_names:
            return block_name
    return None


def _find_nearest_kernel_ancestor(
    component_path: str,
    component_by_path: Mapping[str, dict[str, object]],
    *,
    kernel_type: str,
) -> str | None:
    matching_ancestor: str | None = None
    for ancestor_path in _iter_ancestor_paths(component_path):
        ancestor = component_by_path.get(ancestor_path)
        if ancestor is None:
            continue
        if ancestor.get("kernel_type") == kernel_type:
            if _subtree_contains_task_kernel(ancestor_path, component_by_path):
                continue
            matching_ancestor = ancestor_path
    return matching_ancestor


def _iter_ancestor_paths(component_path: str) -> Sequence[str]:
    ancestors: list[str] = []
    current_path = component_path
    while "." in current_path:
        current_path = current_path.rsplit(".", 1)[0]
        ancestors.append(current_path)
    return ancestors


def _subtree_contains_task_kernel(
    subtree_path: str,
    component_by_path: Mapping[str, dict[str, object]],
) -> bool:
    subtree_prefix = f"{subtree_path}."
    for component_path, component in component_by_path.items():
        if component_path != subtree_path and not component_path.startswith(subtree_prefix):
            continue
        if component.get("num_tasks") is not None:
            return True
    return False


def _collect_product_scope_parts(
    kernel_components: Sequence[dict[str, object]],
    product_path: str,
    feature_name_by_dim: dict[int, str],
    feature_blocks: Mapping[str, set[str]],
) -> list[str]:
    product_parts: list[str] = []

    for component in kernel_components:
        component_path = cast(str, component["kernel_path"])
        if not component_path.startswith(f"{product_path}."):
            continue

        lengthscale_values = cast(list[float] | None, component["lengthscale_model_units"])
        if lengthscale_values is None:
            continue

        feature_dims = cast(list[int], component["feature_dims"])
        part_label = _resolve_block_label(
            feature_dims=feature_dims,
            feature_name_by_dim=feature_name_by_dim,
            feature_blocks=feature_blocks,
        ) or _describe_kernel_scope(feature_dims, feature_name_by_dim)
        if part_label not in product_parts:
            product_parts.append(part_label)

    return product_parts


def _resolve_task_kernel(model: object) -> object | None:
    task_covar_module = getattr(model, "task_covar_module", None)
    if task_covar_module is not None:
        return task_covar_module

    covar_module = getattr(model, "covar_module", None)
    return _find_task_kernel(covar_module)


def _find_task_kernel(kernel: Any | None) -> object | None:
    if kernel is None:
        return None
    if hasattr(kernel, "covar_matrix") and getattr(kernel, "num_tasks", None) is not None:
        return kernel

    for attr_name in ["base_kernel", "data_covar_module", "task_covar_module"]:
        child = getattr(kernel, attr_name, None)
        found = _find_task_kernel(child)
        if found is not None:
            return found

    for attr_name in ["kernels", "covar_module_list"]:
        children = getattr(kernel, attr_name, None)
        if children is None:
            continue
        for child in children:
            found = _find_task_kernel(child)
            if found is not None:
                return found

    return None


def _build_task_label_map(
    df: pd.DataFrame,
    train_x: torch.Tensor,
    task_column: str | None,
    task_feature_index: int | None,
) -> dict[int, object]:
    if task_column is None or task_feature_index is None:
        return {}

    encoded_tasks = train_x[:, task_feature_index].numpy()
    raw_labels = df[task_column].to_numpy()
    task_label_map: dict[int, object] = {}

    for raw_label, encoded_value in zip(raw_labels, encoded_tasks, strict=True):
        task_index = int(round(float(encoded_value)))
        existing = task_label_map.get(task_index)
        if existing is None:
            task_label_map[task_index] = raw_label
            continue
        if existing != raw_label:
            raise ValueError(
                "Task labels are not aligned with the encoded task feature in the model training inputs. "
                f"Task index {task_index} was associated with both '{existing}' and '{raw_label}'."
            )

    return task_label_map


def _build_task_report(
    model: object,
    task_kernel: object | None,
    task_label_map: dict[int, object],
    target_column: str,
    input_dim: int,
    task_feature_index: int | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    if task_kernel is None:
        return None, None, None

    covariance = _evaluate_task_covariance(task_kernel, input_dim=input_dim, task_feature_index=task_feature_index)
    if covariance is None:
        return None, None, None

    task_indices = list(range(covariance.shape[0]))
    task_labels = [task_label_map.get(index, index) for index in task_indices]
    mean_constants = _extract_task_mean_constants(getattr(model, "mean_module", None), len(task_indices))
    noise_values = _extract_noise_values(model)

    rows: list[dict[str, object]] = []
    for index, label in enumerate(task_labels):
        rows.append(
            {
                "task_index": index,
                "task_label": label,
                "target_column": target_column,
                "task_variance": float(covariance[index, index]),
                "mean_constant": mean_constants[index] if mean_constants is not None else None,
                "noise_variance": _select_noise_value(noise_values, index),
            }
        )

    correlation = _covariance_to_correlation(covariance)
    covariance_df = pd.DataFrame(covariance, index=task_labels, columns=task_labels)
    correlation_df = pd.DataFrame(correlation, index=task_labels, columns=task_labels)
    return pd.DataFrame(rows), covariance_df, correlation_df


def _evaluate_task_covariance(
    task_kernel: object,
    input_dim: int,
    task_feature_index: int | None,
) -> np.ndarray | None:
    num_tasks = getattr(task_kernel, "num_tasks", None)
    if isinstance(num_tasks, int):
        task_ids = torch.arange(num_tasks, dtype=torch.long)
        active_dims = _normalize_dims(getattr(task_kernel, "active_dims", None))

        evaluation_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        if active_dims:
            dummy_inputs = torch.zeros((num_tasks, input_dim), dtype=torch.long)
            active_task_dim = active_dims[0]
            dummy_inputs[:, active_task_dim] = task_ids
            evaluation_pairs.append((dummy_inputs, dummy_inputs))

        evaluation_pairs.extend(
            [
                (task_ids, task_ids),
                (task_ids.unsqueeze(-1), task_ids.unsqueeze(-1)),
            ]
        )

        for left, right in evaluation_pairs:
            try:
                covariance = _linear_operator_to_numpy(cast(Any, task_kernel)(left, right))
            except Exception:
                covariance = None
            if covariance is not None and covariance.shape[-2:] == (num_tasks, num_tasks):
                return covariance

    covariance = _linear_operator_to_numpy(getattr(task_kernel, "covar_matrix", None))
    if covariance is not None and isinstance(num_tasks, int) and covariance.shape[-2:] != (num_tasks, num_tasks):
        return None
    return covariance


def _extract_task_mean_constants(mean_module: object, expected_count: int) -> list[float] | None:
    if mean_module is None:
        return None
    base_means = getattr(mean_module, "base_means", None)
    if base_means is None:
        constant = getattr(mean_module, "constant", None)
        constant_value = _tensor_to_scalar(constant)
        if constant_value is None:
            return None
        return [constant_value] * expected_count

    constants: list[float] = []
    for base_mean in base_means:
        constant_value = _tensor_to_scalar(getattr(base_mean, "constant", None))
        if constant_value is None:
            return None
        constants.append(constant_value)

    if len(constants) != expected_count:
        return None
    return constants


def _build_noise_summary(model: object, task_label_map: dict[int, object], target_column: str) -> pd.DataFrame:
    noise_values = _extract_noise_values(model)
    if noise_values is None:
        return pd.DataFrame(columns=["output", "noise_variance"])

    if len(noise_values) == 1:
        output_label = target_column if not task_label_map else "all_tasks"
        return pd.DataFrame([{"output": output_label, "noise_variance": noise_values[0]}])

    rows = []
    for index, noise_value in enumerate(noise_values):
        rows.append({"output": task_label_map.get(index, index), "noise_variance": noise_value})
    return pd.DataFrame(rows)


def _extract_noise_values(model: object) -> list[float] | None:
    likelihood = getattr(model, "likelihood", None)
    if likelihood is None:
        return None

    if hasattr(likelihood, "noise"):
        noise_values = _tensor_to_flat_list(likelihood.noise)
        if noise_values is not None:
            return [float(value) for value in noise_values]

    noise_covar = getattr(likelihood, "noise_covar", None)
    if noise_covar is None:
        return None

    if hasattr(noise_covar, "noise"):
        noise_values = _tensor_to_flat_list(noise_covar.noise)
        if noise_values is not None:
            return [float(value) for value in noise_values]

    return None


def _select_noise_value(noise_values: list[float] | None, index: int) -> float | None:
    if noise_values is None:
        return None
    if len(noise_values) == 1:
        return float(noise_values[0])
    if index < len(noise_values):
        return float(noise_values[index])
    return None


def _linear_operator_to_numpy(value: object) -> np.ndarray | None:
    if value is None:
        return None
    dense_operator = cast(Any, value)
    if hasattr(dense_operator, "to_dense"):
        dense_value = dense_operator.to_dense()
        if isinstance(dense_value, torch.Tensor):
            return dense_value.detach().cpu().numpy().astype(float)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(float)
    return None


def _covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.sqrt(np.clip(np.diag(covariance), _EPSILON, None))
    denominator = np.outer(diagonal, diagonal)
    return covariance / denominator


def _build_interpretation_notes(
    feature_summary: pd.DataFrame,
    task_summary: pd.DataFrame | None,
) -> list[str]:
    notes = [
        "Shorter lengthscales indicate the GP can change more quickly as that input moves; longer lengthscales "
        "indicate smoother dependence.",
    ]
    if task_summary is not None:
        notes.append(
            "Task variance and task correlation summarize how strongly the multitask kernel shares statistical "
            "structure across outputs."
        )
    return notes


def _prepare_feature_display_table(feature_summary: pd.DataFrame, max_feature_rows: int) -> pd.DataFrame:
    if feature_summary.empty:
        return feature_summary.copy()

    columns = [
        "feature",
        "kernel_scope",
        "kernel_type",
        "lengthscale_model_units",
    ]
    return (
        feature_summary.loc[:, columns]
        .sort_values(by=["lengthscale_model_units", "feature"], ascending=[True, True], kind="stable")
        .head(max_feature_rows)
        .reset_index(drop=True)
    )


def _prepare_kernel_display_table(kernel_summary: pd.DataFrame) -> pd.DataFrame:
    if kernel_summary.empty:
        return kernel_summary.copy()

    columns = [
        "kernel_path",
        "kernel_type",
        "active_dims",
        "lengthscale_model_units",
        "outputscale",
        "variance",
        "alpha",
        "period_length",
        "num_tasks",
        "rank",
    ]
    return kernel_summary.loc[:, columns].reset_index(drop=True)


def _prepare_task_display_table(task_summary: pd.DataFrame | None) -> pd.DataFrame | None:
    if task_summary is None or task_summary.empty:
        return task_summary
    return task_summary.sort_values(by="task_index", kind="stable").reset_index(drop=True)


def _build_summary_markdown(
    model_summary: object,
    feature_summary: pd.DataFrame,
    task_summary: pd.DataFrame | None,
) -> str:
    summary = model_summary if isinstance(model_summary, dict) else {}
    model_type = summary.get("model_type", "unknown")
    covar_module_type = summary.get("covar_module_type", "unknown")
    target_column = summary.get("target_column", "unknown")
    task_column = summary.get("task_column")

    lines = [
        "## GP Interpretation Summary",
        f"- Model type: {model_type}",
        f"- Covariance module: {covar_module_type}",
        f"- Target column: {target_column}",
        f"- Feature rows shown: {len(feature_summary)}",
    ]
    if task_column is not None:
        task_count = 0 if task_summary is None else len(task_summary)
        lines.append(f"- Task column: {task_column} ({task_count} tasks)")

    if not feature_summary.empty:
        top_row = feature_summary.iloc[0]
        top_lengthscale = float(top_row["lengthscale_model_units"])
        lines.append(f"- Most responsive feature shown: {top_row['feature']} ({top_lengthscale:.4g} model units)")

    return "\n".join(lines)


def _build_notes_markdown(interpretation_notes: object) -> str:
    if not isinstance(interpretation_notes, list) or not interpretation_notes:
        return ""
    return "\n".join(["## Interpretation Notes", *[f"- {note}" for note in interpretation_notes]])


def _build_task_correlation_figure(task_correlation: pd.DataFrame | None) -> go.Figure | None:
    if task_correlation is None or task_correlation.empty:
        return None

    labels = [str(label) for label in task_correlation.index.tolist()]
    z_values = task_correlation.to_numpy(dtype=float)
    figure = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=labels,
            y=labels,
            zmin=0,
            zmax=1.0,
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": "Corr"},
            text=np.round(z_values, 3),
            texttemplate="%{text}",
            hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Task Correlation",
        xaxis_title="Output Task",
        yaxis_title="Output Task",
    )
    return figure


def _tensor_to_scalar(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy().reshape(-1)
        if array.size == 0:
            return None
        return float(array[0])
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _tensor_to_scalar_or_list(value: object) -> float | list[float] | None:
    values = _tensor_to_flat_list(value)
    if values is None:
        return None
    if len(values) == 1:
        return float(values[0])
    return [float(item) for item in values]


def _tensor_to_flat_list(value: object) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().numpy().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(item) for item in np.asarray(value, dtype=float).reshape(-1).tolist()]
    try:
        return [float(cast(Any, value))]
    except (TypeError, ValueError):
        return None
