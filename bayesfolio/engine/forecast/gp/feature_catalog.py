"""Pure feature-catalog helpers for planner-driven multitask GP workflows.

This module resolves planner-declared feature groups and feature names into the
ordered non-task feature columns used by the active multitask GP workflow.
It does not perform I/O or model fitting.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from bayesfolio.contracts.results.gp_workflow import PlannerBlockSpec, ResolvedFeatureBlock


def get_non_task_feature_columns(
    df: pd.DataFrame,
    *,
    target_column: str,
    task_column: str,
    allowed_feature_columns: list[str] | None = None,
) -> list[str]:
    """Return ordered non-task feature columns used for GP tensor creation.

    Args:
        df: Input dataframe containing the target, task, and feature columns.
        target_column: Target column name.
        task_column: Task identifier column name.
        allowed_feature_columns: Optional explicit whitelist of allowed
            non-task feature columns.

    Returns:
        Ordered non-task feature columns preserving dataframe order.

    Raises:
        ValueError: If required columns are missing, allowed columns are
            unknown, or no usable feature columns remain.
    """

    missing = [column for column in (target_column, task_column) if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataframe columns: {missing}")

    excluded = {target_column, task_column, "__task_idx__"}
    ordered_columns = [column for column in df.columns if column not in excluded]

    if allowed_feature_columns is not None:
        unknown = [column for column in allowed_feature_columns if column not in df.columns]
        if unknown:
            raise ValueError(f"Unknown allowed feature columns: {unknown}")
        forbidden = [column for column in allowed_feature_columns if column in excluded]
        if forbidden:
            raise ValueError(f"Allowed feature columns cannot include target/task columns: {forbidden}")
        allowed_lookup = set(allowed_feature_columns)
        ordered_columns = [column for column in ordered_columns if column in allowed_lookup]

    if not ordered_columns:
        raise ValueError("No usable non-task feature columns were found")

    return ordered_columns


def validate_feature_groups(
    feature_groups: dict[str, list[str]],
    *,
    feature_columns: list[str],
    task_column: str,
    target_column: str,
) -> dict[str, list[str]]:
    """Validate and normalize feature groups against available columns.

    Args:
        feature_groups: Semantic feature-group mapping.
        feature_columns: Ordered usable non-task feature columns.
        task_column: Task identifier column name.
        target_column: Target column name.

    Returns:
        Validated feature-group mapping filtered to usable feature columns.

    Raises:
        ValueError: If a group references forbidden or unknown columns.
    """

    feature_lookup = set(feature_columns)
    normalized: dict[str, list[str]] = {}
    for group_name, group_columns in feature_groups.items():
        forbidden = [column for column in group_columns if column in {task_column, target_column, "__task_idx__"}]
        if forbidden:
            raise ValueError(f"Feature group {group_name} includes forbidden columns: {forbidden}")
        unknown = [column for column in group_columns if column not in feature_lookup]
        if unknown:
            raise ValueError(f"Feature group {group_name} includes unknown feature columns: {unknown}")
        ordered = [column for column in feature_columns if column in set(group_columns)]
        normalized[group_name] = ordered
    return normalized


def resolve_planner_blocks(
    blocks: list[PlannerBlockSpec],
    *,
    feature_columns: list[str],
    feature_groups: dict[str, list[str]],
) -> list[ResolvedFeatureBlock]:
    """Resolve planner block feature selections into ordered tensor dimensions.

    Args:
        blocks: Planner-selected block specifications.
        feature_columns: Ordered non-task feature columns.
        feature_groups: Validated semantic feature-group mapping.

    Returns:
        Resolved feature blocks with ordered feature names and dimensions.

    Raises:
        ValueError: If groups or features are unknown, resolve to empty sets,
            or overlap across blocks.
    """

    feature_lookup = {column: index for index, column in enumerate(feature_columns)}
    feature_set = set(feature_columns)
    assignments: Counter[str] = Counter()
    resolved_blocks: list[ResolvedFeatureBlock] = []

    for block in blocks:
        selected_names = _resolve_block_feature_names(
            block,
            feature_columns=feature_columns,
            feature_groups=feature_groups,
            feature_set=feature_set,
        )
        for feature_name in selected_names:
            assignments[feature_name] += 1
        dims = [feature_lookup[name] for name in selected_names]
        resolved_blocks.append(
            ResolvedFeatureBlock(
                name=block.name,
                variable_type=block.variable_type,
                feature_names=selected_names,
                dims=dims,
            )
        )

    duplicates = sorted(name for name, count in assignments.items() if count > 1)
    if duplicates:
        raise ValueError(f"Resolved planner blocks overlap on feature columns: {duplicates}")

    return resolved_blocks


def _resolve_block_feature_names(
    block: PlannerBlockSpec,
    *,
    feature_columns: list[str],
    feature_groups: dict[str, list[str]],
    feature_set: set[str],
) -> list[str]:
    """Resolve one planner block to ordered feature names.

    Args:
        block: Planner block specification.
        feature_columns: Ordered non-task feature columns.
        feature_groups: Validated feature-group mapping.
        feature_set: Set of usable non-task feature columns.

    Returns:
        Ordered resolved feature names.

    Raises:
        ValueError: If the block references unknown groups or features, or if
            the resolved feature set is empty.
    """

    unknown_groups = [group_name for group_name in block.source_groups if group_name not in feature_groups]
    if unknown_groups:
        raise ValueError(f"Planner block {block.name} references unknown groups: {unknown_groups}")

    unknown_features = [feature_name for feature_name in block.source_features if feature_name not in feature_set]
    if unknown_features:
        raise ValueError(f"Planner block {block.name} references unknown features: {unknown_features}")

    selected = set(block.source_features)
    for group_name in block.source_groups:
        selected.update(feature_groups[group_name])

    if not selected:
        selected = set(feature_columns)

    ordered = [column for column in feature_columns if column in selected]
    if not ordered:
        raise ValueError(f"Planner block {block.name} resolved to zero usable features")
    return ordered
