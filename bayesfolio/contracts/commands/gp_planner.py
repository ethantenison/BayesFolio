"""Planner request contracts for planner-driven Gaussian Process workflows.

This module defines command-layer boundary schemas used to request a structured
multitask Gaussian Process design from an LLM-backed planner. The schemas are
data-only and carry no business logic or I/O.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import ContractModel, SchemaName, VersionedContract


class DatasetSummaryItem(ContractModel):
    """Single dataset summary entry passed to the GP planner.

    Attributes:
        key: Summary field name.
        value: JSON-serializable summary value rendered for the planner.
        description: Optional explanation of the summary field.
    """

    key: str
    value: str | int | float | bool | list[object] | dict[str, object]
    description: str | None = None


class GPPlannerRequest(VersionedContract):
    """Command contract for generating a structured GP design plan.

    Attributes:
        schema: Contract schema identifier.
        schema_version: Semantic schema version.
        target_column: Name of the target return column in decimal units.
        task_column: Name of the task or asset identifier column.
        user_instruction_text: Free-form user GP instructions and prior beliefs.
        feature_groups: Mapping of semantic feature-group names to feature
            column names. Group membership preserves user-provided ordering.
        allowed_feature_columns: Optional explicit whitelist of usable non-task
            feature columns. When omitted, all non-task numeric columns are
            eligible.
        dataset_summary: Structured dataset summary values provided as planner
            context.
        planner_prompt: System prompt text supplied by the caller. This keeps
            planner policy outside business logic.
    """

    schema: Literal[SchemaName.GP_PLANNER_COMMAND] = SchemaName.GP_PLANNER_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    target_column: str
    task_column: str
    user_instruction_text: str
    feature_groups: dict[str, list[str]] = Field(default_factory=dict)
    allowed_feature_columns: list[str] | None = None
    dataset_summary: list[DatasetSummaryItem] = Field(default_factory=list)
    planner_prompt: str
