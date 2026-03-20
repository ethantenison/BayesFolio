from __future__ import annotations

import pandas as pd
import pytest

from bayesfolio.contracts.results.gp_workflow import PlannerBlockSpec
from bayesfolio.engine.forecast.gp.feature_catalog import (
    get_non_task_feature_columns,
    resolve_planner_blocks,
    validate_feature_groups,
)


def test_get_non_task_feature_columns_preserves_order_and_allowed_subset() -> None:
    df = pd.DataFrame(
        {
            "macro_1": [1.0, 2.0],
            "etf_1": [3.0, 4.0],
            "macro_2": [5.0, 6.0],
            "y_excess_lead": [0.01, 0.02],
            "asset_id": ["A", "B"],
        }
    )

    columns = get_non_task_feature_columns(
        df,
        target_column="y_excess_lead",
        task_column="asset_id",
        allowed_feature_columns=["etf_1", "macro_2"],
    )

    assert columns == ["etf_1", "macro_2"]


def test_resolve_planner_blocks_maps_groups_to_dims() -> None:
    feature_columns = ["macro_1", "macro_2", "etf_1"]
    feature_groups = validate_feature_groups(
        {"macro": ["macro_1", "macro_2"], "etf": ["etf_1"]},
        feature_columns=feature_columns,
        task_column="asset_id",
        target_column="y_excess_lead",
    )

    blocks = [
        PlannerBlockSpec(name="macro", variable_type="macro", source_groups=["macro"], components=[]),
        PlannerBlockSpec(name="etf", variable_type="etf", source_groups=["etf"], components=[]),
    ]

    resolved = resolve_planner_blocks(blocks, feature_columns=feature_columns, feature_groups=feature_groups)

    assert resolved[0].feature_names == ["macro_1", "macro_2"]
    assert resolved[0].dims == [0, 1]
    assert resolved[1].feature_names == ["etf_1"]
    assert resolved[1].dims == [2]


def test_resolve_planner_blocks_rejects_overlapping_assignments() -> None:
    feature_columns = ["macro_1", "etf_1"]
    blocks = [
        PlannerBlockSpec(name="one", variable_type="generic", source_features=["macro_1"], components=[]),
        PlannerBlockSpec(name="two", variable_type="generic", source_features=["macro_1"], components=[]),
    ]

    with pytest.raises(ValueError, match="overlap"):
        resolve_planner_blocks(blocks, feature_columns=feature_columns, feature_groups={})
