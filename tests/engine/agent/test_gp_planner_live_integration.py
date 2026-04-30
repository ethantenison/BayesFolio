from __future__ import annotations

import os

import pytest

from bayesfolio.contracts.commands.gp_planner import GPPlannerRequest
from bayesfolio.engine.agent.gp_planner import plan_gp_design_with_status


def _integration_request() -> GPPlannerRequest:
    return GPPlannerRequest(
        target_column="y_excess_lead",
        task_column="asset_id",
        user_instruction_text="Use a matern kernel with ard for all input variables",
        feature_groups={"macro": ["macro_1"], "etf": ["etf_1"]},
        allowed_feature_columns=["macro_1", "etf_1"],
        dataset_summary=[],
        planner_prompt="Return JSON only and strictly follow the requested schema.",
    )


def test_plan_gp_design_with_status_live_endpoint_opt_in() -> None:
    if os.getenv("BAYESFOLIO_RUN_LIVE_PLANNER_TEST") != "1":
        pytest.skip("Set BAYESFOLIO_RUN_LIVE_PLANNER_TEST=1 to run live planner integration test.")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Set OPENAI_API_KEY for live planner integration testing.")

    response, status = plan_gp_design_with_status(_integration_request(), timeout_seconds=30.0)

    assert status == "ok"
    assert response.planner_status == "ok"
    assert response.selected_design is not None
