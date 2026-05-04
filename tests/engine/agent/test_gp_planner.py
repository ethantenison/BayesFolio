from __future__ import annotations

import json

from bayesfolio.contracts.commands.gp_planner import GPPlannerRequest
from bayesfolio.engine.agent.gp_planner import plan_gp_design_with_status


def _make_request() -> GPPlannerRequest:
    return GPPlannerRequest(
        target_column="y_excess_lead",
        task_column="asset_id",
        user_instruction_text="Use a matern kernel with ard for all input variables",
        feature_groups={"macro": ["macro_1"], "etf": ["etf_1"]},
        allowed_feature_columns=["macro_1", "etf_1"],
        dataset_summary=[],
        planner_prompt="Return JSON only.",
    )


def test_plan_gp_design_with_status_falls_back_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response, status = plan_gp_design_with_status(_make_request())

    assert status == "missing_openai_api_key"
    assert response.planner_status == "ok"
    assert response.selected_design is not None
    assert response.selected_design.covar_config.blocks


def test_plan_gp_design_with_status_parses_valid_llm_json(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    payload = {
        "planner_status": "ok",
        "clarification_questions": [],
        "dataset_assumptions": [],
        "instruction_mode": "explicit_constraints",
        "selected_design": {
            "name": "explicit_all_inputs_matern",
            "design_intent": "Follow the explicit request exactly.",
            "objective_assessment": {
                "fidelity_to_user_priors": {"score": 5, "reason": "Exact match."},
                "fit_stability": {"score": 4, "reason": "Simple single-block kernel."},
                "interpretability": {"score": 4, "reason": "Single block."},
                "expressiveness": {"score": 3, "reason": "Moderately expressive."},
                "repairability": {"score": 5, "reason": "Easy to simplify."},
            },
            "mean_config": {"kind": "multitask_constant"},
            "covar_config": {
                "blocks": [
                    {
                        "name": "features",
                        "variable_type": "generic",
                        "source_groups": [],
                        "source_features": ["macro_1", "etf_1"],
                        "block_structure": "additive",
                        "use_outputscale": False,
                        "components": [
                            {
                                "kind": "matern",
                                "ard": True,
                                "use_outputscale": True,
                                "matern_nu": 2.5,
                                "lengthscale_policy": {"policy": "adaptive"},
                                "period_prior": None,
                            }
                        ],
                    }
                ],
                "global_structure": "additive",
                "interaction_policy": "none",
                "custom_interactions": [],
            },
            "explicit_instruction_trace": [],
            "why_this_design_wins": [],
            "tradeoffs_accepted": [],
            "adjustments_from_user_request": [],
            "fit_risk": "low",
            "fit_risk_reasons": [],
            "repair_plan": [],
        },
    }

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}]}).encode("utf-8")

    monkeypatch.setattr(
        "bayesfolio.engine.agent.gp_planner.urllib_request.urlopen",
        lambda *args, **kwargs: _FakeResponse(),
    )

    response, status = plan_gp_design_with_status(_make_request())

    assert status == "ok"
    assert response.selected_design is not None
    assert response.selected_design.name == "explicit_all_inputs_matern"
