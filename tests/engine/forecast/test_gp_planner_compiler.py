from __future__ import annotations

from bayesfolio.contracts.results.gp_workflow import (
    GPPlannerResponse,
    PlannerCovarSpec,
    PlannerKernelComponentSpec,
    PlannerMeanSpec,
    PlannerObjectiveAssessment,
    PlannerSelectedDesign,
    ScoreReason,
)
from bayesfolio.engine.forecast.gp.multitask_builder import InteractionPolicy, MaternKernelComponentConfig
from bayesfolio.engine.forecast.gp.planner_compiler import compile_planner_response


def _make_response() -> GPPlannerResponse:
    return GPPlannerResponse(
        planner_status="ok",
        clarification_questions=[],
        dataset_assumptions=[],
        instruction_mode="preference_optimization",
        selected_design=PlannerSelectedDesign(
            name="planner_design",
            design_intent="Baseline planner output.",
            objective_assessment=PlannerObjectiveAssessment(
                fidelity_to_user_priors=ScoreReason(score=3, reason="Baseline."),
                fit_stability=ScoreReason(score=3, reason="Baseline."),
                interpretability=ScoreReason(score=3, reason="Baseline."),
                expressiveness=ScoreReason(score=3, reason="Baseline."),
                repairability=ScoreReason(score=3, reason="Baseline."),
            ),
            mean_config=PlannerMeanSpec(kind="multitask_constant"),
            covar_config=PlannerCovarSpec(
                blocks=[
                    {
                        "name": "macro",
                        "variable_type": "macro",
                        "source_groups": ["macro"],
                        "source_features": [],
                        "block_structure": "additive",
                        "use_outputscale": True,
                        "components": [
                            PlannerKernelComponentSpec(
                                kind="linear",
                                use_outputscale=False,
                            )
                        ],
                    }
                ],
                global_structure="hierarchical",
                interaction_policy="sparse",
                custom_interactions=[],
            ),
            explicit_instruction_trace=[],
            why_this_design_wins=[],
            tradeoffs_accepted=[],
            adjustments_from_user_request=[],
            fit_risk="medium",
            fit_risk_reasons=[],
            repair_plan=[],
        ),
    )


def test_compile_planner_response_enforces_all_inputs_matern_ard() -> None:
    compiled = compile_planner_response(
        _make_response(),
        feature_columns=["macro_1", "etf_1", "etf_2"],
        feature_groups={"macro": ["macro_1"], "etf": ["etf_1", "etf_2"]},
        instruction_text="Use a matern kernel with ard for all input variables",
    )

    assert len(compiled.covar_config.blocks) == 1
    component = compiled.covar_config.blocks[0].components[0]
    assert isinstance(component, MaternKernelComponentConfig)
    assert component.ard is True
    assert component.dims == [0, 1, 2]
    assert compiled.covar_config.interaction_policy is InteractionPolicy.NONE


def test_compile_planner_response_enforces_block_specific_matern_nu() -> None:
    compiled = compile_planner_response(
        _make_response(),
        feature_columns=["macro_1", "etf_1"],
        feature_groups={"macro": ["macro_1"], "etf": ["etf_1"]},
        instruction_text="Use matern 5/2 on macro and matern 1/2 on etf",
    )

    blocks = {block.name: block for block in compiled.covar_config.blocks}
    assert blocks["macro"].components[0].matern_nu == 2.5
    assert blocks["etf"].components[0].matern_nu == 0.5


def test_compile_planner_response_enforces_linear_plus_matern_on_macro() -> None:
    compiled = compile_planner_response(
        _make_response(),
        feature_columns=["macro_1", "etf_1"],
        feature_groups={"macro": ["macro_1"], "etf": ["etf_1"]},
        instruction_text="Use a linear plus matern kernel on macro",
    )

    macro_block = next(block for block in compiled.covar_config.blocks if block.name == "macro")
    assert {component.kind.value for component in macro_block.components} == {"linear", "matern"}
