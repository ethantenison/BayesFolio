"""LLM-backed planner client for structured multitask GP designs.

This module is responsible for calling an OpenAI-compatible chat-completions
endpoint and validating the planner's JSON response against BayesFolio GP
planner contracts. It does not fit models or perform workflow orchestration.
"""

from __future__ import annotations

import json
import os
from urllib import error as urllib_error
from urllib import request as urllib_request

from pydantic import ValidationError

from bayesfolio.contracts.commands.gp_planner import GPPlannerRequest
from bayesfolio.contracts.results.gp_workflow import (
    GPPlannerResponse,
    PlannerCovarSpec,
    PlannerKernelComponentSpec,
    PlannerLengthscalePolicySpec,
    PlannerMeanSpec,
    PlannerObjectiveAssessment,
    PlannerSelectedDesign,
    ScoreReason,
)
from bayesfolio.engine.agent.prompts import DEFAULT_GP_PLANNER_PROMPT


def plan_gp_design(request: GPPlannerRequest, timeout_seconds: float = 20.0) -> GPPlannerResponse:
    """Return a validated planner response for a multitask GP workflow.

    Args:
        request: Planner request contract containing structured planner input.
        timeout_seconds: HTTP timeout in seconds for the planner call.

    Returns:
        Validated planner response. When the LLM is unavailable, returns a
        deterministic fallback response.
    """

    response, _status = plan_gp_design_with_status(request=request, timeout_seconds=timeout_seconds)
    return response


def plan_gp_design_with_status(
    request: GPPlannerRequest,
    timeout_seconds: float = 20.0,
) -> tuple[GPPlannerResponse, str]:
    """Return a validated planner response and a client status code.

    Args:
        request: Planner request contract containing structured planner input.
        timeout_seconds: HTTP timeout in seconds for the planner call.

    Returns:
        Tuple of validated planner response and a planner client status code.
        Status values are fail-soft and mirror the style used by other LLM
        helper modules in this repository.
    """

    if not request.user_instruction_text.strip():
        return _build_fallback_response(request, instruction_mode="preference_optimization"), "empty_message"

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        return _build_fallback_response(request, instruction_mode=_instruction_mode(request)), "missing_openai_api_key"

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("BAYESFOLIO_GP_PLANNER_MODEL", "gpt-4o-mini")

    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": request.planner_prompt or DEFAULT_GP_PLANNER_PROMPT},
            {"role": "user", "content": _render_request_payload(request)},
        ],
        "temperature": 0,
    }

    body = json.dumps(payload).encode("utf-8")
    request_obj = urllib_request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(request_obj, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        return _build_fallback_response(request, instruction_mode=_instruction_mode(request)), f"http_error_{exc.code}"
    except (urllib_error.URLError, TimeoutError):
        return _build_fallback_response(request, instruction_mode=_instruction_mode(request)), "url_error"
    except json.JSONDecodeError:
        return _build_fallback_response(request, instruction_mode=_instruction_mode(request)), "invalid_response_json"

    choices = response_payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return _build_fallback_response(request, instruction_mode=_instruction_mode(request)), "no_choices"

    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        return _build_fallback_response(request, instruction_mode=_instruction_mode(request)), "empty_content"

    try:
        raw_response = json.loads(content)
    except json.JSONDecodeError:
        return _build_fallback_response(request, instruction_mode=_instruction_mode(request)), "invalid_content_json"

    if not isinstance(raw_response, dict):
        return _build_fallback_response(request, instruction_mode=_instruction_mode(request)), "non_object_json"

    try:
        validated = GPPlannerResponse.model_validate(raw_response)
    except ValidationError:
        return _build_fallback_response(request, instruction_mode=_instruction_mode(request)), "invalid_schema"

    return validated, "ok"


def _render_request_payload(request: GPPlannerRequest) -> str:
    """Render a structured JSON payload for the planner user message.

    Args:
        request: Planner request contract.

    Returns:
        JSON string containing the structured planner context.
    """

    return json.dumps(
        {
            "target_column": request.target_column,
            "task_column": request.task_column,
            "user_instruction_text": request.user_instruction_text,
            "feature_groups": request.feature_groups,
            "allowed_feature_columns": request.allowed_feature_columns,
            "dataset_summary": [item.model_dump(mode="json") for item in request.dataset_summary],
        },
        sort_keys=True,
    )


def _instruction_mode(request: GPPlannerRequest) -> str:
    lowered = request.user_instruction_text.lower()
    explicit_markers = (
        "use a ",
        "use an ",
        "use matern",
        "use additive",
        "use linear",
        "no interactions",
    )
    has_explicit_marker = any(marker in lowered for marker in explicit_markers)
    return "explicit_constraints" if has_explicit_marker else "preference_optimization"


def _build_fallback_response(request: GPPlannerRequest, *, instruction_mode: str) -> GPPlannerResponse:
    """Build a deterministic fallback planner response.

    Args:
        request: Planner request contract.
        instruction_mode: Planner instruction mode inferred from the request.

    Returns:
        Conservative fallback planner response.
    """

    fallback_groups = list(request.feature_groups)
    selected_groups = fallback_groups or ["all_features"]
    design = PlannerSelectedDesign(
        name="fallback_conservative_matern",
        design_intent="Deterministic fallback when planner output is unavailable.",
        objective_assessment=PlannerObjectiveAssessment(
            fidelity_to_user_priors=ScoreReason(score=3, reason="Fallback preserves explicit enforcement downstream."),
            fit_stability=ScoreReason(score=5, reason="Single conservative Matérn structure is robust."),
            interpretability=ScoreReason(score=4, reason="Single-block additive structure is easy to inspect."),
            expressiveness=ScoreReason(score=2, reason="Fallback intentionally favors stability over richness."),
            repairability=ScoreReason(score=5, reason="Few moving parts make deterministic repair straightforward."),
        ),
        mean_config=PlannerMeanSpec(kind="multitask_constant"),
        covar_config=PlannerCovarSpec(
            blocks=[
                {
                    "name": "features",
                    "variable_type": "generic",
                    "source_groups": [] if selected_groups == ["all_features"] else selected_groups,
                    "source_features": [] if selected_groups != ["all_features"] else [],
                    "block_structure": "additive",
                    "use_outputscale": False,
                    "components": [
                        PlannerKernelComponentSpec(
                            kind="matern",
                            ard=True,
                            use_outputscale=True,
                            matern_nu=2.5,
                            lengthscale_policy=PlannerLengthscalePolicySpec(policy="adaptive"),
                        )
                    ],
                }
            ],
            global_structure="additive",
            interaction_policy="none",
            custom_interactions=[],
        ),
        explicit_instruction_trace=["Planner fallback response generated; explicit instructions enforced downstream."],
        why_this_design_wins=["Fallback prioritizes stable fitting with the active multitask builder."],
        tradeoffs_accepted=["Reduced expressiveness while planner output is unavailable."],
        adjustments_from_user_request=[],
        fit_risk="low",
        fit_risk_reasons=["Single Matérn block is conservative and repairable."],
        repair_plan=[
            "increase min_inferred_noise_level",
            "simplify interactions",
            "collapse to one conservative Matérn block",
        ],
    )
    return GPPlannerResponse(
        planner_status="ok",
        clarification_questions=[],
        dataset_assumptions=["Fallback planner response used because live planner output was unavailable."],
        instruction_mode=instruction_mode,
        selected_design=design,
    )
