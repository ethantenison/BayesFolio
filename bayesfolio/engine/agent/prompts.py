from __future__ import annotations

DEFAULT_PLANNER_PROMPT = """You are BayesFolio planner. Produce deterministic structured plan outputs only."""
DEFAULT_INTENT_PROMPT = """Extract objective, risk measure, and constraints from structured user request."""
DEFAULT_GP_PLANNER_PROMPT = (
    "You are BayesFolio GP Planning Agent. Return only JSON matching the requested planner schema. "
    "Do not emit markdown, prose, or code fences."
)
