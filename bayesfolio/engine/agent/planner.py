from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PlanStep(BaseModel):
    """Deterministic planning step for no-LLM agent mode."""

    name: str
    status: str = Field(default="pending")
    details: str = Field(default="")

    model_config = ConfigDict(extra="forbid")


class PipelinePlan(BaseModel):
    """Serializable planner output for orchestration."""

    steps: list[PlanStep] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


def default_plan() -> PipelinePlan:
    """Return default deterministic plan for schema-first engine."""

    return PipelinePlan(
        steps=[
            PlanStep(name="universe"),
            PlanStep(name="forecast"),
            PlanStep(name="scenarios"),
            PlanStep(name="optimize"),
            PlanStep(name="backtest"),
            PlanStep(name="report"),
        ]
    )
