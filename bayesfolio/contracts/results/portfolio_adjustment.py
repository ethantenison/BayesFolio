"""Structured results for constrained LLM portfolio adjustments.

The result records the model's evidence assessment and the deterministic blend
applied to portfolio weights. All weights and returns use decimal units
(``0.10 = 10%``); this module contains no inference or I/O behavior.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import AwareDatetime, Field

from bayesfolio.contracts.base import ContractModel, SchemaName, VersionedContract


class EvidenceAssessment(ContractModel):
    """Model assessment of one cited research item.

    Attributes:
        evidence_id: Identifier from the input evidence packet.
        direction: Whether the item supports or contradicts the optimizer target.
        materiality: Estimated importance to the rebalance decision.
        affected_assets: Assets or sleeves to which the assessment applies.
        rationale: Concise explanation grounded in the cited item.
    """

    evidence_id: str = Field(min_length=1)
    direction: Literal["supports", "contradicts", "mixed", "neutral"]
    materiality: Literal["low", "medium", "high"]
    affected_assets: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class ModelAdjustmentDecision(ContractModel):
    """Restricted decision emitted by an LLM before deterministic blending.

    Attributes:
        selected_blend_fraction: Chosen fraction of the current-to-optimizer move.
        timing: Whether to rebalance now, stage execution, or defer for an event.
        confidence: Qualitative confidence in the adjustment recommendation.
        decision_summary: Concise recommendation suitable for an audit report.
        model_signal_assessment: Assessment of forecast signal relative to uncertainty.
        uncertainty_assessment: Assessment of portfolio-level posterior risk.
        evidence_assessments: Source-by-source comparison with the optimizer.
        cited_evidence_ids: Evidence identifiers supporting the decision.
        key_risks: Important failure modes or unresolved risks.
        defer_until: Named event or condition when timing is ``defer``; otherwise null.
    """

    selected_blend_fraction: float = Field(ge=0.0, le=1.0)
    timing: Literal["rebalance_now", "stage", "defer"]
    confidence: Literal["low", "medium", "high"]
    decision_summary: str = Field(min_length=1)
    model_signal_assessment: str = Field(min_length=1)
    uncertainty_assessment: str = Field(min_length=1)
    evidence_assessments: list[EvidenceAssessment] = Field(min_length=1)
    cited_evidence_ids: list[str] = Field(min_length=1)
    key_risks: list[str] = Field(min_length=1)
    defer_until: str | None = None


class PortfolioAdjustmentResult(VersionedContract):
    """Auditable final portfolio adjustment produced by the fixed procedure.

    Attributes:
        schema: Stable schema identifier.
        schema_version: Semantic version of this contract.
        strategy_id: Strategy identifier from the adjustment command.
        as_of_date: Last market-data date used by the GP and optimizer.
        decision_at: Point-in-time evidence cutoff.
        provider: Model provider label.
        model: Provider model identifier or local model path.
        prompt_version: Immutable prompt version used for the decision.
        input_hash: SHA-256 digest of canonical command JSON.
        prompt_hash: SHA-256 digest of the exact system and user prompts.
        response_hash: SHA-256 digest of the accepted raw model response.
        validation_attempts: Number of initial-plus-repair model calls used.
        decision: Validated restricted model decision.
        current_weights: Pre-decision weights in decimal units.
        candidate_weights: Raw optimizer weights in decimal units.
        adjusted_weights: Deterministic blended weights in decimal units.
        one_way_turnover: One-way turnover from current to adjusted weights.
        retrieval_tool_names: Ordered successful semantic retrieval calls.
        evidence_ids: Evidence identifiers made available to the model.
    """

    schema: Literal[SchemaName.PORTFOLIO_ADJUSTMENT_RESULT] = SchemaName.PORTFOLIO_ADJUSTMENT_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    strategy_id: str = Field(min_length=1)
    as_of_date: date
    decision_at: AwareDatetime
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_attempts: int = Field(ge=1)
    decision: ModelAdjustmentDecision
    current_weights: dict[str, float]
    candidate_weights: dict[str, float]
    adjusted_weights: dict[str, float]
    one_way_turnover: float = Field(ge=0.0, le=1.0)
    retrieval_tool_names: list[str]
    evidence_ids: list[str]
