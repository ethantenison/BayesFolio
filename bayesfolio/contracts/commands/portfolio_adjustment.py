"""Contracts for a point-in-time LLM portfolio-adjustment decision.

This module defines data-only inputs for the agent layer. All portfolio weights
and returns use decimal units (``0.10 = 10%``). It performs no model calls,
research retrieval, portfolio optimization, or file/network I/O.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from bayesfolio.contracts.base import ContractModel, SchemaName, VersionedContract


class AssetForecastEvidence(ContractModel):
    """Point-in-time forecast information for one investable asset.

    Attributes:
        asset: Asset ticker or stable portfolio identifier.
        expected_excess_return: Predictive mean monthly excess return in decimal units.
        predictive_std: Predictive monthly standard deviation in decimal units.
    """

    asset: str = Field(min_length=1)
    expected_excess_return: float
    predictive_std: float = Field(gt=0.0)


class PortfolioRiskEvidence(ContractModel):
    """Posterior portfolio diagnostics for one candidate blend.

    Attributes:
        label: Stable label such as ``current``, ``optimizer``, or ``blend_0_50``.
        blend_fraction: Fraction of the move from current to optimizer weights.
        expected_return: Predictive mean monthly portfolio return in decimal units.
        volatility: Predictive monthly portfolio standard deviation in decimal units.
        probability_negative: Posterior probability of a negative monthly return.
        cvar_05: Mean monthly return in the worst five percent of scenarios, in decimal units.
    """

    label: str = Field(min_length=1)
    blend_fraction: float = Field(ge=0.0, le=1.0)
    expected_return: float
    volatility: float = Field(ge=0.0)
    probability_negative: float = Field(ge=0.0, le=1.0)
    cvar_05: float


class ResearchEvidence(ContractModel):
    """A dated research item available by the portfolio decision cutoff.

    Attributes:
        evidence_id: Stable identifier cited by the adjustment model.
        source_type: Source category used by the fixed retrieval procedure.
        source_name: Publisher or data-provider name.
        title: Human-readable source title.
        url: Source URL or stable archive locator.
        published_at: Timestamp when the information became publicly available.
        summary: Point-in-time factual summary supplied to the model.
        affected_assets: Assets or sleeves directly implicated by the evidence.
    """

    evidence_id: str = Field(min_length=1)
    source_type: Literal[
        "brokerage",
        "central_bank",
        "economic_release",
        "rates",
        "schwab_research",
        "financial_news",
    ]
    source_name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    published_at: AwareDatetime
    summary: str = Field(min_length=1)
    affected_assets: list[str] = Field(default_factory=list)


class RetrievalRecord(ContractModel):
    """Audit record for one deterministic evidence-retrieval operation.

    Attributes:
        tool_name: Stable semantic tool name required by the policy.
        arguments: Non-secret scalar arguments supplied to the tool.
        requested_at: Timestamp when retrieval started.
        completed_at: Timestamp when retrieval completed.
        success: Whether the retrieval completed successfully.
        evidence_ids: Evidence identifiers produced by the call.
        content_hash: Optional SHA-256 digest of the archived raw response.
    """

    tool_name: str = Field(min_length=1)
    arguments: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    requested_at: AwareDatetime
    completed_at: AwareDatetime
    success: bool
    evidence_ids: list[str] = Field(default_factory=list)
    content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_timing(self) -> "RetrievalRecord":
        """Require completion to occur no earlier than request time."""

        if self.completed_at < self.requested_at:
            raise ValueError("completed_at cannot precede requested_at")
        return self


class AdjustmentModelConfig(ContractModel):
    """Reproducibility settings for the structured LLM call.

    Attributes:
        provider: Provider label used in the audit record.
        model: Provider model identifier or local model path.
        temperature: Sampling temperature requested from the provider.
        seed: Provider seed requested for repeatability when supported.
        max_output_tokens: Maximum generated output tokens.
        max_validation_attempts: Maximum initial-plus-repair model calls.
        prompt_version: Version of the immutable adjustment prompt.
    """

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 16_638
    max_output_tokens: int = Field(default=2_000, ge=256, le=16_384)
    max_validation_attempts: int = Field(default=2, ge=1, le=3)
    prompt_version: Literal["1.0.0"] = "1.0.0"


class PortfolioAdjustmentPolicy(ContractModel):
    """Deterministic action space and retrieval requirements for adjustment.

    Attributes:
        allowed_blend_fractions: Permitted moves from current to optimizer weights.
        required_retrievals: Exact ordered semantic tool sequence required before inference.
        evidence_lookback_days: Maximum research lookback before the decision timestamp.
        allow_defer: Whether the model may recommend waiting for a named event.
    """

    allowed_blend_fractions: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    required_retrievals: list[str] = Field(
        default_factory=lambda: [
            "schwab_account_snapshot",
            "federal_reserve_releases",
            "bls_releases",
            "treasury_yield_curve",
            "schwab_market_commentary",
            "financial_news_search",
        ]
    )
    evidence_lookback_days: int = Field(default=90, ge=1, le=365)
    allow_defer: bool = True

    @model_validator(mode="after")
    def validate_action_space(self) -> "PortfolioAdjustmentPolicy":
        """Require a unique, sorted blend grid spanning the baseline and optimizer."""

        if self.allowed_blend_fractions != sorted(set(self.allowed_blend_fractions)):
            raise ValueError("allowed_blend_fractions must be unique and sorted")
        if not self.allowed_blend_fractions or self.allowed_blend_fractions[0] < 0.0:
            raise ValueError("allowed_blend_fractions must contain values in [0, 1]")
        if self.allowed_blend_fractions[-1] > 1.0:
            raise ValueError("allowed_blend_fractions must contain values in [0, 1]")
        if not math.isclose(self.allowed_blend_fractions[0], 0.0, abs_tol=1e-12):
            raise ValueError("allowed_blend_fractions must include 0.0")
        if not math.isclose(self.allowed_blend_fractions[-1], 1.0, abs_tol=1e-12):
            raise ValueError("allowed_blend_fractions must include 1.0")
        if len(self.required_retrievals) != len(set(self.required_retrievals)):
            raise ValueError("required_retrievals must not contain duplicates")
        return self


class GeneratePortfolioAdjustmentCommand(VersionedContract):
    """Generate a constrained LLM adjustment to optimizer portfolio weights.

    Attributes:
        schema: Stable schema identifier.
        schema_version: Semantic version of this contract.
        strategy_id: Stable identifier shared with the GP/optimizer strategy.
        as_of_date: Last market-data date used by the GP and optimizer.
        decision_at: Point-in-time cutoff for all account and research evidence.
        current_weights: Pre-decision portfolio weights in decimal units, summing to one.
        candidate_weights: Raw optimizer weights in decimal units, summing to one.
        forecasts: Asset-level predictive means and standard deviations.
        portfolio_risk: Posterior diagnostics for every allowed blend under review.
        evidence: Dated account, macroeconomic, Schwab, and financial-news evidence.
        retrievals: Ordered audit trail for the fixed evidence tool calls.
        account_tax_treatment: Tax context relevant to turnover interpretation.
        model_config_input: Model identity and deterministic generation settings.
        policy: Allowed action space and evidence-collection requirements.
    """

    schema: Literal[SchemaName.PORTFOLIO_ADJUSTMENT_COMMAND] = SchemaName.PORTFOLIO_ADJUSTMENT_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    strategy_id: str = Field(min_length=1)
    as_of_date: date
    decision_at: AwareDatetime
    current_weights: dict[str, float] = Field(min_length=1)
    candidate_weights: dict[str, float] = Field(min_length=1)
    forecasts: list[AssetForecastEvidence] = Field(min_length=1)
    portfolio_risk: list[PortfolioRiskEvidence] = Field(min_length=2)
    evidence: list[ResearchEvidence] = Field(min_length=1)
    retrievals: list[RetrievalRecord] = Field(min_length=1)
    account_tax_treatment: Literal["taxable", "tax_deferred_ira", "tax_free_roth_ira"]
    model_config_input: AdjustmentModelConfig
    policy: PortfolioAdjustmentPolicy = Field(default_factory=PortfolioAdjustmentPolicy)

    @model_validator(mode="after")
    def validate_point_in_time_packet(self) -> "GeneratePortfolioAdjustmentCommand":
        """Reject invalid weights, look-ahead evidence, or a changed retrieval procedure."""

        _validate_weight_map(self.current_weights, label="current_weights")
        _validate_weight_map(self.candidate_weights, label="candidate_weights")
        if self.decision_at.date() < self.as_of_date:
            raise ValueError("decision_at cannot precede as_of_date")

        forecast_assets = [item.asset for item in self.forecasts]
        if len(forecast_assets) != len(set(forecast_assets)):
            raise ValueError("forecasts must contain unique assets")
        known_assets = set(self.current_weights) | set(self.candidate_weights)
        unknown_forecast_assets = sorted(set(forecast_assets) - known_assets)
        if unknown_forecast_assets:
            raise ValueError(f"forecast assets are absent from both portfolios: {unknown_forecast_assets}")

        risk_labels = [item.label for item in self.portfolio_risk]
        if len(risk_labels) != len(set(risk_labels)):
            raise ValueError("portfolio_risk labels must be unique")
        available_blends = {item.blend_fraction for item in self.portfolio_risk}
        missing_blends = [
            blend
            for blend in self.policy.allowed_blend_fractions
            if not any(math.isclose(blend, available, abs_tol=1e-12) for available in available_blends)
        ]
        if missing_blends:
            raise ValueError(f"portfolio_risk is missing allowed blends: {missing_blends}")

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence must contain unique evidence_id values")
        future_evidence = [item.evidence_id for item in self.evidence if item.published_at > self.decision_at]
        if future_evidence:
            raise ValueError(f"evidence published after decision_at is not allowed: {future_evidence}")
        evidence_floor = self.decision_at - timedelta(days=self.policy.evidence_lookback_days)
        stale_evidence = [item.evidence_id for item in self.evidence if item.published_at < evidence_floor]
        if stale_evidence:
            raise ValueError(f"evidence predating the policy lookback is not allowed: {stale_evidence}")

        successful_retrievals = [item.tool_name for item in self.retrievals if item.success]
        if successful_retrievals != self.policy.required_retrievals:
            raise ValueError(
                "successful retrieval sequence must exactly match policy.required_retrievals; "
                f"got={successful_retrievals}"
            )
        unknown_retrieval_evidence = sorted(
            {
                evidence_id
                for retrieval in self.retrievals
                for evidence_id in retrieval.evidence_ids
                if evidence_id not in set(evidence_ids)
            }
        )
        if unknown_retrieval_evidence:
            raise ValueError(f"retrievals reference unknown evidence IDs: {unknown_retrieval_evidence}")
        return self


def _validate_weight_map(weights: dict[str, float], *, label: str) -> None:
    """Validate non-negative decimal weights that sum to one."""

    negative_assets = sorted(asset for asset, weight in weights.items() if weight < 0.0)
    if negative_assets:
        raise ValueError(f"{label} contains negative weights: {negative_assets}")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{label} must sum to 1.0")
