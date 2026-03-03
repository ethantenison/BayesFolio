from __future__ import annotations

import enum
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return the current UTC datetime with timezone info.

    Returns:
        Current UTC datetime.
    """
    return datetime.now(timezone.utc)


class SchemaName(str, enum.Enum):
    """Canonical schema identifiers for all BayesFolio contract types.

    Each value uniquely identifies a contract schema across the system.
    """

    COMMON_META = "bayesfolio.common.meta"
    COMMON_ENVELOPE = "bayesfolio.common.envelope"
    COMMON_PROBLEM_DETAILS = "bayesfolio.common.problem_details"
    SCENARIO_RECORD = "bayesfolio.scenario.record"
    SCENARIO_UI_INPUT = "bayesfolio.scenario.ui_input"
    UNIVERSE_RECORD = "bayesfolio.universe.record"
    UNIVERSE_UI_INPUT = "bayesfolio.universe.ui_input"
    OPTIMIZE_COMMAND = "bayesfolio.optimize.command"
    OPTIMIZE_RESULT = "bayesfolio.optimize.result"
    OPTIMIZE_PACKET = "bayesfolio.optimize.packet"
    FORECAST_COMMAND = "bayesfolio.forecast.command"
    FORECAST_RESULT = "bayesfolio.forecast.result"
    FORECAST_PACKET = "bayesfolio.forecast.packet"
    FORECAST_GP_FIT_REPORT = "bayesfolio.forecast.gp_fit_report"
    FORECAST_PREDICTIVE_DISTRIBUTION = "bayesfolio.forecast.predictive_distribution"
    BACKTEST_COMMAND = "bayesfolio.backtest.command"
    BACKTEST_RESULT = "bayesfolio.backtest.result"
    BACKTEST_PACKET = "bayesfolio.backtest.packet"
    REPORT_COMMAND = "bayesfolio.report.command"
    REPORT_RESULT = "bayesfolio.report.result"
    REPORT_PACKET = "bayesfolio.report.packet"
    ANALYZE_COMMAND = "bayesfolio.analyze.command"
    ANALYZE_RESULT = "bayesfolio.analyze.result"
    ANALYZE_PACKET = "bayesfolio.analyze.packet"
    JOB_SUBMIT_REQUEST = "bayesfolio.job.submit_request"
    JOB_HANDLE = "bayesfolio.job.handle"
    JOB_STATUS = "bayesfolio.job.status"
    JOB_RESULT_REF = "bayesfolio.job.result_ref"
    ARTIFACT_RECORD = "bayesfolio.artifact.record"
    ARTIFACT_POINTER = "bayesfolio.artifact.pointer"
    CHAT_TURN = "bayesfolio.chat.turn"
    CHAT_MESSAGE_USER = "bayesfolio.chat.message.user"
    CHAT_MESSAGE_ASSISTANT = "bayesfolio.chat.message.assistant"
    CHAT_INTENT_PARSED = "bayesfolio.chat.intent.parsed"
    CHAT_TOOL_CALL = "bayesfolio.chat.tool_call"
    CHAT_TOOL_RESULT = "bayesfolio.chat.tool_result"
    BELIEFS_COMMAND = "bayesfolio.beliefs.command"
    BELIEFS_RESULT = "bayesfolio.beliefs.result"
    UNIVERSE_COMMAND = "bayesfolio.universe.command"
    SCENARIO_COMMAND = "bayesfolio.scenario.command"


class ContractModel(BaseModel):
    """Base model for all BayesFolio contract types.

    Enforces strict field validation and consistent serialization settings
    across all contract schemas.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, str_strip_whitespace=True)


class Meta(ContractModel):
    """Metadata envelope attached to every contract.

    Attributes:
        schema: Schema identifier; always COMMON_META.
        schema_version: Semantic version of the schema.
        created_at: UTC datetime when the contract was created.
        correlation_id: Optional correlation ID for request tracing.
        request_id: Optional request identifier.
        producer: Optional name of the producing service or component.
        git_sha: Optional git commit SHA of the producing service.
        dataset_fingerprint: Optional SHA-256 hex digest of the input dataset.
        model_fingerprint: Optional SHA-256 hex digest of the model artifact.
    """

    schema: SchemaName = Field(default=SchemaName.COMMON_META, const=True)
    schema_version: str = "0.1.0"
    created_at: datetime = Field(default_factory=utc_now)
    correlation_id: str | None = None
    request_id: str | None = None
    producer: str | None = None
    git_sha: str | None = None
    dataset_fingerprint: str | None = None
    model_fingerprint: str | None = None


class Envelope(ContractModel):
    """Generic message envelope wrapping any contract payload.

    Attributes:
        meta: Contract metadata.
        data: Arbitrary payload data as a key-value mapping.
        diagnostics: Optional diagnostic or debug information.
    """

    meta: Meta
    data: dict[str, object]
    diagnostics: dict[str, object] = Field(default_factory=dict)


class FieldError(ContractModel):
    """Describes a single field-level validation error.

    Attributes:
        field: Name of the field that failed validation.
        message: Human-readable error message.
        error_type: Optional machine-readable error type code.
    """

    field: str
    message: str
    error_type: str | None = None


class ProblemDetails(ContractModel):
    """RFC 7807 problem details response.

    Attributes:
        problem_type: URI reference identifying the problem type.
        title: Short human-readable summary.
        status: HTTP status code.
        detail: Human-readable explanation of this specific occurrence.
        instance: URI reference identifying the specific occurrence.
        correlation_id: Optional correlation ID for request tracing.
        errors: List of field-level validation errors.
    """

    problem_type: str
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    correlation_id: str | None = None
    errors: list[FieldError] = Field(default_factory=list)


class VersionedContract(ContractModel):
    """Abstract base for all versioned BayesFolio contracts.

    Subclasses must override ``schema`` with a ``Field(default=SchemaName.X, const=True)``
    to enforce schema identity at the boundary.

    Attributes:
        schema: Schema identifier; subclasses override with a specific SchemaName.
        schema_version: Semantic version of the schema.
    """

    schema: SchemaName
    schema_version: str = "0.1.0"
