"""Result contracts for engine execution outputs.

This package exports versioned result schemas returned by pipeline workflows.
"""

from bayesfolio.contracts.results.features import (
    ArtifactPointer,
    FeatureColumnSpec,
    FeaturesDatasetResult,
    IndexInfo,
)
from bayesfolio.contracts.results.gp_workflow import (
    GPFitValidationSummary,
    GPPlannerResponse,
    GPRepairAttempt,
    GPWorkflowResult,
    PlannerSelectedDesign,
    ResolvedFeatureBlock,
)

__all__ = [
    "ArtifactPointer",
    "FeatureColumnSpec",
    "FeaturesDatasetResult",
    "GPFitValidationSummary",
    "GPPlannerResponse",
    "GPRepairAttempt",
    "GPWorkflowResult",
    "IndexInfo",
    "PlannerSelectedDesign",
    "ResolvedFeatureBlock",
]
