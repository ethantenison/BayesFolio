"""Result contracts for engine execution outputs.

This package exports versioned result schemas returned by pipeline workflows.
"""

from bayesfolio.contracts.results.features import (
    ArtifactPointer,
    FeatureColumnSpec,
    FeaturesDatasetResult,
    IndexInfo,
)

__all__ = [
    "ArtifactPointer",
    "FeatureColumnSpec",
    "FeaturesDatasetResult",
    "IndexInfo",
]
