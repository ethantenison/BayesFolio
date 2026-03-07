"""Command contracts for engine execution inputs.

This package exports versioned command schemas consumed at pipeline boundaries.
"""

from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand

__all__ = ["BuildFeaturesDatasetCommand"]
