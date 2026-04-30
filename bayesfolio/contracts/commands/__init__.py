"""Command contracts for engine execution inputs.

This package exports versioned command schemas consumed at pipeline boundaries.
"""

from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.contracts.commands.gp_planner import GPPlannerRequest

__all__ = ["BuildFeaturesDatasetCommand", "GPPlannerRequest"]
