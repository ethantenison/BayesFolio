"""Application composition root for the supported monthly portfolio workflow.

This module assembles concrete providers, artifact storage, and runtime I/O,
then delegates all monthly feature, GP, scenario, and optimization decisions to
the engine. It is the canonical zero-configuration programmatic entry point.
"""

from __future__ import annotations

from bayesfolio.contracts.commands.monthly_portfolio import MonthlyPortfolioCommand
from bayesfolio.contracts.results.monthly_portfolio import MonthlyPortfolioResult
from bayesfolio.engine.automation.monthly_portfolio import (
    MonthlyPortfolioRuntimeDependencies,
    execute_monthly_portfolio,
)
from bayesfolio.engine.features import make_default_feature_providers
from bayesfolio.io.artifact_store import ParquetArtifactStore
from bayesfolio.io.monthly_portfolio_runtime import build_monthly_portfolio_run_io


def build_default_monthly_portfolio_dependencies(
    command: MonthlyPortfolioCommand,
) -> MonthlyPortfolioRuntimeDependencies:
    """Build all concrete dependencies for a monthly portfolio run.

    Args:
        command: Validated monthly workflow configuration.

    Returns:
        Complete dependency bundle ready for engine execution.
    """

    return MonthlyPortfolioRuntimeDependencies(
        feature_providers=make_default_feature_providers(cache_root="artifacts/cache"),
        artifact_store=ParquetArtifactStore(base_dir=command.artifacts.artifact_root),
        run_io=build_monthly_portfolio_run_io(command.artifacts),
    )


def complete_monthly_portfolio_dependencies(
    command: MonthlyPortfolioCommand,
    dependencies: MonthlyPortfolioRuntimeDependencies | None,
) -> MonthlyPortfolioRuntimeDependencies:
    """Complete absent or legacy dependency bundles with runtime I/O.

    Args:
        command: Validated monthly workflow configuration.
        dependencies: Optional dependency bundle. Existing provider and store
            instances are preserved; a missing runtime adapter is supplied.

    Returns:
        Complete dependency bundle ready for engine execution.
    """

    if dependencies is None:
        return build_default_monthly_portfolio_dependencies(command)
    if dependencies.run_io is not None:
        return dependencies
    return MonthlyPortfolioRuntimeDependencies(
        feature_providers=dependencies.feature_providers,
        artifact_store=dependencies.artifact_store,
        run_io=build_monthly_portfolio_run_io(command.artifacts),
    )


def run_monthly_portfolio(
    command: MonthlyPortfolioCommand,
    *,
    dependencies: MonthlyPortfolioRuntimeDependencies | None = None,
) -> MonthlyPortfolioResult:
    """Run the canonical monthly workflow with application-level composition.

    Args:
        command: Validated monthly workflow configuration.
        dependencies: Optional custom or partially supplied runtime dependencies.

    Returns:
        Structured monthly result containing forecasts, weights, and artifacts.
    """

    runtime = complete_monthly_portfolio_dependencies(command, dependencies)
    return execute_monthly_portfolio(command, dependencies=runtime)
