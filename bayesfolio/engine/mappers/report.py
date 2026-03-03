from __future__ import annotations

from bayesfolio.contracts.commands.report import ReportCommand
from bayesfolio.contracts.results.report import ArtifactPointer, ReportResult


def command_to_engine_inputs(command: ReportCommand) -> dict[str, object]:
    """Map a ReportCommand to keyword arguments for the report assembler.

    Args:
        command: The report command contract.

    Returns:
        Dict of engine keyword arguments (run_id, include_artifacts).
    """
    return {
        "run_id": command.run_id,
        "include_artifacts": command.include_artifacts,
    }


def engine_output_to_result(
    headline_metrics: dict[str, float],
    artifacts: list[ArtifactPointer] | None = None,
) -> ReportResult:
    """Map raw report assembler output to a ReportResult contract.

    Args:
        headline_metrics: Key performance metrics as decimals.
        artifacts: Optional list of artifact pointers.

    Returns:
        ReportResult contract.
    """
    return ReportResult(
        headline_metrics=headline_metrics,
        artifacts=artifacts or [],
    )
