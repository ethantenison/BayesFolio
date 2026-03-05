from __future__ import annotations

from pathlib import Path
from typing import Any

from bayesfolio.contracts.results.report import ReportResult
from bayesfolio.io.artifacts import save_plotly_figures_html
from bayesfolio.io.backends import ArtifactBackend


def persist_report_diagnostic_figures(
    report_result: ReportResult,
    figures: dict[str, Any],
    *,
    output_dir: str | Path = "reports/diagnostics",
    root_uri: str | Path | None = None,
    backend: ArtifactBackend | None = None,
) -> ReportResult:
    """Persist diagnostic figures and append artifact pointers to report result.

    Args:
        report_result: Existing report result contract.
        figures: Mapping of figure name to Plotly-compatible figures.
        output_dir: Output directory for generated ``.html`` files.
        root_uri: Optional backend root URI/path for relative outputs.
        backend: Optional explicit backend implementation.

    Returns:
        Updated report result with appended artifact pointers.
    """

    if not figures:
        return report_result

    pointers = save_plotly_figures_html(
        figures=figures,
        output_dir=output_dir,
        root_uri=root_uri,
        backend=backend,
    )
    return report_result.model_copy(update={"artifacts": [*report_result.artifacts, *pointers]})
