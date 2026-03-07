from __future__ import annotations

from typing import Any

from bayesfolio.contracts.results.backtest import BacktestResult
from bayesfolio.contracts.results.features import FeaturesDatasetResult
from bayesfolio.contracts.results.report import DiagnosticFigureSummary, ReportResult
from bayesfolio.engine.report.visualization.eda import build_feature_diagnostic_figures


def assemble_report(
    backtest_result: BacktestResult,
    features_result: FeaturesDatasetResult | None = None,
) -> ReportResult:
    """Build report result from a backtest result."""

    diagnostic_figures: list[DiagnosticFigureSummary] = []
    if features_result is not None:
        figures = build_feature_diagnostic_figures(features_result)
        diagnostic_figures = _build_diagnostic_figure_summaries(figures)

    return ReportResult(
        headline_metrics={
            "cumulative_return": backtest_result.cumulative_return,
            "annualized_return": backtest_result.annualized_return,
            "annualized_volatility": backtest_result.annualized_volatility,
            "sharpe_ratio": backtest_result.sharpe_ratio,
            "max_drawdown": backtest_result.max_drawdown,
            "calmar_ratio": backtest_result.calmar_ratio,
            "sortino_ratio": backtest_result.sortino_ratio,
        },
        market_structure=features_result.market_structure if features_result is not None else None,
        diagnostic_figures=diagnostic_figures,
    )


def _build_diagnostic_figure_summaries(figures: dict[str, Any]) -> list[DiagnosticFigureSummary]:
    summaries: list[DiagnosticFigureSummary] = []
    for name, figure in figures.items():
        title = getattr(getattr(figure, "layout", None), "title", None)
        title_text = getattr(title, "text", None) if title is not None else None
        summaries.append(
            DiagnosticFigureSummary(
                name=name,
                title=str(title_text) if title_text else name,
                trace_count=len(getattr(figure, "data", [])),
            )
        )
    return summaries
