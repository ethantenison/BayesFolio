"""Historical-only MVP chat orchestration for BayesFolio.

This module implements a minimal multi-agent workflow behind a chat interface.
It includes deterministic parsing, universe construction, data-quality checks,
optional feature building, historical Riskfolio optimization, and report
assembly. A chat-tool wrapper exposes the workflow through ``ChatTurn`` and
``ChatToolCall`` contracts executed by the orchestrator state machine.

Execution Flow:
    1. Parse request text into ``HistoricalMvpRequest``.
    2. Universe Agent builds historical returns matrix + ``UniverseRecord``.
    3. Data Quality Agent computes gate diagnostics.
    4. Feature Agent optionally builds/persists feature artifact (fail-soft).
    5. Optimization Agent runs historical Riskfolio allocation.
    6. Report Agent assembles markdown + tabular outputs.
    7. Chat wrapper executes this pipeline via ``ChatToolCall`` and
       ``run_orchestration_cycle`` to produce a ``ChatTurn``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from bayesfolio.contracts.chat.protocol import (
    ChatMessageAssistant,
    ChatMessageUser,
    ChatToolCall,
    ChatTurn,
)
from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.contracts.commands.optimize import OptimizeCommand
from bayesfolio.contracts.commands.universe import UniverseCommand
from bayesfolio.contracts.results.features import FeaturesDatasetResult
from bayesfolio.contracts.results.optimize import OptimizeResult
from bayesfolio.contracts.ui.universe import UniverseRecord
from bayesfolio.core.settings import Horizon, Interval
from bayesfolio.engine.agent.orchestrator import run_orchestration_cycle
from bayesfolio.engine.asset_allocation import optimize_from_historical_returns
from bayesfolio.engine.features import (
    build_features_dataset,
    build_long_panel,
    fetch_etf_features,
    fetch_macro_features,
)
from bayesfolio.engine.features.dataset_builder import FeatureProviders
from bayesfolio.engine.features.universe_loader import build_universe_snapshot
from bayesfolio.io import (
    EtfFeaturesProvider,
    MacroProvider,
    ParquetArtifactStore,
    RegistryToolExecutor,
    ReturnsProvider,
)


@dataclass(frozen=True)
class HistoricalMvpRequest:
    """Request payload for historical-only MVP portfolio construction.

    Attributes:
        tickers: Asset tickers to include in the universe.
        start_date: Inclusive start date for historical returns.
        end_date: Inclusive end date for historical returns.
        objective: Riskfolio objective (for example ``Sharpe``).
        risk_measure: Riskfolio risk measure (for example ``CVaR``).
        min_weight: Minimum portfolio weight as decimal.
        max_weight: Maximum portfolio weight as decimal.
        build_features: Whether to run the feature agent.
        lookback_days: Additional lookback window for feature generation.
        use_local_cache: Whether to read/write local market-data cache.
        cache_dir: Base directory for local market-data cache files.
    """

    tickers: list[str]
    start_date: date
    end_date: date
    objective: str = "Sharpe"
    risk_measure: str = "CVaR"
    min_weight: float = 0.0
    max_weight: float = 0.35
    build_features: bool = True
    lookback_days: int = 365
    use_local_cache: bool = True
    cache_dir: str = "artifacts/cache"


@dataclass(frozen=True)
class DataQualityResult:
    """Data quality diagnostics for historical returns.

    Attributes:
        pass_gate: Whether returns pass minimum quality checks.
        n_periods: Number of historical periods.
        n_assets: Number of assets.
        missing_rate_by_asset: Missing-rate diagnostics by ticker.
        stale_assets: Assets with no variation in returns.
        insufficient_history_assets: Assets with too few observations.
    """

    pass_gate: bool
    n_periods: int
    n_assets: int
    missing_rate_by_asset: dict[str, float]
    stale_assets: list[str]
    insufficient_history_assets: list[str]


@dataclass(frozen=True)
class HistoricalMvpResult:
    """Coordinator output payload for the Streamlit MVP workflow.

    Attributes:
        request: Normalized request used by the coordinator.
        universe: Universe snapshot from the universe agent.
        data_quality: Data quality diagnostics.
        optimize_result: Optimization output weights.
        report_markdown: Human-readable report text.
        weights_table: Weights table for display.
        agent_logs: Ordered coordinator logs.
        warnings: Non-fatal warnings surfaced during execution.
        features_result: Optional feature artifact result.
    """

    request: HistoricalMvpRequest
    universe: UniverseRecord
    data_quality: DataQualityResult
    optimize_result: OptimizeResult
    report_markdown: str
    weights_table: pd.DataFrame
    agent_logs: list[str]
    warnings: list[str] = field(default_factory=list)
    features_result: FeaturesDatasetResult | None = None


def parse_chat_request(message: str, today: date | None = None) -> HistoricalMvpRequest:
    """Parse a free-form chat request into a normalized coordinator request.

    Args:
        message: User message containing tickers and optional settings.
        today: Optional date override for deterministic tests.

    Returns:
        Parsed and normalized request object.

    Raises:
        ValueError: If no tickers can be parsed.
    """

    anchors = today or date.today()
    parsed_tickers = _extract_tickers(message)
    if not parsed_tickers:
        msg = "Could not parse any tickers. Include tickers like SPY, QQQ, TLT."
        raise ValueError(msg)

    parsed_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", message)
    if len(parsed_dates) >= 2:
        start_date = date.fromisoformat(parsed_dates[0])
        end_date = date.fromisoformat(parsed_dates[1])
    else:
        end_date = anchors
        start_date = end_date - timedelta(days=365 * 5)

    objective = _extract_objective(message)
    risk_measure = _extract_risk_measure(message)
    return HistoricalMvpRequest(
        tickers=parsed_tickers,
        start_date=start_date,
        end_date=end_date,
        objective=objective,
        risk_measure=risk_measure,
    )


def assess_data_quality(returns: pd.DataFrame, min_observations: int = 24) -> DataQualityResult:
    """Run lightweight data quality checks for historical returns.

    Args:
        returns: Returns matrix indexed by date and columns as assets.
        min_observations: Minimum non-null observations required per asset.

    Returns:
        Structured data quality result.
    """

    n_periods = int(returns.shape[0])
    n_assets = int(returns.shape[1])
    missing_rate_by_asset = {str(asset): float(rate) for asset, rate in returns.isna().mean().to_dict().items()}

    stale_assets: list[str] = []
    insufficient_history_assets: list[str] = []
    for asset in returns.columns:
        series = pd.to_numeric(returns[asset], errors="coerce").dropna()
        if len(series) < min_observations:
            insufficient_history_assets.append(str(asset))
        if series.nunique() <= 1:
            stale_assets.append(str(asset))

    pass_gate = (
        n_periods >= min_observations
        and n_assets >= 2
        and len(stale_assets) == 0
        and len(insufficient_history_assets) == 0
    )

    return DataQualityResult(
        pass_gate=pass_gate,
        n_periods=n_periods,
        n_assets=n_assets,
        missing_rate_by_asset=missing_rate_by_asset,
        stale_assets=stale_assets,
        insufficient_history_assets=insufficient_history_assets,
    )


def run_historical_mvp_pipeline(
    request: HistoricalMvpRequest,
    progress: Callable[[str], None] | None = None,
) -> HistoricalMvpResult:
    """Run the MVP multi-agent historical portfolio workflow.

    Args:
        request: Parsed historical pipeline request.
        progress: Optional callback for streaming progress messages.

    Returns:
        End-to-end MVP output for chatbot rendering.
    """

    agent_logs: list[str] = []
    warnings: list[str] = []

    def _log(message: str) -> None:
        agent_logs.append(message)
        if progress is not None:
            progress(message)

    _log("Universe Agent: building monthly historical return matrix.")
    universe, returns_matrix = _run_universe_agent(request)

    _log("Data Quality Agent: evaluating missingness, staleness, and history depth.")
    quality = assess_data_quality(returns_matrix)
    if not quality.pass_gate:
        warnings.append("Data quality checks flagged issues; optimization proceeds in MVP mode.")

    features_result: FeaturesDatasetResult | None = None
    if request.build_features:
        _log("Feature Agent: building and persisting features dataset.")
        try:
            features_result = _run_feature_agent(request)
        except Exception as exc:
            warnings.append(f"Feature Agent failed and was skipped: {exc}")

    _log("Optimization Agent: running historical Riskfolio optimization.")
    optimize_command = OptimizeCommand(
        objective=request.objective,
        risk_measure=request.risk_measure,
        min_weight=request.min_weight,
        max_weight=request.max_weight,
        hist=True,
    )
    optimize_result = optimize_from_historical_returns(returns=returns_matrix, request=optimize_command)

    _log("Report Agent: assembling summary report output.")
    weights_table = pd.DataFrame(
        {
            "asset": optimize_result.asset_order,
            "weight": optimize_result.weights,
        }
    ).sort_values("weight", ascending=False, ignore_index=True)
    report_markdown = _render_report_markdown(
        request=request,
        universe=universe,
        data_quality=quality,
        optimize_result=optimize_result,
        warnings=warnings,
        features_result=features_result,
    )

    return HistoricalMvpResult(
        request=request,
        universe=universe,
        data_quality=quality,
        optimize_result=optimize_result,
        report_markdown=report_markdown,
        weights_table=weights_table,
        agent_logs=agent_logs,
        warnings=warnings,
        features_result=features_result,
    )


def run_historical_mvp_chat_turn(message: str, progress: Callable[[str], None] | None = None) -> ChatTurn:
    """Run one chat turn through the orchestrator using a real tool execution path.

    Args:
        message: User chat message containing portfolio request context.
        progress: Optional callback for streaming progress updates.

    Returns:
        ChatTurn containing executed tool result payload and assistant message.
    """

    request = parse_chat_request(message)
    turn = ChatTurn(user_message=ChatMessageUser(content=message))
    turn.tool_calls = [
        ChatToolCall(
            call_id="call_001",
            tool_name="run_historical_mvp_pipeline",
            arguments={"request": _request_to_payload(request)},
        )
    ]

    executor = RegistryToolExecutor(
        handlers={
            "run_historical_mvp_pipeline": lambda arguments: _run_mvp_tool(
                arguments=arguments,
                progress=progress,
            )
        }
    )

    updated_turn = run_orchestration_cycle(turn=turn, tool_executor=executor)
    if not updated_turn.tool_results:
        updated_turn.assistant_message = ChatMessageAssistant(content="No tool result was produced.")
        return updated_turn

    latest = updated_turn.tool_results[-1]
    if latest.success:
        assistant_text = str(latest.payload.get("report_markdown", "Historical MVP run completed."))
    else:
        assistant_text = f"MVP run failed: {latest.error_message or 'unknown tool error'}"
    updated_turn.assistant_message = ChatMessageAssistant(content=assistant_text)
    return updated_turn


def _run_universe_agent(request: HistoricalMvpRequest) -> tuple[UniverseRecord, pd.DataFrame]:
    """Build universe snapshot and historical returns matrix for optimization.

    Args:
        request: Normalized historical MVP request.

    Returns:
        Tuple containing:
            - UniverseRecord with canonical asset order and metadata.
            - Returns matrix indexed by date with assets as columns and
              decimal excess-return labels.

    Raises:
        ValueError: If upstream returns fetch or filtered return matrix is empty.
    """

    returns_provider = ReturnsProvider(
        fetcher=build_long_panel,
        cache_dir=_cache_dir_for(request=request, dataset="returns"),
    )
    returns_long = returns_provider.get_y_excess_lead_long(
        tickers=request.tickers,
        start=request.start_date.isoformat(),
        end=request.end_date.isoformat(),
        horizon=Horizon.MONTHLY,
    )
    if returns_long.empty:
        msg = "Universe Agent returned no return rows for the requested range."
        raise ValueError(msg)

    returns_matrix = (
        returns_long.pivot(index="date", columns="asset_id", values="y_excess_lead")
        .sort_index()
        .loc[request.start_date.isoformat() : request.end_date.isoformat()]
        .dropna(how="all")
    )
    if returns_matrix.empty:
        msg = "No historical returns available after pivot/filtering."
        raise ValueError(msg)

    universe_request = UniverseCommand(
        tickers=request.tickers,
        start_date=request.start_date.isoformat(),
        end_date=request.end_date.isoformat(),
        return_unit="decimal",
    )
    universe = build_universe_snapshot(returns=returns_matrix, request=universe_request)
    return universe, returns_matrix


def _run_feature_agent(request: HistoricalMvpRequest) -> FeaturesDatasetResult:
    """Run optional feature engineering and persistence stage.

    Args:
        request: Normalized historical MVP request.

    Returns:
        FeaturesDatasetResult with artifact pointer and diagnostics.
    """

    feature_command = BuildFeaturesDatasetCommand(
        tickers=request.tickers,
        lookback_date=request.start_date - timedelta(days=request.lookback_days),
        start_date=request.start_date,
        end_date=request.end_date,
        interval=Interval.DAILY,
        horizon=Horizon.MONTHLY,
        artifact_name=(f"mvp_features_{request.start_date.isoformat()}_{request.end_date.isoformat()}"),
    )
    providers = FeatureProviders(
        returns_provider=ReturnsProvider(
            fetcher=build_long_panel,
            cache_dir=_cache_dir_for(request=request, dataset="returns"),
        ),
        macro_provider=MacroProvider(
            fetcher=fetch_macro_features,
            max_retries=1,
            retry_backoff_seconds=0.0,
            cache_dir=_cache_dir_for(request=request, dataset="macro"),
        ),
        etf_features_provider=EtfFeaturesProvider(
            fetcher=fetch_etf_features,
            cache_dir=_cache_dir_for(request=request, dataset="etf_features"),
        ),
    )
    artifact_store = ParquetArtifactStore(base_dir="artifacts/features")
    return build_features_dataset(feature_command, providers=providers, artifact_store=artifact_store)


def _render_report_markdown(
    request: HistoricalMvpRequest,
    universe: UniverseRecord,
    data_quality: DataQualityResult,
    optimize_result: OptimizeResult,
    warnings: list[str],
    features_result: FeaturesDatasetResult | None,
) -> str:
    """Render compact markdown summary for chat display.

    Args:
        request: Original pipeline request.
        universe: Universe snapshot contract.
        data_quality: Data quality diagnostics contract.
        optimize_result: Portfolio optimization result.
        warnings: Non-fatal warnings generated by earlier agents.
        features_result: Optional features artifact metadata.

    Returns:
        Markdown report string summarizing configuration, diagnostics, and top
        portfolio weights.
    """

    weights = pd.Series(optimize_result.weights, index=optimize_result.asset_order).sort_values(ascending=False)
    top_assets = ", ".join(f"{asset}: {weight:.1%}" for asset, weight in weights.head(3).items())

    feature_text = "Feature artifact not available."
    if features_result is not None:
        feature_text = (
            f"Feature artifact: {features_result.artifact.uri} "
            f"(rows={features_result.artifact.row_count}, cols={features_result.artifact.column_count})."
        )

    warning_text = "None" if not warnings else " | ".join(warnings)
    return (
        "### Historical MVP Portfolio Report\n"
        f"- Universe size: {len(universe.asset_order)} assets\n"
        f"- Date range: {request.start_date.isoformat()} to {request.end_date.isoformat()}\n"
        f"- Objective / Risk: {request.objective} / {request.risk_measure}\n"
        f"- Data quality gate: {'PASS' if data_quality.pass_gate else 'FLAGGED'}\n"
        f"- Top weights: {top_assets}\n"
        f"- {feature_text}\n"
        f"- Warnings: {warning_text}"
    )


def _request_to_payload(request: HistoricalMvpRequest) -> dict[str, object]:
    """Convert typed request dataclass into JSON-safe tool payload.

    Args:
        request: Historical MVP request object.

    Returns:
        Dictionary payload with ISO-formatted dates and primitive values.
    """

    payload = asdict(request)
    payload["start_date"] = request.start_date.isoformat()
    payload["end_date"] = request.end_date.isoformat()
    return payload


def _payload_to_request(payload: dict[str, object]) -> HistoricalMvpRequest:
    """Rehydrate typed request from tool payload dictionary.

    Args:
        payload: Tool payload containing request fields.

    Returns:
        Parsed HistoricalMvpRequest with typed dates and numeric settings.
    """

    tickers_value = payload.get("tickers", [])
    if isinstance(tickers_value, list):
        tickers = [str(ticker) for ticker in tickers_value]
    else:
        tickers = []

    start_raw = payload.get("start_date")
    end_raw = payload.get("end_date")
    if start_raw is None or end_raw is None:
        msg = "Request payload must include start_date and end_date."
        raise ValueError(msg)

    min_weight_raw = payload.get("min_weight", 0.0)
    max_weight_raw = payload.get("max_weight", 0.35)
    lookback_days_raw = payload.get("lookback_days", 365)
    use_local_cache_raw = payload.get("use_local_cache", True)
    cache_dir_raw = payload.get("cache_dir", "artifacts/cache")

    min_weight = float(str(min_weight_raw))
    max_weight = float(str(max_weight_raw))
    lookback_days = int(str(lookback_days_raw))
    use_local_cache = bool(use_local_cache_raw)
    cache_dir = str(cache_dir_raw)

    return HistoricalMvpRequest(
        tickers=tickers,
        start_date=date.fromisoformat(str(start_raw)),
        end_date=date.fromisoformat(str(end_raw)),
        objective=str(payload.get("objective", "Sharpe")),
        risk_measure=str(payload.get("risk_measure", "CVaR")),
        min_weight=min_weight,
        max_weight=max_weight,
        build_features=bool(payload.get("build_features", True)),
        lookback_days=lookback_days,
        use_local_cache=use_local_cache,
        cache_dir=cache_dir,
    )


def _cache_dir_for(request: HistoricalMvpRequest, dataset: str) -> str | None:
    """Resolve optional local cache directory for a market-data dataset.

    Args:
        request: Historical MVP request with cache settings.
        dataset: Dataset bucket name (for example ``returns``).

    Returns:
        Dataset-specific local cache path, or ``None`` when cache is disabled.
    """

    if not request.use_local_cache:
        return None
    return str(Path(request.cache_dir) / dataset)


def _run_mvp_tool(arguments: dict[str, object], progress: Callable[[str], None] | None) -> dict[str, object]:
    """Execute historical MVP workflow for a chat-tool invocation.

    Args:
        arguments: Tool call arguments containing ``request`` payload.
        progress: Optional callback for streamed status messages.

    Returns:
        JSON-serializable payload including report markdown, weights,
        data-quality diagnostics, warnings, and optional feature artifact.

    Raises:
        ValueError: If request payload is missing or malformed.
    """

    request_payload = arguments.get("request")
    if not isinstance(request_payload, dict):
        msg = "Tool arguments must include a 'request' dictionary."
        raise ValueError(msg)

    request = _payload_to_request({str(k): v for k, v in request_payload.items()})
    result = run_historical_mvp_pipeline(request=request, progress=progress)

    feature_artifact: dict[str, object] | None = None
    if result.features_result is not None:
        feature_artifact = {
            "uri": result.features_result.artifact.uri,
            "fingerprint": result.features_result.artifact.fingerprint,
            "row_count": result.features_result.artifact.row_count,
            "column_count": result.features_result.artifact.column_count,
        }

    return {
        "report_markdown": result.report_markdown,
        "weights": result.weights_table.to_dict(orient="records"),
        "data_quality": {
            "pass_gate": result.data_quality.pass_gate,
            "n_periods": result.data_quality.n_periods,
            "n_assets": result.data_quality.n_assets,
            "stale_assets": result.data_quality.stale_assets,
            "insufficient_history_assets": result.data_quality.insufficient_history_assets,
            "missing_rate_by_asset": result.data_quality.missing_rate_by_asset,
        },
        "warnings": result.warnings,
        "agent_logs": result.agent_logs,
        "feature_artifact": feature_artifact,
    }


def _extract_tickers(message: str) -> list[str]:
    """Extract ticker symbols from free-form user text.

    Args:
        message: User prompt text.

    Returns:
        Order-preserving deduplicated ticker list in uppercase.
    """

    explicit = re.search(r"tickers?\s*[:=]\s*([A-Za-z,\s]+)", message, flags=re.IGNORECASE)
    tokens: list[str]
    if explicit:
        tokens = [token.strip().upper() for token in explicit.group(1).split(",") if token.strip()]
    else:
        raw_tokens = re.findall(r"\b[A-Z]{1,6}\b", message.upper())
        blocked = {
            "FOR",
            "FROM",
            "TO",
            "RISK",
            "MEASURE",
            "OBJECTIVE",
            "SHARPE",
            "CVAR",
            "MVR",
            "MV",
            "MAXRET",
            "MINRISK",
            "UTILITY",
            "PORTFOLIO",
            "BUILD",
        }
        tokens = [token for token in raw_tokens if token not in blocked]

    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped


def _extract_objective(message: str) -> str:
    """Infer optimization objective from chat text keywords.

    Args:
        message: User prompt text.

    Returns:
        Riskfolio objective string. Defaults to ``"Sharpe"``.
    """

    lowered = message.lower()
    if "minrisk" in lowered or "min risk" in lowered:
        return "MinRisk"
    if "maxret" in lowered or "max ret" in lowered or "max return" in lowered:
        return "MaxRet"
    if "utility" in lowered:
        return "Utility"
    return "Sharpe"


def _extract_risk_measure(message: str) -> str:
    """Infer optimization risk measure from chat text keywords.

    Args:
        message: User prompt text.

    Returns:
        Risk measure string. Defaults to ``"CVaR"``.
    """

    lowered = message.lower().replace("-", "")
    if "cvar" in lowered:
        return "CVaR"
    if "mv" in lowered or "variance" in lowered:
        return "MV"
    if "mad" in lowered:
        return "MAD"
    if "cdar" in lowered:
        return "CDaR"
    return "CVaR"
