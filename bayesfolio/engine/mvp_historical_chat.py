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
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Literal

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
from bayesfolio.core.settings import Horizon, Interval, Objective, RiskfolioConfig, RiskMeasure
from bayesfolio.engine.agent.intent_extractor import extract_intent_overrides_with_status
from bayesfolio.engine.agent.orchestrator import run_orchestration_cycle
from bayesfolio.engine.asset_allocation import optimize_from_historical_returns
from bayesfolio.engine.backtest import run_weighted_backtest
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
from bayesfolio.io.providers.chat_knowledge_provider import ChatKnowledgeProvider

_DEFAULT_RISKFOLIO = RiskfolioConfig()
_RISKFOLIO_KNOWLEDGE_PROVIDER = ChatKnowledgeProvider()
ParserMode = Literal["rule-based", "llm-based"]


@dataclass(frozen=True)
class HistoricalMvpRequest:
    """Request payload for historical-only MVP portfolio construction.

    Attributes:
        tickers: Asset tickers to include in the universe.
        start_date: Inclusive start date for historical returns.
        end_date: Inclusive end date for historical returns.
        objective: Riskfolio objective (for example ``Sharpe``).
        risk_measure: Riskfolio risk measure (for example ``MV``).
        model: Riskfolio model code (for example ``Classic``).
        rf: Risk-free rate in decimal units for the same return frequency.
        hist: Whether to use historical scenarios for non-MV risk measures.
        kelly: Optional Kelly mode (``approx`` or ``exact``).
        min_weight: Minimum portfolio weight as decimal.
        max_weight: Maximum portfolio weight as decimal (Riskfolio ``upperlng``).
        nea: Target number of assets for Riskfolio optimization.
        build_features: Whether to run the feature agent.
        lookback_days: Additional lookback window for feature generation.
        use_local_cache: Whether to read/write local market-data cache.
        cache_dir: Base directory for local market-data cache files.
        parser_mode: Parsing mode used for request normalization.
        llm_overrides_applied: Whether LLM intent overrides were applied.
    """

    tickers: list[str]
    start_date: date
    end_date: date
    objective: str = _DEFAULT_RISKFOLIO.obj.value
    risk_measure: str = _DEFAULT_RISKFOLIO.rm.value
    model: str = _DEFAULT_RISKFOLIO.model.value
    rf: float = _DEFAULT_RISKFOLIO.rf
    hist: bool = _DEFAULT_RISKFOLIO.hist
    kelly: str | None = None
    min_weight: float = 0.0
    max_weight: float = _DEFAULT_RISKFOLIO.upperlng
    nea: int = _DEFAULT_RISKFOLIO.nea
    build_features: bool = True
    lookback_days: int = 365
    use_local_cache: bool = True
    cache_dir: str = "artifacts/cache"
    parser_mode: ParserMode = "rule-based"
    llm_overrides_applied: bool = False


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
        portfolio_metrics: Backtest metrics where return and volatility values are
            decimals (0.10 = 10%), max_drawdown is decimal (negative or zero),
            and ratio values are dimensionless.
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
    portfolio_metrics: dict[str, float]
    report_markdown: str
    weights_table: pd.DataFrame
    agent_logs: list[str]
    warnings: list[str] = field(default_factory=list)
    features_result: FeaturesDatasetResult | None = None


def parse_chat_request(
    message: str,
    today: date | None = None,
    parser_mode: ParserMode = "rule-based",
) -> HistoricalMvpRequest:
    """Parse a free-form chat request into a normalized coordinator request.

    Args:
        message: User message containing tickers and optional settings.
        today: Optional date override for deterministic tests.
        parser_mode: Parsing mode selector. ``"rule-based"`` skips LLM
            extraction while ``"llm-based"`` applies LLM override extraction.

    Returns:
        Parsed and normalized request object.

    Raises:
        ValueError: If no tickers can be parsed.
        ValueError: If ``parser_mode`` is ``"llm-based"`` and no LLM
            overrides can be extracted.
    """

    anchors = today or date.today()
    parsed_tickers = _extract_tickers(message)
    if not parsed_tickers:
        msg = "Could not parse any tickers. Include tickers like SPY, QQQ, TLT."
        raise ValueError(msg)

    parsed_dates = re.findall(r"\b(?:\d{4}-\d{2}-\d{2}|\d{8})\b", message)
    if len(parsed_dates) >= 2:
        start_date = _parse_chat_date(parsed_dates[0])
        end_date = _parse_chat_date(parsed_dates[1])
    else:
        end_date = anchors
        start_date = end_date - timedelta(days=365 * 5)

    objective = _extract_objective(message)
    risk_measure = _extract_risk_measure(message)
    model = _extract_model(message)
    rf = _extract_rf(message)
    hist = _extract_hist(message)
    kelly = _extract_kelly(message)
    min_weight = 0.0
    nea = _extract_nea(message)
    max_weight = _extract_upperlng(message)
    llm_overrides_applied = False
    if parser_mode == "llm-based":
        llm_overrides, llm_status = extract_intent_overrides_with_status(message)
        if not llm_overrides:
            msg = (
                "LLM-based parser selected, but no LLM overrides were extracted. "
                f"Reason: {llm_status}. "
                "Check OPENAI_API_KEY/model access or switch to rule-based mode."
            )
            raise ValueError(msg)

        objective = str(llm_overrides.get("objective", _DEFAULT_RISKFOLIO.obj.value))
        risk_measure = str(llm_overrides.get("risk_measure", _DEFAULT_RISKFOLIO.rm.value))
        model = str(llm_overrides.get("model", _DEFAULT_RISKFOLIO.model.value))
        rf = float(llm_overrides.get("rf", _DEFAULT_RISKFOLIO.rf))
        hist = _coerce_bool(llm_overrides.get("hist", _DEFAULT_RISKFOLIO.hist), default=_DEFAULT_RISKFOLIO.hist)
        kelly_raw = llm_overrides.get("kelly", None)
        kelly = str(kelly_raw) if isinstance(kelly_raw, str) else None
        min_weight = float(llm_overrides.get("min_weight", 0.0))
        max_weight = float(llm_overrides.get("max_weight", _DEFAULT_RISKFOLIO.upperlng))
        nea = int(llm_overrides.get("nea", _DEFAULT_RISKFOLIO.nea))
        llm_overrides_applied = True

    return HistoricalMvpRequest(
        tickers=parsed_tickers,
        start_date=start_date,
        end_date=end_date,
        objective=objective,
        risk_measure=risk_measure,
        model=model,
        rf=rf,
        hist=hist,
        kelly=kelly,
        min_weight=min_weight,
        max_weight=max_weight,
        nea=nea,
        parser_mode=parser_mode,
        llm_overrides_applied=llm_overrides_applied,
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
        model=request.model,
        rf=request.rf,
        kelly=request.kelly,
        min_weight=request.min_weight,
        max_weight=request.max_weight,
        nea=request.nea,
        hist=request.hist,
    )
    optimize_result = optimize_from_historical_returns(returns=returns_matrix, request=optimize_command)

    _log("Backtest Agent: computing weighted portfolio performance metrics.")
    backtest_result = run_weighted_backtest(realized_returns=returns_matrix, optimization=optimize_result)
    portfolio_metrics = {
        "cumulative_return": backtest_result.cumulative_return,
        "annualized_return": backtest_result.annualized_return,
        "annualized_volatility": backtest_result.annualized_volatility,
        "max_drawdown": backtest_result.max_drawdown,
        "sharpe_ratio": backtest_result.sharpe_ratio,
        "sortino_ratio": backtest_result.sortino_ratio,
        "calmar_ratio": backtest_result.calmar_ratio,
    }

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
        portfolio_metrics=portfolio_metrics,
        report_markdown=report_markdown,
        weights_table=weights_table,
        agent_logs=agent_logs,
        warnings=warnings,
        features_result=features_result,
    )


def run_historical_mvp_chat_turn(
    message: str,
    progress: Callable[[str], None] | None = None,
    parser_mode: ParserMode = "rule-based",
) -> ChatTurn:
    """Run one chat turn through the orchestrator using a real tool execution path.

    Args:
        message: User chat message containing portfolio request context.
        progress: Optional callback for streaming progress updates.
        parser_mode: Parsing mode selector.

    Returns:
        ChatTurn containing executed tool result payload and assistant message.
    """

    request = parse_chat_request(message, parser_mode=parser_mode)
    is_ambiguous = _is_ambiguous_request(message=message)
    knowledge_payload: dict[str, object] | None = None
    normalization_payload: dict[str, object] = {
        "applied": {},
        "reason": "not_ambiguous" if not is_ambiguous else "no_change",
    }

    if is_ambiguous:
        knowledge_payload = _run_riskfolio_knowledge_tool(
            arguments={"message": message},
            provider=_RISKFOLIO_KNOWLEDGE_PROVIDER,
        )
        request, normalization_payload = _apply_knowledge_normalization(
            request=request,
            knowledge_payload=knowledge_payload,
        )
        knowledge_payload["normalization"] = normalization_payload

    turn = ChatTurn(user_message=ChatMessageUser(content=message))
    turn.tool_calls = []
    if is_ambiguous:
        turn.tool_calls.append(
            ChatToolCall(
                call_id="call_000",
                tool_name="retrieve_riskfolio_knowledge",
                arguments={
                    "message": message,
                    "knowledge_payload": knowledge_payload,
                },
            )
        )
    turn.tool_calls.append(
        ChatToolCall(
            call_id="call_001",
            tool_name="run_historical_mvp_pipeline",
            arguments={"request": _request_to_payload(request)},
        )
    )

    executor = RegistryToolExecutor(
        handlers={
            "retrieve_riskfolio_knowledge": lambda arguments: _run_riskfolio_knowledge_tool(
                arguments=arguments,
                provider=_RISKFOLIO_KNOWLEDGE_PROVIDER,
            ),
            "run_historical_mvp_pipeline": lambda arguments: _run_mvp_tool(
                arguments=arguments,
                progress=progress,
            ),
        }
    )

    updated_turn = run_orchestration_cycle(turn=turn, tool_executor=executor)
    updated_turn.diagnostics["parser_mode"] = parser_mode
    updated_turn.diagnostics["llm_overrides_applied"] = request.llm_overrides_applied
    updated_turn.diagnostics["ambiguous_request"] = is_ambiguous
    updated_turn.diagnostics["knowledge_normalization"] = normalization_payload
    if not updated_turn.tool_results:
        updated_turn.assistant_message = ChatMessageAssistant(content="No tool result was produced.")
        return updated_turn

    latest = updated_turn.tool_results[-1]
    if latest.success:
        assistant_text = str(latest.payload.get("report_markdown", "Historical MVP run completed."))
        retrieval_payload = _find_retrieval_payload(turn=updated_turn)
        if retrieval_payload is not None:
            assistant_text = f"{assistant_text}\n\n{_render_knowledge_summary(retrieval_payload)}"
    else:
        assistant_text = f"MVP run failed: {latest.error_message or 'unknown tool error'}"
    updated_turn.assistant_message = ChatMessageAssistant(content=assistant_text)
    return updated_turn


def _run_riskfolio_knowledge_tool(
    arguments: dict[str, object],
    provider: ChatKnowledgeProvider,
) -> dict[str, object]:
    """Execute deterministic Riskfolio knowledge retrieval for chat grounding.

    Args:
        arguments: Tool arguments including query text and optional precomputed
            payload.
        provider: IO provider used to retrieve snippets and normalization hints.

    Returns:
        Retrieval payload with snippets and canonical suggestion hints.

    Raises:
        ValueError: If query message is missing.
    """

    precomputed = arguments.get("knowledge_payload")
    if isinstance(precomputed, dict):
        return {str(key): value for key, value in precomputed.items()}

    message = arguments.get("message")
    if not isinstance(message, str) or not message.strip():
        msg = "Knowledge tool requires a non-empty 'message' argument."
        raise ValueError(msg)

    top_k_raw = arguments.get("top_k", 5)
    top_k = int(top_k_raw) if isinstance(top_k_raw, int | float | str) else 5
    return provider.retrieve_and_suggest(query=message, top_k=max(top_k, 1))


def _is_ambiguous_request(message: str) -> bool:
    """Determine whether a chat request lacks explicit optimization settings.

    Args:
        message: User request text.

    Returns:
        ``True`` when objective or risk measure appears underspecified.
    """

    lowered = message.lower()
    objective_explicit = bool(re.search(r"\b(min\s*risk|minrisk|max\s*ret(?:urn)?|maxret|utility|sharpe)\b", lowered))
    risk_explicit = bool(
        re.search(
            r"\b(cvar|mv|variance|mad|cdar|edar|rldar|evar|rlvar|mdd|uci|gmd|sortino|flpm|slpm)\b",
            lowered,
        )
    )
    return not (objective_explicit and risk_explicit)


def _apply_knowledge_normalization(
    request: HistoricalMvpRequest,
    knowledge_payload: dict[str, object],
) -> tuple[HistoricalMvpRequest, dict[str, object]]:
    """Apply canonical objective/risk normalization from retrieval hints.

    Args:
        request: Parsed request prior to normalization.
        knowledge_payload: Retrieval payload including ``suggested_overrides``.

    Returns:
        Tuple of normalized request and normalization diagnostics.
    """

    suggested = knowledge_payload.get("suggested_overrides")
    if not isinstance(suggested, dict):
        return request, {"applied": {}, "reason": "no_suggestions"}

    normalized = request
    applied: dict[str, dict[str, str]] = {}

    objective_raw = suggested.get("objective")
    if isinstance(objective_raw, str) and objective_raw in {objective.value for objective in Objective}:
        if objective_raw != request.objective:
            normalized = replace(normalized, objective=objective_raw)
            applied["objective"] = {"from": request.objective, "to": objective_raw}

    risk_raw = suggested.get("risk_measure")
    if isinstance(risk_raw, str) and risk_raw in {risk.value for risk in RiskMeasure}:
        if risk_raw != request.risk_measure:
            normalized = replace(normalized, risk_measure=risk_raw)
            applied["risk_measure"] = {"from": request.risk_measure, "to": risk_raw}

    model_raw = suggested.get("model")
    if isinstance(model_raw, str) and model_raw in {"Classic", "BL", "FM", "BLFM"}:
        if model_raw != request.model:
            normalized = replace(normalized, model=model_raw)
            applied["model"] = {"from": request.model, "to": model_raw}

    rf_raw = suggested.get("rf")
    if isinstance(rf_raw, int | float):
        rf_value = float(rf_raw)
        if rf_value != request.rf:
            normalized = replace(normalized, rf=rf_value)
            applied["rf"] = {"from": f"{request.rf}", "to": f"{rf_value}"}

    hist_raw = suggested.get("hist")
    if isinstance(hist_raw, bool) and hist_raw != request.hist:
        normalized = replace(normalized, hist=hist_raw)
        applied["hist"] = {"from": f"{request.hist}", "to": f"{hist_raw}"}

    kelly_raw = suggested.get("kelly")
    if isinstance(kelly_raw, str) and kelly_raw in {"approx", "exact"}:
        if kelly_raw != request.kelly:
            normalized = replace(normalized, kelly=kelly_raw)
            applied["kelly"] = {"from": f"{request.kelly}", "to": kelly_raw}

    reason = "applied" if applied else "no_change"
    return normalized, {"applied": applied, "reason": reason}


def _find_retrieval_payload(turn: ChatTurn) -> dict[str, object] | None:
    """Return the retrieval payload from tool results if available."""

    for result in turn.tool_results:
        if result.tool_name == "retrieve_riskfolio_knowledge" and result.success:
            return result.payload
    return None


def _render_knowledge_summary(payload: dict[str, object]) -> str:
    """Render compact retrieval provenance for assistant output."""

    snippets_raw = payload.get("snippets")
    snippets = snippets_raw if isinstance(snippets_raw, list) else []
    normalization_raw = payload.get("normalization")
    normalization = normalization_raw if isinstance(normalization_raw, dict) else {}
    applied_raw = normalization.get("applied")
    applied = applied_raw if isinstance(applied_raw, dict) else {}

    source_entries: list[str] = []
    for snippet in snippets[:2]:
        if not isinstance(snippet, dict):
            continue
        source = snippet.get("source")
        if isinstance(source, str):
            source_entries.append(source)

    source_text = "none"
    if source_entries:
        source_text = ", ".join(source_entries)

    if applied:
        normalized_text = ", ".join(f"{field}: {change.get('to', '')}" for field, change in applied.items())
    else:
        normalized_text = "none"

    return f"Knowledge used: sources={source_text}; normalized={normalized_text}."


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
        f"- Model / rf / hist / kelly: {request.model} / {request.rf:.4f} / {request.hist} / {request.kelly}\n"
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
    max_weight_raw = payload.get("max_weight", _DEFAULT_RISKFOLIO.upperlng)
    nea_raw = payload.get("nea", _DEFAULT_RISKFOLIO.nea)
    lookback_days_raw = payload.get("lookback_days", 365)
    use_local_cache_raw = payload.get("use_local_cache", True)
    cache_dir_raw = payload.get("cache_dir", "artifacts/cache")
    parser_mode_raw = str(payload.get("parser_mode", "rule-based"))

    min_weight = float(str(min_weight_raw))
    max_weight = float(str(max_weight_raw))
    nea = int(str(nea_raw))
    lookback_days = int(str(lookback_days_raw))
    use_local_cache = bool(use_local_cache_raw)
    cache_dir = str(cache_dir_raw)
    parser_mode: ParserMode = "llm-based" if parser_mode_raw == "llm-based" else "rule-based"

    return HistoricalMvpRequest(
        tickers=tickers,
        start_date=date.fromisoformat(str(start_raw)),
        end_date=date.fromisoformat(str(end_raw)),
        objective=str(payload.get("objective", _DEFAULT_RISKFOLIO.obj.value)),
        risk_measure=str(payload.get("risk_measure", _DEFAULT_RISKFOLIO.rm.value)),
        model=str(payload.get("model", _DEFAULT_RISKFOLIO.model.value)),
        rf=float(str(payload.get("rf", _DEFAULT_RISKFOLIO.rf))),
        hist=_coerce_bool(payload.get("hist", _DEFAULT_RISKFOLIO.hist), default=_DEFAULT_RISKFOLIO.hist),
        kelly=(str(payload.get("kelly")) if isinstance(payload.get("kelly"), str) else None),
        min_weight=min_weight,
        max_weight=max_weight,
        nea=nea,
        build_features=bool(payload.get("build_features", True)),
        lookback_days=lookback_days,
        use_local_cache=use_local_cache,
        cache_dir=cache_dir,
        parser_mode=parser_mode,
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
        "metrics": result.portfolio_metrics,
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
        raw_tokens = re.findall(r"\b[A-Z]{2,6}\b", message.upper())
        blocked = {
            "FOR",
            "FROM",
            "TO",
            "WITH",
            "AND",
            "OF",
            "ASSET",
            "ASSETS",
            "EFFECTIVE",
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
            "MAX",
            "WEIGHT",
            "UPPERLNG",
            "NEA",
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
        Risk measure string. Defaults to ``"MV"``.
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
    return _DEFAULT_RISKFOLIO.rm.value


def _extract_model(message: str) -> str:
    """Infer Riskfolio model from chat text keywords.

    Args:
        message: User prompt text.

    Returns:
        Model code. Defaults to ``Classic``.
    """

    lowered = message.lower()
    if "blfm" in lowered or "black litterman factor" in lowered:
        return "BLFM"
    if re.search(r"\bfm\b", lowered) or "factor model" in lowered:
        return "FM"
    if "black litterman" in lowered or re.search(r"\bbl\b", lowered):
        return "BL"
    return _DEFAULT_RISKFOLIO.model.value


def _extract_rf(message: str) -> float:
    """Extract risk-free rate in decimal units from chat text.

    Args:
        message: User prompt text.

    Returns:
        Decimal risk-free rate. Defaults to configured Riskfolio value.
    """

    match = re.search(
        r"\b(?:rf|risk\s*free(?:\s*rate)?)\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*(%)?",
        message,
        flags=re.IGNORECASE,
    )
    if match is None:
        return float(_DEFAULT_RISKFOLIO.rf)

    value = float(match.group(1))
    if match.group(2) == "%" or value > 1.0:
        value = value / 100.0
    return value


def _extract_hist(message: str) -> bool:
    """Extract Riskfolio ``hist`` flag from chat text.

    Args:
        message: User prompt text.

    Returns:
        Parsed boolean flag. Defaults to configured Riskfolio value.
    """

    lowered = message.lower()
    if "hist false" in lowered or "no historical" in lowered or "non historical" in lowered:
        return False
    if "hist true" in lowered or "historical" in lowered:
        return True
    return bool(_DEFAULT_RISKFOLIO.hist)


def _extract_kelly(message: str) -> str | None:
    """Extract Kelly mode from chat text.

    Args:
        message: User prompt text.

    Returns:
        ``approx``, ``exact``, or ``None`` when not requested.
    """

    lowered = message.lower()
    if "kelly exact" in lowered or "exact kelly" in lowered:
        return "exact"
    if "kelly approx" in lowered or "approx kelly" in lowered or "approximate kelly" in lowered:
        return "approx"
    return None


def _coerce_bool(value: object, default: bool) -> bool:
    """Coerce arbitrary user/tool input into a boolean value.

    Args:
        value: Candidate boolean-like value.
        default: Fallback value when parsing is not possible.

    Returns:
        Parsed or fallback boolean.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
        return default
    if isinstance(value, int | float):
        return bool(value)
    return default


def _extract_nea(message: str) -> int:
    """Extract Riskfolio ``nea`` override from chat text.

    Args:
        message: User prompt text.

    Returns:
        Parsed ``nea`` value when present; otherwise default config value.
    """

    patterns = [
        r"\bnea(?:\s*(?:of|:=|=|:))?\s*(\d+)\b",
        r"\bnumber\s+of\s+effective\s+assets(?:\s*(?:of|:=|=|:))?\s*(\d+)\b",
        r"\beffective\s+assets(?:\s*(?:of|:=|=|:))?\s*(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match is not None:
            parsed = int(match.group(1))
            return parsed if parsed >= 1 else int(_DEFAULT_RISKFOLIO.nea)

    return int(_DEFAULT_RISKFOLIO.nea)


def _parse_chat_date(raw_value: str) -> date:
    """Parse supported date token formats from chat text.

    Args:
        raw_value: Date token in ``YYYY-MM-DD`` or ``YYYYMMDD`` format.

    Returns:
        Parsed calendar date.
    """

    if "-" in raw_value:
        return date.fromisoformat(raw_value)
    return datetime.strptime(raw_value, "%Y%m%d").date()


def _extract_upperlng(message: str) -> float:
    """Extract Riskfolio upper long bound from chat text.

    Supports ``upperlng`` and ``max_weight`` aliases, with decimal or percent
    input values.

    Args:
        message: User prompt text.

    Returns:
        Parsed upper-long bound in decimal units, or default when absent/invalid.
    """

    match = re.search(
        r"\b(?:upperlng|max_weight)\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*(%)?",
        message,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.search(
            r"\bmax(?:\s+weight)?\s*(?:of|=|:)?\s*(-?\d+(?:\.\d+)?)\s*(%)?\s*(?:upperlng)?\b",
            message,
            flags=re.IGNORECASE,
        )
    if match is None:
        return float(_DEFAULT_RISKFOLIO.upperlng)

    value = float(match.group(1))
    if match.group(2) == "%" or value > 1.0:
        value = value / 100.0

    if value <= 0.0 or value > 1.0:
        return float(_DEFAULT_RISKFOLIO.upperlng)
    return value
