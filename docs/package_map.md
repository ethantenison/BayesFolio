# BayesFolio Package Map

Status: Active

Audience: Contributors, maintainers, and coding agents

Canonical topic: Package ownership and preferred public entry points

Last verified: 2026-10-10

This is the canonical, high-signal routing map for `bayesfolio/`. Keep it concise and update it whenever package boundaries, ownership, or preferred public entry points change. Detailed API behavior belongs in code docstrings.

## Top-Level Ownership

| Package | Responsibility | Must Not Do |
|---|---|---|
| `bayesfolio/application/` | Assemble supported engine workflows with concrete providers and I/O adapters. | Own modeling, optimization, or persistence implementation. |
| `bayesfolio/core/` | Primitive types and shared settings enums/models. | Import other `bayesfolio` domains. |
| `bayesfolio/contracts/` | Pydantic boundary schemas for commands, results, UI, and chat payloads. | Business logic or I/O. |
| `bayesfolio/engine/` | Business logic, domain workflows, and orchestration. | Add new provider, persistence, or network implementations. |
| `bayesfolio/io/` | External providers, persistence, artifact handling, and MLflow adapters. | Own portfolio or modeling decisions. |
| `bayesfolio/utils.py` | Shared pure helpers that do not belong to a domain. | Become a general dumping ground or perform hidden I/O. |

## Contracts

- `contracts/base.py`: strict base models, schema identities, envelopes, and shared error shapes.
- `contracts/commands/`: validated execution inputs, including feature, GP planner, and monthly portfolio commands.
- `contracts/results/`: validated outputs, including feature, GP workflow, forecast, optimization, report, and monthly portfolio results.
- `contracts/ui/`: UI-facing universe and scenario shapes.
- `contracts/chat/`: parsed intent and tool-protocol schemas for agent and LLM boundaries.

Framework payloads from OpenAI, LangChain, LangSmith, LlamaIndex, or similar runtimes must be translated at adapters or orchestrators rather than stored in contracts.

## Application

- `application/monthly_portfolio.py`: canonical composition root for default providers, feature storage, monthly runtime I/O, and engine execution.

## Engine

- `engine/agent/`: intent extraction, planning, and LLM-backed GP design helpers.
- `engine/automation/`: monthly business sequencing through injected provider, store, and runtime-I/O interfaces; it contains no concrete persistence or MLflow implementation.
- `engine/automation/monthly_ports.py`: structural runtime-I/O protocol required by monthly engine execution.
- `engine/features/`: feature engineering, time alignment, dataset assembly, and provider protocols.
- `engine/forecast/`: forecast payloads, metrics, scaling, cross-validation, and GP modeling workflows.
- `engine/asset_allocation/`: portfolio optimization logic and Riskfolio adaptation.
- `engine/backtest/`: backtest execution and performance summaries.
- `engine/scenarios/`: scenario sampling from predictive distributions.
- `engine/report/`: report assembly and visualization logic.
- `engine/mappers/`: the explicit contract-to-engine and engine-to-contract field-mapping bridge.
- `engine/pipeline.py`: schema-first cross-domain pipeline orchestration.
- `engine/mvp_historical_chat.py`: historical chat orchestration module; it is not exported from the root `bayesfolio.engine` namespace.

Engine leaf domains should not cross-import sibling domains. Cross-domain composition belongs in an approved orchestration module. See [architecture guardrails](copilot_architecture.md) for the normative rules and transitional exceptions.

## I/O

- `io/providers/`: external ETF, macroeconomic, return, and chat-knowledge providers.
- `io/artifact_store.py` and `io/parquet_store.py`: parquet persistence and metadata.
- `io/artifacts.py` and `io/report_artifacts.py`: JSON, CSV, and report-figure persistence.
- `io/backends.py`: local and fsspec artifact backends.
- `io/fingerprints.py`: deterministic content digests.
- `io/mlflow_logger.py`: MLflow contract and artifact logging.
- `io/monthly_portfolio_runtime.py`: local artifact and MLflow implementation of the monthly runtime-I/O protocol.
- `io/agent_tool_executor.py`: registry-backed execution of agent tool calls.

## Preferred Public Entry Points

### Cross-domain workflows

- Schema-first pipeline: `bayesfolio.engine.run_schema_first_pipeline`
- Monthly portfolio workflow: `bayesfolio.application.run_monthly_portfolio`
- Injected monthly engine core: `bayesfolio.engine.automation.execute_monthly_portfolio`
- Deprecated monthly compatibility path: `bayesfolio.engine.automation.run_monthly_portfolio`
- Historical chat workflow: `bayesfolio.engine.mvp_historical_chat.run_historical_mvp_chat_turn`

### Features and forecasting

- Feature dataset: `bayesfolio.engine.features.build_features_dataset`
- Default feature providers: `bayesfolio.engine.features.make_default_feature_providers`
- Planned GP workflow: `bayesfolio.engine.forecast.run_planned_multitask_gp_workflow`
- Planned GP from a DataFrame: `bayesfolio.engine.forecast.run_planned_multitask_gp_from_dataframe`
- GP interpretation: `bayesfolio.engine.forecast.build_gp_interpretation_report`

### Portfolio and evaluation

- Scenario optimization: `bayesfolio.engine.asset_allocation.optimize_from_scenarios`
- Historical-return optimization: `bayesfolio.engine.asset_allocation.optimize_from_historical_returns`
- Weighted backtest: `bayesfolio.engine.backtest.run_weighted_backtest`

### Contracts

- Monthly command: `bayesfolio.contracts.commands.MonthlyPortfolioCommand`
- Feature command: `bayesfolio.contracts.commands.BuildFeaturesDatasetCommand`
- GP planner request: `bayesfolio.contracts.commands.GPPlannerRequest`
- Monthly result: `bayesfolio.contracts.results.MonthlyPortfolioResult`
- Chat contracts: `bayesfolio.contracts.chat`
- UI contracts: `bayesfolio.contracts.ui`

### I/O

- Parquet artifact store: `bayesfolio.io.ParquetArtifactStore`
- Providers: `bayesfolio.io.EtfFeaturesProvider`, `bayesfolio.io.MacroProvider`, `bayesfolio.io.ReturnsProvider`
- Artifact backends: `bayesfolio.io.LocalArtifactBackend`, `bayesfolio.io.FsspecArtifactBackend`
- Fingerprints: `bayesfolio.io.sha256_digest`, `bayesfolio.io.sha256_fingerprint`

## Documentation Workflow

1. Use [documentation authority](README.md) to find the canonical owner for a topic.
2. Keep package summaries and curated exports in `__init__.py` short.
3. Keep architecture rules and rationale in [architecture guardrails](copilot_architecture.md).
4. Keep modeling rationale in [first principles](first_principles.md).
5. Keep API detail and contract field semantics in code docstrings.
6. Update this map when ownership or preferred entry points change.
