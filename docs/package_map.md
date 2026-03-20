# BayesFolio Package Map

This file is the canonical, high-signal map of responsibilities across
`bayesfolio/`. Keep this page short and update it whenever package boundaries
or public entry points change.

## Top-Level Ownership

| Package | Responsibility | Must Not Do |
|---|---|---|
| `bayesfolio/core/` | Primitive types and shared settings enums/models. | Import other `bayesfolio` domains. |
| `bayesfolio/contracts/` | Boundary contracts (Pydantic schemas) for commands/results/ui/chat. | Business logic or IO. |
| `bayesfolio/engine/` | Pure business logic and orchestration. | File/network IO and MLflow calls. |
| `bayesfolio/io/` | External IO: providers, persistence, artifact handling. | Business logic. |

## Contracts Subpackages

- `contracts/commands/`: execution inputs (command contracts)
- `contracts/results/`: execution outputs (result contracts)
- `contracts/ui/`: UI-facing data shapes
- `contracts/chat/`: chat intent and tool protocol schemas (`ChatMessage*`, `ChatTool*`, `ChatTurn`)
- `contracts/commands/gp_planner.py`: planner request contract for structured multitask GP design selection
- `contracts/results/gp_workflow.py`: planner response and planner-driven GP workflow execution results

### Agent Boundary (Phase 1)

Agent and LLM runtime integrations must treat `contracts/chat/` as the canonical
I/O surface. Framework-specific payloads (LangChain/LangSmith/OpenAI/LlamaIndex)
should be translated at adapters/orchestrators, not leaked into contracts.

## Engine Subpackages

- `engine/features/`: feature engineering and dataset build logic
- `engine/forecast/`: forecast stage adapters and forecast payload builders
- `engine/asset_allocation/`: portfolio optimization logic
- `engine/backtest/`: backtest execution and performance summaries
- `engine/scenarios/`: scenario sampling from predictive distributions
- `engine/report/`: report payload assembly/composition from engine outputs (no I/O)
- `engine/mappers/`: contract ↔ engine field mapping bridge (no business orchestration)

## IO Subpackages

- `io/providers/`: external data provider adapters
- `io/artifact_store.py`: parquet artifact persistence and fingerprint metadata
- `io/fingerprints.py`: deterministic file digests

## Public Entry Points (Current)

- Pipeline orchestration: `engine.run_schema_first_pipeline`
- Historical chat orchestration: `engine.run_historical_mvp_chat_turn`
- Feature workflow: `engine.features.build_features_dataset`
- Feature provider factory: `engine.features.make_default_feature_providers`
- GP interpretation helpers: `engine.forecast.build_gp_interpretation_report`, `engine.forecast.render_gp_interpretation_report`
- Planner-driven GP workflow: `engine.forecast.run_planned_multitask_gp_workflow`
- IO dataset persistence: `io.ParquetArtifactStore`
- IO providers: `io.EtfFeaturesProvider`, `io.MacroProvider`, `io.ReturnsProvider`
- IO fingerprint helpers: `io.sha256_digest`, `io.sha256_fingerprint`

## Documentation Workflow (Efficient + RAG-Friendly)

1. Keep package-level summaries in `__init__.py` short (1-3 lines + `__all__`).
2. Keep architecture rules in `docs/copilot_architecture.md`.
3. Keep this package map updated for ownership/search/discovery.
4. Keep API detail in module/class/function docstrings.

This split keeps docs low-maintenance and creates stable chunks for retrieval.
