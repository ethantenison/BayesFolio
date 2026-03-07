# Copilot Architecture Guardrails

## Layer Ownership

- `bayesfolio/contracts/`: boundary schemas only (Pydantic models, no business logic, no pandas dependency in contracts).
- `bayesfolio/engine/`: pure business logic and data transformations; no direct file/network I/O.
- `bayesfolio/io/`: all external data access and artifact persistence.

## Feature Pipeline Placement

- Put new build commands/results in `contracts/commands/` and `contracts/results/`.
- Put feature engineering/transforms and orchestration in `engine/features/`.
- Put fetch/persist adapters and providers in `io/`.

## Transitional Rule

Current `engine/features/asset_prices.py` and `engine/features/market_fundamentals.py` include legacy I/O helpers.
Do not copy that pattern into new code. Transitional wiring should happen by dependency injection from composition
code, so `io` providers can call legacy fetchers without importing engine modules directly.

## Return Unit Convention

All returns are decimal units (`0.02 = 2%`) across commands/results and datasets.
Never mix decimal and percent-point units implicitly.

## Look-Ahead Alignment

Feature rows used for prediction at time `t` must only contain information available before `t`.
Apply predictor lagging (`t-1`) in engine feature transforms, and keep `y_excess_lead` as the target outcome
at `t` (next-period excess return label).

## Package Map

For a concise ownership map and public entry-point index, see `docs/package_map.md`.

## MVP Historical Chat Orchestration Note

`engine/mvp_historical_chat.py` is an approved orchestration module for the
historical MVP chat workflow. It may compose engine steps with injected
`io` adapters/providers and tool executors to run a full chat turn, while
keeping business logic in engine domains and persistence/fetch implementation
inside `io` modules.

## Agent Contract Boundary

- Multi-agent and LLM workflows must exchange chat/tool data through
	`contracts/chat/` schemas (`ChatMessage*`, `ChatTool*`, `ChatTurn`, `ParsedIntent`).
- Keep framework SDK payloads (LangChain/LangSmith/OpenAI/LlamaIndex) out of
	contracts; adapt them at orchestration or IO adapter boundaries.

## Assembler vs Mapper Boundary

Use this rule to separate responsibilities inside `engine/`:

- **Assembler (`engine/report/assembler.py`)**:
	- Composes one or more engine outputs into a report-shaped engine payload.
	- May select and combine values from multiple engine-domain results (for example,
		backtest metrics plus feature diagnostics).
	- Must stay pure: no file/network I/O and no persistence side effects.

- **Mapper (`engine/mappers/report.py`)**:
	- Translates between boundary contracts and engine call signatures/results.
	- Should only perform schema/field mapping (rename/re-shape/default handling),
		not cross-domain composition logic.

Practical guidance:

- If logic answers "what goes into the report," put it in the assembler.
- If logic answers "how this contract maps to/from engine args/results," put it
	in a mapper.
