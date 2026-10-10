# Copilot Architecture Guardrails

Status: Active

Audience: Contributors, maintainers, and coding agents

Canonical topic: Architecture rationale and boundaries

Last verified: 2026-10-10

## Layer Ownership

- `bayesfolio/contracts/`: boundary schemas only (Pydantic models, no business logic, no pandas dependency in contracts).
- `bayesfolio/application/`: composition roots that assemble engine workflows with concrete I/O adapters.
- `bayesfolio/engine/`: pure business logic and data transformations; no direct file/network I/O.
- `bayesfolio/io/`: all external data access and artifact persistence.

Dependency direction for supported workflows is `application -> engine` and `application -> io`. Engine and I/O modules
communicate through engine-owned structural protocols; I/O does not import engine.

## Monthly Automation Boundary

- `application/monthly_portfolio.py` is the canonical default-wiring and user-facing execution entry point.
- `engine/automation/monthly_portfolio.py` owns monthly feature, GP, forecast, scenario, and optimization sequencing.
- `engine/automation/monthly_ports.py` defines the runtime side effects required by that engine workflow.
- `io/monthly_portfolio_runtime.py` owns MLflow lifecycle and contract logging, Git revision discovery, parquet loading,
  directory creation, index-free CSV persistence, JSON persistence, fingerprints, and artifact pointers.
- The application layer injects providers, feature storage, and runtime I/O. Engine code does not inspect storage or
  tracking implementation details, and the I/O adapter does not make modeling or portfolio decisions.

The deprecated `engine.automation.run_monthly_portfolio` entry point is a narrowly scoped compatibility façade. It
locally delegates default composition to `application/` and must not acquire direct I/O behavior. Remove this exception
when the historical engine import path can be retired in a future breaking release.

## Backtest and Final-Decision Boundary

- `engine/backtest/runner.py` evaluates serialized walk-forward periods. Every
  period identifies a decision date, a later realization date, fixed weights,
  and subsequent realized returns; the engine does not infer date shifts.
- `engine/automation/monthly_decision.py` is the pure final-review gate. It
  requires matching candidate/backtest strategy identifiers and rejects
  evidence unavailable strictly before the candidate as-of date. Approval
  requires at least one distinct benchmark on the identical evaluation grid.
- `application/monthly_decision.py` is the preferred public entry point for the
  review gate.
- A `MonthlyPortfolioResult` contains candidate optimizer weights. Only an
  approved `MonthlyPortfolioDecisionResult` contains finalized weights, and
  approval must be explicit human input. Brokerage execution remains outside
  this boundary.

## Feature Pipeline Placement

- Put new build commands/results in `contracts/commands/` and `contracts/results/`.
- Put feature engineering/transforms and orchestration in `engine/features/`.
- Put fetch/persist adapters and providers in `io/`.

## Brokerage and Performance Extension Boundary

- `engine/brokerage/ports.py` defines the read-only structural account protocol.
  A future `io` implementation satisfies it by shape and does not import the
  engine package.
- `engine/brokerage/planner.py` converts an approved portfolio decision plus a
  fresh account snapshot into dollar-notional planning instructions. Its result
  is permanently marked `executable=False`; it is not an order request.
- `engine/performance/ports.py` defines the structural boundary for externally
  measured, cash-flow-aware portfolio/benchmark return periods.
- `engine/performance/tracker.py` aggregates returns, gain/loss changes, fees,
  active risk, beta, and descriptive single-benchmark Jensen alpha without file,
  network, or brokerage access.
- Future Schwab authentication, account retrieval, and order execution belong
  in `io/` plus an application composition root. Order submission requires a
  separate explicit authorization contract and is not implied by final weights
  or a rebalance plan.

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
