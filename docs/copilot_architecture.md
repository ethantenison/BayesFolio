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
