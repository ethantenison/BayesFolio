# BayesFolio New Architecture

## Boundary Layers

| Layer | Purpose |
|-------|---------|
| `bayesfolio/core/` | Primitive types and configuration settings. No business logic. |
| `bayesfolio/contracts/` | Schema-first boundary contracts between layers. Pydantic models only. |
| `bayesfolio/engine/` | Pure business logic. No I/O. Consumes contracts, returns contracts. |
| `bayesfolio/io/` | External I/O — file system, object stores, MLflow, Yahoo Finance. |

## Contract Categories

| Subdirectory | Purpose |
|-------------|---------|
| `contracts/commands/` | Commands that trigger execution (inputs to engine functions). |
| `contracts/results/` | Results and outputs from engine functions. |
| `contracts/chat/` | LLM tool call I/O — parsed intents, tool results. |
| `contracts/ui/` | UI-specific shapes — forms, display records, user inputs. |

## Schema Identity Strategy

- All schemas are identified by a `SchemaName` `StrEnum` value (e.g., `"bayesfolio.optimize.command"`).
- Each `VersionedContract` subclass overrides `schema` with `Field(default=SchemaName.X, const=True)`.
- This enforces identity at the boundary: no contract can be deserialized under the wrong schema name.
- Versioning is embedded in `schema_version` (semver string, e.g., `"0.1.0"`).
- No `/v1/` folder versioning is used — version is a field, not a path segment.

## Return Units Convention

All return values in contracts are **decimal** (0.02 = 2%), never percent-points (2.0 = 2%).
This is enforced by field documentation and the `return_unit: str = Field(default="decimal")` field
present on every contract that carries returns.

- `mean: list[float]` — decimals (0.02 = 2%)
- `weights: list[float]` — decimals summing to 1.0
- `cumulative_return: float` — decimal (0.10 = 10%)
- `annualized_return: float` — decimal (0.10 = 10%)
- `annualized_volatility: float` — decimal (0.15 = 15%)

If a conversion is necessary (e.g., from a data vendor returning percent-points), do it
explicitly at the I/O boundary and document the transformation.

## Look-Ahead Bias

Features must be aligned to `t-1` before being used to predict `t`.

Every feature or label used in training must be time-indexed such that:
- Features at row `t` contain only information available **before** time `t`.
- Labels at row `t` represent the outcome **at** time `t`.

This alignment must happen in `bayesfolio/engine/features/` before features leave that layer.
No feature engineering or lagging should occur inside the model training code.
