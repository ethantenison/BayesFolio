# GitHub Copilot Instructions for BayesFolio

BayesFolio aims to be an AI-powered portfolio optimization platform. 

These instructions are **STRICT**. Generated code must follow them.

## Documentation Source of Truth (Anti-Duplication)

To keep maintenance cost low, use this split:

- Keep this file (`.github/copilot-instructions.md`) for **normative rules only**
  (must/must-not constraints for architecture, typing, testing, and units).
- Keep package/component descriptions and entry-point maps in
  `docs/package_map.md`.
- Keep architecture rationale and guardrails in `docs/copilot_architecture.md`.

When adding or changing functionality:

- Update this file only if a **rule** changes.
- Update docs files if **structure, ownership, or public entry points** change.
- Do not duplicate long package maps or explanatory prose in this file.

---

## Language & Style

- Python **3.13+**
- Use **native typing**: `list[str]`, `dict[str, float]`, `X | None`
- Do NOT use `typing.List`, `Optional`, `Union`
- Use `from __future__ import annotations` in new modules
- Prefer clarity over cleverness
- Avoid hidden state; prefer explicit inputs/outputs

### Docstrings
- All **public** functions/classes require **Google-style docstrings**
- Include shape/units expectations for arrays/DataFrames (especially returns)

### Numeric / Finance Conventions (IMPORTANT)
- Returns must be clearly documented as either:
  - **decimal** (0.02 = 2%), or
  - **percent points** (2.0 = 2%)
- Never silently mix units. If converting, do it explicitly and document it.
- Time indexing must avoid look-ahead bias. Every feature/label must be aligned.

---

## Code Quality & Linting (STRICT)

This repository uses **ruff** with strict pre-commit hooks.
All generated code MUST pass linting.

### Must Avoid
- Unused variables (F841)
- Unused imports (F401)
- Line length > 120 (E501)
- Imports not at top of file (E402)
- Bare `except:` (E722)
- Shadowing builtins (A001/A002/A003)

### Formatting Rules
- Max line length: **120**
- Organize imports: stdlib → third-party → local
- No commented-out dead code
- No debug prints in library code; use `logging` when needed

### Determinism
- Fix RNG seeds when randomness affects results (torch / numpy / random)
- Provide `seed` in configs where relevant
- Avoid nondeterministic ops unless explicitly justified

### Pylance Quality Gate (STRICT)

- Generated/modified Python code must be **Pylance-clean** in edited files.
- After edits, validate diagnostics in touched files before finalizing.
- Prefer fixing root typing issues (narrowing, casts, typed helpers) over suppressions.
- Avoid blanket `# type: ignore` usage; if unavoidable, use the narrowest code and explain why.
- Keep workspace analysis focused on maintained code paths:
  - include `bayesfolio/` and `tests/`
  - exclude experimental folders (`experiments/`, `scratch/`) from default type-check scope

---

## Architecture Rules (STRICT)

### Separation of Concerns

Each top-level package has a single responsibility — never mix them:

| Package | Responsibility |
|---|---|
| `bayesfolio/contracts/` | Data schemas (Pydantic models) only. No business logic, no I/O. |
| `bayesfolio/engine/` | Business logic only. Reads/writes contract objects. No direct I/O. |
| `bayesfolio/core/` | App-wide settings, configs (`settings.py`), and primitive types (`types.py`). |
| `bayesfolio/io/` | Storage, artifact persistence, and MLflow logging. No business logic. |

- `contracts/` must **not** import from `engine/` or `io/`.
- `io/` must **not** import from `engine/`.
- `engine/` must **not** do direct file I/O or MLflow calls — delegate to `io/`.
- `core/` must **not** import from any other `bayesfolio` package.

### No Cross-Imports Between Domains

Engine submodules (`agent`, `asset_allocation`, `backtest`, `features`, `forecast`, `report`, `scenarios`) are **leaf domains** and must not import from each other. (`mappers` is an intentional exception — see below.)

- Cross-domain orchestration belongs in `engine/pipeline.py` or a dedicated pipeline module.
- `engine/mappers/` is the only place allowed to bridge engine outputs to contract schemas; it may import from `contracts/` and individual engine leaf modules.
- Never import a sibling engine module directly (e.g., `backtest` must not import from `forecast`).

### Where Code Goes

| What you are adding | Where it goes |
|---|---|
| New data schema (command, result, or UI input) | `bayesfolio/contracts/commands/`, `contracts/results/`, or `contracts/ui/` |
| New primitive type alias | `bayesfolio/core/types.py` |
| New config/settings class | `bayesfolio/core/settings.py` |
| Forecasting logic | `bayesfolio/engine/forecast/` |
| Portfolio optimization logic | `bayesfolio/engine/asset_allocation/` |
| Backtesting logic | `bayesfolio/engine/backtest/` |
| Feature engineering | `bayesfolio/engine/features/` |
| Scenario sampling | `bayesfolio/engine/scenarios/` |
| Reporting or visualization | `bayesfolio/engine/report/` |
| AI agent / intent extraction | `bayesfolio/engine/agent/` |
| Contract ↔ engine mapping | `bayesfolio/engine/mappers/` |
| Artifact storage / MLflow logging | `bayesfolio/io/` |
| Shared pure utilities | `bayesfolio/utils.py` |

---

## Schemas

- All schemas must subclass `VersionedContract` (versioned, cross-boundary contracts) or `ContractModel` (internal base types) from `bayesfolio/contracts/base.py`.
- Every new schema name must be registered in the `SchemaName` enum in `contracts/base.py` using the pattern `bayesfolio.<domain>.<name>`.
- The `schema` field must use `Literal[SchemaName.X]` with the matching default to enforce schema identity at boundaries.
- Commands go in `contracts/commands/`, results in `contracts/results/`, UI inputs in `contracts/ui/`.
- Use canonical primitive types from `core/types.py` (`Ticker`, `Weight`, `HorizonDays`, `ReturnDecimal`) wherever applicable.
- Schema models are **data-only**: no methods with side effects, no imports from `engine/` or `io/`.

### Schema Documentation Rules (STRICT)

- Every contract/schema class must include a Google-style docstring with an `Attributes:` section.
- The `Attributes:` section must document **every public field** defined on that class.
- Field documentation must include, when applicable:
  - units (especially returns: decimal vs percent points)
  - allowed literals/enums
  - nullability/optional semantics
  - shape expectations for arrays/tables
- For nested schema fields, state what nested contract is expected and its role.
- Keep attribute docs concise and schema-specific; avoid boilerplate text.
- When a schema field is renamed/added/removed, update the class docstring in the same change.

---

## Testing Expectations

Use `pytest` and keep tests architecture-aligned.

### Test Layout (Mirror Source Tree)

- Tests live under `tests/` and mirror source paths exactly.
  - `bayesfolio/contracts/commands/features.py` → `tests/contracts/commands/test_features.py`
  - `bayesfolio/engine/features/dataset_builder.py` → `tests/engine/features/test_dataset_builder.py`
  - `bayesfolio/io/artifact_store.py` → `tests/io/test_artifact_store.py`
- Naming conventions:
  - file: `test_<module>.py`
  - class: `Test<ClassName>` (optional)
  - function: `test_<behavior>`

### Layer-Safe Test Rules (STRICT)

- Contract tests (`tests/contracts/**`):
  - Validate schema identity (`schema` + `schema_version`) and `extra="forbid"` behavior.
  - Do not import `engine` or `io`.
- Engine tests (`tests/engine/**`):
  - Test pure business logic only with synthetic in-memory DataFrames.
  - Inject fakes/stubs for providers/stores; no file/network I/O in unit tests.
  - Never call Yahoo/FRED/MLflow from engine tests.
- IO tests (`tests/io/**`):
  - Test persistence/fetch adapters and path/fingerprint behavior.
  - Do not import engine business logic in IO tests.

### Determinism and Numeric Assertions

- Tests must be deterministic:
  - set seeds explicitly for `random`, `numpy`, and `torch` when relevant.
  - never rely on implicit global RNG state.
- Never assert raw float equality; use `pytest.approx` or `numpy.testing.assert_allclose` with explicit tolerances.
- Returns assertions must respect unit conventions (default: decimal, e.g. `0.02 = 2%`).

### Runtime and External Dependency Policy

- Keep unit tests fast and local-first.
- No network calls and no dependency on external services.
- Use small synthetic fixtures; prefer factory helpers over large fixture files.
- Use `tmp_path` for filesystem tests and cleanly isolate side effects.

### Execution Guidance

- Run targeted tests first, then broader suite:
  - `pytest tests/engine/features/test_dataset_builder.py -q`
  - `pytest tests/contracts -q`
  - `pytest -q`

---

## Performance Integrity

- Avoid accidental quadratic operations over long panels
- Avoid copying large DataFrames unless necessary
- Cache reusable computations explicitly (no hidden global caches)
- Make expensive operations obvious and documented

---

## Responsible AI & Transparency

BayesFolio outputs must be explainable and auditable:
- record model assumptions (kernel, rank, likelihood)
- record scaling choice (global vs per_task)
- record dataset cutoffs and dates used
- include uncertainty diagnostics (posterior std, correlations, stress scenarios)
- do not claim guaranteed performance

---

## Defaults (When Unsure)

- Keep APIs small and composable
- Add type hints + docstrings by default

---
