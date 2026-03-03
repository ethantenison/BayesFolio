# GitHub Copilot Instructions for BayesFolio

BayesFolio aims to be an AI-powered portfolio optimization platform. 

These instructions are **STRICT**. Generated code must follow them.

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

---

## Architecture Rules (STRICT)

### Separation of Concerns
<insert separation of concerns rules here>

### No Cross-Imports Between Domains
<insert rules to avoid cross-imports between features, models, backtesting, portfolio, evaluation, visualization, configs, mlflow>

### Where Code Goes
<Map for code organization here, e.g.>
---

## Schemas
<Insert schema rules here>

---

## Testing Expectations

<insert testing rules here, e.g.>

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
