# GitHub Copilot Instructions for BayesFolio

## Language & Style
- Python 3.13+
- Poetry 2.3.2
- Use native Python typing (list[str], X | None)
- Do NOT use typing.List, Optional, Union
- All public APIs require Google-style docstrings
- Prefer clarity over cleverness
---

## Code Quality & Linting (STRICT)

This repository uses **ruff** for linting with strict pre-commit hooks. All generated code MUST pass linting before commit.

### Pre-Generation Checks

Before writing any Python code, verify:

1. **No Unused Variables or Imports**
   - ❌ `result = function()` if `result` is never used → `function()`
   - ❌ `import pytest` if pytest is never referenced → Remove import
   - ✅ Every variable assigned must be used
   - ✅ Every import must be referenced

2. **Line Length: 120 Characters Maximum**
   - ❌ Long log messages or f-strings on single line
   - ✅ Split long lines using parentheses or multi-line f-strings:
     ```python
     # Good:
     logger.info(
         f"Connected to {host}:{port}/"
         f"{database}"
     )
     ```

3. **Import Organization**
   - ❌ Imports scattered throughout file
   - ❌ Module-level imports after code
   - ✅ All imports at top of file
   - ✅ Order: stdlib → third-party → local

4. **No Dead Code**
   - ❌ Commented-out code blocks
   - ❌ Variables assigned for "future use"
   - ✅ Only include code that is actively used

5. **Type Hints**
   - ✅ All function parameters and return values typed
   - ✅ Use `from __future__ import annotations` for forward references

### Common Ruff Violations to Avoid

- **F841**: Local variable assigned but never used
- **F401**: Module imported but unused
- **E501**: Line too long (>120 characters)
- **E402**: Module-level import not at top of file
- **E722**: Bare `except:` (use specific exceptions)

### Pre-Commit Verification

Before completing code generation:
1. Mentally check for unused variables and imports
2. Verify line lengths are ≤120 characters
3. Confirm all imports are at file top
4. Remove any debug/test code that isn't needed

**If code doesn't pass ruff linting, it's incorrect.**

---
## Development Environment

### Poetry Environment
BayesFolio uses Poetry for dependency management. **All Python commands must be run within the Poetry environment.**

#### Running Python Code
- **Correct:** `poetry run python script.py`
- **Correct:** `poetry run pytest tests/`
- **Wrong:** `python script.py` (will not find dependencies)
- **Wrong:** `python3 -m pytest` (will not find dependencies)

#### Common Commands
```bash
# Install dependencies
poetry install

# Run Python script
poetry run python path/to/script.py

# Run tests
poetry run pytest tests/

# Run specific test file
poetry run pytest tests/unit/test_file.py
```

## Architecture Rules (STRICT)
- Workflows/orchestrators coordinate only — no computation
- All computation for workflows lives in helper modules
- Do NOT introduce cross-imports between:
  data / ml / optimization / interpret
- New functionality must use helpers + registries
- Do NOT modify core workflows to add features

## Config & Schemas
- All configs and results are Pydantic models
- `*Config` classes define workflow behavior
- `*WorkflowResult` classes are transport schemas crossing ML ↔ backend boundaries
- Transport schemas must be JSON-serializable
- NEVER use pickle or dill
- All transport schemas require schema_version
- Changing transport schemas requires regression test updates

## Where Code Goes
- Data ingestion & validation → BayesFolio/data/
- Surrogate models & tuning → BayesFolio/ml/
- Bayesian Optimization → BayesFolio/bo/
- DOE & Space Filling Design -> sureact/doe
- Interpreting Variables -> BayesFolio/interpret
- Workflow orchestration (coordination only) → BayesFolio/workflows/
- Workflow helper logic → helpers modules within workflows
- Config schemas (`*Config`) → BayesFolio/schemas/configs/
- Transport schemas (`*WorkflowResult`) → BayesFolio/schemas/results/
- Tests → tests/unit/, tests/integration/, tests/regression/, tests/workflow
- New Feature Documentation → docs/new_feature_info/ *use descriptive filenames*

## Testing Expectations
- Helper logic requires unit tests
- Workflows require workflow tests
- End to End test require integration tests
- Config & Transport schema require regression tests
- Fix random seeds for determinism

## Defaults
- Prefer small, testable helpers
- Avoid modifying existing APIs
- Maintain backward compatibility for transport schemas
- Follow coding_guidelines.md strictly

## Variable Definitions
- See [docs/variable_definitions.md](../docs/variable_definitions.md) for comprehensive guide on:
  - Differences between variable type, role, and usage
  - Supported input/output/meta variable types
  - Parameter settings and caveats
  - Current limitations (temporal inputs, output types, Non-Modeled Variables)
 
## Sustainability & Efficiency Guidelines

BayesFolio is a long-lived, agent-facing system. Code must be safe to evolve,
cheap to reason about, and efficient at scale.

### Performance & Resource Use
- Prefer linear-time algorithms; avoid accidental quadratic behavior
- Avoid materializing large intermediate objects when streaming or iterating is sufficient
- Cache expensive computations when results are reused (explicit caches only)
- Do not recompute schema-derived metadata repeatedly — compute once and reuse

### Memory & Serialization
- Be mindful of object lifetimes in workflows and helpers
- Avoid copying large arrays or DataFrames unless strictly necessary
- Transport schemas must remain compact and JSON-friendly

### API & Schema Stability
- New fields must be additive and backward-compatible
- Prefer optional fields over breaking changes
- Avoid overloading a single field with multiple meanings

### Abstraction Discipline
- Helpers may grow; workflows must not
- If logic is reused in more than one place, extract it
- Do not encode business rules implicitly — make them explicit and testable

### Agent Awareness
- Assume code and schemas may be queried by LLMs
- Prefer explicit names and structured metadata over clever inference
- Avoid hidden coupling that an agent cannot observe

### Cost Awareness
- Assume workflows may be executed repeatedly in CI, batch, or agent loops
- Avoid unnecessary randomness or non-determinism
- Make expensive operations obvious and documented

**NOTE**: When in doubt, prefer the simplest solution that preserves extensibility.
