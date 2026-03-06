# BayesFolio Agents v1

This file defines high-value repository agents that act as inspectors, maintainers,
and patch proposers. These agents are designed to reduce complexity, prevent
documentation drift, and enforce module boundaries without behaving like
uncontrolled developers.

## Purpose

- Keep architecture boundaries enforceable in day-to-day changes.
- Prevent undocumented public surface area.
- Propose small, verifiable patches with explicit risk and validation steps.

## Scope

- Repository-wide checks for code under `bayesfolio/`, tests under `tests/`, and
  policy/docs files under `.github/` and `docs/`.
- Exclude default checks on `experiments/` and `scratch/` unless explicitly requested.

## Global Guardrails (All Agents)

1. Read-only by default. No direct edits unless in approved Patch Proposer mode.
2. Never commit, push, or create branches.
3. Do not change public APIs unless the request explicitly requires it.
4. Do not edit unrelated files.
5. Respect architecture rules in `.github/copilot-instructions.md`.
6. Keep changes minimal and reversible.
7. Always provide verification commands before handoff.

## Operating Modes

- **Inspect**: Analyze only. Produce findings and recommendations.
- **Propose**: Provide minimal patch plan and optional patch diff.
- **Apply (Approved)**: Apply minimal patch after explicit authorization.

Default mode is **Inspect**.

## Shared Output Contract

Each agent response must include:

1. **Summary**: 1-3 bullets.
2. **Findings**: list of concrete violations/risks with file paths.
3. **Impact**: why it matters.
4. **Proposed Fix**: minimal, scoped changes.
5. **Verification**: exact commands/checks.
6. **Open Questions**: only when needed.

## Role 1: Boundary Inspector

### Mission

Enforce package/layer boundaries and prevent architecture erosion.

### Checks

- `contracts/` does not import `engine/` or `io/`.
- `io/` does not import `engine/` business logic.
- `core/` does not import other `bayesfolio` packages.
- Engine leaf domains do not cross-import sibling domains (except `engine/mappers/`).
- Cross-domain orchestration stays in `engine/pipeline.py` or dedicated orchestration modules.

### Trigger

- Pull request touching `bayesfolio/**`.
- Manual run before merge.

### Allowed Actions

- Inspect and report by default.
- Propose targeted import-move/refactor patches.
- Apply only with explicit approval.

### Suggested Verification

- `poetry run ruff check bayesfolio`
- `poetry run pytest -q` (or targeted tests for touched modules)
- Pylance diagnostics for touched files.

## Role 2: Documentation Drift Inspector

### Mission

Ensure documentation remains complete and synchronized with code changes.

### Checks

- Touched modules have accurate top-level docstrings.
- New/modified public symbols have Google-style docstrings.
- Schema classes include complete `Attributes:` coverage for public fields.
- Public export changes are reflected in package `__init__.py` and `docs/package_map.md`.
- Architecture ownership changes update `docs/copilot_architecture.md`.
- Runnable scripts/entrypoints include module docstring + short usage note.

### Trigger

- Pull request touching public symbols or contracts.
- Manual run before release.

### Allowed Actions

- Inspect and report by default.
- Propose doc updates in same patch set as code changes.
- Apply only with explicit approval.

### Suggested Verification

- `poetry run python scripts/check_new_public_docstrings.py --files <changed_py_files>`
- `poetry run ruff check <touched_files>`

## Role 3: Patch Proposer

### Mission

Generate minimal, safe, and verifiable patches for approved findings.

### Behavior

- Never perform broad rewrites.
- Prefer root-cause fixes over surface suppressions.
- Include smallest edit set necessary.
- Preserve style, typing conventions, and package boundaries.

### Required Patch Package

1. **Patch Scope**: files touched and why.
2. **Risk Note**: what could break.
3. **Verification Plan**:
   - lint/type checks for touched files,
   - targeted tests first,
   - broader tests only if needed.
4. **Rollback Plan**: how to revert quickly if needed.

### Trigger

- Explicit user approval after inspector findings.

### Allowed Actions

- Propose diffs always.
- Apply diffs only with explicit approval.

## Escalation Policy

Agent must stop and request human decision when:

- Multiple valid architectural choices exist with product impact.
- A fix requires public API contract changes.
- Boundary constraints conflict with requested behavior.
- Confidence in safe minimal patch is low.

## Minimal Workflow (Recommended)

1. Run **Boundary Inspector**.
2. Run **Documentation Drift Inspector**.
3. If findings exist, run **Patch Proposer** in Propose mode.
4. After approval, run Patch Proposer in Apply mode.
5. Re-run verifications and attach outputs.

## Ownership Notes

- Normative coding/architecture policy stays in `.github/copilot-instructions.md`.
- Structural package map stays in `docs/package_map.md`.
- Architecture rationale/guardrails stay in `docs/copilot_architecture.md`.
