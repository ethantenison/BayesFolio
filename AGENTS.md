# BayesFolio Agents v1

This file defines high-value repository agents that act as inspectors, maintainers,
and patch proposers. These agents are designed to reduce complexity, prevent
documentation drift, and enforce module boundaries while completing authorized
repository work without unnecessary approval pauses.

## Purpose

- Keep architecture boundaries enforceable in day-to-day changes.
- Prevent undocumented public surface area.
- Propose small, verifiable patches with explicit risk and validation steps.

## Scope

- Repository-wide checks for code under `bayesfolio/`, tests under `tests/`, and
  policy/docs files under `.github/` and `docs/`.
- Exclude default checks on `experiments/` and `scratch/` unless explicitly requested.

## Global Guardrails (All Agents)

1. Apply in-scope work by default when the user requests implementation, refactoring,
   fixes, or continuation of an active goal. Use read-only mode only when the user
   requests inspection, review, diagnosis, or a proposal without implementation.
2. Branch creation, commits, and pushes are allowed as normal delivery steps. Merging
   a pull request always requires explicit human approval.
3. Do not make breaking public API changes unless the request explicitly requires
   them or the human user approves the specific change.
4. Do not edit unrelated files.
5. Respect architecture rules in `.github/copilot-instructions.md`.
6. Keep changes minimal and reversible.
7. Always provide verification commands before handoff.

## Operating Modes

- **Inspect**: Analyze only when the user requests inspection, review, diagnosis,
  or a proposal without implementation.
- **Propose**: Provide minimal patch plan and optional patch diff.
- **Apply**: Implement requested in-scope work, including an active goal, without a
  separate approval pause.

Default mode follows the user's request. Explicit implementation, refactoring, fix,
or active-goal continuation requests use **Apply**; analysis-only requests use
**Inspect**.

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

- Pull requests touching `bayesfolio/**`.
- Manual run before merge.

### Allowed Actions

- Inspect and report for analysis-only requests.
- Propose targeted import-move/refactor patches.
- Apply in-scope fixes when implementation or active-goal work is requested.

### Suggested Verification

- `uv run --locked ruff check bayesfolio`
- `uv run --locked pytest -q` (or targeted tests for touched modules)
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

- Pull requests touching public symbols or contracts.
- Manual run before release.

### Allowed Actions

- Inspect and report for analysis-only requests.
- Propose doc updates in the same patch set as code changes.
- Apply in-scope documentation fixes with the implementation they describe.

### Suggested Verification

- `uv run --locked python scripts/check_new_public_docstrings.py --files <changed_py_files>`
- `uv run --locked ruff check <touched_files>`

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

- An implementation, refactoring, fix, or active-goal continuation request, or
  explicit user approval after inspector findings.

### Allowed Actions

- Propose diffs always.
- Apply in-scope diffs when the user requested implementation or continuation of an
  active goal.

### Suggested Verification

- `uv run --locked ruff check <touched_files>`
- `uv run --locked pytest -q` (or targeted tests for touched modules)
- Pylance diagnostics for touched files.

## Role 4: Simplicity & Redundancy Inspector

### Mission

Keep the codebase lean by identifying duplication, unnecessary indirection, and maintainability drag.

### Checks

- Duplicate or near-duplicate logic across modules/functions.
- Redundant modules/files that overlap in responsibility.
- Dead code paths, stale helpers, and unused exports.
- One-off wrappers/adapters that add layers without clear reuse.
- Complexity hotspots (long functions, branching-heavy flows, repeated special-casing).

### Trigger

- Pull requests touching multiple modules in the same domain.
- Periodic cleanup runs before release.
- Tiny, single-file changes that do not touch public surface area may skip this role unless duplication or complexity risk is present.

### Allowed Actions

- Inspect and report for analysis-only requests with concrete consolidation candidates.
- Propose minimal deduplication/simplification patches with risk notes.
- Apply in-scope simplifications when implementation or active-goal work is requested.

### Suggested Verification

- `uv run --locked ruff check bayesfolio tests`
- `uv run --locked pytest -q` (or targeted tests for simplified areas)
- Pylance diagnostics for touched files.

## Escalation Policy

Agent must stop and request human decision when:

- Multiple valid architectural choices exist with product impact.
- A fix requires a breaking public API contract change not already authorized by the
  request.
- Boundary constraints conflict with requested behavior.
- Confidence in safe minimal patch is low.
- A destructive or materially out-of-scope action is required.
- A pull request is ready to merge. Creating or updating the pull request does not
  authorize merging it.

## Minimal Workflow (Recommended)

1. Run **Boundary Inspector**.
2. Run **Documentation Drift Inspector**.
3. Run **Simplicity & Redundancy Inspector**.
4. If findings exist during an implementation or active-goal request, run **Patch
   Proposer** in Apply mode for the smallest in-scope patch.
5. For analysis-only requests, remain in Propose mode until the user requests
   implementation.
6. Re-run verifications and attach outputs.

Note: Role 4 may be skipped for tiny, single-file changes that do not touch public surface area, unless there is clear duplication or complexity risk.

## Ownership Notes

- Normative coding/architecture policy stays in `.github/copilot-instructions.md`.
- Structural package map stays in `docs/package_map.md`.
- Architecture rationale/guardrails stay in `docs/copilot_architecture.md`.
