# BayesFolio Documentation

Status: Active

Audience: Contributors, maintainers, and coding agents

Canonical topic: Documentation authority, ownership, and navigation

Last verified: 2026-10-10

## Purpose

BayesFolio documentation is long-term repository memory. It should be concise, canonical, searchable, and updated in the same change as the behavior it describes. This system is for software and modeling documentation; it is not a curriculum or training-content framework.

## Authority

Use the following order when sources disagree:

1. Repository policy defines contributor and agent requirements:
   - [AGENTS.md](../AGENTS.md) controls agent modes, guardrails, and review workflow.
   - [Copilot instructions](../.github/copilot-instructions.md) control normative architecture, typing, testing, units, and documentation requirements.
2. Code, contracts, and tests define implemented behavior.
3. One canonical document per topic defines meaning, intended use, and limitations.
4. READMEs and guides provide routing and concise derived explanations.
5. Experiments and dated analyses provide evidence and historical context; they are not normative policy.

Higher-authority sources win. Correct a stale lower-authority source rather than preserving competing versions of the same rule.

## Canonical Topics

| Topic | Canonical owner | Audience |
|---|---|---|
| Installation and common workflows | [Root README](../README.md) | Users and contributors |
| Documentation policy and navigation | `docs/README.md` | Contributors, maintainers, and agents |
| Prediction, uncertainty, and multitask GP rationale | [First principles](first_principles.md) | Users, modelers, and agents |
| Architecture rationale and boundaries | [Architecture guardrails](copilot_architecture.md) | Contributors and agents |
| Package ownership and public entry points | [Package map](package_map.md) | Contributors and agents |
| Read-only Schwab connection | [Schwab account access](schwab_read_only.md) | Users and contributors |
| Public API behavior | Module, class, and function docstrings | API consumers and agents |
| Contract field semantics | Contract class `Attributes:` docstrings | API consumers and agents |
| Experiments and dated findings | [`experiments/README.md`](../experiments/README.md) | Researchers |
| Refactor evidence | `docs/refactor/` | Maintainers; historical unless promoted |

## Document Status

- **Active**: authoritative and maintained with current repository behavior.
- **Provisional**: currently usable but expected to change.
- **Draft**: incomplete and non-authoritative.
- **Historical**: retained as evidence and not a description of current behavior.
- **Superseded**: replaced by another named document.

“Active” means current in this repository. It does not prove that a particular release, environment, or brokerage account uses the documented behavior.

## Audience Labels

Canonical documents should identify one or more of these audiences:

- **Users** run supported workflows and consume outputs.
- **Modelers** evaluate statistical assumptions and evidence.
- **Contributors** change maintained code and documentation.
- **Maintainers** own policy, releases, and long-term structure.
- **Coding agents** need deterministic routing, boundaries, and completion gates.

## Editing Rule

- Update implementation, tests, docstrings, and the owning canonical document in the same change when applicable.
- Put detailed behavior in one canonical location. Elsewhere, use a short summary and a link.
- Do not duplicate policy matrices, formulas, field definitions, or long API inventories.
- New or changed public APIs need useful Google-style docstrings.
- Public contract fields must be covered by the class `Attributes:` section.
- Architecture documents change only when boundaries, ownership, or public entry points change.
- Historical experiment conclusions remain with the experiment unless they are explicitly promoted into current behavior.
- Unrelated legacy documentation gaps do not automatically expand a small change.

## Agent Routing

- For contributor or agent rules, read [AGENTS.md](../AGENTS.md).
- For a normative boundary decision, read the [Copilot instructions](../.github/copilot-instructions.md), then the [architecture guardrails](copilot_architecture.md).
- For current package ownership and preferred entry points, read the [package map](package_map.md) and the relevant package `__init__.py`.
- For retail IRA account connection and secret handling, read [Schwab account access](schwab_read_only.md).
- For field semantics, read the defining contract class.
- For modeling rationale, read [first principles](first_principles.md).
- For historical evidence, search `experiments/` and `docs/refactor/`; do not infer current policy from them.

## Minimum Document Header

Canonical documents should begin with:

```text
Status: Active | Provisional | Draft | Historical | Superseded
Audience: <one or more defined audiences>
Canonical topic: <the single topic owned here>
Last verified: YYYY-MM-DD
```

Historical or superseded documents should also name their current replacement when one exists.

## Completion Gate

A change is incomplete when it creates an undocumented public surface or makes an affected canonical document inaccurate. Review must check module docstrings, public symbol docstrings, contract field coverage, package exports, package-map entries, and architecture ownership where relevant.

The repository currently enforces most documentation coverage through review. Scripts under `scripts/` support docstring audits, but the existence of a script does not imply that CI runs it.

## Maintenance Commands

Use uv for repository commands:

```bash
uv run --locked python scripts/audit_doc_coverage.py
uv run --locked python scripts/check_new_public_docstrings.py --files <changed_py_files>
uv run --locked ruff check <touched_files>
uv run --locked pytest -q <targeted_tests>
```
