# BayesFolio Experiment Archive

Status: Active

Audience: Researchers, maintainers, and coding agents

## Purpose

`experiments/` preserves dated research code and selected evidence without
making every historical study part of the maintained package. Experiment
findings are non-normative until they are promoted into package code, tests,
and the relevant canonical documentation.

All new experiment work belongs here. The older `notebooks/` directory is a
frozen historical archive retained to preserve recorded script and output
paths; do not add new studies there.

## What Stays Versioned

Each experiment should use a dated directory such as
`YYYY-MM-short-topic/`. Version the material needed to understand or reproduce
the decision:

- a README or experiment plan with the hypothesis and status;
- source code and configuration;
- data cutoffs, seeds, package revision, and exact run commands;
- small manifests, decision logs, and curated result summaries; and
- explicit notes about whether a finding was rejected, retained as evidence,
  or promoted into maintained behavior.

Generated `outputs/`, `runs/`, `mlruns/`, and `logs/` are ignored by default.
Already-tracked historical evidence remains tracked. Put newly selected durable
evidence in an experiment-level `evidence/` directory; use `git add -f` only
when an artifact must retain its generated path.

## Maintenance Boundary

- Package code must never import from `experiments/`.
- Experiments may call supported package entry points or pin a historical Git
  revision when exact replay requires it.
- Routine Ruff, Pyright, test, documentation, and source-distribution scopes
  exclude experiments. Validate an experiment explicitly when changing it.
- Package refactors do not require bulk migration of historical experiments.
- Do not copy experiment-only defaults into package code without tests and a
  documented promotion decision.

## Promotion Checklist

Before treating an experimental result as current behavior:

1. identify the winning evidence and the alternatives considered;
2. implement the behavior in `bayesfolio/` behind the appropriate contract;
3. add deterministic or regression coverage under `tests/`;
4. update the canonical documentation owner; and
5. record the promotion commit in the experiment README or decision log.
