# Reproduce and verify the November v2 decision

Run from `/Users/et/Documents/BayesFolio`:

```sh
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync python experiments/2026-11-portfolio-multiobjective-v2/v2_run.py --verify
```

This performs no fitting, downloads, trading or output replacement. It verifies
the complete delivery fingerprints and protected old evidence, rebuilds pooled
means/covariance from the original expert caches and six new checkpoints,
recalculates November's causal eligible-14 historical-EW variance, and recreates
all ten finalists' 53 monthly allocations plus the November allocation. Raw
weights must match within `1e-12`; financial metrics must reproduce. The raw
CSV is checked independently with a separately expressed strict conic problem.

`--execute --workers 14` resumes the finite experiment from immutable checkpoints;
a completed folder is verified. Initial external auditing, code checks and visual
QA are explicit acceptance gates, never synthesized as automatic success.

## Evidence map

- `GOAL.md`, `PREREGISTRATION.json`, `SPLITS.json`, `SELECTION_RULE.json`,
  `PROTOCOL_CLARIFICATIONS.json`, `ANCHORS.json`: original scope, chronology,
  fixed normalization, decision rules and pre-score numerical conventions.
- `CONTROL_REGISTRATION.json`, `CONTROLS.json`: unchanged three-expert and older
  two-expert controls, replayed with MGK excluded from allocation only.
- `rounds/ROUND01.json` through `ROUND04.json`: timestamped complete adaptive
  menus, registered before scoring. Reviews preserve development-only findings.
- `EXPERIMENT_LEDGER.jsonl`: all 874 new policy records and two controls, including
  every preserved exploratory result. A policy tuple includes predictions,
  covariance, diversification, risk ceiling and optional exposure limit.
- `outputs/development/`: every complete 41-month path, raw optimizer evidence,
  costs, mean/covariance, family metrics, source hashes and numerical checks.
- `GP_VARIANTS.json`, `CACHE_INVENTORY.json`, `GP_CACHE_AUDIT.json`,
  `FIT_CONTRACT.json`, `FITS_COMPLETED.json`: exact fixed reference specifications,
  cache compatibility, limited compact-rank1 admission and six new attempts.
- `outputs/fit_admissions/`, `outputs/fit_checkpoints/`, `outputs/fits/`: conservative
  attempt counting, raw model parameters, actual kernel/prior telemetry, optimizer
  limits and the all-15-task forecasts. References remain rank2; all maxiter15.
- `MATCHED_CONTROL_AUDIT.json`: exact rank pairs and the 16 preserved unmatched
  exploratory leaves, uniformly excluded from promotion and rank attribution.
  This is an explicit conservative post-registration design restriction.
- `ROBUSTNESS_MENU.json`, `outputs/robustness/`: frozen fixed-reference 512-draw
  paired block3/6 sensitivity, annual blocks, leave-one-month and cost guardrails.
- `SHORTLIST_FROZEN.json` and SHA256: ten policies locked before confirmation.
- `outputs/confirmation/`: only those ten policies, complete 53-month paths and
  the final 12-month block. No subsequent policy/blend tuning is permitted.
- `FINAL_SELECTION.json`, `CONFIG.json`, `NOVEMBER_WEIGHTS.csv`,
  `outputs/NOVEMBER_MOMENTS.npz`, `outputs/NOVEMBER_OPTIMIZER.json`: exact final
  choice, all-15 raw decimal targets including MGK=0 and numerical inputs.
- `BENCHMARKS.json`, `outputs/benchmarks/`: only Schwab aggressive and eligible-14
  top-seven momentum; same dates and explicit cash-aware costs.
- `STRESS.json`, `outputs/stress/`: fixed-November-weight historical and stylized
  sensitivity exercises, not historical GP backtests or future probabilities.
- `DISTRIBUTION_DIAGNOSTICS.json/.md`: 14-asset finalist and separate 15-task
  expert Gaussian scores/coverage/projection diagnostics. No recalibration or
  choice change; before/after selection hashes are recorded.
- `ROOT_VALIDATION.json`, `VALIDATION.json`, `outputs/INDEPENDENT_*AUDIT.json`:
  separate numerical, temporal, model-scope, shortlist and final acceptance.
- `outputs/NUMERICAL_PARITY_NOTE.json`: original small conic representation
  discrepancy and uniformly declared `5e-6` independent weight-parity tolerance;
  raw constraints and exact deterministic reproduction remain unchanged.
- `CODE_CHECKS.json`, `VISUAL_QA.json`: executed tests/lint/docstrings and plot QA.
- `COMPLETED.json`: final source/data/code/artifact/environment fingerprints.
  Logs, locks, bytecode/caches and mutable `STATUS.json` are not sealed.

No pristine holdout claim is made: these dates were inspected in earlier
research. The locked latest12 replay prevents further within-run adaptation.
Original data/feature lag and vendor-vintage limits are retained. October30
closing information does not imply an executable October30 fill for a decision
made afterward and potentially executed November2.
