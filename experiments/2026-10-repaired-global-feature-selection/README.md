# Repaired family and global input selection

Active study specification: `study.json`. This supersedes the cached-only and
separate-family-only restrictions of the October confirmation study **for this
new experiment only**. Original artifacts, results, and production defaults are
preserved. Phase 3 permits 12 new configurations per family per round; the global
loop permits six per round. Both loops have three rounds and a two-stalled-round
early-stop rule.

The original adopted prompt, with only the authorized family-cap amendment, is
preserved in `GOAL_CONTRACT.md`. That contract defines completion; an interim
green test run or a single global winner does not replace its full scope.

## Completion contract

1. Freeze an independently versioned dataset with 84 initial training months
   (May 2017–April 2024), followed by 24 common monthly scored dates
   (May 2024–April 2026), all 15 approved ETFs, and source/hash/coverage validation.
2. Inventory and deduplicate valid historical masks; rebaseline all four families
   with three seeds and fixed kernel construction and fitting controls.
3. Interpret deterministic evidence and preregister adaptive family batches.
   Use paired additions/removals, redundancy substitutions, and motivated blocks;
   keep lag eligible rather than forcing its removal.
4. Build conditional feature-by-family evidence, including unresolved cases.
5. Test shared-core, union, reduced-union, and exception hypotheses in the actual
   15-task GP; then run the bounded global adaptive loop. Report pooled and
   within-family IC and actual combined-portfolio Sharpe separately.
6. Deliver Pareto tables, exact inputs, coverage/sharpness diagnostics, decision
   logs, reproducible/resumable commands, and recommendations with limitations.

IC, Gaussian NLL, and conventional annualized arithmetic excess-return Sharpe
are the only selection objectives. Return and RMSE are diagnostic only. Do not
use turnover. Count inputs without silently making count a fourth objective.
Keep `t_index`; exclude IJR, REIT targets, and commodity targets. Commodity
predictors are eligible. Do not use local Qwen. Planning is hosted Codex reasoning;
arithmetic, fits, constraints, and evidence validation are deterministic.

The all-input **feature** comparator matches the current `MonthlyForecastConfig`
lists exactly. It uses the same research kernel/control policy as every other
mask (signed LKJ, no time modulation, rank 2, 15 iterations, 256 scenarios), not
the production model's positive/no-prior task kernel, lengthscale modulation,
rank 5, 75 iterations, and 5,000 scenarios. Results establish conditional input
importance under the controlled research setup; they are not a claim to have
beaten the deployed production model or a reason to deploy without confirmation.

## Data policy

Download sources once to ignored `artifacts/v1/raw/`; offline reconstruction uses
these immutable snapshots. Preserve original return targets and rebuild eligible
predictors using sufficient pre-2017 warm-up. Predictor formulas follow the
existing engineering definitions, including their unusual proxy semantics:
`spy_ret` is the last daily log return of the month, `erp` subtracts the annual
3-month bill yield from that signal, `pct_above_50dma` is price/MA50 times 100
(not the proportion of index constituents above an MA), and real-rate/breakeven
inputs are ETF-return proxies, not observed real yields. Do not silently give
these variables new definitions.

Use explicit month-end aggregation and a single one-period feature lag. The
cross-sectional momentum rank is computed over the same fixed 15-ETF universe
before family splitting, so shared predictor values stay equal across studies.
This is a new, documented universe convention; old ranks are not pooled.

ICE spread snapshots are for local internal research only; do not publish their
raw observations. Current FRED observations and the two historical snapshots
must agree on overlap. FRED CPI is latest-vintage, not a point-in-time vintage;
one-month feature lag is retained and revision limitations remain explicit.
October 2025 CPI is genuinely unavailable (BLS shutdown gap). Compute inflation
signals without level filling, then hold the last valid signal for at most two
months, recording every carried month in the audit. Do not interpolate CPI levels
or interpret the September-to-November change as a one-month change.

No unknown-to-zero filling, future backfill, silent date dropping, or blending
old and new fronts. Feature calculations use only trailing observations.
Suspicious invalid or unsupported series are quarantined explicitly. Corrections
after freezing require another dataset version.

## Resource and evidence policy

At most 15 independent fits across the whole workflow, with one Torch thread
and one scenario worker each; lower concurrency under memory pressure. Smoke
test before batches. Persist plans, source hashes, resolved commands, predictions,
and atomic status. Only reuse exact matching source/config/data/date/seed runs.

Historical windows have already been inspected. This remains adaptive research,
not an untouched holdout. Calibrated BO, formal new holdout design, production
integration, and live trading remain deferred. A defensible negative finding is
valid; neither seed repeats nor narrow intervals prove generalization.

## Current execution

- Phase 1: v1 repaired dataset and the v2 historical feature-pool extension are
  frozen and validated. Every shared v1/v2 cell is bitwise identical.
- Phase 2: historical three-seed reruns are complete. The corrected registry reconciles
  July and October: 320 family masks plus the 15-task all-input control. These counts
  exclude still-quarantined historical liquidity/turnover blocks.
- Scope-audit correction: five valid historical macro-only manifests had been
  excluded by an unintended local-block requirement. Two deduplicated supplemental
  family baselines have now completed all six seeds; neither changes the family fronts.
  See results/HISTORICAL_MACRO_ONLY_SCOPE_CORRECTION.md; old results stay preserved.
- Phase 3: all three adaptive rounds for all four families are complete. Each
  round improved the frontier; stop at the cap, not an asserted convergence point.
- Phase 4: final component synthesis is frozen with 463 exact feature contexts,
  full fronts and uncertainty diagnostics, plus hosted cross-family interpretation.
- Phase 5: six initial global masks and all three adaptive six-mask rounds are
  complete, with all three seeds and independently verified objectives. Global
  scope is twenty-five masks/seventy-five seed runs. The complete study has
  489 universe/mask configurations and 1,467 seed runs; all searches stop at their
  caps, not asserted convergence. See results/FINAL_RESEARCH_REPORT.md,
  results/FINAL_PARETO_TABLES.md, results/final_recommendations.json and
  results/FINAL_COMPLETION_AUDIT.md for the actual conclusions and limitations.

The v2 pool adds `vol_1m`, `vol_accel`, `vol_ratio_1m_3m`, `vix3m`,
`vix_ts_chg_1p`, `dealer_gamma_proxy`, oil/copper/gold levels, EM-FX level,
and the RSP/SPY level ratio. No new raw sources or changed common values were
introduced. V1 fits retain v1 hashes; new masks using v2 inputs retain v2 hashes.

Dates denote the existing runner's observation/decision dates. A target at t is
the forward excess return ending at the following business month-end. Thus the
24 scored observation dates May 2024–April 2026 correspond to forward outcomes
ending June 2024–May 2026. See `results/TARGET_ALIGNMENT_AUDIT.md`.

```bash
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/repair_dataset.py --fetch
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/repair_dataset.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/inventory_history.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/extend_feature_pool.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/register_pool_history.py
.venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_repair_dataset.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/workflow.py --check
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/workflow.py --execute --workers 15
```

The workflow validates all registered plans, executes them with at most 15 total
fit workers, and resumes only exact matching outputs. After a batch, hosted Codex
reviews the persisted evidence and registers the next bounded plan; it does not
launch unvalidated plans or depend on a local model producing tool calls. The
single command resumes the currently registered work; the active Codex goal is
responsible for the reasoning turns and creating subsequent adaptive plans.

After registered results are complete, refresh cumulative reports and the
conditional cross-family synthesis:

```bash
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/analyze_results.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/synthesize_components.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_joint_objectives.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_model_contract.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_execution_coverage.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/package_final_results.py
```

The synthesis refuses incomplete/stale cumulative inputs and records source
hashes. `results/component_synthesis/` contains every matched feature context,
feature-by-family evidence states, adaptive block effects, full fronts, and
monthly pooled/within-family calibration and interval widths. Context counts are
descriptive only: they are not feature-importance votes or selection weights.

The joint objective audit independently recomputes monthly Spearman IC from
scipy ranks, marginal Gaussian NLL, actual weights-times-excess-return portfolio
arithmetic, conventional sample-SD Sharpe, return/RMSE diagnostics, and family
IC/NLL/coverage. It compares these with published reports without importing their
metric calculators. It records pending/failing joint seeds explicitly and exits
nonzero until every registered joint seed and its current report pass. Run it
after refreshing cumulative evidence, and again at final handoff.

The no-fit model-construction audit independently checks actual signed task
priors, likelihood/noise and causal transforms in all five universes. Its overlay
also documents misleading legacy prior/correlation-group metadata without
rewriting run manifests or changing execution-critical sources. See
`results/MODEL_CONTRACT_AUDIT.md` and `results/model_contract_audit.json`.

The execution-coverage audit independently revalidates every registered batch,
compares required historical-mask IDs against the registry, refreshes completed
round comparisons from their frozen evidence, and checks cap/early-stop closure.
It exits nonzero for missing/live work. Its output certifies execution scope
only: it does not certify recommendations, scientific validity, or goal completion.
The final report and requirement-by-requirement review remain separate deliverables.

The final packager checks fresh complete-scope, arithmetic, model and synthesis
audits, then joins all exact Pareto masks/seed diagnostics with the hosted
planning/final_decisions.json choices. It does not create a weighted winner or
promote production. Re-running it reproduces the final tables and structured
recommendations without fitting a model. All final scope audits passed; the
scientific completion review remains explicit in FINAL_COMPLETION_AUDIT.md.
