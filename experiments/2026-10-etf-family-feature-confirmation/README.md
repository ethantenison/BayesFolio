# ETF Family Feature Confirmation

Status: U.S. follow-up and international confirmation/ablation stages complete

## International equity orchestration

The international task universe is `VEA`, `VWO`, `VSS`, and `EWX`. A cached
artifact builder joins the verified July international feature panel with the
missing ETF columns from the local production cache, then freezes the last 24
complete dates. The phase controller validates hashes, preflights every
configuration, executes or safely reuses the three-seed runs, and writes the
Pareto and hypervolume reports.

Run the confirmation matrix:

```bash
uv run --locked python \
  experiments/2026-10-etf-family-feature-confirmation/orchestrate_international_experiments.py \
  --config experiments/2026-10-etf-family-feature-confirmation/international_confirmation_config.json \
  --execute --seeds 27,127,227
```

Run the targeted macro ablations after the confirmation baseline exists:

```bash
uv run --locked python \
  experiments/2026-10-etf-family-feature-confirmation/orchestrate_international_experiments.py \
  --config experiments/2026-10-etf-family-feature-confirmation/international_targeted_ablation_config.json \
  --execute --seeds 27,127,227
```

The completed results select the corrected momentum-plus-breadth/flow model as
the primary international candidate. The oil plus gold/crude block is also
nondominated because it slightly improves NLL while materially reducing IC and
Sharpe; it is retained as a calibration-oriented research alternative. A
subsequent oil-only run materially improves IC and slightly improves Sharpe but
worsens NLL, creating a separate ranking-oriented Pareto point. See `STATUS.md`
and the international result reports for the full metrics.

## One-command U.S. orchestration

The U.S. seed panel is encoded in `us_confirmation_config.json`; its execution does not require Qwen to inspect or
reinterpret the repository. The controller validates the frozen cached input and date hashes, snapshots the exact
configuration and execution-critical sources, preflights all four configurations, resumes or runs each requested seed,
validates every completed run, and writes the conventional-Sharpe Pareto report. Qwen is used only afterward, where
reasoning is valuable, for bounded next-round research and planning.

Preview all 12 resolved commands without writing or fitting:

```bash
uv run --locked python \
  experiments/2026-10-etf-family-feature-confirmation/orchestrate_us_confirmation.py \
  --seeds 27,127,227
```

Execute the complete three-seed U.S. confirmation followed by one bounded xhigh Qwen planning turn:

```bash
uv run --locked python \
  experiments/2026-10-etf-family-feature-confirmation/run_qwen_confirmation_workflow.py \
  --execute
```

Each execution writes an immutable-start snapshot, resolved commands, per-command logs, and atomic status checkpoints
under `executions/<execution-id>/`. The controller stops if the sources, configuration, cached feature artifact,
artifact manifest, or date file changes after snapshotting. Existing output is reused only when it passes the complete
run-validation contract. The Qwen handoff text is in `QWEN_RUN_PROMPT.md`.

After the deterministic aggregate (stage 3) passes, the workflow calculates dominated hypervolume using the fixed
raw-objective reference `(IC=0, Gaussian NLL=0, Sharpe=0)`. It then gives Qwen exactly one `xhigh` reasoning turn to
research and preregister 4-8 next-round candidates. The batch must include exploitation, frontier bridging, and genuine
exploration. Qwen may plan but cannot launch those proposed fits; deterministic validation and human review remain
gates before another experiment round.

## Purpose

Confirm whether family-specific multitask Gaussian-process inputs reduce noise and improve useful forecasts relative to
running the full shared input set for every ETF target. This experiment continues the July 2026 U.S. and international
equity studies, then extends the same disciplined process to additional ETF families.

This document is the source-of-truth specification for the Qwen-assisted confirmation work. The planning checkpoint
is complete. U.S. Phase 0 and the four seed-27 U.S. pilots were subsequently authorized and completed; broader seed sweeps
and later phases remain separately gated.

## Existing Evidence to Inspect

At minimum, read and reconcile these sources before proposing new runs:

- `experiments/2026-07-us-equity-etf-features/experiment_plan.md`
- `experiments/2026-07-us-equity-etf-features/decision_log.md`
- `experiments/2026-07-us-equity-etf-features/redundancy_pruning_decisions.md`
- `experiments/2026-07-us-equity-etf-features/run_first_round_walkforward.py`
- `experiments/2026-07-us-equity-etf-features/run_macro_split_kernel_batch.py`
- `experiments/2026-07-us-equity-etf-features/runs/`
- `experiments/2026-07-international-equity-etf-features/README.md`
- `experiments/2026-07-international-equity-etf-features/runs/international_equity_dynamic_feature_screen_20260708T2115Z/`
- `experiments/2026-07-international-equity-etf-features/runs/international_equity_remaining_feature_screen_20260708T2320Z/`
- `experiments/2026-07-international-equity-etf-features/runs/xgboost_shap_optuna_rankspace_400_20260709T0036Z/`
- current monthly-GP defaults and feature configuration under `bayesfolio/`

Historical candidate labels are not sufficient specifications. Reconstruct every proposed candidate from its saved
feature columns, covariance components, interaction terms, task-covariance configuration, time modulation, optimizer
budget, seed, and walk-forward dates.

## Fixed Decisions

- Exclude `IJR` from this work. The U.S. equity task universe is `SPY`, `MGK`, `VTV`, and `IWM`.
- Include `t_index` in every GP candidate and baseline.
- Do not run a `t_index` ablation. Time-input experiments are separate future work.
- Use cached local data and existing artifacts for confirmation planning. Do not refresh data from external services.
- Keep production integration, combined-portfolio reruns, and deployment out of scope. Those begin after this
  experiment program has produced confirmed family-level decisions.
- Preserve all existing modified and untracked files. Do not reset, discard, overwrite, commit, push, merge, or create
  a separate worktree during the planning checkpoint.

## Multi-Objective Evaluation

Candidate selection is a multi-objective problem. Evaluate every candidate on these three objectives:

1. maximize out-of-sample cross-sectional information coefficient (`IC`);
2. minimize out-of-sample Gaussian negative log likelihood (`NLL`); and
3. maximize out-of-sample portfolio Sharpe ratio (`Sharpe`).

For this experiment, Sharpe means the conventional annualized arithmetic-return ratio:
`mean(monthly excess return) / sample standard deviation(monthly excess return) * sqrt(12)`. The portfolio return series
is built from `y_excess_lead`, which already subtracts the integrated risk-free return, so no second risk-free subtraction
is applied. Do not use the underlying runner's legacy `CAGR / annualized volatility` field as a confirmation objective.

Do not combine the three objectives into an arbitrary weighted score. Report the Pareto frontier. Candidate A dominates
candidate B only when A is at least as good as B on all three objectives and strictly better on at least one. If multiple
candidates remain Pareto-optimal, retain them as finalists for an explicit later decision rather than inventing a
tie-breaker.

Report each objective by seed, scored window, and ETF where the metric is defined, together with aggregate means and
dispersion. Stability analysis describes uncertainty in the three objectives; it is not a fourth objective.

Record cumulative portfolio return and RMSE as descriptive diagnostics only. They must not determine dominance,
ranking, rejection, or promotion. Do not consider turnover when comparing or selecting candidates. Historical files may
contain turnover columns, but the confirmation workflow must exclude turnover from its evaluation tables and decision
logic.

## Fair-Comparison Requirements

- Compare candidates on identical task universes, target definitions, data cutoffs, scored dates, expanding-window
  construction, scenario generation, optimizer settings, and random seeds.
- Compare family-specific candidates with the current all-input multitask GP on the same evaluation panel.
- Use deterministic seeds and save them in every resolved configuration.
- Separate candidate discovery from confirmation. Do not select and judge a candidate on the same short screen.
- Report missing windows, failed fits, and non-comparable historical artifacts rather than silently dropping them.
- Prevent look-ahead bias in every feature, label, and rebalance date.
- Save enough configuration to reproduce targets, feature columns, kernel structure, interactions, task covariance,
  time handling, likelihood, seed, dates, optimizer budget, and metrics.

## Work Program

### 1. Confirm the U.S. Candidate

Use only `SPY`, `MGK`, `VTV`, and `IWM`. Reconstruct and confirm the four-task `u005 AC` candidate over a longer history
and multiple deterministic seeds. Compare it with the appropriate historical compact baseline, a simpler compact-feature
baseline if distinct, and the current all-input GP. Keep `t_index` in every model.

The historical label `u005 AC` is only a pointer. Verify its exact feature and kernel definition from the saved run
artifacts before including it in the run matrix.

### 2. Confirm the International Candidate

Use `VEA`, `VWO`, `VSS`, and `EWX`. Confirm the corrected momentum-plus-breadth/flow candidate against
momentum-plus-market, momentum-only, and the current all-input GP. Use longer history and multiple deterministic seeds,
with `t_index` retained throughout. Do not treat the earlier hard-coded `split_groups` screen as valid evidence against
international features.

### 3. Run Targeted International GP Ablations

Translate the valid XGBoost/Optuna/SHAP results into a small, preregistered set of GP ablations. Treat SHAP as hypothesis
generation, not proof of GP feature importance. Test coherent blocks rather than another broad combinatorial search:

- `dxy` and `em_fx_ret`;
- `breakeven_proxy`;
- `hy_spread`;
- `oil_ret` and `gold_crude_ratio`; and
- only a few economically justified interactions.

Compare these candidates with the confirmed breadth/flow candidate under the same evaluation protocol.

### 4. Hold the Time Input Constant

Keep `t_index` included and record that its separate ablation is deferred. Perform no additional time-input experiment
in this work program.

### 5. Evaluate Remaining ETF Families

Proceed sequentially after the equity confirmations:

1. investment-grade bonds and Treasuries;
2. high-yield and emerging-market debt;
3. REITs; and
4. commodities.

For each family, define a nonredundant target universe, audit feature correlation and missingness, establish a
family-specific compact baseline, test small economically coherent feature blocks, and confirm the Pareto-leading set
over longer history and multiple seeds. Do not reuse the equity feature set without family-specific justification.

## Planning Checkpoint

The initial Qwen run must not launch full GP confirmations. It may inspect files and run cheap read-only or help commands.
It may write only the following planning artifacts in this directory:

- `PLAN.md`: evidence inventory, invalid/non-comparable artifacts, exact candidate reconstructions, proposed run matrix,
  seeds and date ranges, exact commands, expected run count, approximate compute burden, required minimal code changes,
  validation commands, and data gaps;
- `STATUS.md`: current milestone, completed inspection, decisions, risks, blockers, and the next action awaiting approval.

The run matrix must show how IC, NLL, and Sharpe will be aggregated and used to calculate the Pareto frontier. It must
label return and RMSE as record-only diagnostics and exclude turnover.

Stop after writing and validating `PLAN.md` and `STATUS.md`. Do not begin the full U.S. confirmation until the user has
reviewed the proposed matrix and explicitly authorized execution.

## Planning Verification

Before declaring the planning checkpoint complete:

- re-read this specification and confirm every fixed decision appears in `PLAN.md`;
- confirm the run commands reference files and arguments that exist in the current checkout;
- confirm all proposed candidates include `t_index` and none include `IJR`;
- confirm the selection procedure uses Pareto dominance over IC, NLL, and Sharpe only;
- confirm return and RMSE are record-only and turnover is excluded;
- report `git status --short` without modifying or cleaning pre-existing work; and
- follow the response contract in the repository `AGENTS.md`.
