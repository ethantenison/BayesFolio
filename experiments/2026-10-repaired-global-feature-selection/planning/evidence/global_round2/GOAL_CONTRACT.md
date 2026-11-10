# Executed goal contract

Source: original goal drafted in this chat, turn 01a14f4d-e84e-7d62-aa5e-d26b8d3bbf5f.
The user activated this goal and amended only the phase-three family limit from
six to twelve new configurations per round. Global limits remain six. The text
below preserves that adopted scope; the execution checklist does not replace it.

GOAL: Identify a compact, evidence-backed input set for the combined multitask ETF Gaussian-process model, while establishing which inputs help individual ETF families and which provide broadly shared predictive value.

Work in /Users/et/Documents/BayesFolio. Implement and execute the experiment workflow—not merely a plan—within the boundaries below.

SUCCESS CONDITION

Deliver:
1. A separate, versioned, validated repaired experiment dataset.
2. Fresh, comparable results for all four ETF families.
3. An evidence-directed family refinement loop.
4. A cross-family synthesis followed by actual combined-model feature-selection experiments.
5. Reproducible Pareto tables, conditional feature-importance evidence, and recommendations that clearly distinguish findings from uncertainty.

Success does not require proving that a smaller model wins. A defensible negative result is acceptable. Do not claim an exhaustive search or a globally optimal feature set.

SCOPE AND FIXED DECISIONS

Families:
- U.S. equity: SPY, MGK, VTV, IWM.
- International equity: VEA, VWO, VSS, EWX.
- U.S. fixed income: BND, IEF, LQD, HYG.
- International fixed income: BNDX, VWOB, HYEM.

The combined experimental model contains these 15 ETFs.
Exclude IJR, REIT ETF targets, and commodity ETF targets. Commodity predictors remain eligible.
Retain t_index in every configuration; do not test its removal.
Keep lag_y_excess_lead eligible. Test its value rather than assuming it must stay or disappear.

Use the current hosted ChatGPT/Codex model for interpretation and planning, not local Qwen. Deterministic code must build datasets, validate plans, execute fits, compute metrics, and maintain the experiment registry.

Follow AGENTS.md and .github/copilot-instructions.md. Preserve unrelated changes, original datasets, and historical results. Keep production deployment, live trading, and changes to production defaults out of scope.

The earlier cached-only restriction is superseded specifically for acquiring and verifying the historical data needed for this repaired experiment. Freeze the resulting sources before fitting.

PHASE 1 — BUILD THE REPAIRED DATASET

Inspect the existing experiment infrastructure under:
experiments/2026-10-etf-family-feature-confirmation/

Reuse and minimally extend its builders, runners, validators, and summaries rather than introducing a competing framework.

Create a new versioned experiment directory. Never overwrite the old artifacts or reuse their result directories.

Use:
- Initial training observations: May 2017–April 2024, 84 monthly dates per ETF.
- Evaluation: May 2024–April 2026, 24 common monthly scored dates.
- Expanding training thereafter.
- Seeds: 27, 127, 227.

Verify these are actual usable observations under the runner’s target-availability convention, not merely 84 rows. Feature warm-up history must precede the training start.

Repair:
- mom12m_skip1m using the existing daily adjusted-price definition:
  price.shift(21) / price.shift(252) - 1.
  Do not substitute a monthly-return approximation.
- hy_spread using the exact BAMLH0A0HYM2 index, with its existing decimal units, monthly change, and rolling 12-month z-score.

Previously checked recovery leads:
- Yahoo daily adjusted prices covered all eight equity ETFs.
- Historical credit snapshot:
  https://github.com/maaurocp/Trading_Protocol/blob/bf64e83fa4c2a6e72c37d3883476dc81bd9d2e31/data/raw/fred_BAMLH0A0HYM2.csv
- Independent historical snapshot:
  https://github.com/csaladenes/eco-archive/blob/main/BAMLH0A0HYM2.csv

Reverify these sources. FRED currently limits this ICE series to three years, so a current download alone cannot supply the entire history. Preserve provenance, overlap checks, and applicable internal-use restrictions. Do not silently substitute another credit index or publish restricted raw data.

Audit every other eligible predictor too. Check missingness, suspicious zero-filled spans, constants, warm-up sufficiency, units, transformations, and publication/availability alignment. Distinguish genuine zeros from placeholders.

Do not silently drop dates, fill unknown values with zero, backfill from future observations, or apply the one-period feature lag twice. Fit learned preprocessing on training data only.

If an input cannot be validated, quarantine it explicitly and document affected configurations. Do not silently redefine the all-input baseline.

Save raw-source snapshots where permitted, source and dataset hashes, feature definitions, transformation versions, repaired-cell comparisons, and a machine-readable coverage report. Freeze the dataset before running experiments; any later repair creates a new version.

PHASE 2 — REBASELINE ALL FOUR FAMILIES

Inventory and deduplicate the valid historical feature configurations from the existing family studies. Re-evaluate them on the new common dataset, dates, and three seeds. Exclude invalid historical screens with explicit reasons.

Include each family’s:
- Compute-matched current all-input baseline.
- Prior primary, challengers, and Pareto configurations.
- Relevant simple baselines.
- Paired with-lag/no-lag comparisons, particularly for U.S. fixed income.
- Separate oil-only and gold/oil-ratio tests where relevant.

Do not assume prior winners or importance conclusions survive.

Keep feature selection separate from model tuning. Use a fixed, documented kernel-construction rule, covariance structure, interaction policy, likelihood, fitting budget, and portfolio settings within each comparison. Changing active feature dimensions must not silently change unrelated kernel choices.

Start from the established screening controls:
rank 2, maxiter 15, 256 scenarios, and the saved signed-LKJ setup, after verifying their exact implementation.

Old and new metrics must remain separate. Changing both data quality and training history means their difference cannot be attributed solely to either change. Document that limitation.

PHASE 3 — ADAPTIVE FAMILY REFINEMENT

For each family, run up to three additional rounds, with at most twelve new configurations per round and all three seeds per configuration.

After every completed batch:
1. Validate and aggregate deterministic evidence.
2. Interpret the Pareto frontier, paired ablations, failures, and instability.
3. Write a concise hypothesis ledger and the next machine-readable experiment plan.
4. Validate that plan before launching its fits.
5. Update the cumulative results and checkpoint.

Mix exploitation of promising subsets with exploration of plausible alternatives. Test individual removals, coherent blocks, substitutions among redundant inputs, and a limited number of motivated combinations. Do not rely exclusively on greedy backward elimination.

Treat importance as conditional on the other inputs, training history, and kernel—not as an intrinsic property established by inclusion frequency, correlation, ARD length scales, or SHAP alone.

PHASE 4 — CROSS-FAMILY SYNTHESIS

Build a feature-by-family evidence table using matched add/remove comparisons:
- Helpful.
- Redundant or harmful in the tested context.
- Interaction-dependent or unstable.
- Untested or unresolved.

Include effect sizes for IC, NLL, and Sharpe, seed consistency, and links to exact comparisons.

Generate global hypotheses from this evidence, including:
- A compact shared core.
- The union of promising family inputs.
- Smaller unions that remove redundant proxies.
- Shared inputs plus potentially important family-specific exceptions.
- Features weak separately that might help through cross-task information sharing.

Do not call a feature universally useful merely because it appears in every family’s shortlist. Absence from a shortlist is not proof of irrelevance.

PHASE 5 — COMBINED-MODEL FEATURE-SELECTION LOOP

Fit the actual 15-ETF multitask GP. Family results generate hypotheses; combined-model experiments decide whether those hypotheses transfer.

First evaluate the repaired all-input baseline and the preregistered global candidates. Then run up to three adaptive rounds, at most six new configurations per round, each with seeds 27, 127, 227.

Reason over both combined and family-level results after each batch. Test global removals, additions, redundancy substitutions, and selected interactions using matched controls.

Evaluate a genuinely shared feature set first. Do not simulate family-specific exclusion by replacing values with zero. A family-masked kernel is a different architecture and requires a separately labelled proposal rather than a silent implementation.

Report combined-model objectives alongside every family’s outcomes. Distinguish pooled cross-sectional IC from within-family IC; do not let pooled improvements conceal family deterioration. Do not average family Sharpes and call that the combined portfolio Sharpe.

EVALUATION

Optimize three objectives without an arbitrary weighted score:
- Maximize out-of-sample IC.
- Minimize Gaussian NLL.
- Maximize conventional annualized Sharpe.

Sharpe:
mean(monthly excess portfolio return) /
sample_std(monthly excess portfolio return) * sqrt(12).

Verify the target’s return convention; do not subtract the risk-free rate twice.

Record return and RMSE as diagnostics only. Exclude turnover from selection.

Report Pareto fronts, feature counts, paired objective changes, per-seed dispersion, and window/family breakdowns. Simplicity is an explicit preference when reporting competitive alternatives, not a hidden fourth objective.

Use a frozen hypervolume reference and objective transformation within each study. Do not change them after observing candidates or compare hypervolume across incompatible panels.

Assess predictive uncertainty with NLL and coverage/sharpness diagnostics where supported. Narrower predictive distributions alone are not evidence of better uncertainty. Distinguish forecast uncertainty from seed variability and uncertainty in estimated performance.

Do not add a heavyweight calibrated Bayesian optimizer or a new formal holdout program in this goal. Retain causal walk-forward discipline and label the repeatedly inspected evaluation period as adaptive research evidence—not untouched confirmation or production readiness.

EXECUTION, LIMITS, AND RECOVERY

Use up to 15 fit workers across the entire workflow, not 15 per family. Prevent nested thread oversubscription and reduce concurrency if memory pressure requires it.

Preflight commands and use a smoke run before large batches. Persist plans, resolved configurations, evidence, logs, and atomic status checkpoints. Resume only when dataset, code, configuration, seed, and date hashes match.

Do not repeat discovery after context compaction; resume from the saved state.

Continue routine validated batches without asking for approval after each run. Stay within the round/configuration caps. Stop adaptive work early after two consecutive rounds produce neither frontier improvement nor a competitive simplification.

Stop new launches at the caps and summarize remaining hypotheses. Seek direction only for unavailable essential data, paid access, materially different architecture, destructive actions, or scope expansion.

DELIVERABLES AND COMPLETION CHECK

Provide:
- Versioned repaired dataset and successful validation report.
- Historical-configuration inventory and exclusions.
- Reproducible family and combined-model results.
- Pareto tables with exact input lists and uncertainty diagnostics.
- Feature-by-family evidence table.
- Hypothesis and decision logs for every adaptive round.
- Recommended shared core, unresolved family-specific exceptions, and compact global finalists.
- A one-command resumable workflow, tests, and exact reproduction commands.

Verify executed outputs against these requirements before declaring completion. Report failed or missing fits explicitly. Recommend production changes only as a separate future decision.

