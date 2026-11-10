# Track 1: GP settings under the scenario allocator

## Question and boundary

Does changing task-covariance factor rank or the GP fitting budget improve monthly
portfolio decisions under the existing scenario allocator? Do the improvements
persist for individual broad/compact forecasts and their fixed two-model blend?

This folder owns the Track 1 design and independent result review. The campaign
orchestrator owns one shared fitting/allocation queue; this track must not start
another pool or edit any previously sealed experiment. No fits or allocations
were launched to prepare this plan.

## Verified starting point

The completed 28-month comparison already applies rank **2**, maxiter **15** to
both broad14 and compact5, then feeds the blended predictive mean and 50/50
observed predictive covariance into the scenario allocator. Consequently that
case is a cached control, not a new hypothesis.

The historical July pipeline is not the same model. Its regression fixture
records rank **5**, maxiter **75**, 18 training tasks/16 final tasks, 27 non-time
inputs, lengthscale-only time modulation and no task-covariance prior. The current
experiment keeps the repaired dataset, 15 tasks, signed-LKJ prior and current
kernel/transforms fixed. Changing only rank does not recreate July's model.

Sources, relative to the repository root:

- `tests/fixtures/regression/july_2026/config.json`
- `experiments/2026-10-portfolio-evaluation-v1/CURRENT_BASELINES.json`
- `experiments/2026-10-portfolio-evaluation-v1/monthly_extension_v1/outputs/fit_contract.json`
- `experiments/2026-10-portfolio-evaluation-v1/monthly_extension_v1/outputs/paths.npz`
- `experiments/2026-10-portfolio-evaluation-v1/scenario_sharpe_v3/signed_sharpe.py`
- `experiments/2026-10-portfolio-evaluation-v1/scenario_riskfolio_v1/scenario_allocator.py`
- `experiments/2026-10-portfolio-evaluation-v1/scenario_riskfolio_mv_v2/policy.json`

## Controlled grid

Cross ranks `[1, 2, 5]` with maxiter `[5, 15, 36, 75]`, separately for broad14 and
compact5. All other modeling inputs and settings stay fixed. Each component panel
contains all three original seeds and all 28 forecast origins, with training
strictly before each origin and 84+k expanding months per task.

- 12 rank/budget cells × 2 feature specifications = **24 component panels**.
- Each panel has 3 seeds × 28 months = **84 monthly fits**.
- Two complete rank2/maxiter15 panels are reusable: **22 × 84 = 1,848 new fits**.
- Track 2 reuses these same panels; it must not fit duplicate components.

The 75-iteration level matches the historical July fitting budget and crosses it
with ranks 2 and 5, so those effects can be isolated within the **current** model.
The other architecture, dataset and task differences from July remain frozen;
this still does not replicate July's GP. Component IDs are
`broad_r{rank}_i{maxiter}_exp` and `compact_r{rank}_i{maxiter}_exp`.
The parent campaign additionally reserves four audit refits and six complementary
component panels, for 1,848 + 4 + 504 = **2,356 new campaign fits**. Those audit and
complement fits are not Track 1's fit count. Freeze this revised grid before
examining results; do not add a new level based on which cell currently leads.

Maxiter is an upper bound, not a promise that the optimizer takes that many steps
or reaches its optimum. Preserve actual iterations, convergence status, retries,
fit objective, runtime and numerical diagnostics for every fit. Rank means the
task-covariance factor rank; the diagonal contribution remains present. Do not
describe rank 1 as a one-dimensional complete task covariance.

## Scenario cases and fixed blend

For each of the 12 cells, evaluate these three distributions:

1. **Broad only:** broad14 mean and observed predictive covariance.
2. **Compact only:** compact5 mean and observed predictive covariance.
3. **Symmetric settings, fixed blend:** both components use that cell's settings;
   apply the frozen Sharpe-leader family means and 50/50 observed covariance.

For the blended case, `mu_i = (1-alpha_family)*mu_broad_i + alpha_family*mu_compact_i`.
The compact fractions in family order are `[0.5, 0, 0.375, 0.25]` for U.S. equity,
international equity, U.S. fixed income and international fixed income.
`Sigma = 0.5*Sigma_broad + 0.5*Sigma_compact`.

**Each recipe is constant across every month and seed.** Track 1 performs no blend
optimization. It therefore isolates model-setting changes from recipe changes;
Track 2 will report separately on constant-recipe retuning/asymmetric pairs.

These definitions give **36 scenario cases × 84 = 3,024 seed-month allocations**.
The existing blended rank2/maxiter15 case supplies 84 cached allocations, leaving
**2,940 new allocations**, if cache identity checks pass. Broad-only and
compact-only baseline allocations are new unless an exactly matching sealed
28-month case is found. Reuse only exact compatible evidence, not similar labels.

The arithmetic covariance blend is the approved working Gaussian covariance. It
is not claimed to be the covariance of a true probabilistic model mixture, nor of
a weighted average of two correlated GP random variables with unknown cross-model
covariance. Do not introduce a between-model-mean term, matched family covariance
weights or recalibration during this controlled track.

## Frozen allocator and paired scenarios

- Draw 5,000 joint Gaussian decimal discounted-excess-return scenarios from the
  complete observed posterior, including likelihood noise.
- Use the existing sampling helper and identical `(seed, origin)` RNG key across
  cases. With fixed asset order this reuses base normal draws, making comparisons
  less affected by different Monte Carlo draws. Do not moment-match, clip
  eigenvalues, diagonalize covariance, silently add jitter or discard scenarios.
- Optimize the same signed scenario mean / scenario standard deviation objective:
  long-only, fully invested, minimum effective assets 10, ETF cap 25%, risk-free
  offset zero because the scenarios are already discounted excess returns.
- Use real Riskfolio Classic/MV/Sharpe for feasible positive-mean cases; use the
  existing deterministic sign-safe solver for nonpositive feasible mean cases.
- No CVaR, old-month weights, minimum-volatility replacement, constraint
  relaxation, extra variance ceiling, costs or taxes are introduced.
- Record the raw/returned weight audit, solver route and warnings. A nonpositive
  signed-Sharpe solve is nonconvex: multistart agreement is not a proof of global
  optimality. Negative Sharpe can improve by increasing volatility; show such
  months rather than silently replacing the economic objective.

## Comparisons that isolate the cause

For each source type, report paired rank contrasts `(1,2)` and `(5,2)` at
each fixed maxiter, and budget contrasts `(5,15)`, `(36,15)` and `(75,15)` at each fixed rank.
Also show the interaction: does raising rank help only after raising the fitting
budget? Use every seed/month, not only the most favorable initialization.

Within each cell show broad-only versus compact-only versus the fixed blend.
Because predictions are identical for a given source/cell regardless of allocator,
ranking IC is a model/signal diagnostic, not a portfolio-allocation achievement.

Standing references are the frozen blended rank2/maxiter15 hard-rank strategy,
the frozen blended rank2/maxiter15 scenario strategy, Schwab aggressive proxy and
simple momentum top7 equal-weight. Keep both current GP controls in the report.
Scenario NE10/cap25% differs from hard rank NE7/no cap/variance ceiling: their
comparison is a **whole-strategy comparison**, not a pure allocator experiment.

## Evaluation and independent review

Primary results are a Pareto frontier maximizing conventional annualized Sharpe,
compounded total return and mean monthly cross-sectional Spearman IC. Calculate
metrics only after the complete 28-month path exists. Report the three-seed mean,
each seed, median, dispersion and paired differences from the appropriate control.
Seeds are model initialization replicates, not independent market histories.

Also report drawdown, worst month, annualized volatility, effective asset count,
equity exposure, fit runtime/convergence, marginal/joint predictive NLL and raw
uncertainty diagnostics. NLL is not a promotion target and uncertainty calibration
is not fitted. Report the predeclared first-16 tuning/later-12 validation split
separately from the complete 28-month results; none is a pristine holdout because
this history has already been inspected. Avoid significance claims based on the
number of simulated scenarios, ETF observations or initialization seeds.

For every result candidate, require complete valid weights for all 84 seed-months.
On failure diagnose and make only a documented numerical/orchestration repair
that preserves economics, then resume from checkpoints. If not repairable, mark
the case incomplete and exclude it from winner tables; never compare truncated
paths or hide failures behind the preceding weights.

The Track 1 review should answer, before reading other tracks' winner selection:

1. Does rank or budget have a consistent effect, or only isolated favorable cells?
2. Is a benefit shared by return/Sharpe/ranking, or does it trade objectives off?
3. Does it survive each seed and chronological segment?
4. Does the fixed blend still add value versus its components?
5. Is any apparent gain explained by altered exposure/concentration or solver
   branch behavior rather than improved predictive information?

No promotion is automatic. This is adaptive exploratory research on a short reused
market path; future prospective evidence is still needed.

## Verification and rollback

The shared `case_evaluator.py` adapter reads campaign `context.npz` and component
panels, validates static mean/covariance blends, reuses exact baseline weights and
stores complete-or-failed case evidence. `run_track.py` exposes
`run_track(output_root, workers=14)` for the parent after all fitting ends. It
preregisters all 36 scenario cases plus one hard-rank reference, then writes
`outputs/track1/RESULTS.md`, `summary.json` and `table.csv`. The first-16-only Pareto
selector from Track 2 chooses at most three review cases before later-period
reporting; no portfolio configuration is automatically promoted.

Artifact verification, from the repository root:

```sh
uv run --locked --no-sync python -m json.tool experiments/2026-10-gp-tuning-v1/track1_scenarios/plan.json >/dev/null
uv run --locked --no-sync pytest -q experiments/2026-10-gp-tuning-v1/track1_scenarios
uv run --locked --no-sync ruff check experiments/2026-10-gp-tuning-v1/track1_scenarios
git diff --check -- experiments/2026-10-gp-tuning-v1/track1_scenarios
```

The campaign implementation must additionally verify input hashes, exact cached
baseline arrays, fit counts, complete seed/date grids, finite Cholesky-positive
predictive covariance, scenario RNG reproduction, raw portfolio constraints and
return-to-wealth reconstruction. Freeze the campaign/track plans before computing
results. Rollback is to keep using the untouched prior monthly-extension outputs;
these design files do not change production code, APIs or previous research.
