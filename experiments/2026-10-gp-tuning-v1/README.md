# Fixed-recipe multitask GP tuning

Approved October 25, 2026. Research only; production and existing baseline files
remain unchanged. No git mutations, data downloads, CVaR, fallback portfolios,
constraint relaxation, or month-specific blending are authorized by this run.

## Three separate questions

1. Scenarios: isolate rank and fitting-budget effects within the current model
   architecture; broad-only, compact-only and the frozen family-blended mean /
   50-50 covariance recipe. This is not a recreation of the old July GP, which
   also had different inputs, tasks and kernels.
2. Asymmetric components: screen 144 broad/compact setting pairs at the fixed
   incumbent recipe. Shortlist using first16 months only, then try a bounded
   menu of static family mean recipes and separately varied covariance weights.
3. Complements: add broad/compact rolling48/rolling60 models and two distinct
   feature representations. Evaluate incremental value relative to both the
   frozen two-model reference and a first16-selected Track2 recipe.

Each track has its own preregistered PLAN.md and machine-readable specifications,
execution/report module, outputs/trackN/RESULTS.md, and full evidence tables.
The combined report follows all three track reports, never replaces them.

## Registered fit budget

The common model grid is broad/compact × rank 1/2/5 × maxiter 5/15/36/75.
Two rank2/maxiter15 component panels are reused from sealed caches. Therefore:

- 22 new grid components × 3 seeds × 28 months = 1,848 fits.
- Six complementary components × 3 seeds × 28 months = 504 fits.
- Four baseline parity-audit refits = 4 fits.
- Total planned new monthly fits: **2,356**, below the approved roughly 2,500.

One representative month per new specification is smoke-tested before the bulk
queue. These 28 fits count toward, not on top of, the 2,352. Failed attempts are
preserved and may retry the identical settings once within the remaining budget.
Actual optimizer steps/status, training dates, fitted parameters, seeds, timings,
RSS, predictions and full observed predictive covariance are retained.
Maxiter is a numerical fitting cap; stopping there is not claimed as convergence.

One shared maximum-14-process fit queue uses one Torch/BLAS thread per worker.
Portfolio-evaluation pools start only after that queue closes. No agent launches
an independent fitting pool. A campaign lock prevents concurrent runs. All
monthly jobs checkpoint independently; incomplete runs resume without repeating
completed fits. Failed allocation cases retain their complete failure record and
never receive partial performance scores.

## Common data, portfolios, and evaluation

15 ETFs, IWM retained, no REIT/commodity targets. All28 origins May2024–Aug2026
predict June2024–September2026 returns. Seeds 27/127/227 share the same market
history; they are not three independent time samples. Initial expanding training
is 84 months, increasing to111. Rolling variants retain exactly the latest48/60
strictly pre-origin target months and the original global seed/time index.
They do not receive a different allocator risk-budget history.

Every candidate's blending coefficients are constant across **all months and
all seeds**. Family-specific prediction fractions are permitted; covariance
pooling is separately controlled. No monthly adaptive blender is fitted.
These are arithmetic moment pools, not claims of a fully specified probabilistic
mixture or known cross-model error covariance. No uncertainty recalibration is fitted.

Selections use only the first16 origin outcomes. The selected static recipes are
also scored on the later12 months and all28 months. All28 results are exploratory;
this history has already been viewed, so later12 is not a pristine holdout.
Objective tradeoffs are conventional Sharpe, compounded total return and monthly
cross-sectional rank IC. Drawdown, subperiods, seed dispersion, predictive NLL
and risk diagnostics are recorded, but low NLL does not independently promote a model.

Hard rank retains NE>=7, long-only, full investment, no ETF cap, and its unchanged
expanding historical equal-weight variance ceiling. Scenarios retain 5,000 joint
draws, maximum signed Sharpe with MV risk, NE>=10 and ETF cap25%. These economic
constraints differ: comparing these arms is a complete-strategy comparison,
not an isolated allocator experiment. Nonpositive-mean direct signed-Sharpe
multistart agreement is not a global-optimality certificate.

Standing benchmarks remain Schwab aggressive ETF proxy and simple momentum top7
equal weight. NAV starts at $10,000, self-financing, monthly rebalanced, gross of
explicit trading costs/taxes. Labels are decimal discounted excess, converted to
total return using the frozen RF convention. October is incomplete and excluded.

## One command, progress and verification

From repository root, prefix with
`UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1`:

```sh
uv run --locked --no-sync python experiments/2026-10-gp-tuning-v1/run_study.py --launch --workers 14
uv run --locked --no-sync python experiments/2026-10-gp-tuning-v1/run_study.py --status
uv run --locked --no-sync python experiments/2026-10-gp-tuning-v1/campaign.py --verify
uv run --locked --no-sync python experiments/2026-10-gp-tuning-v1/run_study.py --verify
uv run --locked --no-sync pytest -q experiments/2026-10-gp-tuning-v1
uv run --locked --no-sync ruff check --no-respect-gitignore experiments/2026-10-gp-tuning-v1
```

The final --verify requires completion. While running, campaign.py --verify
checks protected inputs and all completed fit artifacts. Durable status/log are
outputs/workflow_status.json, outputs/status.json and outputs/campaign.log.
Resume the same finite workflow after repairing a documented implementation
failure; it never silently changes model settings or portfolio constraints.

Boundary/documentation/simplicity checks: this is isolated experimental I/O
orchestration, with pure recipe/window diagnostics, and reused sealed GP/allocator
helpers. No maintained package exports or production APIs change. Targeted tests,
Ruff and public-docstring checks validate additions; Pylance is not available in
this environment. Rollback is to continue using the untouched prior baseline.
