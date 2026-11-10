# Track 2: asymmetric compact/broad GP settings

## Question and invariant controls

Does changing each component's rank and fitting budget improve the existing
two-GP ensemble, and does retuning one broadly applicable blending recipe add
anything beyond that component change?

All cases retain the same 15 ETFs, repaired/extended dataset, three seeds,
28 forecast origins, original stable window indices, `t_index`, target scaling,
kernel, chronological training rule, and observed predictive covariance.
The broad component has 14 non-time features and the compact component has five.
The initial training history is 84 months and expands to 111; it is not a fixed
86-month window. Component fits are exclusively owned by the shared parent queue.

Each component spans ranks **1, 2, 5** and maxiter **5, 15, 36, 75**. The original
rank-2/maxiter-15 components are reused, not refitted. Every proposed ensemble
uses one fixed four-family mean-weight vector and one fixed covariance fraction
for all seeds and months. There is no monthly blend selection or adaptation.

## Staged comparison

1. **Fixed-blend screen: 144 pairs.** Cross all 12 broad settings with all 12
   compact settings. Compact mean fractions stay `(0.5, 0, 0.375, 0.25)` for US
   equities, international equities, US fixed income, international fixed income.
   Covariance stays `0.5 * broad + 0.5 * compact`. This isolates GP-setting effects.
2. **Select up to three component pairs using first-16-month metrics only.** Keep
   the original pair as an explicit control, even if dominated. Fill other slots
   from the three-objective Pareto frontier using normalized ideal proximity and
   farthest-first coverage. This bounded selection is not an assertion of global
   optimality. Never use last-12-month or full-window outcomes for this admission.
3. **Static mean tuning: 97 recipes per admitted pair.** Evaluate the 81 family
   combinations of fractions `{0, 0.5, 1}`, both approved recipes, and the unique
   single-family perturbations of the approved leader by `+/-1/16` and `+/-1/8`.
   Covariance remains 50/50. This menu contains pure broad, pure compact, uniform
   50/50, and family-dependent mixes; it is a bounded coarse/local search, not a
   claim to exhaust all continuous mixtures. The incumbent duplicate from the
   screen can be reused rather than allocated again.
4. **Isolated covariance tuning: at most 12 extra recipes per admitted pair.**
   Select up to three mean finalists using first-16-month metrics only. For each,
   hold its means fixed and test compact covariance fractions 0, 0.25, 0.75, and 1,
   alongside the already measured 0.5. This tests covariance without conflating
   it with a mean-blend change. Convex arithmetic pooling preserves positive
   semidefiniteness; no independence assumption, disagreement variance term,
   covariance repair, or uncertainty recalibration is silently introduced.

Maximum unique hard-rank recipes: **144 + 3 * (96 + 12) = 468**. The mean-grid
incumbent is an existing screen case, so only 96 of each pair's 97 means are new.
The corresponding uncached allocation ceiling is **39,312 monthly solves** over
three seeds and 28 origins; it introduces **zero additional GP fits** beyond the
shared component registry. Cache equivalent recipes by numeric recipe plus
component IDs, not only by the stage-specific ID. Screening infeasible/failed
cases must be reported as incomplete rather than selectively averaged.

The hard-rank arm is the primary allocator in this track. The parent may use
these fixed recipes in matched scenario diagnostics; do not multiply the full
468-case menu by the expensive scenario optimizer without an explicit bounded
cross-track plan. Hard rank retains NE >=7, no ETF cap, its historical-EW variance
ceiling, and no fallback or constraint relaxation. The separate scenario arm
retains NE >=10, cap 25%, maximum signed Sharpe with MV risk and no CVaR. Results
across those arms compare complete strategies with differing constraints.

## Selection, evaluation, and claims

- All candidate admission and recipe tuning use the **first 16 origins**:
  May 2024-August 2025, realized returns June 2024-September 2025.
- Freeze those case IDs and static weights before evaluating the **last 12
  origins**: September 2025-August 2026, realized returns October 2025-September
  2026. Report every admitted candidate, not just the temporal-validation winner.
- Report first 16, last 12, and all 28 separately, for each seed and their mean.
  The objectives are conventional Sharpe, compounded total return (decimal), and
  average monthly cross-sectional ranking IC; NLL is diagnostic, not a substitute
  objective or promotion guardrail. Rank correlations are computed within month
  across the 15 ETFs, not pooled over the entire time panel.
- Also report drawdown, volatility, seed consistency, numerical feasibility,
  covariance/risk diagnostics, actual optimizer iterations/status, and comparison
  with the frozen current blended hard-rank strategy, Schwab aggressive proxy,
  and standalone top-seven momentum benchmark. Optimizer maxiter is a budget,
  not proof that every fit ran that many iterations or reached convergence.
- All 28 months have previously been inspected, so the temporal split is a
  disciplined **retrospective temporal validation**, not a pristine holdout.
  Full-window Pareto leaders may be shown descriptively but must not be presented
  as selected-before-validation or as unbiased future performance estimates.
- Distinguish the best fixed-incumbent-blend setting from the retuned-blend
  setting: attribute improvements to GP settings, mean recipes, and covariance
  recipes separately. Report if an apparent gain depends on one seed/subperiod.

## Files and boundaries

`recipe_math.py` is a pure experiment helper. It enumerates static case recipes,
applies blends, and constructs deterministic Pareto shortlists from complete
metrics. It performs no fits, portfolio allocation, network access, I/O, or
production-package modifications. `plan.json` is the machine-readable budget and
selection contract. `run_track2.py` exposes `run_track(output_root, workers=14)`
for the parent to call after the shared fit queue completes. It uses a single
bounded allocation pool, reuses numerically identical recipes across stages,
requires all 84 allocations for eligibility, and writes immutable stage-selection
evidence plus `outputs/track2/{RESULTS.md,summary.json,tables.csv,COMPLETED.json}`.
It calls the parent's shared `case_evaluator.evaluate_case`, never a GP fit.
No work starts on import, and no standalone CLI launches an independent pool.
The parent campaign owns shared fitting and cross-track conclusions; this folder
cannot independently launch competing compute pools.

## Verification and reversibility

Run from repository root:

```sh
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync pytest -q experiments/2026-10-gp-tuning-v1/track2_asymmetric
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache uv run --locked --no-sync ruff check --no-respect-gitignore experiments/2026-10-gp-tuning-v1/track2_asymmetric
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache uv run --locked --no-sync python scripts/check_new_public_docstrings.py --files experiments/2026-10-gp-tuning-v1/track2_asymmetric/recipe_math.py experiments/2026-10-gp-tuning-v1/track2_asymmetric/run_track2.py experiments/2026-10-gp-tuning-v1/track2_asymmetric/test_recipe_math.py experiments/2026-10-gp-tuning-v1/track2_asymmetric/test_run_track2.py
```

No shared code or sealed evidence is modified. Rollback consists of ceasing use
of this new experiment-only folder; retain its artifacts as a recoverable record.
