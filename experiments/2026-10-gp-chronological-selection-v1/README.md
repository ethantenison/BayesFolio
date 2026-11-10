# Chronological GP feature, setting, and ensemble selection

Approved October 25, 2026. Isolated research; no production changes, trading,
git mutations, forest stacker, new data downloads, or automatic promotion.

## Questions and fixed controls

1. Does a validated July-style richer representation add information beyond
   the previously selected broad14/compact5 inputs?
2. Do richer, broad, compact, and shorter-window experts behave differently
   under matched forecasting and allocation controls?
3. Can an earlier-data-only selection procedure choose features, GP settings,
   and fixed blend recipes that work in subsequent chronological blocks?

Use the frozen repaired/extended dataset, the same 15 ETFs (IWM, not IJR;
no REIT/commodity targets), signed-LKJ GP construction, and three seeds.
Keep t_index. Exclude raw hy_spread and hy_spread_chg_1p; retain the existing
eligible z-score. Richer means 25 economic predictors plus t_index, not a
recreation of July's different universe, positive task kernel, or time kernel.
No full-panel clipping, feature imputation, covariance repair or target changes.

The bounded catalog contains 19 experts:

- broad14 and compact5 at rank 1/2 and maxiter 5/15 (eight experts);
- broad14 rolling48 at rank2/maxiter15;
- eligible July-style rich inputs: expanding r2/i15, r1/i15, r2/i5, and
  rolling48 r2/i15;
- six matched r2/i15 rich ablations: risk/liquidity, 36-month momentum,
  breadth/flow, macro regime, commodities/FX, and lagged target removal.

All catalog definitions are hypotheses, not claims of optimal inputs. The
ablation comparisons isolate inputs at fixed settings. Hyperparameter coverage
is deliberately bounded, not exhaustive.

## Earlier forecasts and temporal selection

Generate 52 origins, May2022--Aug2026. The first expanding fit uses 60 months
(May2017--Apr2022), increasing to111; rolling experts always use48. The extra
24 earlier forecasts provide meta-training outcomes without using GP in-sample
predictions. At May2024, expanding history is exactly the prior84 months.
Keep original May2024-based global seed indexes, including negative indexes for
the earlier origins, so all reusable later fits remain exactly matched.

Reuse nine exact 28-origin component panels. Planned new fits: **2,208**;
plus two independent cache-parity audit fits. A maximum of14 single-threaded
workers share one queue. Two attempts per failed job, identical settings;
at most2,500 attempts including audits. Persist every failure and resume only
when code, data, settings and all completed artifact hashes match.

Selection never sees the next block's returns:

- Initial selection: 24 earlier origins, May2022--Apr2024.
- Four successive test blocks: seven origins each, May2024--Aug2026.
- For each block, select only on the strictly earlier forecast/outcome ledger.
- Feature masks and GP settings are selected through expert inclusion; search
  static nonnegative family weights, at most three experts per family.
- Fixed deterministic coarse recipes plus bounded beam refinements jointly
  tune all included weights. Save Sharpe, ranking, and equal-objective-rank
  compromise recipes before test scoring. No monthly weight optimization.
- Each recipe is frozen across the whole seven-month block and all seeds.
- Also carry the *initial* recipes unchanged through all28 later months. These
  fully static controls distinguish retraining from simply finding a good blend.
- Record first/second-half inner-period metrics and single-expert results;
  do not use test metrics to revise the catalog or selection algorithm.

This is a **retrospective nested chronological selection simulation**, not a
pristine holdout. The candidate library and research hypotheses were informed
by previously inspected history; existing compact/broad masks inherit that
selection. Latest-vintage macro data and the existing target convention retain
their documented limitations. Only future, genuinely unseen data can provide
fresh confirmation. Seeds are numerical repetitions, not independent markets.

## Common allocator and reporting

Keep the original hard-rank allocator: long-only, fully invested, NE>=7, no
position cap, and the unchanged causal historical-EW variance ceiling. All
methods use the same fixed50/50 broad-r2-i15/compact-r2-i15 covariance pool.
This isolates predictive blending and input selection, not covariance tuning.
No CVaR, previous-month fallback, positive-return floor, turnover objective,
uncertainty recalibration, or changing the risk ceiling. Negative returns remain
valid observations. Portfolio computation follows completion of monthly fits.

Compare conventional Sharpe, compounded total return, monthly cross-sectional
rank IC, seed/subperiod stability and drawdown. Retain RMSE, joint/marginal NLL,
portfolio-risk and fixed-reference distribution diagnostics. NLL is not a
standalone selection objective. Gross of explicit costs/taxes, as before.
Standing benchmarks: Schwab aggressive proxy and standalone top7 momentum.
Report selected input lists/settings/family blend weights at every cutoff,
and matched feature ablations without calling selection frequency importance.

## Reproduction and completion

From the repository root, prefix commands with
`UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1`:

```sh
uv run --locked --no-sync python experiments/2026-10-gp-chronological-selection-v1/run_study.py --prepare
uv run --locked --no-sync pytest -q experiments/2026-10-gp-chronological-selection-v1
uv run --locked --no-sync python experiments/2026-10-gp-chronological-selection-v1/run_study.py --pilot --workers 14
uv run --locked --no-sync python experiments/2026-10-gp-chronological-selection-v1/run_study.py --execute --workers 14
uv run --locked --no-sync python experiments/2026-10-gp-chronological-selection-v1/run_study.py --status
uv run --locked --no-sync python experiments/2026-10-gp-chronological-selection-v1/run_study.py --verify
```

Completion requires passed outputs/COMPLETED.json, complete fits, all frozen
cutoffs, full portfolio paths, RESULTS.md, COMPARISON.png, and unchanged protected
inputs. FITS_COMPLETED.json alone does not mean the experiment is complete.
Source edits are confined here; maintained package boundaries/exports do not
change. Rollback: continue using the untouched current baseline.
