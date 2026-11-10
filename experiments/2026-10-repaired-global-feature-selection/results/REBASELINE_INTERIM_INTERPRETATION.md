# Repaired-data rebaseline: interim interpretation

This note interprets completed three-seed evidence, not unfinished runs. It is
not the final family selection or a production-promotion recommendation. The
additional historical masks and adaptive rounds remain required work.

## Combined 15-task all-input control

Verified seeds 27, 127, 227, 24 common scored months and 84 initial training months:

| Metric | Three-seed mean |
| --- | ---: |
| Pooled cross-sectional IC | 0.108780 |
| Gaussian NLL | -2.141413 |
| Conventional annualized excess-return Sharpe | 1.127881 |

The within-family ICs from these same **joint-model** predictions differ:

| Family | Within-family IC | NLL | Empirical 95% coverage |
| --- | ---: | ---: | ---: |
| U.S. equity | 0.172222 | -1.621425 | 0.899306 |
| International equity | 0.041667 | -1.700185 | 0.881944 |
| U.S. fixed income | -0.097222 | -2.668109 | 0.927083 |
| International fixed income | 0.222222 | -2.720776 | 0.935185 |

Implication: pooled ranking success does not establish within-family ranking
success. The global search must preserve these separate breakdowns. Coverage
below the nominal level also prevents interpreting narrower intervals alone as
better predictive uncertainty. There is no feature-importance inference from
this single all-input configuration.

## First international-equity rebaseline batch

All 25 initially registered masks have complete three-seed evidence. Seven are
nondominated in this provisional set. Exact inputs and seed metrics are in
`cumulative/international_equity/`.

- The 27-input compute-matched all-input comparator has IC 0.019444, NLL
  -1.740618, and Sharpe 1.171160. Multiple compact candidates dominate it on the
  three aggregate objectives; this does not settle the full remaining search.
- Adding oil return to the corrected breadth/flow/ERP context improves NLL and
  Sharpe but slightly reduces aggregate IC. Oil is useful evidence, not a
  universal three-objective improvement in every context.
- Adding the gold/crude ratio is harmful on all three objectives in several
  matched contexts, with all three seeds agreeing in those tested pairs.
  This is conditional evidence, not proof against that input in every family.
- `spy_ret` and `rsp_spy_roc_1p` have several favorable exact paired comparisons.
  Removing them indiscriminately for sparsity is not supported by this batch.
- `erp` and `spy_flow_z_12p` show objective tradeoffs that vary with the other
  retained signals. Do not carry forward the previous short-history decision to
  drop both as a blanket rule for the repaired 84-month policy.
- DXY and the breakeven proxy hurt in the particular tested equity contexts.
  These results do not establish their value for fixed income.
- The simple local with-lag/no-lag pair trades better NLL for worse IC when lag
  is included. That is not enough to require or prohibit lag globally.

## Evidence and next decisions

The July scope reconciliation adds valid historical masks rather than silently
restricting the study to October's final candidates. The expanded v2 feature pool
adds 11 validated historical predictors, with **every shared v1 cell bitwise
unchanged**. New v2 runs retain their own artifact hashes; v1 runs are never
relabelled as v2 resumes. Liquidity blocks that require unsupported clipping or
turnover handling remain explicitly quarantined.

Finish the registered history-based masks before planning the international
adaptive round. Use the complete new front and exact paired effects to select
exploitation, redundancy substitutions, and genuinely new exploration. Keep
fixed-income conclusions pending until those families have their own complete
three-seed evidence. Do not infer the shared feature core from one family's front.

Sources of experiment evidence:

- `cumulative/global/objectives_by_seed.csv`
- `cumulative/global/family_diagnostics_by_seed.csv`
- `cumulative/international_equity/configuration_aggregates.csv`
- `cumulative/international_equity/conditional_feature_pairs.csv`
- `historical_scope_audit.json`
- `historical_pool_scope_audit.json`
- `feature_pool_extension_audit.json`
