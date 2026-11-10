# Feature-definition redundancy audit

| identity                          |   rows_checked |   max_absolute_residual | passed   |
|:----------------------------------|---------------:|------------------------:|:---------|
| tnote10y = y10_nominal            |           1620 |                       0 | True     |
| chmom = mom12m - mom6m            |           1620 |                       0 | True     |
| term_spread = tnote10y - tbill3m  |           1620 |                       0 | True     |
| erp = spy_ret - tbill3m           |           1620 |                       0 | True     |
| vix_slope = log(vix3m) - log(vix) |           1620 |                       0 | True     |
| gold_crude_ratio = gold / oil     |           1620 |                       0 | True     |

Monthly `_1m`/`_1p` and `_12m`/`_12p` aliases are canonicalized before mask deduplication.
The two Treasury level names were not silently canonicalized out of historical masks; their
removal is a useful explicit paired experiment under the fixed kernel/fitting budget.

Exact definitions are not ablation outcomes. Fit paired removals before claiming unchanged IC/NLL/Sharpe; duplicate axes alter GP geometry and optimization at the fixed budget.

Priority hypotheses:

- Keep only one of the identical Treasury-level columns in broad rate-aware candidates.
- Test chmom versus mom6m/mom12m bases without assuming a nonlinear GP is invariant.
- Test ERP/bill/market-return redundancy where multiple components are present.
- Treat ratio-versus-components tests as representation changes, not independent economic data.
