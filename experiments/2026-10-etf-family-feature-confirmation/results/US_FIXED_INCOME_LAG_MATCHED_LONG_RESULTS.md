# Us Fixed Income Refinement Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                              |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:-------------------------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| USFI-MATCH-LONG-TREND-NO_LAG               |            3 | 27,127,227 |  0.183333 | 0.054645 |           -2.529196 |           0.016369 | 0.395310 |     0.056405 |                            0.032608 |               0.018035 | True                | False                 |
| USFI-MATCH-LONG-TREND-WITH_LAG             |            3 | 27,127,227 |  0.147222 | 0.033679 |           -2.673565 |           0.007682 | 0.429514 |     0.024294 |                            0.034860 |               0.015503 | True                | True                  |
| USFI-MATCH-LONG-NO_MOM12-NO_LAG            |            3 | 27,127,227 |  0.161111 | 0.055486 |           -2.529273 |           0.015818 | 0.399378 |     0.015578 |                            0.032863 |               0.017972 | True                | False                 |
| USFI-MATCH-LONG-NO_MOM12-WITH_LAG          |            3 | 27,127,227 |  0.216667 | 0.066144 |           -2.658920 |           0.007016 | 0.429309 |     0.046250 |                            0.035102 |               0.015927 | True                | True                  |
| USFI-MATCH-LONG-NO_CS-NO_LAG               |            3 | 27,127,227 |  0.141667 | 0.058333 |           -2.551570 |           0.007696 | 0.378138 |     0.023452 |                            0.030956 |               0.017920 | True                | False                 |
| USFI-MATCH-LONG-NO_CS-WITH_LAG             |            3 | 27,127,227 |  0.175000 | 0.036324 |           -2.671665 |           0.003918 | 0.454434 |     0.034500 |                            0.037021 |               0.015595 | True                | True                  |
| USFI-MATCH-LONG-AUTOCORR-NO_LAG            |            3 | 27,127,227 |  0.158333 | 0.036324 |           -2.605945 |           0.010554 | 0.390941 |     0.031008 |                            0.032179 |               0.016916 | True                | False                 |
| USFI-MATCH-LONG-AUTOCORR-WITH_LAG          |            3 | 27,127,227 |  0.213889 | 0.067871 |           -2.627451 |           0.046687 | 0.429903 |     0.010898 |                            0.035265 |               0.015871 | True                | True                  |
| USFI-MATCH-LONG-VOL-NO_LAG                 |            3 | 27,127,227 |  0.050000 | 0.080364 |           -2.584896 |           0.013162 | 0.291397 |     0.052632 |                            0.023820 |               0.017308 | True                | False                 |
| USFI-MATCH-LONG-VOL-WITH_LAG               |            3 | 27,127,227 |  0.230556 | 0.050918 |           -2.632968 |           0.026280 | 0.427207 |     0.054102 |                            0.035214 |               0.015833 | True                | True                  |
| USFI-MATCH-LONG-NO_MOM12_AUTOCORR-NO_LAG   |            3 | 27,127,227 |  0.127778 | 0.091793 |           -2.619546 |           0.031253 | 0.361030 |     0.021790 |                            0.029389 |               0.016791 | True                | False                 |
| USFI-MATCH-LONG-NO_MOM12_AUTOCORR-WITH_LAG |            3 | 27,127,227 |  0.222222 | 0.118537 |           -2.653782 |           0.024571 | 0.416432 |     0.025311 |                            0.033936 |               0.015779 | True                | True                  |
| USFI-MATCH-LONG-BALANCED-NO_LAG            |            3 | 27,127,227 |  0.183333 | 0.014434 |           -2.573003 |           0.013538 | 0.395273 |     0.027802 |                            0.032978 |               0.017482 | True                | False                 |
| USFI-MATCH-LONG-BALANCED-WITH_LAG          |            3 | 27,127,227 |  0.211111 | 0.039382 |           -2.657938 |           0.013377 | 0.458606 |     0.033669 |                            0.037691 |               0.015742 | True                | True                  |
| USFI-MATCH-LONG-MINIMAL-NO_LAG             |            3 | 27,127,227 |  0.113889 | 0.045896 |           -2.554328 |           0.031196 | 0.381168 |     0.024284 |                            0.031300 |               0.017754 | True                | False                 |
| USFI-MATCH-LONG-MINIMAL-WITH_LAG           |            3 | 27,127,227 |  0.191667 | 0.066667 |           -2.668034 |           0.023751 | 0.422049 |     0.040469 |                            0.034851 |               0.015789 | True                | True                  |

## Nondominated configurations

USFI-MATCH-LONG-TREND-WITH_LAG, USFI-MATCH-LONG-NO_MOM12-WITH_LAG, USFI-MATCH-LONG-NO_CS-WITH_LAG, USFI-MATCH-LONG-AUTOCORR-WITH_LAG, USFI-MATCH-LONG-VOL-WITH_LAG, USFI-MATCH-LONG-NO_MOM12_AUTOCORR-WITH_LAG, USFI-MATCH-LONG-BALANCED-WITH_LAG, USFI-MATCH-LONG-MINIMAL-WITH_LAG

## Hypervolume

- Observed dominated hypervolume: `0.280640676`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BND, IEF, LQD, HYG form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
