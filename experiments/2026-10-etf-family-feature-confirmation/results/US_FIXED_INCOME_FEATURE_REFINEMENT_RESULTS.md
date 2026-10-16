# Us Fixed Income Refinement Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                     |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:----------------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| USFI-TREND-NO-MOM36               |            3 | 27,127,227 |  0.205556 | 0.055486 |           -2.619566 |           0.048283 | 0.443697 |     0.013657 |                            0.036133 |               0.015978 | True                | True                  |
| USFI-TREND                        |            3 | 27,127,227 |  0.147222 | 0.033679 |           -2.673565 |           0.007682 | 0.429514 |     0.024294 |                            0.034860 |               0.015503 | True                | True                  |
| USFI-COMPACT-NO-LAG-Y-EXCESS-LEAD |            3 | 27,127,227 |  0.130556 | 0.033679 |           -2.532124 |           0.006715 | 0.345077 |     0.009507 |                            0.028488 |               0.017833 | True                | False                 |
| USFI-COMPACT-NO-CHMOM             |            3 | 27,127,227 |  0.177778 | 0.093665 |           -2.656025 |           0.023889 | 0.430185 |     0.024092 |                            0.034977 |               0.015834 | True                | True                  |
| USFI-COMPACT-NO-MOM12M            |            3 | 27,127,227 |  0.216667 | 0.066144 |           -2.658920 |           0.007016 | 0.429309 |     0.046250 |                            0.035102 |               0.015927 | True                | True                  |
| USFI-COMPACT-NO-CS-MOM-RANK       |            3 | 27,127,227 |  0.175000 | 0.036324 |           -2.671665 |           0.003918 | 0.454434 |     0.034500 |                            0.037021 |               0.015595 | True                | True                  |
| USFI-COMPACT-NO-MA-SIGNAL         |            3 | 27,127,227 |  0.150000 | 0.065085 |           -2.614660 |           0.078809 | 0.424566 |     0.027414 |                            0.034477 |               0.016130 | True                | False                 |
| USFI-COMPACT+BASPREAD             |            3 | 27,127,227 |  0.144444 | 0.076980 |           -2.648058 |           0.020783 | 0.399615 |     0.039443 |                            0.032504 |               0.016012 | True                | False                 |
| USFI-COMPACT+RET-KURT             |            3 | 27,127,227 |  0.133333 | 0.101379 |           -2.623385 |           0.024803 | 0.402665 |     0.051334 |                            0.032734 |               0.016326 | True                | False                 |
| USFI-COMPACT+MAX-DD-6M            |            3 | 27,127,227 |  0.175000 | 0.072648 |           -2.625202 |           0.049881 | 0.411668 |     0.062353 |                            0.033570 |               0.015921 | True                | False                 |
| USFI-COMPACT+RET-AUTOCORR         |            3 | 27,127,227 |  0.213889 | 0.067871 |           -2.627451 |           0.046687 | 0.429903 |     0.010898 |                            0.035265 |               0.015871 | True                | True                  |
| USFI-COMPACT+VOL-Z                |            3 | 27,127,227 |  0.230556 | 0.050918 |           -2.632968 |           0.026280 | 0.427207 |     0.054102 |                            0.035214 |               0.015833 | True                | True                  |

## Nondominated configurations

USFI-TREND-NO-MOM36, USFI-TREND, USFI-COMPACT-NO-CHMOM, USFI-COMPACT-NO-MOM12M, USFI-COMPACT-NO-CS-MOM-RANK, USFI-COMPACT+RET-AUTOCORR, USFI-COMPACT+VOL-Z

## Hypervolume

- Observed dominated hypervolume: `0.276936663`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BND, IEF, LQD, HYG form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
