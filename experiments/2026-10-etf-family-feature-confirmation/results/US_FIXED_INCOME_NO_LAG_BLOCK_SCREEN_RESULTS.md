# Us Fixed Income Block Screen Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                     |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:----------------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| USFI-COMPACT-NO-CS+RET-AUTOCORR   |            3 | 27,127,227 |  0.211111 | 0.039382 |           -2.657938 |           0.013377 | 0.458606 |     0.033669 |                            0.037691 |               0.015742 | True                | True                  |
| USFI-COMPACT+VOL-Z                |            3 | 27,127,227 |  0.230556 | 0.050918 |           -2.632968 |           0.026280 | 0.427207 |     0.054102 |                            0.035214 |               0.015833 | True                | True                  |
| USFI-TREND                        |            3 | 27,127,227 |  0.147222 | 0.033679 |           -2.673565 |           0.007682 | 0.429514 |     0.024294 |                            0.034860 |               0.015503 | True                | True                  |
| USFI-COMPACT-NO-LAG-Y-EXCESS-LEAD |            3 | 27,127,227 |  0.130556 | 0.033679 |           -2.532124 |           0.006715 | 0.345077 |     0.009507 |                            0.028488 |               0.017833 | True                | False                 |
| USFI-NOLAG-CORE                   |            3 | 27,127,227 |  0.130556 | 0.033679 |           -2.532124 |           0.006715 | 0.345077 |     0.009507 |                            0.028488 |               0.017833 | True                | False                 |
| USFI-NOLAG-CORE-AUTOCORR          |            3 | 27,127,227 |  0.158333 | 0.036324 |           -2.605945 |           0.010554 | 0.390941 |     0.031008 |                            0.032179 |               0.016916 | True                | False                 |
| USFI-NOLAG-CORE-VOL               |            3 | 27,127,227 |  0.050000 | 0.080364 |           -2.584896 |           0.013162 | 0.291397 |     0.052632 |                            0.023820 |               0.017308 | True                | False                 |
| USFI-NOLAG-CORE-AUTOCORR-VOL      |            3 | 27,127,227 |  0.058333 | 0.082074 |           -2.637704 |           0.030896 | 0.377327 |     0.019169 |                            0.030435 |               0.016675 | True                | False                 |
| USFI-NOLAG-CORE-RISK-DYNAMICS     |            3 | 27,127,227 |  0.075000 | 0.008333 |           -2.634180 |           0.041291 | 0.411749 |     0.050460 |                            0.033345 |               0.016688 | True                | False                 |
| USFI-NOLAG-ALL-LOCAL              |            3 | 27,127,227 |  0.027778 | 0.017347 |           -2.601392 |           0.014489 | 0.308477 |     0.008837 |                            0.024961 |               0.017224 | True                | False                 |
| USFI-NOLAG-CORE-CREDIT-LEVEL-Z    |            3 | 27,127,227 |  0.194444 | 0.020972 |           -2.247133 |           0.037539 | 0.341417 |     0.060485 |                            0.028067 |               0.018324 | True                | False                 |
| USFI-NOLAG-CORE-CREDIT-ALL        |            3 | 27,127,227 |  0.119444 | 0.062546 |           -2.185450 |           0.061816 | 0.340450 |     0.099858 |                            0.028527 |               0.017704 | True                | False                 |
| USFI-NOLAG-CORE-INFLATION         |            3 | 27,127,227 | -0.041667 | 0.016667 |           -2.500042 |           0.029571 | 0.249452 |     0.004625 |                            0.020276 |               0.018154 | True                | False                 |
| USFI-NOLAG-CORE-MARKET-RISK       |            3 | 27,127,227 |  0.113889 | 0.020972 |           -2.511140 |           0.037449 | 0.426529 |     0.065396 |                            0.035342 |               0.018426 | True                | False                 |
| USFI-NOLAG-CORE-CREDIT-MARKET     |            3 | 27,127,227 |  0.213889 | 0.050231 |           -2.451636 |           0.019841 | 0.414761 |     0.065791 |                            0.034721 |               0.016650 | True                | False                 |
| USFI-NOLAG-CORE-GLOBAL            |            3 | 27,127,227 |  0.077778 | 0.009623 |           -2.579413 |           0.002610 | 0.347198 |     0.020205 |                            0.027381 |               0.015446 | True                | False                 |
| USFI-NOLAG-CORE-ALL-MACRO         |            3 | 27,127,227 |  0.069444 | 0.029266 |           -2.584970 |           0.018422 | 0.341656 |     0.057703 |                            0.027560 |               0.015941 | True                | False                 |
| USFI-NOLAG-ALL-MATCH              |            3 | 27,127,227 |  0.022222 | 0.084300 |           -2.561646 |           0.007955 | 0.256956 |     0.094748 |                            0.020817 |               0.016083 | True                | False                 |
| USFI-NOLAG-NO-CS-AUTOCORR-VOL     |            3 | 27,127,227 |  0.083333 | 0.052042 |           -2.596786 |           0.015481 | 0.350049 |     0.023276 |                            0.028629 |               0.017167 | True                | False                 |
| USFI-NOLAG-NO-MOM12-AUTOCORR-VOL  |            3 | 27,127,227 | -0.002778 | 0.045896 |           -2.630592 |           0.008101 | 0.341326 |     0.090688 |                            0.027880 |               0.016372 | True                | False                 |

## Nondominated configurations

USFI-COMPACT-NO-CS+RET-AUTOCORR, USFI-COMPACT+VOL-Z, USFI-TREND

## Hypervolume

- Observed dominated hypervolume: `0.280192567`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BND, IEF, LQD, HYG form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
