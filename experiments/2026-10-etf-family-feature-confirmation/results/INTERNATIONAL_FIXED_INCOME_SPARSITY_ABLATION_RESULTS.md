# International Fixed Income Refinement Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                                            |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:---------------------------------------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| INTFI-ALL-MACRO-NO-HY-CHANGE-NO-COPPER                   |            3 | 27,127,227 |  0.354167 | 0.036084 |           -2.601980 |           0.013482 | 0.870139 |     0.006510 |                            0.069877 |               0.015702 | True                | False                 |
| INTFI-ALL-MACRO-NO-HY-CHANGE-NO-ERP-NO-CPI-MOM-NO-COPPER |            3 | 27,127,227 |  0.388889 | 0.024056 |           -2.605973 |           0.009110 | 0.859886 |     0.011347 |                            0.069001 |               0.015244 | True                | False                 |
| INTFI-TREND+MARKET-RISK                                  |            3 | 27,127,227 |  0.159722 | 0.024056 |           -2.688399 |           0.008227 | 0.847268 |     0.009645 |                            0.067871 |               0.015314 | True                | True                  |
| INTFI-PRIMARY-NO-LAG-Y-EXCESS-LEAD                       |            3 | 27,127,227 |  0.402778 | 0.043368 |           -2.614493 |           0.011814 | 0.877914 |     0.007306 |                            0.070189 |               0.015585 | True                | True                  |
| INTFI-PRIMARY-NO-CHMOM                                   |            3 | 27,127,227 |  0.305556 | 0.031823 |           -2.605256 |           0.007824 | 0.864558 |     0.011295 |                            0.069476 |               0.015496 | True                | False                 |
| INTFI-PRIMARY-NO-MOM12M                                  |            3 | 27,127,227 |  0.319444 | 0.012028 |           -2.605187 |           0.008076 | 0.869476 |     0.006148 |                            0.069826 |               0.015913 | True                | False                 |
| INTFI-PRIMARY-NO-MOM36M                                  |            3 | 27,127,227 |  0.270833 | 0.020833 |           -2.604966 |           0.014553 | 0.850846 |     0.010472 |                            0.068239 |               0.016156 | True                | False                 |
| INTFI-PRIMARY-NO-CS-MOM-RANK                             |            3 | 27,127,227 |  0.256944 | 0.043368 |           -2.619222 |           0.013803 | 0.857385 |     0.004163 |                            0.068692 |               0.015505 | True                | True                  |
| INTFI-PRIMARY-NO-MA-SIGNAL                               |            3 | 27,127,227 |  0.347222 | 0.012028 |           -2.602811 |           0.010256 | 0.869289 |     0.005185 |                            0.069809 |               0.015776 | True                | False                 |
| INTFI-PRIMARY-NO-HY-SPREAD                               |            3 | 27,127,227 |  0.250000 | 0.020833 |           -2.621498 |           0.001264 | 0.850356 |     0.012496 |                            0.068214 |               0.015840 | True                | True                  |
| INTFI-PRIMARY-NO-HY-SPREAD-Z-12M                         |            3 | 27,127,227 |  0.104167 | 0.020833 |           -2.614670 |           0.012018 | 0.849718 |     0.017288 |                            0.068161 |               0.016326 | True                | False                 |
| INTFI-PRIMARY-NO-VIX-SLOPE                               |            3 | 27,127,227 |  0.319444 | 0.052429 |           -2.597533 |           0.009349 | 0.864738 |     0.014207 |                            0.069480 |               0.015692 | True                | False                 |
| INTFI-PRIMARY-NO-VIX-TS-Z-12M                            |            3 | 27,127,227 |  0.312500 | 0.036084 |           -2.606603 |           0.002896 | 0.860633 |     0.014214 |                            0.069152 |               0.015481 | True                | False                 |
| INTFI-PRIMARY-NO-VIX                                     |            3 | 27,127,227 |  0.333333 | 0.041667 |           -2.603788 |           0.006217 | 0.869467 |     0.008025 |                            0.069844 |               0.015618 | True                | False                 |
| INTFI-PRIMARY-NO-SPY-FLOW-Z-12M                          |            3 | 27,127,227 |  0.354167 | 0.041667 |           -2.611239 |           0.013310 | 0.862872 |     0.004289 |                            0.069171 |               0.015584 | True                | False                 |
| INTFI-PRIMARY-NO-SPY-RET                                 |            3 | 27,127,227 |  0.347222 | 0.031823 |           -2.557916 |           0.006454 | 0.869948 |     0.007970 |                            0.069904 |               0.016680 | True                | False                 |
| INTFI-PRIMARY-NO-ERP                                     |            3 | 27,127,227 |  0.361111 | 0.031823 |           -2.601858 |           0.007592 | 0.866266 |     0.018403 |                            0.069625 |               0.015460 | True                | False                 |
| INTFI-PRIMARY-NO-CPI-YOY                                 |            3 | 27,127,227 |  0.326389 | 0.066970 |           -2.589764 |           0.003920 | 0.859541 |     0.014771 |                            0.068990 |               0.015811 | True                | False                 |
| INTFI-PRIMARY-NO-CPI-MOM                                 |            3 | 27,127,227 |  0.347222 | 0.024056 |           -2.604219 |           0.011379 | 0.865959 |     0.004465 |                            0.069489 |               0.015487 | True                | False                 |
| INTFI-PRIMARY-NO-OIL-RET                                 |            3 | 27,127,227 |  0.312500 | 0.036084 |           -2.596089 |           0.005433 | 0.869289 |     0.005185 |                            0.069809 |               0.015846 | True                | False                 |
| INTFI-PRIMARY-NO-GOLD-CRUDE-RATIO                        |            3 | 27,127,227 |  0.319444 | 0.031823 |           -2.613124 |           0.002285 | 0.867345 |     0.006880 |                            0.069413 |               0.015427 | True                | False                 |
| INTFI-PRIMARY-NO-PCT-ABOVE-50DMA                         |            3 | 27,127,227 |  0.284722 | 0.060141 |           -2.602368 |           0.005981 | 0.858246 |     0.014184 |                            0.068870 |               0.015853 | True                | False                 |
| INTFI-PRIMARY-NO-EM-FX-RET                               |            3 | 27,127,227 |  0.291667 | 0.041667 |           -2.605904 |           0.014823 | 0.852922 |     0.010925 |                            0.068317 |               0.015560 | True                | False                 |

## Nondominated configurations

INTFI-TREND+MARKET-RISK, INTFI-PRIMARY-NO-LAG-Y-EXCESS-LEAD, INTFI-PRIMARY-NO-CS-MOM-RANK, INTFI-PRIMARY-NO-HY-SPREAD

## Hypervolume

- Observed dominated hypervolume: `0.935075361`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BNDX, VWOB, HYEM form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
