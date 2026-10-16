# International Fixed Income Refinement Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                                            |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:---------------------------------------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| INTFI-ALL-MATCH                                          |            3 | 27,127,227 |  0.236111 | 0.086736 |           -2.560539 |           0.006161 | 0.861886 |     0.013547 |                            0.069174 |               0.014546 | True                | False                 |
| INTFI-TREND+MARKET-RISK                                  |            3 | 27,127,227 |  0.159722 | 0.024056 |           -2.688399 |           0.008227 | 0.847268 |     0.009645 |                            0.067871 |               0.015314 | True                | True                  |
| INTFI-TREND+GLOBAL                                       |            3 | 27,127,227 |  0.229167 | 0.041667 |           -2.649867 |           0.016120 | 0.849766 |     0.005312 |                            0.068428 |               0.013867 | True                | True                  |
| INTFI-ALL-MACRO-NO-HY-SPREAD-CHG-1M                      |            3 | 27,127,227 |  0.277778 | 0.102768 |           -2.605438 |           0.009109 | 0.860532 |     0.015201 |                            0.069081 |               0.015558 | True                | True                  |
| INTFI-ALL-MACRO-NO-ERP                                   |            3 | 27,127,227 |  0.298611 | 0.060141 |           -2.547286 |           0.005306 | 0.856887 |     0.013118 |                            0.068821 |               0.014510 | True                | False                 |
| INTFI-ALL-MACRO-NO-CPI-MOM                               |            3 | 27,127,227 |  0.305556 | 0.052429 |           -2.558317 |           0.006393 | 0.852179 |     0.010280 |                            0.068355 |               0.014745 | True                | False                 |
| INTFI-ALL-MACRO-NO-COPPER-RET                            |            3 | 27,127,227 |  0.284722 | 0.066970 |           -2.551269 |           0.013368 | 0.859017 |     0.011126 |                            0.068955 |               0.015208 | True                | False                 |
| INTFI-ALL-MACRO-NO-HY-CHANGE-NO-ERP                      |            3 | 27,127,227 |  0.291667 | 0.055120 |           -2.583651 |           0.001750 | 0.857117 |     0.004063 |                            0.068701 |               0.015347 | True                | False                 |
| INTFI-ALL-MACRO-NO-HY-CHANGE-NO-CPI-MOM                  |            3 | 27,127,227 |  0.277778 | 0.024056 |           -2.602178 |           0.002565 | 0.862439 |     0.006431 |                            0.069168 |               0.015353 | True                | True                  |
| INTFI-ALL-MACRO-NO-HY-CHANGE-NO-COPPER                   |            3 | 27,127,227 |  0.354167 | 0.036084 |           -2.601980 |           0.013482 | 0.870139 |     0.006510 |                            0.069877 |               0.015702 | True                | True                  |
| INTFI-ALL-MACRO-NO-ERP-NO-CPI-MOM                        |            3 | 27,127,227 |  0.250000 | 0.055120 |           -2.547223 |           0.002609 | 0.849497 |     0.013003 |                            0.068082 |               0.014355 | True                | False                 |
| INTFI-ALL-MACRO-NO-ERP-NO-COPPER                         |            3 | 27,127,227 |  0.340278 | 0.063647 |           -2.546838 |           0.017841 | 0.862474 |     0.014658 |                            0.069301 |               0.014869 | True                | False                 |
| INTFI-ALL-MACRO-NO-CPI-MOM-NO-COPPER                     |            3 | 27,127,227 |  0.250000 | 0.020833 |           -2.549018 |           0.011340 | 0.860518 |     0.015323 |                            0.069131 |               0.015095 | True                | False                 |
| INTFI-ALL-MACRO-NO-HY-CHANGE-NO-ERP-NO-CPI-MOM           |            3 | 27,127,227 |  0.256944 | 0.063647 |           -2.583112 |           0.014127 | 0.855504 |     0.001481 |                            0.068600 |               0.015114 | True                | False                 |
| INTFI-ALL-MACRO-NO-HY-CHANGE-NO-ERP-NO-CPI-MOM-NO-COPPER |            3 | 27,127,227 |  0.388889 | 0.024056 |           -2.605973 |           0.009110 | 0.859886 |     0.011347 |                            0.069001 |               0.015244 | True                | True                  |
| INTFI-ALL-MATCH-NO-HY-SPREAD-CHG-1M                      |            3 | 27,127,227 |  0.250000 | 0.041667 |           -2.600634 |           0.001331 | 0.861874 |     0.006119 |                            0.069101 |               0.015172 | True                | False                 |
| INTFI-ALL-MATCH-NO-ERP                                   |            3 | 27,127,227 |  0.256944 | 0.084197 |           -2.544062 |           0.009843 | 0.864159 |     0.003147 |                            0.069345 |               0.014192 | True                | False                 |
| INTFI-ALL-MATCH-NO-CPI-MOM                               |            3 | 27,127,227 |  0.243056 | 0.043368 |           -2.557517 |           0.009941 | 0.859202 |     0.010825 |                            0.068960 |               0.014477 | True                | False                 |
| INTFI-ALL-MATCH-NO-COPPER-RET                            |            3 | 27,127,227 |  0.256944 | 0.012028 |           -2.551269 |           0.015141 | 0.866015 |     0.000998 |                            0.069490 |               0.014767 | True                | False                 |

## Nondominated configurations

INTFI-TREND+MARKET-RISK, INTFI-TREND+GLOBAL, INTFI-ALL-MACRO-NO-HY-SPREAD-CHG-1M, INTFI-ALL-MACRO-NO-HY-CHANGE-NO-CPI-MOM, INTFI-ALL-MACRO-NO-HY-CHANGE-NO-COPPER, INTFI-ALL-MACRO-NO-HY-CHANGE-NO-ERP-NO-CPI-MOM-NO-COPPER

## Hypervolume

- Observed dominated hypervolume: `0.894648828`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BNDX, VWOB, HYEM form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
