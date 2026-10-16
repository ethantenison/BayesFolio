# International Fixed Income Refinement Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                       |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:------------------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| INTFI-TREND+ALL-MACRO               |            3 | 27,127,227 |  0.243056 | 0.066970 |           -2.556656 |           0.004855 | 0.854791 |     0.012243 |                            0.068591 |               0.015009 | True                | False                 |
| INTFI-ALL-MATCH                     |            3 | 27,127,227 |  0.236111 | 0.086736 |           -2.560539 |           0.006161 | 0.861886 |     0.013547 |                            0.069174 |               0.014546 | True                | True                  |
| INTFI-TREND+MARKET-RISK             |            3 | 27,127,227 |  0.159722 | 0.024056 |           -2.688399 |           0.008227 | 0.847268 |     0.009645 |                            0.067871 |               0.015314 | True                | True                  |
| INTFI-TREND+GLOBAL                  |            3 | 27,127,227 |  0.229167 | 0.041667 |           -2.649867 |           0.016120 | 0.849766 |     0.005312 |                            0.068428 |               0.013867 | True                | True                  |
| INTFI-ALL-MACRO-NO-HY-SPREAD        |            3 | 27,127,227 |  0.222222 | 0.024056 |           -2.569461 |           0.008985 | 0.841804 |     0.008657 |                            0.067433 |               0.015242 | True                | False                 |
| INTFI-ALL-MACRO-NO-HY-SPREAD-CHG-1M |            3 | 27,127,227 |  0.277778 | 0.102768 |           -2.605438 |           0.009109 | 0.860532 |     0.015201 |                            0.069081 |               0.015558 | True                | True                  |
| INTFI-ALL-MACRO-NO-HY-SPREAD-Z-12M  |            3 | 27,127,227 |  0.166667 | 0.020833 |           -2.555152 |           0.010065 | 0.840762 |     0.009019 |                            0.067260 |               0.015597 | True                | False                 |
| INTFI-ALL-MACRO-NO-VIX-SLOPE        |            3 | 27,127,227 |  0.229167 | 0.020833 |           -2.562261 |           0.006901 | 0.854154 |     0.008775 |                            0.068523 |               0.014847 | True                | False                 |
| INTFI-ALL-MACRO-NO-VIX-TS-Z-12M     |            3 | 27,127,227 |  0.284722 | 0.012028 |           -2.547470 |           0.002668 | 0.849212 |     0.009893 |                            0.068105 |               0.014860 | True                | False                 |
| INTFI-ALL-MACRO-NO-VIX              |            3 | 27,127,227 |  0.256944 | 0.012028 |           -2.558244 |           0.008365 | 0.854155 |     0.011634 |                            0.068523 |               0.014865 | True                | False                 |
| INTFI-ALL-MACRO-NO-SPY-FLOW-Z-12M   |            3 | 27,127,227 |  0.263889 | 0.043368 |           -2.546203 |           0.003322 | 0.846152 |     0.012740 |                            0.067811 |               0.015046 | True                | False                 |
| INTFI-ALL-MACRO-NO-SPY-RET          |            3 | 27,127,227 |  0.284722 | 0.052429 |           -2.545635 |           0.008687 | 0.857216 |     0.013102 |                            0.068865 |               0.015009 | True                | False                 |
| INTFI-ALL-MACRO-NO-ERP              |            3 | 27,127,227 |  0.298611 | 0.060141 |           -2.547286 |           0.005306 | 0.856887 |     0.013118 |                            0.068821 |               0.014510 | True                | True                  |
| INTFI-ALL-MACRO-NO-CPI-YOY          |            3 | 27,127,227 |  0.180556 | 0.031823 |           -2.546540 |           0.003782 | 0.848074 |     0.006481 |                            0.067987 |               0.014948 | True                | False                 |
| INTFI-ALL-MACRO-NO-CPI-MOM          |            3 | 27,127,227 |  0.305556 | 0.052429 |           -2.558317 |           0.006393 | 0.852179 |     0.010280 |                            0.068355 |               0.014745 | True                | True                  |
| INTFI-ALL-MACRO-NO-COPPER-RET       |            3 | 27,127,227 |  0.284722 | 0.066970 |           -2.551269 |           0.013368 | 0.859017 |     0.011126 |                            0.068955 |               0.015208 | True                | True                  |
| INTFI-ALL-MACRO-NO-OIL-RET          |            3 | 27,127,227 |  0.194444 | 0.024056 |           -2.549407 |           0.005585 | 0.850795 |     0.010207 |                            0.068221 |               0.015075 | True                | False                 |
| INTFI-ALL-MACRO-NO-GOLD-CRUDE-RATIO |            3 | 27,127,227 |  0.243056 | 0.043368 |           -2.555910 |           0.011412 | 0.846693 |     0.011265 |                            0.067914 |               0.014790 | True                | False                 |
| INTFI-ALL-MACRO-NO-PCT-ABOVE-50DMA  |            3 | 27,127,227 |  0.250000 | 0.020833 |           -2.539083 |           0.008853 | 0.847638 |     0.011239 |                            0.068021 |               0.015285 | True                | False                 |
| INTFI-ALL-MACRO-NO-EM-FX-RET        |            3 | 27,127,227 |  0.229167 | 0.020833 |           -2.537581 |           0.008404 | 0.844577 |     0.003954 |                            0.067765 |               0.015007 | True                | False                 |
| INTFI-ALL-MACRO+BASPREAD            |            3 | 27,127,227 |  0.236111 | 0.024056 |           -2.555974 |           0.006952 | 0.854791 |     0.012243 |                            0.068591 |               0.014994 | True                | False                 |
| INTFI-ALL-MACRO+RET-KURT            |            3 | 27,127,227 |  0.236111 | 0.066970 |           -2.553129 |           0.002117 | 0.854791 |     0.012243 |                            0.068591 |               0.014852 | True                | False                 |
| INTFI-ALL-MACRO+MAX-DD-6M           |            3 | 27,127,227 |  0.236111 | 0.073164 |           -2.560226 |           0.012461 | 0.855937 |     0.010270 |                            0.068702 |               0.014941 | True                | False                 |
| INTFI-ALL-MACRO+RET-AUTOCORR        |            3 | 27,127,227 |  0.229167 | 0.020833 |           -2.544600 |           0.006652 | 0.853322 |     0.008244 |                            0.068277 |               0.014908 | True                | False                 |
| INTFI-ALL-MACRO+VOL-Z               |            3 | 27,127,227 |  0.236111 | 0.073164 |           -2.553929 |           0.014249 | 0.850774 |     0.015393 |                            0.068245 |               0.014924 | True                | False                 |

## Nondominated configurations

INTFI-ALL-MATCH, INTFI-TREND+MARKET-RISK, INTFI-TREND+GLOBAL, INTFI-ALL-MACRO-NO-HY-SPREAD-CHG-1M, INTFI-ALL-MACRO-NO-ERP, INTFI-ALL-MACRO-NO-CPI-MOM, INTFI-ALL-MACRO-NO-COPPER-RET

## Hypervolume

- Observed dominated hypervolume: `0.698327590`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BNDX, VWOB, HYEM form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
