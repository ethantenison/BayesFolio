# Us Fixed Income Block Screen Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                           |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:----------------------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| USFI-RATE-BALANCED-NO-LAG               |            3 | 27,127,227 |  0.108333 | 0.058333 |           -2.663281 |           0.015354 | 0.432755 |     0.012067 |                            0.034662 |               0.015804 | True                | True                  |
| USFI-RATE-BALANCED-WITH-LAG             |            3 | 27,127,227 |  0.113889 | 0.037577 |           -2.611770 |           0.032028 | 0.365429 |     0.020643 |                            0.029405 |               0.015815 | True                | False                 |
| USFI-RATE-HIGH-IC-NO-LAG                |            3 | 27,127,227 |  0.102778 | 0.041107 |           -2.647049 |           0.019696 | 0.417684 |     0.052209 |                            0.033463 |               0.015767 | True                | False                 |
| USFI-RATE-HIGH-IC-WITH-LAG              |            3 | 27,127,227 |  0.130556 | 0.033679 |           -2.639765 |           0.020238 | 0.380926 |     0.051548 |                            0.030807 |               0.015576 | True                | True                  |
| USFI-RATE-TREASURY-CURVE-NO-LAG         |            3 | 27,127,227 |  0.116667 | 0.087797 |           -2.424715 |           0.013954 | 0.415580 |     0.020565 |                            0.033846 |               0.019597 | True                | False                 |
| USFI-RATE-TREASURY-CURVE-WITH-LAG       |            3 | 27,127,227 |  0.022222 | 0.026788 |           -2.479251 |           0.034626 | 0.386993 |     0.016685 |                            0.031541 |               0.018570 | True                | False                 |
| USFI-RATE-DURATION-CURVE-NO-LAG         |            3 | 27,127,227 |  0.077778 | 0.059122 |           -2.414905 |           0.014267 | 0.454503 |     0.019011 |                            0.038175 |               0.019623 | True                | True                  |
| USFI-RATE-DURATION-CURVE-WITH-LAG       |            3 | 27,127,227 |  0.036111 | 0.038490 |           -2.463639 |           0.013990 | 0.348604 |     0.013174 |                            0.028339 |               0.018677 | True                | False                 |
| USFI-RATE-REAL-RATE-NO-LAG              |            3 | 27,127,227 |  0.130556 | 0.059122 |           -2.624058 |           0.028827 | 0.433566 |     0.017504 |                            0.035924 |               0.016206 | True                | True                  |
| USFI-RATE-REAL-RATE-WITH-LAG            |            3 | 27,127,227 | -0.000000 | 0.028868 |           -2.627255 |           0.050579 | 0.388938 |     0.003960 |                            0.031674 |               0.016776 | True                | False                 |
| USFI-RATE-BREAKEVEN-NO-LAG              |            3 | 27,127,227 |  0.133333 | 0.008333 |           -2.451504 |           0.053388 | 0.449518 |     0.025305 |                            0.036263 |               0.018858 | True                | True                  |
| USFI-RATE-BREAKEVEN-WITH-LAG            |            3 | 27,127,227 |  0.116667 | 0.022048 |           -2.560250 |           0.055878 | 0.425231 |     0.013122 |                            0.033797 |               0.016916 | True                | False                 |
| USFI-RATE-RATE-VOL-NO-LAG               |            3 | 27,127,227 |  0.094444 | 0.061426 |           -2.553684 |           0.038589 | 0.316699 |     0.060974 |                            0.025788 |               0.016919 | True                | False                 |
| USFI-RATE-RATE-VOL-WITH-LAG             |            3 | 27,127,227 |  0.080556 | 0.033679 |           -2.548339 |           0.008591 | 0.353872 |     0.034798 |                            0.029001 |               0.016933 | True                | False                 |
| USFI-RATE-CURVE-RATE-VOL-NO-LAG         |            3 | 27,127,227 |  0.141667 | 0.033333 |           -2.483645 |           0.020341 | 0.362938 |     0.047475 |                            0.029789 |               0.017398 | True                | True                  |
| USFI-RATE-CURVE-RATE-VOL-WITH-LAG       |            3 | 27,127,227 |  0.105556 | 0.029266 |           -2.525998 |           0.003638 | 0.342263 |     0.059125 |                            0.027548 |               0.017036 | True                | False                 |
| USFI-RATE-CURVE-REAL-INFLATION-NO-LAG   |            3 | 27,127,227 |  0.033333 | 0.041667 |           -2.654627 |           0.024608 | 0.342609 |     0.034444 |                            0.027922 |               0.015625 | True                | False                 |
| USFI-RATE-CURVE-REAL-INFLATION-WITH-LAG |            3 | 27,127,227 |  0.019444 | 0.024056 |           -2.662173 |           0.008128 | 0.301016 |     0.053033 |                            0.024072 |               0.015795 | True                | False                 |
| USFI-RATE-ALL-RATES-NO-LAG              |            3 | 27,127,227 | -0.005556 | 0.055486 |           -2.622469 |           0.016101 | 0.265885 |     0.089953 |                            0.021887 |               0.016506 | True                | False                 |
| USFI-RATE-ALL-RATES-WITH-LAG            |            3 | 27,127,227 | -0.011111 | 0.058531 |           -2.658390 |           0.014398 | 0.310564 |     0.038771 |                            0.024992 |               0.016554 | True                | False                 |
| USFI-RATE-ALL-RATES-CREDIT-NO-LAG       |            3 | 27,127,227 | -0.122222 | 0.034694 |           -2.628524 |           0.015098 | 0.340112 |     0.019885 |                            0.028907 |               0.017286 | True                | False                 |
| USFI-RATE-ALL-RATES-CREDIT-WITH-LAG     |            3 | 27,127,227 | -0.047222 | 0.012729 |           -2.656579 |           0.019238 | 0.356138 |     0.041348 |                            0.030022 |               0.017181 | True                | False                 |

## Nondominated configurations

USFI-RATE-BALANCED-NO-LAG, USFI-RATE-HIGH-IC-WITH-LAG, USFI-RATE-DURATION-CURVE-NO-LAG, USFI-RATE-REAL-RATE-NO-LAG, USFI-RATE-BREAKEVEN-NO-LAG, USFI-RATE-CURVE-RATE-VOL-NO-LAG

## Hypervolume

- Observed dominated hypervolume: `0.167152281`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BND, IEF, LQD, HYG form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
