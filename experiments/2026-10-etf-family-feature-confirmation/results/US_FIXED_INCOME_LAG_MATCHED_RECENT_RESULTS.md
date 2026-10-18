# Us Fixed Income Refinement Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                                |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:---------------------------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| USFI-RATE-DURATION-CURVE-NO-LAG              |            3 | 27,127,227 |  0.077778 | 0.059122 |           -2.414905 |           0.014267 | 0.454503 |     0.019011 |                            0.038175 |               0.019623 | True                | True                  |
| USFI-RATE-DURATION-CURVE-WITH-LAG            |            3 | 27,127,227 |  0.036111 | 0.038490 |           -2.463639 |           0.013990 | 0.348604 |     0.013174 |                            0.028339 |               0.018677 | True                | False                 |
| USFI-RATE-REAL-RATE-NO-LAG                   |            3 | 27,127,227 |  0.130556 | 0.059122 |           -2.624058 |           0.028827 | 0.433566 |     0.017504 |                            0.035924 |               0.016206 | True                | True                  |
| USFI-RATE-REAL-RATE-WITH-LAG                 |            3 | 27,127,227 | -0.000000 | 0.028868 |           -2.627255 |           0.050579 | 0.388938 |     0.003960 |                            0.031674 |               0.016776 | True                | False                 |
| USFI-RATE-BREAKEVEN-NO-LAG                   |            3 | 27,127,227 |  0.133333 | 0.008333 |           -2.451504 |           0.053388 | 0.449518 |     0.025305 |                            0.036263 |               0.018858 | True                | True                  |
| USFI-RATE-BREAKEVEN-WITH-LAG                 |            3 | 27,127,227 |  0.116667 | 0.022048 |           -2.560250 |           0.055878 | 0.425231 |     0.013122 |                            0.033797 |               0.016916 | True                | False                 |
| USFI-RATE-CURVE-RATE-VOL-NO-LAG              |            3 | 27,127,227 |  0.141667 | 0.033333 |           -2.483645 |           0.020341 | 0.362938 |     0.047475 |                            0.029789 |               0.017398 | True                | True                  |
| USFI-RATE-CURVE-RATE-VOL-WITH-LAG            |            3 | 27,127,227 |  0.105556 | 0.029266 |           -2.525998 |           0.003638 | 0.342263 |     0.059125 |                            0.027548 |               0.017036 | True                | False                 |
| USFI-MATCH-RECENT-TREND-NO_LAG               |            3 | 27,127,227 |  0.102778 | 0.110659 |           -2.621723 |           0.039581 | 0.393222 |     0.084149 |                            0.031586 |               0.015831 | True                | False                 |
| USFI-MATCH-RECENT-TREND-WITH_LAG             |            3 | 27,127,227 |  0.108333 | 0.058333 |           -2.630501 |           0.022039 | 0.325078 |     0.070978 |                            0.026401 |               0.015847 | True                | False                 |
| USFI-MATCH-RECENT-NO_MOM12-NO_LAG            |            3 | 27,127,227 |  0.066667 | 0.087797 |           -2.631447 |           0.018339 | 0.369149 |     0.059850 |                            0.029158 |               0.015927 | True                | False                 |
| USFI-MATCH-RECENT-NO_MOM12-WITH_LAG          |            3 | 27,127,227 |  0.055556 | 0.034694 |           -2.625196 |           0.025618 | 0.374290 |     0.048584 |                            0.030097 |               0.015798 | True                | False                 |
| USFI-MATCH-RECENT-NO_CS-NO_LAG               |            3 | 27,127,227 |  0.108333 | 0.044096 |           -2.638486 |           0.017681 | 0.432829 |     0.076033 |                            0.035431 |               0.015930 | True                | True                  |
| USFI-MATCH-RECENT-NO_CS-WITH_LAG             |            3 | 27,127,227 |  0.075000 | 0.068211 |           -2.622123 |           0.023719 | 0.338986 |     0.035798 |                            0.027198 |               0.015993 | True                | False                 |
| USFI-MATCH-RECENT-AUTOCORR-NO_LAG            |            3 | 27,127,227 |  0.097222 | 0.033679 |           -2.662609 |           0.013773 | 0.389063 |     0.050502 |                            0.031802 |               0.015584 | True                | False                 |
| USFI-MATCH-RECENT-AUTOCORR-WITH_LAG          |            3 | 27,127,227 |  0.002778 | 0.091414 |           -2.638685 |           0.036751 | 0.353081 |     0.084187 |                            0.028221 |               0.015704 | True                | False                 |
| USFI-MATCH-RECENT-VOL-NO_LAG                 |            3 | 27,127,227 |  0.102778 | 0.041107 |           -2.647049 |           0.019696 | 0.417684 |     0.052209 |                            0.033463 |               0.015767 | True                | False                 |
| USFI-MATCH-RECENT-VOL-WITH_LAG               |            3 | 27,127,227 |  0.130556 | 0.033679 |           -2.639765 |           0.020238 | 0.380926 |     0.051548 |                            0.030807 |               0.015576 | True                | True                  |
| USFI-MATCH-RECENT-NO_MOM12_AUTOCORR-NO_LAG   |            3 | 27,127,227 |  0.111111 | 0.055486 |           -2.687470 |           0.018338 | 0.373760 |     0.036854 |                            0.030015 |               0.015498 | True                | True                  |
| USFI-MATCH-RECENT-NO_MOM12_AUTOCORR-WITH_LAG |            3 | 27,127,227 |  0.061111 | 0.054220 |           -2.641679 |           0.016655 | 0.349778 |     0.059475 |                            0.027973 |               0.015538 | True                | False                 |
| USFI-MATCH-RECENT-BALANCED-NO_LAG            |            3 | 27,127,227 |  0.108333 | 0.058333 |           -2.663281 |           0.015354 | 0.432755 |     0.012067 |                            0.034662 |               0.015804 | True                | True                  |
| USFI-MATCH-RECENT-BALANCED-WITH_LAG          |            3 | 27,127,227 |  0.113889 | 0.037577 |           -2.611770 |           0.032028 | 0.365429 |     0.020643 |                            0.029405 |               0.015815 | True                | False                 |
| USFI-MATCH-RECENT-MINIMAL-NO_LAG             |            3 | 27,127,227 |  0.080556 | 0.080938 |           -2.640811 |           0.006290 | 0.351857 |     0.010844 |                            0.028020 |               0.015973 | True                | False                 |
| USFI-MATCH-RECENT-MINIMAL-WITH_LAG           |            3 | 27,127,227 |  0.077778 | 0.050918 |           -2.652123 |           0.003686 | 0.358496 |     0.022301 |                            0.028852 |               0.015694 | True                | False                 |

## Nondominated configurations

USFI-RATE-DURATION-CURVE-NO-LAG, USFI-RATE-REAL-RATE-NO-LAG, USFI-RATE-BREAKEVEN-NO-LAG, USFI-RATE-CURVE-RATE-VOL-NO-LAG, USFI-MATCH-RECENT-NO_CS-NO_LAG, USFI-MATCH-RECENT-VOL-WITH_LAG, USFI-MATCH-RECENT-NO_MOM12_AUTOCORR-NO_LAG, USFI-MATCH-RECENT-BALANCED-NO_LAG

## Hypervolume

- Observed dominated hypervolume: `0.168181345`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BND, IEF, LQD, HYG form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
