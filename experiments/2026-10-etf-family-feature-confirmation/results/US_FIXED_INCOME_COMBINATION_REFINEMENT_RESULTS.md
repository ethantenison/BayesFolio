# Us Fixed Income Refinement Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                            |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:-----------------------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| USFI-TREND                               |            3 | 27,127,227 |  0.147222 | 0.033679 |           -2.673565 |           0.007682 | 0.429514 |     0.024294 |                            0.034860 |               0.015503 | True                | True                  |
| USFI-TREND-NO-MOM36                      |            3 | 27,127,227 |  0.205556 | 0.055486 |           -2.619566 |           0.048283 | 0.443697 |     0.013657 |                            0.036133 |               0.015978 | True                | False                 |
| USFI-COMPACT-NO-CHMOM                    |            3 | 27,127,227 |  0.177778 | 0.093665 |           -2.656025 |           0.023889 | 0.430185 |     0.024092 |                            0.034977 |               0.015834 | True                | False                 |
| USFI-COMPACT-NO-MOM12M                   |            3 | 27,127,227 |  0.216667 | 0.066144 |           -2.658920 |           0.007016 | 0.429309 |     0.046250 |                            0.035102 |               0.015927 | True                | True                  |
| USFI-COMPACT-NO-CS-MOM-RANK              |            3 | 27,127,227 |  0.175000 | 0.036324 |           -2.671665 |           0.003918 | 0.454434 |     0.034500 |                            0.037021 |               0.015595 | True                | True                  |
| USFI-COMPACT+RET-AUTOCORR                |            3 | 27,127,227 |  0.213889 | 0.067871 |           -2.627451 |           0.046687 | 0.429903 |     0.010898 |                            0.035265 |               0.015871 | True                | True                  |
| USFI-COMPACT+VOL-Z                       |            3 | 27,127,227 |  0.230556 | 0.050918 |           -2.632968 |           0.026280 | 0.427207 |     0.054102 |                            0.035214 |               0.015833 | True                | True                  |
| USFI-COMPACT-NO-MOM12+VOL-Z              |            3 | 27,127,227 |  0.186111 | 0.031549 |           -2.632259 |           0.037717 | 0.395336 |     0.024576 |                            0.032180 |               0.015911 | True                | False                 |
| USFI-COMPACT-NO-MOM12+RET-AUTOCORR       |            3 | 27,127,227 |  0.222222 | 0.118537 |           -2.653782 |           0.024571 | 0.416432 |     0.025311 |                            0.033936 |               0.015779 | True                | True                  |
| USFI-COMPACT-NO-CS+VOL-Z                 |            3 | 27,127,227 |  0.169444 | 0.042763 |           -2.627161 |           0.064981 | 0.411604 |     0.013026 |                            0.033608 |               0.015939 | True                | False                 |
| USFI-COMPACT-NO-CS+RET-AUTOCORR          |            3 | 27,127,227 |  0.211111 | 0.039382 |           -2.657938 |           0.013377 | 0.458606 |     0.033669 |                            0.037691 |               0.015742 | True                | True                  |
| USFI-COMPACT-NO-MOM12-NO-CS              |            3 | 27,127,227 |  0.191667 | 0.066667 |           -2.668034 |           0.023751 | 0.422049 |     0.040469 |                            0.034851 |               0.015789 | True                | True                  |
| USFI-COMPACT-NO-MOM12-NO-CS+VOL-Z        |            3 | 27,127,227 |  0.144444 | 0.017347 |           -2.655905 |           0.017126 | 0.370030 |     0.083321 |                            0.030082 |               0.015654 | True                | False                 |
| USFI-COMPACT-NO-MOM12-NO-CS+RET-AUTOCORR |            3 | 27,127,227 |  0.155556 | 0.026788 |           -2.668665 |           0.013924 | 0.378291 |     0.010308 |                            0.030900 |               0.015721 | True                | False                 |

## Nondominated configurations

USFI-TREND, USFI-COMPACT-NO-MOM12M, USFI-COMPACT-NO-CS-MOM-RANK, USFI-COMPACT+RET-AUTOCORR, USFI-COMPACT+VOL-Z, USFI-COMPACT-NO-MOM12+RET-AUTOCORR, USFI-COMPACT-NO-CS+RET-AUTOCORR, USFI-COMPACT-NO-MOM12-NO-CS

## Hypervolume

- Observed dominated hypervolume: `0.280640676`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BND, IEF, LQD, HYG form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
