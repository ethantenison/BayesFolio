# International Targeted Ablation Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration              |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:---------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| I-BREADTH                  |            3 | 27,127,227 |  0.136111 | 0.041944 |           -1.861688 |           0.021903 | 1.340510 |     0.048731 |                            0.360539 |               0.037982 | True                | True                  |
| I-BREADTH+OIL              |            3 | 27,127,227 |  0.200000 | 0.030046 |           -1.781851 |           0.054191 | 1.342357 |     0.046904 |                            0.360394 |               0.040373 | True                | True                  |
| I-BREADTH+COMMODITIES      |            3 | 27,127,227 |  0.027778 | 0.034694 |           -1.867543 |           0.007396 | 1.213128 |     0.010333 |                            0.317689 |               0.036940 | True                | True                  |
| I-BREADTH+GOLD-CRUDE-RATIO |            3 | 27,127,227 |  0.091667 | 0.060093 |           -1.782716 |           0.037084 | 1.278213 |     0.059745 |                            0.340059 |               0.039829 | True                | False                 |

## Nondominated configurations

I-BREADTH, I-BREADTH+OIL, I-BREADTH+COMMODITIES

## Hypervolume

- Observed dominated hypervolume: `0.493140302`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `bea704cb64a42d1d9aeb112ed76ddb28e9487ca8cedbfcccd137f2ee993dde13`
- All requested runs passed: `True`
- Failures: `[]`
- VEA, VWO, VSS, and EWX form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
