# International Confirmation Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration   |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:----------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| I-BREADTH       |            3 | 27,127,227 |  0.136111 | 0.041944 |           -1.861688 |           0.021903 | 1.340510 |     0.048731 |                            0.360539 |               0.037982 | True                | True                  |
| I-MARKET        |            3 | 27,127,227 |  0.113889 | 0.020972 |           -1.522946 |           0.180913 | 1.265790 |     0.034001 |                            0.332261 |               0.041279 | True                | False                 |
| I-MOM           |            3 | 27,127,227 |  0.025000 | 0.142400 |           -1.730624 |           0.008432 | 1.178977 |     0.043001 |                            0.300093 |               0.041284 | True                | False                 |
| I-ALL-MATCH     |            3 | 27,127,227 |  0.005556 | 0.097658 |           -1.802055 |           0.005441 | 1.168111 |     0.066103 |                            0.298226 |               0.039729 | True                | False                 |

## Nondominated configurations

I-BREADTH

## Hypervolume

- Observed dominated hypervolume: `0.339680511`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `bea704cb64a42d1d9aeb112ed76ddb28e9487ca8cedbfcccd137f2ee993dde13`
- All requested runs passed: `True`
- Failures: `[]`
- VEA, VWO, VSS, and EWX form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
