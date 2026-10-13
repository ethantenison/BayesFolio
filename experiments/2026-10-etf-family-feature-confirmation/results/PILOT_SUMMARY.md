# U.S. Seed-27 Pilot Results

These are pilot results from one seed, not promotion decisions. The Pareto objectives are higher IC, lower
Gaussian NLL, and higher conventional annualized Sharpe. The return series is already risk-free adjusted;
Sharpe uses its sample standard deviation. Cumulative return and RMSE are diagnostic-only.
Turnover is excluded.

## Objective table

| configuration   |   seed |   n_dates |   mean_ic |   mean_gaussian_nll |   sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | diagnostic_metrics_affect_pareto   | validation_passed   | pareto_nondominated_seed27   |
|:----------------|-------:|----------:|----------:|--------------------:|---------:|------------------------------------:|-----------------------:|:-----------------------------------|:--------------------|:-----------------------------|
| U-AC            |     27 |        24 |  0.116667 |           -1.653393 | 1.205594 |                            0.354665 |               0.045245 | False                              | True                | True                         |
| U-C005          |     27 |        24 |  0.233333 |           -1.683705 | 1.141345 |                            0.318381 |               0.045599 | False                              | True                | True                         |
| U-C009          |     27 |        24 |  0.108333 |           -1.703157 | 1.014353 |                            0.272650 |               0.045852 | False                              | True                | True                         |
| U-ALL-MATCH     |     27 |        24 |  0.258333 |           -1.636080 | 1.287046 |                            0.375740 |               0.046269 | False                              | True                | True                         |

## Seed-27 nondominated configurations

U-AC, U-C005, U-C009, U-ALL-MATCH

## Validation

- Common date hash: `bea704cb64a42d1d9aeb112ed76ddb28e9487ca8cedbfcccd137f2ee993dde13`
- Dates: `2024-06-28` through `2026-05-29` (24 windows)
- All run validations passed: `True`
- Validation failures: `[]`
- IJR excluded and t_index included in every pilot.
- Rank 2, maxiter 15, 256 scenarios, and seed 27 were held constant.

## Gate

Do not promote a configuration from this table. Seeds 127 and 227 require separate approval.
