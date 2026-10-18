# U.S. Fixed-Income Lag: Matched Training-History Comparison

Universe: BND, IEF, LQD, HYG. Seeds: 27, 127, 227. Scored months: May 2024 through April 2026.
Rank 2, maxiter 15, 256 scenarios; t_index retained. Objectives: maximize IC and conventional Sharpe,
minimize Gaussian NLL. Cumulative decimal return and decimal RMSE are record-only; turnover excluded.

All eight original Pareto designs were rerun with and without lag for both history starts (96 fresh fits).
Eight previously evaluated rate-model fits per seed are included in the recent comparison only, after
checking their frozen feature hash and every window's training rows and date bounds (24 reused fits).
Long and recent input values are exactly identical on overlapping dates. All 120 fits were audited.
The eight original with-lag long-history results reproduced: True.

## Incumbent input design: direct four-way comparison

Balanced inputs: t_index, chmom, mom12m, ma_signal, ret_autocorr, with optional lag_y_excess_lead.

| history   | include_lag   |   mean_ic |   mean_gaussian_nll |   sharpe |
|:----------|:--------------|----------:|--------------------:|---------:|
| long      | False         |  0.183333 |           -2.573003 | 0.395273 |
| long      | True          |  0.211111 |           -2.657938 | 0.458606 |
| recent    | False         |  0.108333 |           -2.663281 | 0.432755 |
| recent    | True          |  0.113889 |           -2.611770 | 0.365429 |

## Long history: expanding from 2013-08-30

| design            | include_lag   |   mean_ic |   mean_gaussian_nll |   sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only |
|:------------------|:--------------|----------:|--------------------:|---------:|------------------------------------:|-----------------------:|
| trend             | True          |  0.147222 |           -2.673565 | 0.429514 |                            0.034860 |               0.015503 |
| no_mom12          | True          |  0.216667 |           -2.658920 | 0.429309 |                            0.035102 |               0.015927 |
| no_cs             | True          |  0.175000 |           -2.671665 | 0.454434 |                            0.037021 |               0.015595 |
| autocorr          | True          |  0.213889 |           -2.627451 | 0.429903 |                            0.035265 |               0.015871 |
| vol               | True          |  0.230556 |           -2.632968 | 0.427207 |                            0.035214 |               0.015833 |
| no_mom12_autocorr | True          |  0.222222 |           -2.653782 | 0.416432 |                            0.033936 |               0.015779 |
| balanced          | True          |  0.211111 |           -2.657938 | 0.458606 |                            0.037691 |               0.015742 |
| minimal           | True          |  0.191667 |           -2.668034 | 0.422049 |                            0.034851 |               0.015789 |

### Removing lag: no-lag minus with-lag

| design            |   delta_mean_ic |   delta_mean_gaussian_nll |   delta_sharpe |   no_lag_seed_wins_mean_ic |   no_lag_seed_wins_mean_gaussian_nll |   no_lag_seed_wins_sharpe |
|:------------------|----------------:|--------------------------:|---------------:|---------------------------:|-------------------------------------:|--------------------------:|
| trend             |        0.036111 |                  0.144369 |      -0.034204 |                          2 |                                    0 |                         1 |
| no_mom12          |       -0.055556 |                  0.129647 |      -0.029932 |                          0 |                                    0 |                         1 |
| no_cs             |       -0.033333 |                  0.120096 |      -0.076296 |                          1 |                                    0 |                         0 |
| autocorr          |       -0.055556 |                  0.021506 |      -0.038963 |                          1 |                                    1 |                         1 |
| vol               |       -0.180556 |                  0.048072 |      -0.135809 |                          0 |                                    0 |                         0 |
| no_mom12_autocorr |       -0.094444 |                  0.034236 |      -0.055403 |                          1 |                                    0 |                         0 |
| balanced          |       -0.027778 |                  0.084935 |      -0.063332 |                          1 |                                    0 |                         1 |
| minimal           |       -0.077778 |                  0.113707 |      -0.040880 |                          0 |                                    0 |                         1 |

## Recent history: expanding from 2021-03-31

| design            | include_lag   |   mean_ic |   mean_gaussian_nll |   sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only |
|:------------------|:--------------|----------:|--------------------:|---------:|------------------------------------:|-----------------------:|
| duration_curve    | False         |  0.077778 |           -2.414905 | 0.454503 |                            0.038175 |               0.019623 |
| real_rate         | False         |  0.130556 |           -2.624058 | 0.433566 |                            0.035924 |               0.016206 |
| breakeven         | False         |  0.133333 |           -2.451504 | 0.449518 |                            0.036263 |               0.018858 |
| curve_rate_vol    | False         |  0.141667 |           -2.483645 | 0.362938 |                            0.029789 |               0.017398 |
| no_cs             | False         |  0.108333 |           -2.638486 | 0.432829 |                            0.035431 |               0.015930 |
| vol               | True          |  0.130556 |           -2.639765 | 0.380926 |                            0.030807 |               0.015576 |
| no_mom12_autocorr | False         |  0.111111 |           -2.687470 | 0.373760 |                            0.030015 |               0.015498 |
| balanced          | False         |  0.108333 |           -2.663281 | 0.432755 |                            0.034662 |               0.015804 |

### Removing lag: no-lag minus with-lag

| design            |   delta_mean_ic |   delta_mean_gaussian_nll |   delta_sharpe |   no_lag_seed_wins_mean_ic |   no_lag_seed_wins_mean_gaussian_nll |   no_lag_seed_wins_sharpe |
|:------------------|----------------:|--------------------------:|---------------:|---------------------------:|-------------------------------------:|--------------------------:|
| duration_curve    |        0.041667 |                  0.048734 |       0.105898 |                          3 |                                    0 |                         3 |
| real_rate         |        0.130556 |                  0.003197 |       0.044628 |                          3 |                                    2 |                         3 |
| breakeven         |        0.016667 |                  0.108746 |       0.024287 |                          2 |                                    0 |                         2 |
| curve_rate_vol    |        0.036111 |                  0.042354 |       0.020675 |                          2 |                                    0 |                         2 |
| trend             |       -0.005556 |                  0.008778 |       0.068144 |                          1 |                                    1 |                         2 |
| no_mom12          |        0.011111 |                 -0.006251 |      -0.005142 |                          1 |                                    1 |                         2 |
| no_cs             |        0.033333 |                 -0.016363 |       0.093843 |                          2 |                                    3 |                         3 |
| autocorr          |        0.094444 |                 -0.023923 |       0.035982 |                          2 |                                    2 |                         2 |
| vol               |       -0.027778 |                 -0.007284 |       0.036758 |                          1 |                                    2 |                         2 |
| no_mom12_autocorr |        0.050000 |                 -0.045792 |       0.023982 |                          2 |                                    3 |                         1 |
| balanced          |       -0.005556 |                 -0.051511 |       0.067326 |                          1 |                                    3 |                         3 |
| minimal           |        0.002778 |                  0.011312 |      -0.006639 |                          2 |                                    0 |                         1 |

## Matched-history findings

- Long: 0/8 Pareto configurations omit lag; removing lag improves aggregate Sharpe in 0/8 pairs and dominates its with-lag pair in 0 designs.
- Recent: 7/8 Pareto configurations omit lag; removing lag improves aggregate Sharpe in 10/12 pairs and dominates its with-lag pair in 3 designs.

## Recommendation

Retain lag_y_excess_lead for the current long-history U.S. fixed-income incumbent.
The long-history balanced model remains the stronger IC/Sharpe choice; shorter-history no-lag models
remain alternative training policies with calibration tradeoffs. Removing lag is not a justified
family-wide simplification under the current long-history policy.

The previous suggestion that richer rate inputs explain lag redundancy was too strong: local-only
no-lag models also improve when trained from 2021. Training-history sensitivity is directly supported;
the economic mechanism causing that sensitivity has not been established by these experiments.

## Interpretation boundary

Each within-history comparison uses identical observation dates and expanding training rules.
Across histories, feature values and evaluation months are held fixed; training history varies.
Rate variables are cached only from 2021; their performance on 2013-start history remains untested.
The 24 evaluation months have already informed model selection; these results refine the experiment
and require further confirmation before production promotion.

## Verification

```bash
uv run --locked python experiments/2026-10-etf-family-feature-confirmation/summarize_us_fixed_income_lag_history.py
uv run --locked pytest -q experiments/2026-10-etf-family-feature-confirmation/test_confirmation_workflow.py
```
