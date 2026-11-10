# Us Equity round 3

Cumulative configurations: 35 -> 47.

Fixed-reference hypervolume: 0.834919079 -> 0.861677628.

Stalled: False; two consecutive stalled rounds: False.

| configuration          | hypothesis_name                                   | base_configuration     | category                                |   delta_ic |   delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:-----------------------|:--------------------------------------------------|:-----------------------|:----------------------------------------|-----------:|------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| us_equity_8b0e1b779842 | credit_level_front_without_trend                  | us_equity_23a11358aeee | exploitation_combined_pruning           |  0.0111111 |  0.014192   |     -0.0395297 |                   -1 |              1 |               1 |                  1 | True                  |
| us_equity_c6a5b4f1daf1 | credit_level_front_without_six_month_momentum     | us_equity_23a11358aeee | exploration_local_interaction           | -0.0694444 |  0.0334616  |     -0.0720022 |                   -1 |              1 |               0 |                  0 | False                 |
| us_equity_7cac77648247 | credit_level_front_without_credit_zscore          | us_equity_23a11358aeee | exploration_credit_representation       | -0.0222222 |  0.120091   |     -0.0537585 |                   -1 |              1 |               0 |                  0 | False                 |
| us_equity_4570e1c57a92 | credit_level_front_plus_credit_change             | us_equity_23a11358aeee | exploration_credit_complementarity      | -0.0638889 | -0.0440638  |     -0.0539823 |                    1 |              0 |               3 |                  0 | True                  |
| us_equity_ca4082b16bd0 | credit_level_front_plus_monthly_inflation         | us_equity_23a11358aeee | exploration_inflation_complementarity   | -0.127778  |  0.0299221  |     -0.062809  |                    1 |              0 |               0 |                  1 | False                 |
| us_equity_aae6f553c357 | credit_level_front_plus_nominal_yield             | us_equity_23a11358aeee | exploration_rate_complementarity        | -0.0694444 |  0.0216201  |     -0.0786988 |                    1 |              0 |               1 |                  0 | False                 |
| us_equity_0e55a9175696 | credit_level_front_with_unskipped_annual_momentum | us_equity_23a11358aeee | exploration_shared_local_representation | -0.05      |  0.0203027  |     -0.100357  |                    0 |              0 |               0 |                  0 | False                 |
| us_equity_4ef9518f3269 | credit_level_front_plus_rank                      | us_equity_23a11358aeee | exploration_local_interaction           | -0.0111111 | -0.00373076 |     -0.0294604 |                    1 |              1 |               1 |                  1 | True                  |
| us_equity_40972e83592a | credit_level_front_plus_lag                       | us_equity_23a11358aeee | conditional_lag_addition                | -0.122222  | -0.00868984 |     -0.110955  |                    1 |              0 |               2 |                  0 | False                 |
| us_equity_583291afb2cb | six_input_front_without_six_month_momentum        | us_equity_d0a35b28f60b | exploitation_local_pruning              | -0.136111  |  0.0405289  |     -0.0537022 |                   -1 |              0 |               0 |                  0 | False                 |
| us_equity_86c90676a157 | six_input_front_without_credit_change             | us_equity_d0a35b28f60b | exploration_macro_interaction           | -0.0694444 |  0.116848   |     -0.0453005 |                   -1 |              0 |               0 |                  1 | False                 |
| us_equity_b3964012e6b1 | six_input_front_plus_oil                          | us_equity_d0a35b28f60b | exploration_cross_family_transfer       | -0.0444444 | -0.00059479 |     -0.016911  |                    1 |              1 |               1 |                  1 | False                 |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
