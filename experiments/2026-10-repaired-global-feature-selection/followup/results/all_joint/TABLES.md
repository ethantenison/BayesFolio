# All joint model results

Validated configurations: 37; seeds: 111.

Fixed-reference hypervolume: 1.228693994.

## Full Pareto front

| configuration       | hypothesis                |   input_count_excluding_time |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe | pareto_nondominated   |
|:--------------------|:--------------------------|-----------------------------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|:----------------------|
| global_e6c41f2020fd | add_spy_proxy             |                           14 |  0.296528 | 0.008099 |           -2.258812 |           0.002709 | 1.771496 |     0.033957 | True                  |
| global_4397c0b6828e | add_market_inflation_pair |                           15 |  0.286409 | 0.014097 |           -2.243060 |           0.004830 | 1.807445 |     0.058856 | True                  |
| global_bb1b8f9500f2 | remove_dollar_level       |                           12 |  0.281448 | 0.018554 |           -2.307496 |           0.004699 | 1.390330 |     0.090400 | True                  |

## Every configuration

| configuration       | hypothesis                                                      |   input_count_excluding_time |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe | pareto_nondominated   |
|:--------------------|:----------------------------------------------------------------|-----------------------------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|:----------------------|
| global_c55e49933d10 | global_c55e49933d10                                             |                           27 |  0.108780 | 0.002436 |           -2.141413 |           0.015577 | 1.127881 |     0.085396 | False                 |
| global_b48732adb186 | shared_mechanism_core_10                                        |                           10 |  0.204514 | 0.012981 |           -2.238293 |           0.024334 | 1.288369 |     0.226963 | False                 |
| global_3745593ecc26 | literal_selected_family_union_29                                |                           29 |  0.096478 | 0.003983 |           -2.106803 |           0.012106 | 1.178543 |     0.146854 | False                 |
| global_9d4dc5c2a19f | evidence_reduced_union_22                                       |                           22 |  0.182192 | 0.014468 |           -2.157635 |           0.020616 | 1.296970 |     0.100543 | False                 |
| global_2b66b7fc9390 | shared_core_plus_fixed_income_exceptions_14                     |                           14 |  0.114286 | 0.005232 |           -2.093847 |           0.018285 | 1.288462 |     0.027476 | False                 |
| global_389a69848aa0 | shared_core_plus_international_risk_exceptions_15               |                           15 |  0.258433 | 0.015766 |           -2.218703 |           0.018312 | 1.425040 |     0.098589 | False                 |
| global_1b4304ffe9c1 | shared_core_plus_fixed_income_exceptions_and_lag_15             |                           15 |  0.121677 | 0.026211 |           -2.158251 |           0.014108 | 1.299081 |     0.110573 | False                 |
| global_15b18df4a34b | international_exception_front_without_global_rank               |                           14 |  0.266617 | 0.015501 |           -2.232702 |           0.013254 | 1.469891 |     0.111265 | False                 |
| global_20d835629365 | international_exception_front_without_unskipped_annual_momentum |                           14 |  0.256498 | 0.018656 |           -2.220200 |           0.022442 | 1.422856 |     0.097232 | False                 |
| global_67eeea779f2d | international_exception_front_without_gold_oil_ratio            |                           14 |  0.206944 | 0.010364 |           -2.200128 |           0.018258 | 1.340553 |     0.125221 | False                 |
| global_43cd2137e85f | international_exception_front_plus_lag                          |                           16 |  0.262054 | 0.016304 |           -2.239092 |           0.010721 | 1.468439 |     0.077520 | False                 |
| global_d019284686e8 | ten_input_density_core_plus_absolute_credit_level               |                           11 |  0.261210 | 0.008381 |           -2.220792 |           0.022652 | 1.296175 |     0.123052 | False                 |
| global_300876dbdd56 | ten_input_density_core_plus_dealer_proxy                        |                           11 |  0.187302 | 0.003007 |           -2.198568 |           0.023920 | 1.303814 |     0.125519 | False                 |
| global_52d9b7e07091 | lagged_density_front_without_global_rank                        |                           15 |  0.258085 | 0.009674 |           -2.251246 |           0.002668 | 1.526414 |     0.067137 | False                 |
| global_c0bc5df949cf | rank_free_front_without_unskipped_annual_momentum               |                           13 |  0.275397 | 0.015822 |           -2.236822 |           0.018692 | 1.448330 |     0.134929 | False                 |
| global_4a6348639308 | rank_free_front_without_trend_slope                             |                           13 |  0.264385 | 0.013872 |           -2.229558 |           0.021089 | 1.464611 |     0.143039 | False                 |
| global_a8a6b8394d6e | rank_free_front_without_vix_level                               |                           13 |  0.272470 | 0.013992 |           -2.228913 |           0.009264 | 1.567890 |     0.085619 | False                 |
| global_973f696b2e34 | rank_free_front_without_oil_return                              |                           13 |  0.250694 | 0.017862 |           -2.177640 |           0.021551 | 1.278938 |     0.160457 | False                 |
| global_d062f4d835aa | rank_free_front_plus_absolute_credit_level                      |                           15 |  0.274107 | 0.012351 |           -2.233158 |           0.019435 | 1.452375 |     0.079075 | False                 |
| global_9e7175a3ef4a | no_vix_front_without_unskipped_annual_momentum                  |                           12 |  0.274901 | 0.012844 |           -2.234172 |           0.017214 | 1.544716 |     0.170209 | False                 |
| global_ac1e9b719b57 | lagged_density_front_without_vix_level                          |                           14 |  0.261607 | 0.003403 |           -2.243975 |           0.004439 | 1.585268 |     0.095927 | False                 |
| global_33e57e1c2370 | lagged_density_front_without_vix_or_unskipped_annual_momentum   |                           13 |  0.264881 | 0.003324 |           -2.250209 |           0.002646 | 1.622642 |     0.094196 | False                 |
| global_da43a19fc3dd | no_vix_front_without_dealer_proxy                               |                           12 |  0.236359 | 0.024827 |           -2.262087 |           0.016436 | 1.381369 |     0.125196 | False                 |
| global_c86bac72391c | no_vix_front_without_normalized_credit_state                    |                           12 |  0.194692 | 0.022370 |           -2.154281 |           0.004414 | 1.298696 |     0.105981 | False                 |
| global_fdb5fb8b385f | absolute_credit_front_without_vix_level                         |                           14 |  0.278026 | 0.022007 |           -2.232245 |           0.011826 | 1.528323 |     0.063292 | False                 |
| global_a306a45ffc49 | add_hy_level                                                    |                           14 |  0.263740 | 0.005737 |           -2.241139 |           0.006752 | 1.631714 |     0.087301 | False                 |
| global_d77edcc67c1d | add_hy_change                                                   |                           14 |  0.277728 | 0.003381 |           -2.242970 |           0.012334 | 1.643740 |     0.127882 | False                 |
| global_e6c41f2020fd | add_spy_proxy                                                   |                           14 |  0.296528 | 0.008099 |           -2.258812 |           0.002709 | 1.771496 |     0.033957 | True                  |
| global_ccfa2775bb52 | add_monthly_cpi                                                 |                           14 |  0.266667 | 0.003219 |           -2.238164 |           0.007786 | 1.610998 |     0.047056 | False                 |
| global_dc10aef417bc | add_schp_return                                                 |                           14 |  0.167708 | 0.000649 |           -2.164351 |           0.006937 | 1.475153 |     0.100401 | False                 |
| global_6a602a86c2b6 | add_nominal_yield                                               |                           14 |  0.253720 | 0.004110 |           -2.225665 |           0.004198 | 1.432244 |     0.107800 | False                 |
| global_4397c0b6828e | add_market_inflation_pair                                       |                           15 |  0.286409 | 0.014097 |           -2.243060 |           0.004830 | 1.807445 |     0.058856 | True                  |
| global_9254a1b37349 | add_nominal_tips_pair                                           |                           15 |  0.181746 | 0.011757 |           -2.203249 |           0.005828 | 1.397071 |     0.030414 | False                 |
| global_d8737126b4e7 | remove_intermediate_momentum                                    |                           12 |  0.269296 | 0.015593 |           -2.249705 |           0.001863 | 1.539813 |     0.110180 | False                 |
| global_a6fab4154c69 | annual_for_intermediate_momentum                                |                           13 |  0.260863 | 0.009835 |           -2.239131 |           0.004882 | 1.500076 |     0.115452 | False                 |
| global_32cfb52cf8ac | ma_for_trend                                                    |                           13 |  0.265575 | 0.008439 |           -2.249972 |           0.004660 | 1.534868 |     0.064679 | False                 |
| global_bb1b8f9500f2 | remove_dollar_level                                             |                           12 |  0.281448 | 0.018554 |           -2.307496 |           0.004699 | 1.390330 |     0.090400 | True                  |

## Exact frontier inputs

- global_e6c41f2020fd: t_index, lag_y_excess_lead, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, dealer_gamma_proxy, dxy, em_fx_ret, gold_crude_ratio, hy_spread_z_12p, oil_ret, spy_ret, term_spread, vix_ts_z_12p

- global_4397c0b6828e: t_index, lag_y_excess_lead, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, cpi_chg_1p, dealer_gamma_proxy, dxy, em_fx_ret, gold_crude_ratio, hy_spread_z_12p, oil_ret, spy_ret, term_spread, vix_ts_z_12p

- global_bb1b8f9500f2: t_index, lag_y_excess_lead, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, dealer_gamma_proxy, em_fx_ret, gold_crude_ratio, hy_spread_z_12p, oil_ret, term_spread, vix_ts_z_12p

## Family outcomes of frontier and controls

| configuration       | family                     |   within_family_ic |   mean_gaussian_nll |   coverage_95 |   mean_predictive_std |   mean_family_capital_weight |
|:--------------------|:---------------------------|-------------------:|--------------------:|--------------:|----------------------:|-----------------------------:|
| global_33e57e1c2370 | international_equity       |           0.183333 |           -1.771276 |      0.913194 |              0.033406 |                     0.361605 |
| global_33e57e1c2370 | international_fixed_income |           0.090278 |           -2.775744 |      0.962963 |              0.015519 |                     0.031819 |
| global_33e57e1c2370 | us_equity                  |           0.208333 |           -1.742534 |      0.940972 |              0.043055 |                     0.589373 |
| global_33e57e1c2370 | us_fixed_income            |           0.038889 |           -2.842665 |      0.951389 |              0.014949 |                     0.017204 |
| global_4397c0b6828e | international_equity       |           0.077778 |           -1.759741 |      0.916667 |              0.033300 |                     0.373362 |
| global_4397c0b6828e | international_fixed_income |           0.131944 |           -2.795728 |      0.972222 |              0.015568 |                     0.025988 |
| global_4397c0b6828e | us_equity                  |           0.080556 |           -1.728288 |      0.944444 |              0.043099 |                     0.576228 |
| global_4397c0b6828e | us_fixed_income            |           0.111111 |           -2.826650 |      0.951389 |              0.015078 |                     0.024422 |
| global_bb1b8f9500f2 | international_equity       |           0.172222 |           -1.853209 |      0.916667 |              0.033747 |                     0.372482 |
| global_bb1b8f9500f2 | international_fixed_income |           0.208333 |           -2.831071 |      0.958333 |              0.015726 |                     0.030556 |
| global_bb1b8f9500f2 | us_equity                  |           0.133333 |           -1.797828 |      0.961806 |              0.043134 |                     0.546006 |
| global_bb1b8f9500f2 | us_fixed_income            |           0.261111 |           -2.878768 |      0.954861 |              0.015033 |                     0.050957 |
| global_c55e49933d10 | international_equity       |           0.041667 |           -1.700185 |      0.881944 |              0.032888 |                     0.326604 |
| global_c55e49933d10 | international_fixed_income |           0.222222 |           -2.720776 |      0.935185 |              0.015043 |                     0.034141 |
| global_c55e49933d10 | us_equity                  |           0.172222 |           -1.621425 |      0.899306 |              0.041315 |                     0.581067 |
| global_c55e49933d10 | us_fixed_income            |          -0.097222 |           -2.668109 |      0.927083 |              0.014998 |                     0.058188 |
| global_e6c41f2020fd | international_equity       |           0.141667 |           -1.771089 |      0.913194 |              0.033226 |                     0.372236 |
| global_e6c41f2020fd | international_fixed_income |           0.125000 |           -2.804836 |      0.972222 |              0.015652 |                     0.040502 |
| global_e6c41f2020fd | us_equity                  |           0.133333 |           -1.742457 |      0.937500 |              0.043099 |                     0.554791 |
| global_e6c41f2020fd | us_fixed_income            |           0.125000 |           -2.853372 |      0.954861 |              0.015073 |                     0.032472 |

Monthly uncertainty and exact conditional effects are adjacent CSVs.
Three objectives remain active; return/RMSE/width/count are not objectives. Reused-date adaptive research.
