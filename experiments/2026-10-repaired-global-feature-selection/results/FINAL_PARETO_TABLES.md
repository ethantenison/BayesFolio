# Complete final Pareto tables

All points use three complete seeds (27, 127, 227). Mean objectives and sample seed SD are shown.
Counts exclude t_index, which is retained in every exact list. Return and RMSE are diagnostics only.
Return means compounded monthly excess return, not total nominal investment wealth; RMSE is decimal.
Seed dispersion is not economic confidence. Adaptive research fronts are not deployment approval.

## us_equity

Complete configurations: 48; frontier: 9.

Compute-matched all-input control: us_equity_77e1f7f1f070; IC 0.183333, NLL -1.646174, Sharpe 1.342057 (27 non-time inputs).

| Configuration | Inputs | IC ± SD | NLL ± SD | Sharpe ± SD | Excess return | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| us_equity_8b0e1b779842 | 6 | 0.325000 ± 0.038188 | -1.639829 ± 0.017088 | 1.456596 ± 0.043660 | 0.422258 | 0.046093 |
| us_equity_23a11358aeee | 7 | 0.313889 ± 0.041107 | -1.654021 ± 0.050351 | 1.496126 ± 0.050839 | 0.436333 | 0.045695 |
| us_equity_4ef9518f3269 | 8 | 0.302778 ± 0.037577 | -1.657752 ± 0.011820 | 1.466666 ± 0.076737 | 0.419920 | 0.045867 |
| us_equity_c10464d00ae6 | 8 | 0.263889 ± 0.025459 | -1.753270 ± 0.032808 | 1.435086 ± 0.031183 | 0.417439 | 0.040859 |
| us_equity_d0a35b28f60b | 6 | 0.261111 ± 0.061426 | -1.767197 ± 0.015879 | 1.431215 ± 0.072003 | 0.420440 | 0.040483 |
| us_equity_4570e1c57a92 | 8 | 0.250000 ± 0.050000 | -1.698085 ± 0.035090 | 1.442144 ± 0.025921 | 0.412986 | 0.043957 |
| us_equity_5fba6b4b201b | 7 | 0.238889 ± 0.047385 | -1.779372 ± 0.016872 | 1.504956 ± 0.015151 | 0.453978 | 0.040355 |
| us_equity_e7028305d052 | 9 | 0.155556 ± 0.038490 | -1.806277 ± 0.044054 | 1.338789 ± 0.030241 | 0.390401 | 0.038217 |
| us_equity_46cc53338e30 | 7 | 0.047222 ± 0.026788 | -1.780142 ± 0.025040 | 1.353872 ± 0.086126 | 0.374681 | 0.040460 |

### Exact input lists

- us_equity_8b0e1b779842: t_index, mom12m_skip1m, mom6m, cpi_chg_12p, hy_spread, hy_spread_z_12p, term_spread

- us_equity_23a11358aeee: t_index, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, hy_spread, hy_spread_z_12p, term_spread

- us_equity_4ef9518f3269: t_index, cs_mom_rank, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, hy_spread, hy_spread_z_12p, term_spread

- us_equity_c10464d00ae6: t_index, cs_mom_rank, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, hy_spread_chg_1p, hy_spread_z_12p, term_spread

- us_equity_d0a35b28f60b: t_index, mom12m_skip1m, mom6m, cpi_chg_12p, hy_spread_chg_1p, hy_spread_z_12p, term_spread

- us_equity_4570e1c57a92: t_index, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, hy_spread, hy_spread_chg_1p, hy_spread_z_12p, term_spread

- us_equity_5fba6b4b201b: t_index, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, hy_spread_chg_1p, hy_spread_z_12p, term_spread

- us_equity_e7028305d052: t_index, cs_mom_rank, lag_y_excess_lead, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, hy_spread_chg_1p, hy_spread_z_12p, term_spread

- us_equity_46cc53338e30: t_index, mom12m_skip1m, mom6m, trend_slope, cpi_chg_1p, hy_spread_chg_1p, hy_spread_z_12p, term_spread

## international_equity

Complete configurations: 186; frontier: 8.

Compute-matched all-input control: international_equity_642db2c8937b; IC 0.019444, NLL -1.740618, Sharpe 1.171160 (27 non-time inputs).

| Configuration | Inputs | IC ± SD | NLL ± SD | Sharpe ± SD | Excess return | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| international_equity_f7933f4054ed | 16 | 0.261111 ± 0.045896 | -1.645865 ± 0.011067 | 1.474843 ± 0.032804 | 0.399672 | 0.043400 |
| international_equity_9b13070209b1 | 10 | 0.250000 ± 0.060093 | -1.841178 ± 0.008019 | 1.369501 ± 0.020184 | 0.371156 | 0.036021 |
| international_equity_3f07351cc740 | 17 | 0.236111 ± 0.045896 | -1.662995 ± 0.028695 | 1.430755 ± 0.117827 | 0.395038 | 0.042379 |
| international_equity_3c896548ff79 | 12 | 0.225000 ± 0.054645 | -1.842191 ± 0.026216 | 1.327043 ± 0.015460 | 0.356665 | 0.036102 |
| international_equity_66ef3bfb80d7 | 10 | 0.222222 ± 0.020972 | -1.842081 ± 0.020986 | 1.377475 ± 0.016879 | 0.377432 | 0.035897 |
| international_equity_41d789e7d4d7 | 11 | 0.188889 ± 0.012729 | -1.798295 ± 0.031376 | 1.380586 ± 0.018514 | 0.371021 | 0.036121 |
| international_equity_baaff7ca24fa | 18 | 0.186111 ± 0.019245 | -1.668138 ± 0.050677 | 1.428950 ± 0.079021 | 0.385762 | 0.042115 |
| international_equity_d0bc2acfc464 | 14 | 0.155556 ± 0.012729 | -1.689244 ± 0.029403 | 1.413513 ± 0.028494 | 0.376705 | 0.040667 |

### Exact input lists

- international_equity_f7933f4054ed: t_index, cs_mom_rank, mom12m_skip1m, mom6m, trend_slope, breakeven_proxy, copper_ret, cpi_chg_12p, cpi_chg_1p, dxy, em_fx_ret, gold_crude_ratio, oil_ret, schp_ret, term_spread, tnote10y, y10_real_proxy

- international_equity_9b13070209b1: t_index, cs_mom_rank, mom12m_skip1m, mom6m, trend_slope, dealer_gamma_proxy, dxy, em_fx_ret, oil_ret, vix, vix_ts_z_12p

- international_equity_3f07351cc740: t_index, cs_mom_rank, mom12m_skip1m, mom6m, trend_slope, breakeven_proxy, copper_ret, cpi_chg_12p, cpi_chg_1p, dxy, em_fx_ret, gold_crude_ratio, hy_spread_z_12p, oil_ret, schp_ret, term_spread, tnote10y, y10_real_proxy

- international_equity_3c896548ff79: t_index, cs_mom_rank, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, dealer_gamma_proxy, dxy, em_fx_ret, oil_ret, vix, vix_slope, vix_ts_z_12p

- international_equity_66ef3bfb80d7: t_index, cs_mom_rank, mom12m_skip1m, trend_slope, dealer_gamma_proxy, dxy, em_fx_ret, oil_ret, vix, vix_slope, vix_ts_z_12p

- international_equity_41d789e7d4d7: t_index, cs_mom_rank, mom12m_skip1m, mom6m, trend_slope, dealer_gamma_proxy, dxy, em_fx_ret, oil_ret, vix, vix_slope, vix_ts_z_12p

- international_equity_baaff7ca24fa: t_index, cs_mom_rank, mom12m_skip1m, mom6m, trend_slope, breakeven_proxy, copper_ret, cpi_chg_12p, cpi_chg_1p, credit_spread_chg_1p, dxy, em_fx_ret, gold_crude_ratio, hy_spread_z_12p, oil_ret, schp_ret, term_spread, tnote10y, y10_real_proxy

- international_equity_d0bc2acfc464: t_index, cs_mom_rank, mom12m_skip1m, mom6m, trend_slope, copper_ret, cpi_chg_12p, cpi_chg_1p, credit_spread_chg_1p, dxy, em_fx_ret, gold_crude_ratio, oil_ret, term_spread, tnote10y

## us_fixed_income

Complete configurations: 105; frontier: 6.

Compute-matched all-input control: us_fixed_income_2ad11ac7fb0d; IC 0.016667, NLL -2.519054, Sharpe 0.457650 (27 non-time inputs).

| Configuration | Inputs | IC ± SD | NLL ± SD | Sharpe ± SD | Excess return | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| us_fixed_income_78c395992b67 | 8 | 0.350000 ± 0.054645 | -2.609230 ± 0.010260 | 0.564963 ± 0.020114 | 0.047473 | 0.017321 |
| us_fixed_income_89293fc088d7 | 7 | 0.311111 ± 0.053576 | -2.639529 ± 0.017236 | 0.553264 ± 0.020856 | 0.046597 | 0.016982 |
| us_fixed_income_b4b6de59299e | 6 | 0.297222 ± 0.039382 | -2.649068 ± 0.020020 | 0.521521 ± 0.048523 | 0.043942 | 0.016705 |
| us_fixed_income_3199c073b636 | 7 | 0.255556 ± 0.053576 | -2.657009 ± 0.018331 | 0.512473 ± 0.032358 | 0.043132 | 0.016864 |
| us_fixed_income_c2fef69db5f6 | 5 | 0.250000 ± 0.022048 | -2.570066 ± 0.025355 | 0.597448 ± 0.044884 | 0.049558 | 0.015665 |
| us_fixed_income_4e8b02eef7f0 | 6 | 0.222222 ± 0.019245 | -2.576897 ± 0.009785 | 0.573954 ± 0.026156 | 0.047606 | 0.015937 |

### Exact input lists

- us_fixed_income_78c395992b67: t_index, chmom, lag_y_excess_lead, ma_signal, mom12m, oil_ret, schp_ret, y10_nominal, y10_real_proxy

- us_fixed_income_89293fc088d7: t_index, chmom, ma_signal, mom12m, oil_ret, schp_ret, y10_nominal, y10_real_proxy

- us_fixed_income_b4b6de59299e: t_index, ma_signal, mom12m, oil_ret, schp_ret, y10_nominal, y10_real_proxy

- us_fixed_income_3199c073b636: t_index, chmom, ma_signal, mom12m_skip1m, oil_ret, schp_ret, y10_nominal, y10_real_proxy

- us_fixed_income_c2fef69db5f6: t_index, lag_y_excess_lead, ma_signal, mom12m, mom36m, hy_spread_z_12p

- us_fixed_income_4e8b02eef7f0: t_index, lag_y_excess_lead, ma_signal, mom12m, mom36m, vol_1m, hy_spread_z_12p

## international_fixed_income

Complete configurations: 125; frontier: 5.

Compute-matched all-input control: international_fixed_income_0311693f134c; IC 0.055556, NLL -2.628034, Sharpe 0.834097 (27 non-time inputs).

| Configuration | Inputs | IC ± SD | NLL ± SD | Sharpe ± SD | Excess return | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| international_fixed_income_ade5ca6498cf | 11 | 0.437500 ± 0.055120 | -2.735421 ± 0.014059 | 0.858261 ± 0.006105 | 0.069029 | 0.015208 |
| international_fixed_income_12f65dc2d4e4 | 12 | 0.416667 ± 0.075116 | -2.716427 ± 0.017141 | 0.879944 ± 0.006293 | 0.070450 | 0.015129 |
| international_fixed_income_ed4f030d1f2e | 11 | 0.402778 ± 0.012028 | -2.739289 ± 0.020122 | 0.868075 ± 0.006927 | 0.069389 | 0.014982 |
| international_fixed_income_e9e05d05fc25 | 11 | 0.388889 ± 0.024056 | -2.752157 ± 0.021006 | 0.851889 ± 0.008843 | 0.068406 | 0.015554 |
| international_fixed_income_373dc9719ca5 | 11 | 0.361111 ± 0.012028 | -2.752397 ± 0.032424 | 0.860687 ± 0.008745 | 0.069066 | 0.014760 |

### Exact input lists

- international_fixed_income_ade5ca6498cf: t_index, mom12m, cpi_chg_12p, cpi_chg_1p, em_fx_ret, oil_ret, pct_above_50dma, spy_flow_z_12p, spy_ret, vix, vix_slope, vix_ts_z_12p

- international_fixed_income_12f65dc2d4e4: t_index, mom12m, mom36m, cpi_chg_12p, cpi_chg_1p, em_fx_ret, oil_ret, pct_above_50dma, spy_flow_z_12p, spy_ret, vix, vix_slope, vix_ts_z_12p

- international_fixed_income_ed4f030d1f2e: t_index, mom12m, mom36m, cpi_chg_12p, cpi_chg_1p, em_fx_ret, oil_ret, pct_above_50dma, spy_flow_z_12p, spy_ret, vix_slope, vix_ts_z_12p

- international_fixed_income_e9e05d05fc25: t_index, cs_mom_rank, mom12m, cpi_chg_12p, cpi_chg_1p, em_fx_ret, oil_ret, pct_above_50dma, spy_ret, vix, vix_slope, vix_ts_z_12p

- international_fixed_income_373dc9719ca5: t_index, mom12m, mom36m, cpi_chg_12p, cpi_chg_1p, em_fx_ret, oil_ret, pct_above_50dma, spy_ret, vix, vix_slope, vix_ts_z_12p

## global

Complete configurations: 25; frontier: 7.

Compute-matched all-input control: global_c55e49933d10; IC 0.108780, NLL -2.141413, Sharpe 1.127881 (27 non-time inputs).

| Configuration | Inputs | IC ± SD | NLL ± SD | Sharpe ± SD | Excess return | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| global_fdb5fb8b385f | 14 | 0.278026 ± 0.022007 | -2.232245 ± 0.011826 | 1.528323 ± 0.063292 | 0.468136 | 0.032805 |
| global_c0bc5df949cf | 13 | 0.275397 ± 0.015822 | -2.236822 ± 0.018692 | 1.448330 ± 0.134929 | 0.449255 | 0.032558 |
| global_9e7175a3ef4a | 12 | 0.274901 ± 0.012844 | -2.234172 ± 0.017214 | 1.544716 ± 0.170209 | 0.496940 | 0.032657 |
| global_a8a6b8394d6e | 13 | 0.272470 ± 0.013992 | -2.228913 ± 0.009264 | 1.567890 ± 0.085619 | 0.498365 | 0.032865 |
| global_33e57e1c2370 | 13 | 0.264881 ± 0.003324 | -2.250209 ± 0.002646 | 1.622642 ± 0.094196 | 0.513656 | 0.031601 |
| global_52d9b7e07091 | 15 | 0.258085 ± 0.009674 | -2.251246 ± 0.002668 | 1.526414 ± 0.067137 | 0.483350 | 0.031931 |
| global_da43a19fc3dd | 12 | 0.236359 ± 0.024827 | -2.262087 ± 0.016436 | 1.381369 ± 0.125196 | 0.401078 | 0.031365 |

### Exact input lists

- global_fdb5fb8b385f: t_index, mom12m, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, dealer_gamma_proxy, dxy, em_fx_ret, gold_crude_ratio, hy_spread, hy_spread_z_12p, oil_ret, term_spread, vix_ts_z_12p

- global_c0bc5df949cf: t_index, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, dealer_gamma_proxy, dxy, em_fx_ret, gold_crude_ratio, hy_spread_z_12p, oil_ret, term_spread, vix, vix_ts_z_12p

- global_9e7175a3ef4a: t_index, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, dealer_gamma_proxy, dxy, em_fx_ret, gold_crude_ratio, hy_spread_z_12p, oil_ret, term_spread, vix_ts_z_12p

- global_a8a6b8394d6e: t_index, mom12m, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, dealer_gamma_proxy, dxy, em_fx_ret, gold_crude_ratio, hy_spread_z_12p, oil_ret, term_spread, vix_ts_z_12p

- global_33e57e1c2370: t_index, lag_y_excess_lead, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, dealer_gamma_proxy, dxy, em_fx_ret, gold_crude_ratio, hy_spread_z_12p, oil_ret, term_spread, vix_ts_z_12p

- global_52d9b7e07091: t_index, lag_y_excess_lead, mom12m, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, dealer_gamma_proxy, dxy, em_fx_ret, gold_crude_ratio, hy_spread_z_12p, oil_ret, term_spread, vix, vix_ts_z_12p

- global_da43a19fc3dd: t_index, mom12m, mom12m_skip1m, mom6m, trend_slope, cpi_chg_12p, dxy, em_fx_ret, gold_crude_ratio, hy_spread_z_12p, oil_ret, term_spread, vix_ts_z_12p

