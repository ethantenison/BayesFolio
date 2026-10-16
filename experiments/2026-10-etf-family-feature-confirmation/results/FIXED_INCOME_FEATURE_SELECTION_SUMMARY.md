# Fixed-Income ETF Feature Search

This report consolidates the U.S. and international fixed-income searches across block screening,
targeted ablations, feature-level add/remove tests, combination refinement, and sparsity refinement.

The Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher conventional annualized
Sharpe. Cumulative return and RMSE are recorded only; turnover is excluded. All results use cached data,
seeds `27,127,227`, 24 common monthly scored windows, rank 2, maxiter 15, 256 scenarios, and `t_index`.

## U.S. fixed income

Universe: `BND, IEF, LQD, HYG`. Evaluated `31` unique configurations and `93` valid fits.

### Final Pareto front

| configuration                      |   mean_ic |   mean_gaussian_nll |   sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only |
|:-----------------------------------|----------:|--------------------:|---------:|------------------------------------:|-----------------------:|
| USFI-TREND                         |  0.147222 |           -2.673565 | 0.429514 |                            0.034860 |               0.015503 |
| USFI-COMPACT-NO-MOM12M             |  0.216667 |           -2.658920 | 0.429309 |                            0.035102 |               0.015927 |
| USFI-COMPACT-NO-CS-MOM-RANK        |  0.175000 |           -2.671665 | 0.454434 |                            0.037021 |               0.015595 |
| USFI-COMPACT+RET-AUTOCORR          |  0.213889 |           -2.627451 | 0.429903 |                            0.035265 |               0.015871 |
| USFI-COMPACT+VOL-Z                 |  0.230556 |           -2.632968 | 0.427207 |                            0.035214 |               0.015833 |
| USFI-COMPACT-NO-MOM12+RET-AUTOCORR |  0.222222 |           -2.653782 | 0.416432 |                            0.033936 |               0.015779 |
| USFI-COMPACT-NO-CS+RET-AUTOCORR    |  0.211111 |           -2.657938 | 0.458606 |                            0.037691 |               0.015742 |
| USFI-COMPACT-NO-MOM12-NO-CS        |  0.191667 |           -2.668034 | 0.422049 |                            0.034851 |               0.015789 |

### Hypervolume progression

| round                  |   unique_configurations |   hypervolume |
|:-----------------------|------------------------:|--------------:|
| block_screen           |                       7 |   0.173644663 |
| targeted_ablation      |                      14 |   0.242330509 |
| feature_refinement     |                      24 |   0.276936663 |
| combination_refinement |                      31 |   0.280640676 |

### Promotion set

- **primary — USFI-COMPACT-NO-CS+RET-AUTOCORR**: IC `0.211111`, NLL `-2.657938`, Sharpe `0.458606`; inputs: `t_index, lag_y_excess_lead, chmom, mom12m, ma_signal, ret_autocorr`.
- **challenger — USFI-COMPACT+VOL-Z**: IC `0.230556`, NLL `-2.632968`, Sharpe `0.427207`; inputs: `t_index, lag_y_excess_lead, chmom, mom12m, cs_mom_rank, ma_signal, vol_z`.
- **calibration_anchor — USFI-TREND**: IC `0.147222`, NLL `-2.673565`, Sharpe `0.429514`; inputs: `t_index, lag_y_excess_lead, chmom, mom12m, mom36m, cs_mom_rank, ma_signal`.

### Rate-aware lag challenge

A subsequent cached-only challenge added 10 bond-specific rate-state variables and evaluated
22 explicitly paired with-lag/no-lag configurations (66 fits) on the same 24 scored dates.
The rate source limits training to 62 complete months from 2021-03 through 2026-04, so its
raw objectives remain separate from the 2013-history front above.

| configuration                   |   mean_ic |   mean_gaussian_nll |   sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only |
|:--------------------------------|----------:|--------------------:|---------:|------------------------------------:|-----------------------:|
| USFI-RATE-BALANCED-NO-LAG       |  0.108333 |           -2.663281 | 0.432755 |                            0.034662 |               0.015804 |
| USFI-RATE-HIGH-IC-WITH-LAG      |  0.130556 |           -2.639765 | 0.380926 |                            0.030807 |               0.015576 |
| USFI-RATE-DURATION-CURVE-NO-LAG |  0.077778 |           -2.414905 | 0.454503 |                            0.038175 |               0.019623 |
| USFI-RATE-REAL-RATE-NO-LAG      |  0.130556 |           -2.624058 | 0.433566 |                            0.035924 |               0.016206 |
| USFI-RATE-BREAKEVEN-NO-LAG      |  0.133333 |           -2.451504 | 0.449518 |                            0.036263 |               0.018858 |
| USFI-RATE-CURVE-RATE-VOL-NO-LAG |  0.141667 |           -2.483645 | 0.362938 |                            0.029789 |               0.017398 |

Five of the six rate-aware Pareto points omit `lag_y_excess_lead`.

- **balanced_no_lag_challenger — USFI-RATE-REAL-RATE-NO-LAG**: IC `0.130556`, NLL `-2.624058`, Sharpe `0.433566`; inputs: `t_index, chmom, mom12m, ma_signal, ret_autocorr, y10_nominal, schp_ret, y10_real_proxy`.
- **no_lag_calibration_anchor — USFI-RATE-BALANCED-NO-LAG**: IC `0.108333`, NLL `-2.663281`, Sharpe `0.432755`; inputs: `t_index, chmom, mom12m, ma_signal, ret_autocorr`.

The real-rate no-lag model beat its exact with-lag pair on IC and Sharpe in all three seeds
and on NLL in two of three seeds. The balanced no-lag model beat its exact with-lag pair on
NLL and Sharpe in all three seeds, with only a `0.005556` aggregate IC reduction. The prior
lag requirement was therefore conditional on the original feature/training regime, not a
stable U.S. fixed-income property. See `US_FIXED_INCOME_RATE_NO_LAG_RESULTS.md` for details.

## International fixed income

Universe: `BNDX, VWOB, HYEM`. Evaluated `87` unique configurations and `261` valid fits.

### Final Pareto front

| configuration                                        |   mean_ic |   mean_gaussian_nll |   sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only |
|:-----------------------------------------------------|----------:|--------------------:|---------:|------------------------------------:|-----------------------:|
| INTFI-TREND+COMMODITY-RET                            | -0.083333 |           -2.710271 | 0.819405 |                            0.065428 |               0.015601 |
| INTFI-TREND+GLOBAL                                   |  0.229167 |           -2.649867 | 0.849766 |                            0.068428 |               0.013867 |
| INTFI-PRIMARY-NO-LAG-NO-ERP                          |  0.430556 |           -2.607603 | 0.874801 |                            0.070008 |               0.015199 |
| INTFI-PRIMARY-NO-LAG-Y-EXCESS-LEAD                   |  0.402778 |           -2.614493 | 0.877914 |                            0.070189 |               0.015585 |
| INTFI-PRIMARY-NO-LAG-NO-MA-SIGNAL                    |  0.395833 |           -2.623165 | 0.879086 |                            0.070326 |               0.015358 |
| INTFI-PRIMARY-NO-LAG-NO-ERP-NO-CPI-MOM-NO-GOLD-RATIO |  0.319444 |           -2.607666 | 0.885193 |                            0.070665 |               0.015074 |
| INTFI-TREND+MARKET-RISK                              |  0.159722 |           -2.688399 | 0.847268 |                            0.067871 |               0.015314 |
| INTFI-PRIMARY-NO-LAG-NO-MA-NO-GOLD                   |  0.402778 |           -2.632185 | 0.874315 |                            0.070082 |               0.015078 |
| INTFI-PRIMARY-NO-LAG-NO-CPI-NO-GOLD                  |  0.388889 |           -2.633392 | 0.883191 |                            0.070512 |               0.015134 |
| INTFI-PRIMARY-NO-LAG-NO-MA-NO-ERP                    |  0.409722 |           -2.618535 | 0.871866 |                            0.069719 |               0.015061 |

### Hypervolume progression

| round                  |   unique_configurations |   hypervolume |
|:-----------------------|------------------------:|--------------:|
| block_screen           |                       7 |   0.558858315 |
| targeted_ablation      |                      15 |   0.558861053 |
| feature_refinement     |                      36 |   0.698328598 |
| combination_refinement |                      48 |   0.894649816 |
| sparsity_ablation      |                      68 |   0.936761119 |
| sparsity_combination   |                      80 |   1.008291011 |
| sparsity_final         |                      87 |   1.010109927 |

### Promotion set

- **primary — INTFI-PRIMARY-NO-LAG-NO-CPI-NO-GOLD**: IC `0.388889`, NLL `-2.633392`, Sharpe `0.883191`; inputs: `t_index, chmom, mom12m, mom36m, cs_mom_rank, ma_signal, hy_spread, hy_spread_z_12m, vix_slope, vix_ts_z_12m, vix, spy_flow_z_12m, spy_ret, erp, cpi_yoy, oil_ret, pct_above_50dma, em_fx_ret`.
- **high_ic_challenger — INTFI-PRIMARY-NO-LAG-NO-ERP**: IC `0.430556`, NLL `-2.607603`, Sharpe `0.874801`; inputs: `t_index, chmom, mom12m, mom36m, cs_mom_rank, ma_signal, hy_spread, hy_spread_z_12m, vix_slope, vix_ts_z_12m, vix, spy_flow_z_12m, spy_ret, cpi_yoy, cpi_mom, oil_ret, gold_crude_ratio, pct_above_50dma, em_fx_ret`.
- **sparse_sharpe_challenger — INTFI-PRIMARY-NO-LAG-NO-ERP-NO-CPI-MOM-NO-GOLD-RATIO**: IC `0.319444`, NLL `-2.607666`, Sharpe `0.885193`; inputs: `t_index, chmom, mom12m, mom36m, cs_mom_rank, ma_signal, hy_spread, hy_spread_z_12m, vix_slope, vix_ts_z_12m, vix, spy_flow_z_12m, spy_ret, cpi_yoy, oil_ret, pct_above_50dma, em_fx_ret`.
- **calibration_anchor — INTFI-TREND+MARKET-RISK**: IC `0.159722`, NLL `-2.688399`, Sharpe `0.847268`; inputs: `t_index, lag_y_excess_lead, chmom, mom12m, mom36m, cs_mom_rank, ma_signal, vix_slope, vix_ts_z_12m, vix, spy_flow_z_12m, spy_ret, erp, pct_above_50dma`.

## Findings and recommendation

- The long-history U.S. fixed-income search favored local time-series features. Generic credit/inflation
  and broad macro bundles were noise at this budget. The paired rate-aware challenge later showed that
  `lag_y_excess_lead` is not generally required: sparse no-lag local and real-rate models occupied five
  of six Pareto positions.
- International fixed income required macro context, but the broad model was over-specified. The final
  primary removes `hy_spread_chg_1m`, `copper_ret`, `lag_y_excess_lead`, `cpi_mom`, and
  `gold_crude_ratio` while retaining `oil_ret`; it dominates the former 21-input primary.
- `INTFI-TREND+COMMODITY-RET` remains mathematically nondominated only because it has the best NLL;
  its negative IC means it contributes no hypervolume above the fixed zero reference and is not promoted.
- Promote each listed primary and challenger to longer-history/full-budget confirmation.
  Keep the calibration anchor in that confirmation so selection remains genuinely multi-objective.
  Do not replace production inputs from this screen alone.

## Validation and limitations

- Every reported configuration passed exact universe, date, seed, input-column, and model-control checks.
- The search is conditional on the cached 2013-08 through 2026-04 data and the 24 scored months from
  2024-05 through 2026-04. It is evidence-directed rather than exhaustive over every feature subset.
- The rate-aware challenge uses the same scored months but a 2021-03 through 2026-04 complete training
  panel. Its paired comparisons are valid, but its raw metrics are not pooled into the long-history
  front.
- The next confirmation should preserve a common date panel and use the production optimization budget
  before any production promotion decision.
