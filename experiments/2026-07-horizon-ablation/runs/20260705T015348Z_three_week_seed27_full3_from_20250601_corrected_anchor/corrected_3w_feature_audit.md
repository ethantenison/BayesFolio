# Corrected 3-week Feature Audit

- Feature artifact: `/Users/et/.bayesfolio/artifacts/features/horizon_ablation_20260703/portfolio_etf_macro_features_20260705T015348Z_three_week_seed27_full3_from_20250601_corrected_anchor.parquet`
- Feature SHA256: `f9d84e5cee7b67b0c43cf076741d460b9df59857d990db3324771cfa7cf5529c`
- Rows: 1,980 (110 dates x 18 assets)
- Date grid: 2020-03-06 to 2026-06-12, gaps {21: 109}
- Scored windows: 18 from 2025-06-20 to 2026-06-12, gaps {21: 17}
- Checks: 26/26 passed

## Check Results

| check | result | detail |
|---|---:|---|
| feature dates are a single 21-day grid | PASS | {21: 109} |
| feature dates match expected 2018-03-01 anchored 3W-FRI grid | PASS | 110 dates; first=2020-03-06, last=2026-06-12 |
| scored dates are subset of feature dates | PASS | 18 scored dates; gaps={21: 17} |
| no duplicate feature date/asset rows | PASS | duplicates=0 |
| constant asset coverage by date | PASS | 18 assets; counts={18: 110} |
| t_index increments once per 21-day feature date | PASS | t_index range=0..109 |
| returns raw cache has unique keys | PASS | duplicates=0 |
| returns raw cache still contains historical interleaved anchors | PASS | raw gaps={7: 129, 14: 128, 21: 17} |
| ETF raw cache has unique keys | PASS | duplicates=0 |
| ETF raw cache still contains historical interleaved anchors | PASS | raw gaps={7: 129, 14: 128, 21: 17} |
| macro raw cache has unique keys | PASS | duplicates=0 |
| macro raw cache still contains historical interleaved anchors | PASS | raw gaps={7: 129, 14: 128, 21: 17} |
| returns filtered provider grid is single 21-day cadence | PASS | {21: 144} |
| ETF filtered provider grid is single 21-day cadence | PASS | {21: 144} |
| macro filtered provider grid is single 21-day cadence | PASS | {21: 144} |
| target y_excess_lead matches filtered 3W returns cache | PASS | missing=0; bad=0; max_abs_diff=0 |
| lag_y_excess_lead equals prior 21-day target | PASS | bad=0; max_abs_diff=0 |
| lag2_y_excess_lead equals two-prior 21-day target | PASS | bad=0; max_abs_diff=0 |
| target lags have no future overlap on scored windows | PASS | lag1=0/18, lag2=0/18 |
| all ETF feature columns equal prior-anchor filtered cache values | PASS | {'baspread': {'bad': 0, 'max_abs_diff': 0.0}, 'ret_kurt': {'bad': 0, 'max_abs_diff': 0.0}, 'chmom': {'bad': 0, 'max_abs_diff': 0.0}, 'mom12m': {'bad': 0, 'max_abs_diff': 0.0}, 'mom36m': {'bad': 0, 'max_abs_diff': 0.0}, 'cs_mom_rank': {'bad': 0, 'max_abs_diff': 0.0}, 'max_dd_6m': {'bad': 0, 'max_abs_diff': 0.0}, 'ma_signal': {'bad': 0, 'max_abs_diff': 0.0}, 'ret_autocorr': {'bad': 0, 'max_abs_diff': 0.0}, 'vol_z': {'bad': 0, 'max_abs_diff': 0.0}} |
| all macro feature columns equal prior-anchor filtered cache values | PASS | {'hy_spread': {'bad': 0, 'max_abs_diff': 0.0}, 'hy_spread_chg_1p': {'bad': 0, 'max_abs_diff': 0.0}, 'hy_spread_z_12p': {'bad': 0, 'max_abs_diff': 0.0}, 'vix_slope': {'bad': 0, 'max_abs_diff': 0.0}, 'vix_ts_z_12p': {'bad': 0, 'max_abs_diff': 0.0}, 'vix': {'bad': 0, 'max_abs_diff': 0.0}, 'spy_flow_z_12p': {'bad': 0, 'max_abs_diff': 0.0}, 'spy_ret': {'bad': 0, 'max_abs_diff': 0.0}, 'erp': {'bad': 0, 'max_abs_diff': 0.0}, 'cpi_chg_12p': {'bad': 0, 'max_abs_diff': 0.0}, 'cpi_chg_1p': {'bad': 0, 'm... |
| macro values are date-level constants across assets | PASS | max_nunique_by_date=1 |
| no infinities in model input/target columns | PASS | inf_cols=[] |
| scored windows have complete model columns | PASS | nonzero_na={} |
| window_training_history file exists | PASS | columns=['date', 'train_months', 'train_rows', 'train_date_min', 'train_date_max', 'unique_train_dates'] rows=18 |
| prediction rows cover scored date x asset grid | PASS | rows=324 expected=324; assets=18 |

## Interpretation

- The raw cache files still contain old interleaved anchors; this is okay only because providers now filter returned cache rows to the requested `3W-FRI` grid.
- The final corrected artifact reconstructs exactly from the filtered 21-day provider grid.
- ETF and macro predictors in the final artifact are one anchor behind the target date, so same-date and future feature rows are not used.
- Target lags are previous and two-previous 21-day labels by asset, with no future overlap on scored windows.
- Selected ETF rolling features still use fixed trading-day lookbacks such as 21/63/126/252 days; the horizon changes the sampling/anchor, not those semantic lookback definitions.
