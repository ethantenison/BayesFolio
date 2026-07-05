# U.S. Equity Feature Horizon Validation
Validation rebuilds used live ETF-local and macro feature generation plus local cached return labels. `overnight_gap` is excluded.

## BME
- Artifact: `/Users/et/Documents/BayesFolio/experiments/2026-07-us-equity-etf-features/artifacts/us_equity_full_feature_candidates_20260705T135534Z_bme.parquet`
- Shape: `320` rows x `48` columns
- Dates: `2021-03-31` to `2026-06-30`
- Available requested features: `44`
- Missing requested features: `[]`
- Features with missing values: `{}`
- Constant features: `[]`
- HY spread unique counts: `{'hy_spread': 15, 'hy_spread_chg_1p': 16, 'hy_spread_z_12p': 16}`
- Perfect correlations >= 0.999999: `[]`
- High correlations >= 0.98: `[]`
- Contains `mom12m_skip1m`: `True`
- Contains `downside_vol_1m`: `True`
- Contains `overnight_gap`: `False`

## 3W-FRI
- Artifact: `/Users/et/Documents/BayesFolio/experiments/2026-07-us-equity-etf-features/artifacts/us_equity_full_feature_candidates_20260705T135547Z_3w_fri.parquet`
- Shape: `465` rows x `48` columns
- Dates: `2021-03-05` to `2026-06-19`
- Available requested features: `44`
- Missing requested features: `[]`
- Features with missing values: `{}`
- Constant features: `[]`
- HY spread unique counts: `{'hy_spread': 32, 'hy_spread_chg_1p': 38, 'hy_spread_z_12p': 41}`
- Perfect correlations >= 0.999999: `[]`
- High correlations >= 0.98: `[]`
- Contains `mom12m_skip1m`: `True`
- Contains `downside_vol_1m`: `True`
- Contains `overnight_gap`: `False`
