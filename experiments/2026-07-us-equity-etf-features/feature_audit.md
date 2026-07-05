# U.S. Equity ETF Feature Audit

## Current Monthly Selected Features

The current monthly automation config selects:

ETF-local:

- `lag_y_excess_lead`
- `baspread`
- `ret_kurt`
- `chmom`
- `mom12m`
- `mom36m`
- `cs_mom_rank`
- `max_dd_6m`
- `ma_signal`
- `ret_autocorr`
- `vol_z`

Macro/global:

- `hy_spread`
- `hy_spread_chg_1p`
- `hy_spread_z_12p`
- `vix_slope`
- `vix_ts_z_12p`
- `vix`
- `spy_flow_z_12p`
- `spy_ret`
- `erp`
- `cpi_chg_12p`
- `cpi_chg_1p`
- `copper_ret`
- `oil_ret`
- `gold_crude_ratio`
- `pct_above_50dma`
- `em_fx_ret`

## Available ETF-Local Features

BayesFolio can generate more ETF-local features than the current monthly selection:

- price/return: `price`, `log_ret`, `mom1m`, `mom6m`, `mom12m`, `mom12m_skip1m`, `mom36m`, `chmom`, `short_reversal_1m`
- liquidity/flow proxy: `volume`, `dolvol_log`, `turnover`, `sd_turn`, `ill_log`, `baspread`
- volatility/tail: `vol_1w`, `vol_1m`, `vol_3m`, `vol_of_vol`, `vol_z`, `vol_accel`, `vol_ratio_1m_3m`, `downside_vol_1m`, `ret_skew`, `ret_kurt`
- trend/drawdown: `ma_1m`, `ma_3m`, `ma_signal`, `ma_regime`, `trend_slope`, `max_dd_3m`, `max_dd_6m`
- dependence: `ret_autocorr`, `vol_autocorr`
- engineered panel features: `cs_mom_rank`, `lag_y_excess_lead`, `lag2_y_excess_lead`, `t_index`

## Expansion Candidates For U.S. Equity Family

High-priority expansion candidates:

- `mom12m_skip1m`: canonical 12-1 equity momentum definition; avoids mixing the short-term reversal month into long momentum.
- `mom6m`: intermediate trend; likely less stale than `mom36m` for a five-ETF family.
- `short_reversal_1m`: one-month reversal/overextension proxy; useful as a separate block from long momentum.
- `vol_1m`, `vol_3m`, `vol_ratio_1m_3m`, `downside_vol_1m`: volatility regime and downside risk are plausible near-horizon equity signals.
- `max_dd_3m`: shorter drawdown state than the current `max_dd_6m`.
- `trend_slope` and `ma_regime`: cleaner directional trend state than `ma_signal` alone.
- `ill_log`, `dolvol_log`, `turnover`, `sd_turn`: ETF liquidity/flow-pressure proxies when true creations/redemptions are unavailable.

Medium-priority expansion candidates:

- `ret_skew`, `ret_kurt`: tail-shape signals may help in risk regimes but are noisy in small samples.
- `vol_autocorr`: plausible regime persistence feature but likely redundant with volatility levels.

## Removal Or Exclusion Candidates

Exclude from the first U.S. equity family pass:

- `em_fx_ret`: aimed at emerging/international risk, not U.S. equity style/size ranking.
- `de10y`, `jp10y`, `uk10y`, `cn10y` if present in the macro artifact: global sovereign yield changes are more relevant for international/fixed-income sleeves.
- `gold_crude_ratio`, `copper_ret`, `oil_ret`: commodity/risk-cycle proxies may matter for equities, but they are indirect and should be held for a later macro-regime block, not the first compact feature set.
- raw `price`, `volume`, `ma_1m`, `ma_3m`, `dolvol` and `ill`: prefer return/ratio/log-transformed versions to avoid level and scale artifacts across ETF tasks.
- `overnight_gap`: excluded by decision. The rebuilt correlation audit showed it is almost redundant with lagged target-return information after monthly aggregation/alignment.

Keep but isolate in macro/risk block:

- `hy_spread`, `hy_spread_chg_1p`, `hy_spread_z_12p`: credit stress can matter for U.S. equities, but it is not an ETF-local style/size signal.
- `vix`, `vix_slope`, `vix_ts_z_12p`: strong risk-regime candidates; test after ETF-local blocks.
- `spy_ret`, `erp`, `pct_above_50dma`, `spy_flow_z_12p`: broad-market state/breadth features; useful but shared across all tasks, so they cannot create cross-sectional dispersion by themselves unless interacted with ETF-local/task structure.

## Recommended First Feature Sets

### `us_equity_momentum_trend`

- `lag_y_excess_lead`
- `lag2_y_excess_lead`
- `mom12m_skip1m`
- `mom6m`
- `short_reversal_1m`
- `cs_mom_rank`
- `trend_slope`
- `ma_signal`
- `ma_regime`

### `us_equity_vol_drawdown`

- `vol_1m`
- `vol_3m`
- `vol_ratio_1m_3m`
- `downside_vol_1m`
- `vol_of_vol`
- `max_dd_3m`
- `max_dd_6m`
- `ret_skew`
- `ret_kurt`

### `us_equity_liquidity_flow_proxy`

- `ill_log`
- `dolvol_log`
- `turnover`
- `sd_turn`
- `baspread`
- `vol_z`
- `vol_accel`

### `us_equity_macro_risk`

- `vix`
- `vix_slope`
- `vix_ts_z_12p`
- `hy_spread`
- `hy_spread_chg_1p`
- `hy_spread_z_12p`
- `spy_ret`
- `erp`
- `pct_above_50dma`
- `spy_flow_z_12p`

### `us_equity_compact_combined`

- `lag_y_excess_lead`
- `lag2_y_excess_lead`
- `mom12m_skip1m`
- `mom6m`
- `short_reversal_1m`
- `cs_mom_rank`
- `trend_slope`
- `vol_1m`
- `vol_ratio_1m_3m`
- `downside_vol_1m`
- `max_dd_3m`
- `ill_log`
- `dolvol_log`
- `vix`
- `vix_slope`
- `hy_spread_z_12p`
- `spy_ret`
- `pct_above_50dma`

## Notes

The removal phase should not hard-delete features from BayesFolio. It should produce U.S. equity experiment configs that exclude weak or mismatched predictors, then let diagnostics decide whether to reintroduce them.
