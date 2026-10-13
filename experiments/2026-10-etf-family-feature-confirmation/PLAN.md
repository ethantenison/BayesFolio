# ETF Family Feature Confirmation Plan

Status: U.S. Phase 0 and the four seed-27 U.S. pilots complete; broader execution is not authorized by this document.

## 1. Decisions and success criteria

- U.S. universe: `SPY`, `MGK`, `VTV`, `IWM`. `IJR` is excluded.
- International universe: `VEA`, `VWO`, `VSS`, `EWX`.
- `t_index` is present in every candidate and comparator. There is no time-input ablation.
- Use cached local data only. Do not refresh market or macro data.
- Candidate selection is Pareto-only over three objectives: maximize IC, minimize Gaussian NLL, and maximize Sharpe.
- Cumulative return and RMSE are recorded diagnostics only. They cannot dominate, rank, reject, or promote a candidate.
- Turnover is excluded from confirmation evaluation tables and selection logic.
- Discovery and confirmation remain separate. July results generate candidates; they are not pooled with confirmation results.
- Production integration, a combined-portfolio rerun, and deployment remain out of scope.

The recommended primary all-input comparator is **compute matched**: the production feature set and covariance structure,
but rank `2`, `maxiter=15`, `256` posterior scenarios, and the same seed/date panel as the family candidates. The actual
production-budget configuration (rank `5`, `maxiter=75`, `5000` scenarios) is not part of the Pareto comparison. It may
be run later as a separately labelled operational reference after the compute-matched comparison is complete. This
choice requires human review before execution.

## 2. Evidence inventory and comparability

### Verified U.S. candidate

The four-task `u005 AC` definition comes from:

`experiments/2026-07-us-equity-etf-features/runs/macro_split_kernel_r018_threeway_remaining_14_full_20260707T0023Z/full_runs/full_u005_r018_threeway_subset_AC/manifest.json`

- Feature artifact: `experiments/2026-07-us-equity-etf-features/artifacts/us_equity_first_round_feature_candidates_20260705T145015Z.parquet`
- Target: `y_excess_lead`.
- Inputs: `t_index`; ETF block `mom6m,mom12m_skip1m,cs_mom_rank,trend_slope`; macro block
  `vix,vix_slope,spy_ret,pct_above_50dma,hy_spread_chg_1p,hy_spread_z_12p,cpi_chg_12p,term_spread`.
- Task covariance: `signed_lkj_eta_2`, rank `2`.
- Kernel: main effects and pairwise interactions; time x ETF x macro-market; ETF x macro-market x
  macro-credit/inflation; macro-market RQ + Linear ARD; macro-credit/inflation Matern(0.5) ARD.
- Budget: `maxiter=15`, `256` scenarios, seed `27`, no time modulation.
- Historical panel: data `2021-03-31` through `2026-06-30`; 12 scored windows from `2025-07-31` through
  `2026-06-30`.

Historical compact comparators are retained as ablations, not as winners:

- `U-C005`: `momentum_plus_macro`, default all-macro Matern + Linear, signed LKJ, no time modulation, main effects plus
  pairwise interactions.
- `U-C009`: the same feature/kernel family with `--interactions-only`.

### Verified international candidate

The corrected `d005` breadth/flow definition comes from:

`experiments/2026-07-international-equity-etf-features/runs/international_equity_dynamic_feature_screen_20260708T2115Z/full_runs/full_d005_090fdab2/manifest.json`

- Feature artifact: `experiments/2026-07-us-equity-etf-features/artifacts/international_equity_custom_feature_candidates_20260707T182133Z.parquet`.
- Target: `y_excess_lead`.
- Inputs: `t_index`; ETF block `mom6m,mom12m_skip1m,cs_mom_rank,trend_slope`; macro block
  `spy_ret,erp,rsp_spy_roc_1p,pct_above_50dma,spy_flow_z_12p`.
- Task covariance: `signed_lkj_eta_2`, rank `2`.
- Kernel: main effects and pairwise interactions, default grouping and component proposal.
- Budget and historical panel: `maxiter=15`, `256` scenarios, seed `27`, no time modulation, and the same 12-window
  date range as the U.S. evidence.

The earlier international hard-coded `split_groups` screen is invalid evidence against international features. The
XGBoost/Optuna/SHAP study is hypothesis generation only; it does not establish GP importance.

### Non-comparable evidence

Do not merge July results into confirmation aggregates because they use one seed and 12 selected windows. Do not put
the rank-5/maxiter-75/5000-scenario production configuration on the same Pareto frontier as rank-2 confirmation runs.
Scout screens, invalid `split_groups` runs, runs without the same four-task universe, and runs with different scored
dates are descriptive evidence only.

## 3. Common evaluation protocol

1. Use the frozen 24-date U.S. panel in `artifacts/us_scored_dates_24.txt`, covering `2024-06-28` through `2026-05-29`.
   The one-month shift from the provisional panel is required because the cached production inputs end on
   `2026-05-29`. Every seed-27 pilot used this identical panel and training history.
2. Use seeds `27`, `127`, and `227`.
3. Hold target, task universe, dates, expanding-window construction, optimizer budget, portfolio optimizer settings,
   and scenario count constant within each fair comparison.
4. Preserve one resolved manifest per run, including exact input columns, covariance composition, interactions, task
   covariance, time handling, seed, dates, and budget.
5. Measure IC by scored date; Gaussian NLL by scored date and ETF; and conventional annualized Sharpe over the common
   portfolio-return path for each seed. Compute Sharpe as mean monthly return divided by the sample standard deviation
   of monthly returns, multiplied by `sqrt(12)`. These are `y_excess_lead` returns already adjusted for the integrated
   risk-free return, so do not subtract a second risk-free series. Do not use the runner's legacy CAGR/volatility field
   and do not manufacture a one-observation per-window Sharpe. Report means, medians, dispersion, seed ranges, and
   available ETF-level breakdowns.
6. Aggregate each configuration to `(mean IC, mean Gaussian NLL, mean Sharpe)` over the common panel and seed set.
   Candidate A dominates B only if it is no worse on all three objectives and strictly better on at least one. Retain
   every nondominated configuration. Use no weighted score or hidden tie-breaker.
7. Record cumulative return and RMSE next to the objective results with an explicit `diagnostic_only=true` label.
   Exclude turnover columns from the evaluation output.

The wrapper already computes `mean_gaussian_nll` in `forecast_metrics_by_asset.csv` and
`forecast_metrics_overall.csv`. Before execution, add the smallest experiment-scoped extension needed to write NLL by
`date` and by `(date, asset_id)` from `gp_window_predictions.csv`; do not alter GP fitting. Existing
`gp_window_ic.csv`, `portfolio_returns.csv`, and prediction artifacts remain the source inputs for the aggregate.

## 4. Run matrix

| Phase | Family | Configurations | Seeds | Windows | Runs | GP window fits |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0a, complete | U.S. | cache/schema/date preflight only | - | 24 frozen | 0 | 0 |
| 0b, pending | international | cache/schema/date preflight only | - | 24 proposed | 0 | 0 |
| 1 | U.S. | `U-AC`, `U-C005`, `U-C009`, `U-ALL-MATCH` | 3 | 24 | 12 | 288 |
| 2 | international | `I-BREADTH`, `I-MARKET`, `I-MOM`, `I-ALL-MATCH` | 3 | 24 | 12 | 288 |
| 3a | international | four additive targeted ablations | 3 | 24 | 12 | 288 |
| 3b, gated | international | at most two preregistered interactions | 3 | 24 | 0-6 | 0-144 |

Core equity confirmation is 36 runs and 864 GP window fits; including two interaction candidates makes it 42 runs and
1008 fits. A 24-window run is approximately two historical 12-window run units. No reliable wall-clock duration was
stored with the reviewed artifacts, so Phase 1 begins with one seed-27 pilot. If that pilot takes `T24`, budget roughly
`36 * T24` serial time for the core program and revise using observed fit and scenario-generation timing. Do not infer a
wall-clock promise from scenario count alone.

## 5. Execution-ready U.S. confirmation

After Phase 0 writes and verifies the common date file, define:

```bash
RUNNER="experiments/2026-07-us-equity-etf-features/run_first_round_walkforward.py"
US_FEATURES="experiments/2026-10-etf-family-feature-confirmation/artifacts/us_confirmation_cached_inputs_20261012.parquet"
US_DATES="experiments/2026-10-etf-family-feature-confirmation/artifacts/us_scored_dates_24.txt"
OUT="experiments/2026-10-etf-family-feature-confirmation/runs"
SEED=27
```

`U-AC` command template (repeat for `27`, `127`, `227`):

```bash
uv run --locked python "$RUNNER" \
  --feature-path "$US_FEATURES" --output-root "$OUT" --run-id "us_u005_ac_s${SEED}" \
  --tickers SPY,MGK,VTV,IWM --mtgp-rank 2 --scored-date-file "$US_DATES" \
  --maxiter 15 --seed "$SEED" --posterior-scenarios 256 --resume-scenarios \
  --gp-experiment signed_lkj_eta_2 --time-modulation-mode neither \
  --upperlng 0.35 --nea 3 --gp-riskfolio-rm CVaR --gp-riskfolio-obj MaxRet \
  --feature-set momentum_plus_macro \
  --custom-etf-cols mom6m,mom12m_skip1m,cs_mom_rank,trend_slope \
  --custom-macro-cols vix,vix_slope,spy_ret,pct_above_50dma,hy_spread_chg_1p,hy_spread_z_12p,cpi_chg_12p,term_spread \
  --covar-grouping-proposal macro_split_custom_pending \
  --macro-market-kernel-id rq_lin_ard_on --macro-market-ard true \
  --macro-credit-inflation-kernel-id m05_ard_on --macro-credit-inflation-ard true \
  --include-time-etf-market-three-way --include-macro-market-credit-interaction \
  --include-macro-market-credit-etf-three-way
```

Use the same command skeleton for the comparators:

- `U-C005`: remove the three custom split-macro flags and use
  `--covar-grouping-proposal default --macro-component-proposal matern_linear`; retain main effects and pairwise terms.
- `U-C009`: use the `U-C005` command plus `--interactions-only`.
- `U-ALL-MATCH`: use the verified all-input artifact after Phase 0, pass the 11 ETF and 16 macro columns listed below,
  retain rank `2`, `maxiter=15`, and `256` scenarios, and use
  `--kernel-composition-proposal e_plus_m_plus_t_plus_et_plus_mt_plus_emt --time-modulation-mode lengthscale_only`.

All-input ETF columns:

`lag_y_excess_lead,baspread,ret_kurt,chmom,mom12m,mom36m,cs_mom_rank,max_dd_6m,ma_signal,ret_autocorr,vol_z`

All-input macro columns:

`hy_spread,hy_spread_chg_1p,hy_spread_z_12p,vix_slope,vix_ts_z_12p,vix,spy_flow_z_12p,spy_ret,erp,cpi_chg_12p,cpi_chg_1p,copper_ret,oil_ret,gold_crude_ratio,pct_above_50dma,em_fx_ret`

Phase 1 stops after the seed-27 pilot until manifests prove identical dates and budgets, all four tasks are present,
`t_index` is present, `IJR` is absent, and per-date NLL is available.

The seed-27 stop gate passed on 2026-10-12. All four pilots completed 24 windows with identical dates and budgets.
Their manifests contain the four approved tasks, `t_index`, and no `IJR`; per-date and per-date/ETF NLL outputs are
present. All four are nondominated at seed 27, so no promotion decision is made before the remaining seed panel.

## 6. International confirmation and targeted ablations

Define `INTL_FEATURES` as the verified international artifact and use the same `RUNNER`, `OUT`, date-file, budget, seed,
and optimizer settings as Phase 1.

- `I-BREADTH`: the verified `d005` command with `--tickers VEA,VWO,VSS,EWX`, the four momentum/trend ETF columns,
  `--custom-macro-cols spy_ret,erp,rsp_spy_roc_1p,pct_above_50dma,spy_flow_z_12p`,
  `--feature-set momentum_plus_macro_market --covar-grouping-proposal default --macro-component-proposal default`.
- `I-MARKET`: `--feature-set momentum_plus_macro_market` with its native market macro block and no custom macro
  override.
- `I-MOM`: `--feature-set momentum_trend`.
- `I-ALL-MATCH`: the same compute-matched production-feature comparator as Phase 1 on the international task universe.

Phase 3a adds each coherent block to `I-BREADTH`, one at a time:

1. FX: `dxy,em_fx_ret`;
2. inflation: `breakeven_proxy`;
3. credit: `hy_spread`;
4. commodities: `oil_ret,gold_crude_ratio`.

The breadth/flow base from Phase 2 is reused, not rerun. Phase 3b may add no more than two interaction candidates after
their exact covariance blocks and economic hypotheses are written in a preregistration note. Do not launch another
combinatorial search. The wrapper currently exposes block-level rather than individual-feature interaction switches;
if those two candidates cannot be expressed exactly, add a small experiment-local covariance proposal and test its
manifest serialization before running it.

## 7. Time input and remaining ETF families

`t_index` stays fixed in every phase. A time ablation is deferred to a separate experiment and is not a blocker here.

After equity confirmation, plan these families sequentially:

1. investment-grade bonds and Treasuries;
2. high-yield and emerging-market debt;
3. REITs;
4. commodities.

For each family, first select a nonredundant task universe, then audit cached coverage, target correlation, feature
correlation, missingness, and lookbacks. A default planning envelope is five configurations (compact baseline, three
small economic blocks, compute-matched all-input) x three seeds x 24 windows = 15 runs and 360 fits per family, or 60
runs and 1440 fits across all four. These are estimates, not authorization. Family universes and economic blocks require
explicit review before their manifests are frozen; equity features must not be copied without justification.

## 8. Minimal changes and data gates

Before Phase 1:

- Add experiment-local aggregation for NLL by date and `(date, asset_id)`; NLL already exists overall and by asset.
- Add a deterministic Pareto report that reads completed artifacts, verifies common panels, emits all nondominated
  candidates on IC/NLL/Sharpe, labels return/RMSE as diagnostic only, and emits no turnover field.
- If the cached family artifact does not contain every all-input column, build a new versioned artifact from existing
  local caches only. Do not overwrite July artifacts.
- Confirm the 36-month `mom36m` lookback is complete before the first U.S. score date. This passed for `2024-06-28`.
- Confirm cached availability through `2026-06-30` for `term_spread`, `rsp_spy_roc_1p`, and every targeted international
  series. A missing series blocks only its dependent candidate; do not fetch it externally.
- Run `--preflight-only` for every distinct feature schema and compare the scored-date and training-row output before
  fitting.

Stop and request a decision if the 24-month common panel is unavailable, a candidate needs an external refresh, more
than 10% of pilot windows fail, a repeated fit failure survives one clean retry, a configuration cannot serialize its
exact columns/kernel into the manifest, or the two all-input comparator choices remain unresolved.

## 9. Verification

Planning and future implementation checks:

```bash
uv run --locked ruff check experiments/2026-07-us-equity-etf-features/run_first_round_walkforward.py
uv run --locked python experiments/2026-07-us-equity-etf-features/run_first_round_walkforward.py --help
git diff --check -- experiments/2026-10-etf-family-feature-confirmation
git status --short
```

For every completed run, validate manifest task/input lists, `t_index`, absence of `IJR`, exact seed and dates, budget,
and covariance flags. Compare date hashes across candidates. Assert objective tables contain IC, NLL, and Sharpe; mark
return and RMSE record-only; assert turnover is absent; recompute Pareto membership independently from the saved table.

## 10. Approval gate

U.S. Phase 0 and the four seed-27 U.S. pilots were explicitly authorized and are complete. The next gate is human approval
for U.S. seeds `127` and `227`. International runs, later-family work, and the production-budget reference remain
unauthorized.

## 11. Sequential hypervolume-improvement turn

After all 12 Phase-1 U.S. runs pass validation and stage 3 writes the three-seed aggregate, calculate dominated
hypervolume in maximization space `(IC, -NLL, Sharpe)`. Freeze the raw-objective reference at
`(IC=0, Gaussian NLL=0, Sharpe=0)` for this and subsequent rounds so reported improvements remain comparable.

Then run exactly one Qwen turn at `xhigh` reasoning. Give it a bounded, hashed evidence packet containing the measured
run objectives, configuration aggregates, Pareto frontier, hypervolume, exact evaluated configurations, fixed controls,
and a local research allowlist. Treat this as sequential multi-objective experimental design inspired by Bayesian
optimization:

- exploit promising neighborhoods of measured nondominated configurations;
- bridge configurations with complementary objective strengths;
- explore coherent cached feature, kernel, interaction, or task-covariance ideas not represented in the first four;
- propose 4-8 falsifiable candidates, each with exact changes, runner arguments, evidence, expected directional effects,
  compute, execution priority, and stopping rules; and
- do not fabricate numerical expected hypervolume improvement from only four evaluated configurations.

The turn writes `planning/NEXT_EXPERIMENT_PLAN.md` and `planning/next_experiment_candidates.json`. It cannot run GP
fits or modify existing experiment/code files. `validate_next_experiment_plan.py` enforces the seed/date/budget panel,
objective contract, strategy mix, local evidence paths, and prohibited `IJR`, turnover, external-refresh, and time-
ablation concepts. The proposed batch remains behind a new human execution gate.
