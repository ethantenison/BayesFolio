# November 2026 multiobjective portfolio decision

**Validated November target-weight recommendation.**

Selected complete policy: **R4_matched_jointfine1_6_compact_r2_i15_exp** (`policy_b71ce3fe84ee69f58247`).

Promote this fixed policy: it passed development robustness, the single retrospective confirmation and the full-history tradeoff gates. It is selected from the eligible return/drawdown Pareto set by fixed-anchor distance, annual-block stability, then conventional Sharpe/eligible14IC/simplicity. No November weight appearance or new tuning was used.

**Current target concentration:** EWX is 45.206%; equities are 92.521% and fixed income 7.479%. HYEM alone is 7.479%. The selected policy's effective-count minimum is a concentration measure, not an equity/fixed-income balance. Under the two specified common-shock assumptions, the current targets return -18.88% and -28.88%, respectively. **Lower historical drawdown does not establish protection from a common future equity shock.**

## Decision evidence

Compounded return/CAGR and drawdown are the primary objectives. Conventional Sharpe and fourteen-ETF ranking IC are secondary evidence, with NLL/RMSE diagnostic. IC is not a veto that automatically discards a useful return/drawdown tradeoff. No turnover objective, CVaR, uncertainty recalibration, monthly discretionary blending, negative-return floor or previous-weight fallback is used.

Frozen selection rule:

{'confirmation_acceptance': {'all53_distance_gain_over_best_control_min': 0.05, 'complete_all53_and_November_raw_feasibility': True, 'confirmation12_distance_no_worse_than_best_control_plus': 0.5, 'confirmation12_drawdown_loss_no_worse_than_best_control_plus': 0.03, 'no_retuning_after_unlock': True}, 'development_eligibility': {'IC_noninferiority_veto': False, 'annual_blocks_no_worse_by_distance_0p25_min': 2, 'complete_raw_feasible_path': True, 'distance_gain_over_best_control_min': 0.05, 'leave_one_month_distance_gain_nonnegative_fraction_min': 0.8, 'paired_3month_bootstrap_distance_gain_nonnegative_fraction_min': 0.6, 'ten_bps_cost_distance_gain_min': -0.1}, 'final_selection': 'Eligible full53 Pareto minimum fixed-anchor distance; ties within1e-8 by development annual-block stability, Sharpe, eligible14IC, recipe complexity, ID. If no accepted challenger, best matched control by same full53 fixed-anchor distance.', 'frontier_stopping': {'HV_reference': 'reference_CAGR-0.10, reference_DDloss+0.10', 'budgets_are_ceilings_not_targets': True, 'consecutive_negligible_rounds': 2, 'minimum_tracks': ['cached policy investigation', 'allowed GP evidence assessment'], 'negligible_round': 'HV relative gain<1% AND best distance gain<0.05'}, 'normalization': {'CAGR_scale': 0.02, 'anchors_saved_once_before_new_policy_scoring': True, 'aspiration_CAGR': 'reference_CAGR + 0.02', 'aspiration_drawdown_loss': 'max(0, reference_drawdown_loss - 0.02)', 'distance': 'hypot(max(0,(aspiration_CAGR-CAGR)/0.02), max(0,(drawdown_loss-aspiration_drawdown_loss)/0.02))', 'drawdown_scale': 0.02, 'interpretation': 'Balanced operational aspiration, not a personal loss limit or promised return', 'source': 'MGK-free fixed v1 three-model reference development41 ONLY'}, 'primary': ['gross_CAGR', 'drawdown_loss_magnitude'], 'robustness': {'block_lengths': [3, 6], 'bootstrap_draws': 512, 'cost_bps': [0, 5, 10, 20], 'cost_gate_is_sensitivity_only': True, 'cost_objective': False, 'crash_probabilities_calibrated': False, 'seed': 20261101, 'seed_sensitivity_is_not_market_evidence': True, 'stress_drawdown_thresholds': [0.1, 0.15, 0.2]}, 'schema': 'bayesfolio.multiobjective.selection.v2', 'secondary': ['Sharpe', 'eligible14_rank_IC'], 'shortlist': {'mandatory_controls': 2, 'maximum': 10, 'remaining': 'Development eligible Pareto distance/stability plus return/defensive leaders and matched compact-rank pair when supported; deterministic ID ties'}}

All 4 eligible Pareto candidates have fixed-anchor distance 0: each meets both one-sided aspirations. The final choice therefore used the frozen tie chain—development annual-block stability, conventional Sharpe, eligible14 ranking IC, then recipe simplicity/ID. The selected policy was not a unique CAGR/drawdown dominator.

| Frozen criterion                                   | Passed   |
|:---------------------------------------------------|:---------|
| November_registered_constraints_feasible           | True     |
| all53_complete_raw_path                            | True     |
| all53_distance_gain_at_least_0p05                  | True     |
| development_eligible_before_unlock                 | True     |
| latest12_distance_within_best_control_plus0p50     | True     |
| latest12_drawdown_loss_within_best_control_plus3pp | True     |

| Period                                                             | CAGR   | Cumulative return   | Max drawdown   |   Sharpe |   14-ETF rank IC |   Mean effective assets | Worst month   |
|:-------------------------------------------------------------------|:-------|:--------------------|:---------------|---------:|-----------------:|------------------------:|:--------------|
| Development: June 2022–October 2025 (41 months)                    | 13.26% | 53.02%              | -9.02%         |    0.777 |            0.101 |                   3.139 | -4.70%        |
| Retrospective confirmation: November 2025–October 2026 (12 months) | 19.51% | 19.51%              | -8.06%         |    0.987 |            0.251 |                   3.09  | -8.06%        |
| Full stitched history: June 2022–October 2026 (53 months)          | 14.64% | 82.87%              | -9.02%         |    0.838 |            0.135 |                   3.128 | -8.06%        |

The gross cumulative-return comparison is 82.87% selected versus 79.66% Schwab aggressive. At 10 bps per gross traded dollar it is 69.80% versus 79.37%; the selected policy's return edge reverses in that cost case. Its recorded maximum drawdown at 10 bps is -9.65%, versus -13.22% for Schwab. Costs remain the predeclared sensitivity check rather than a new selection objective; the frozen policy is unchanged.

The complete finalist alternatives table is in COMPARISON.md; root-owned flags and distances are reported without recomputing a winner.

## Exact model and allocation policy

Predictions and covariance are pooled equally across seeds 27, 127 and 227 before one allocation per month. The static family prediction blend and covariance recipe below apply across dates and seeds. MGK remains one of the 15 GP training tasks, receives exactly zero portfolio weight, and is excluded from the 14-ETF pooled/family ranking diagnostics. The allocator uses the eligible principal covariance.

The selected hard-rank policy uses minimum effective assets 3, risk multiplier 1.0, and equity cap none. Effective count measures concentration; it does not itself guarantee an equity/fixed-income balance. There is no ETF position cap. The causal risk-ceiling recipe and any static exposure constraint are fixed parts of the selected policy, with feasibility recorded in VALIDATION.json.

The reference experts retain their established economic features, transformations, target definition, publication lags and training-window anchors: compact expanding, broad expanding and broad rolling48. Rolling48 means 48 labeled training months, not a 12-month evaluation block. The exact evaluated GP ranks, kernels/priors and fitting settings—including any explicitly registered limited variant—are recorded in CONFIG.json/GP specs. No unrecorded kernel, feature, window or iteration change is implied by a different allocator or blend. t_index remains in the model; raw hy_spread and its change are not reintroduced.

### Fixed prediction coefficients

| Family                     | broad_r2_i15_exp   | compact_r2_i15_exp   | broad_roll48_r2_i15   |
|:---------------------------|:-------------------|:---------------------|:----------------------|
| U.S. equity                | 0.1820%            | 98.2125%             | 1.6054%               |
| International equity       | 91.1716%           | 8.3820%              | 0.4464%               |
| U.S. fixed income          | 0.0633%            | 5.4930%              | 94.4437%              |
| International fixed income | 0.7567%            | 0.5875%              | 98.6558%              |

These are prediction coefficients, not portfolio weights.

### Fixed covariance coefficients

| Covariance expert            | Fixed coefficient   |
|:-----------------------------|:--------------------|
| broad_r2_i15_exp             | 50.0000%            |
| compact_r2_i15_exp           | 50.0000%            |
| broad_roll48_r2_i15          | 0.0000%             |
| causal historical eligible14 | 0.0000%             |

## November ETF target weights

| ETF   | Weight (decimal)   | Weight (%)   |
|:------|:-------------------|:-------------|
| SPY   | 0.000000008565     | 0.0000%      |
| MGK   | 0.000000000000     | 0.0000%      |
| VTV   | 0.000000000166     | 0.0000%      |
| IWM   | 0.144320008611     | 14.4320%     |
| VEA   | 0.039618780118     | 3.9619%      |
| VWO   | 0.197054373243     | 19.7054%     |
| VSS   | 0.092157482969     | 9.2157%      |
| EWX   | 0.452062482621     | 45.2062%     |
| BND   | -0.000000000048    | -0.0000%     |
| IEF   | -0.000000000075    | -0.0000%     |
| LQD   | 0.000000000047     | 0.0000%      |
| HYG   | 0.000000000152     | 0.0000%      |
| BNDX  | 0.000000000047     | 0.0000%      |
| VWOB  | 0.000000000236     | 0.0000%      |
| HYEM  | 0.074786863347     | 7.4787%      |

All 15 rows, including zeros and MGK=0, are retained. Percentages are display-rounded only; NOVEMBER_WEIGHTS.csv preserves unmodified decimal optimizer output.

### Family exposures

| Family                     | November target weight   |
|:---------------------------|:-------------------------|
| U.S. equity                | 14.4320%                 |
| International equity       | 78.0893%                 |
| U.S. fixed income          | 0.0000%                  |
| International fixed income | 7.4787%                  |

Actual November effective assets are 3.57558, above the policy floor 3. The minimum raw optimizer weight is -7.53e-11, within the predeclared −1e−8 numerical tolerance; it is preserved without clipping or renormalization. The variance-ceiling ratio is 1.0000000013; the risk ceiling is active within the recorded numerical acceptance tolerance.

## Conditional and static stress of the November target

| Test                                                         | Period                                | Portfolio return / drawdown   | Scope                                                  |
|:-------------------------------------------------------------|:--------------------------------------|:------------------------------|:-------------------------------------------------------|
| Assumed equities −20%; all fixed income −5%                  | Hypothetical simultaneous asset shock | -18.88%                       | Conditional arithmetic loss; no assigned probability   |
| Assumed equities −30%; HYG/HYEM −15%; other fixed income −5% | Hypothetical simultaneous asset shock | -28.88%                       | Conditional arithmetic loss; no assigned probability   |
| November weights applied to March 2020                       | March 2020                            | -19.35%                       | Static historical sensitivity; no GP forecast          |
| Static monthly-reset path maximum drawdown                   | 2013-07–2026-10                       | -30.03%                       | Same November targets reset monthly throughout history |
| Worst static 1-month cumulative return                       | 2020-03–2020-03                       | -19.35%                       | Fixed November target historical sensitivity           |
| Worst static 3-month cumulative return                       | 2020-01–2020-03                       | -26.79%                       | Fixed November target historical sensitivity           |
| Worst static 12-month cumulative return                      | 2021-11–2022-10                       | -22.62%                       | Fixed November target historical sensitivity           |

The hypothetical rows apply the stated asset losses directly to the frozen November target weights. The historical rows reset those same targets monthly against adjusted total-return proxy history; they use no historical GP forecasts and are not a backtest of the GP allocation strategy. These conditional/static losses have no assigned probability, calibrated future-crash interpretation or assumed recovery path. They do not establish a personally acceptable loss tolerance. The frozen selection rule and constraints were maintained after inspecting the November weights and these diagnostics; no new selection or constraint was introduced.

Across the full53 months, the selected portfolio has mean squared standardized forecast error z²=2.674 and nominal95% interval coverage 83.02%; fixed eligible equal-weight z²=5.087, versus Gaussian reference values of 1 and 95%. This forecast-error/uncertainty mismatch can reflect mean and variance errors; it does not establish calibrated crash protection, and remains diagnostic without fitting, recalibration or selection changes.

- Historical prices are dividend/split-adjusted total-return proxies from the sealed October31 vintage.
- The same November target is reset monthly throughout history; no historical GP forecasts are used.
- Static historical sensitivity is gross of fees/taxes; November2 execution at October30 closes is not assumed.
- Shock2: HYG/HYEM receive-15%; BND/IEF/LQD/BNDX/VWOB receive-5%. Scenario buckets are not ratings.
- No probability, crash-protection guarantee, currency hedging or recovery path is inferred from these shocks.

[Distribution diagnostics](DISTRIBUTION_DIAGNOSTICS.md) and [machine-readable evidence](DISTRIBUTION_DIAGNOSTICS.json) examine predictive distributions only. They do not trigger fitting, uncertainty recalibration, a changed selection rule or another policy search.

## Timing, validation and reproduction

Logical as-of October 31, 2026 is a Saturday. The latest completed regular-session market bar is October 30, predicting November; no unavailable November target is used for training.

Oct31 Saturday as-of uses Oct30 closing information to forecast November. Possible Nov2 execution has price/gap risk; an after-close decision cannot obtain a retrospective Oct30 fill.

The gap between Friday's information and potential Monday November 2 execution matters: a decision made after the close cannot assume a retrospective Friday-close fill. Target weights are not orders.

VALIDATION.json records freshness/availability, raw-weight numerical tolerances, effective count, risk/exposure feasibility, strict solver acceptance, independent checks and reproducibility. Accepted numerical residuals remain raw; no clipping or renormalization is applied.

```sh
uv run --locked --no-sync python experiments/2026-11-portfolio-multiobjective-v2/v2_run.py --verify
```

## Robustness, costs and limitations

Development includes all first 41 return months. The first five, June–October 2022, remain a partial diagnostic block and remain in every stitched wealth/drawdown path; they are not discarded. The three complete development blocks are November 2022–October 2023, November 2023–October 2024 and November 2024–October 2025. The latest12 block, November 2025–October 2026, was locked while this round's policies were developed and the shortlist/rule frozen. Older research already inspected these dates: final confirmation is retrospective, not a pristine independent holdout. No retuning follows it.

|   Cost (bps per traded dollar) | Cumulative return   | CAGR   | Max drawdown   |   Sharpe |
|-------------------------------:|:--------------------|:-------|:---------------|---------:|
|                              0 | 82.87%              | 14.64% | -9.02%         |    0.838 |
|                             10 | 69.80%              | 12.74% | -9.65%         |    0.696 |
|                             20 | 57.66%              | 10.86% | -10.27%        |    0.554 |
|                              5 | 76.21%              | 13.69% | -9.33%         |    0.767 |

Cost rows charge the first cash-to-assets purchase and subsequent drift-aware purchases plus sales, per gross dollar traded, with exact self-financing funding. They are diagnostic, not objectives.

| Family                     |   Full53 within-family IC |
|:---------------------------|--------------------------:|
| International equity       |                     0.109 |
| International fixed income |                     0.057 |
| U.S. equity                |                     0.264 |
| U.S. fixed income          |                     0.151 |

These are retrospective research paths, gross of explicit costs/taxes unless a cost row says otherwise. Development selection, temporal dependence, prior inspection and limited sample size constrain the strength of inference. Paired block-bootstrap ranges are sensitivity diagnostics, not calibrated future win probabilities or crash protection. Seed variation is numerical sensitivity, not independent market evidence. Better NLL or a more attractive November allocation does not prove future superiority.

## Registered design limitations

```json
[
  "Conservative post-registration metadata-only design restriction. Preserve all unclosed leaves, but do not promote or use them for rank attribution. Round4 promised full pair coverage; its16 unclosed exposure/covariance leaves are a documented coverage limitation, not erased or called complete rank experiments. Exact pairs are required for rank finalists.",
  "All earlier histories were inspected; confirmation is a locked retrospective replay, not pristine holdout.",
  "No documented kernel/prior malfunction was established. Low amplitude and reaching maxiter15 are not sufficient justification; no unsupported kernel/prior or feature/window/budget/rank search was run.",
  "The originalMGK dividend-adjustment vintage difference0.08247% is documented in v1; old history and macro vintages were not revised. Macro point-in-time vintage certification remains limited.",
  "Risk-matrix mixtures are declared allocation recipes, not calibrated Bayesian mixture distributions. Historical covariance's stored MGK diagonal is a noninvestable placeholder, never a conditional observation.",
  "Drawdown bootstrap/static-stress frequencies are not calibrated future crash probabilities."
]
```

No trades, brokerage changes, commits or pushes were made. Prior datasets/experiments and production defaults are protected. See REPRODUCTION.md, the experiment ledger and STATUS.json for exact evidence.

## Recorded settings and budgets

```json
{
  "GPspecs": {
    "broad_r2_i15_exp": {
      "inputs": [
        "t_index",
        "lag_y_excess_lead",
        "mom12m_skip1m",
        "mom6m",
        "trend_slope",
        "cpi_chg_12p",
        "dealer_gamma_proxy",
        "dxy",
        "em_fx_ret",
        "gold_crude_ratio",
        "hy_spread_z_12p",
        "oil_ret",
        "spy_ret",
        "term_spread",
        "vix_ts_z_12p"
      ],
      "kernel_recipe": "original_default",
      "maxiter": 15,
      "rank": 2,
      "rolling_months": null,
      "task_prior_lkj_eta": 2.0,
      "tickers": [
        "SPY",
        "MGK",
        "VTV",
        "IWM",
        "VEA",
        "VWO",
        "VSS",
        "EWX",
        "BND",
        "IEF",
        "LQD",
        "HYG",
        "BNDX",
        "VWOB",
        "HYEM"
      ]
    },
    "broad_roll48_r2_i15": {
      "inputs": [
        "t_index",
        "lag_y_excess_lead",
        "mom12m_skip1m",
        "mom6m",
        "trend_slope",
        "cpi_chg_12p",
        "dealer_gamma_proxy",
        "dxy",
        "em_fx_ret",
        "gold_crude_ratio",
        "hy_spread_z_12p",
        "oil_ret",
        "spy_ret",
        "term_spread",
        "vix_ts_z_12p"
      ],
      "kernel_recipe": "original_default",
      "maxiter": 15,
      "rank": 2,
      "rolling_months": 48,
      "task_prior_lkj_eta": 2.0,
      "tickers": [
        "SPY",
        "MGK",
        "VTV",
        "IWM",
        "VEA",
        "VWO",
        "VSS",
        "EWX",
        "BND",
        "IEF",
        "LQD",
        "HYG",
        "BNDX",
        "VWOB",
        "HYEM"
      ]
    },
    "compact_r1_i15_exp": {
      "inputs": [
        "t_index",
        "mom12m_skip1m",
        "mom6m",
        "cpi_chg_12p",
        "hy_spread_z_12p",
        "term_spread"
      ],
      "kernel_recipe": "original_default",
      "maxiter": 15,
      "rank": 1,
      "rolling_months": null,
      "task_prior_lkj_eta": 2.0,
      "tickers": [
        "SPY",
        "MGK",
        "VTV",
        "IWM",
        "VEA",
        "VWO",
        "VSS",
        "EWX",
        "BND",
        "IEF",
        "LQD",
        "HYG",
        "BNDX",
        "VWOB",
        "HYEM"
      ]
    },
    "compact_r2_i15_exp": {
      "inputs": [
        "t_index",
        "mom12m_skip1m",
        "mom6m",
        "cpi_chg_12p",
        "hy_spread_z_12p",
        "term_spread"
      ],
      "kernel_recipe": "original_default",
      "maxiter": 15,
      "rank": 2,
      "rolling_months": null,
      "task_prior_lkj_eta": 2.0,
      "tickers": [
        "SPY",
        "MGK",
        "VTV",
        "IWM",
        "VEA",
        "VWO",
        "VSS",
        "EWX",
        "BND",
        "IEF",
        "LQD",
        "HYG",
        "BNDX",
        "VWOB",
        "HYEM"
      ]
    }
  },
  "asof": "2026-10-31",
  "cutoff": {
    "November_target_used": false,
    "last_available_training_target": "October2026",
    "last_training_origin": "2026-09-30",
    "market_origin": "2026-10-30",
    "target_month": "2026-11"
  },
  "fit_counts": {
    "admitted_or_started_attempts": 6,
    "attempt_paths": [
      "/Users/et/Documents/BayesFolio/experiments/2026-11-portfolio-multiobjective-v2/outputs/fits/compact_r1_i15_exp_s127_2026-09-30/attempt_01",
      "/Users/et/Documents/BayesFolio/experiments/2026-11-portfolio-multiobjective-v2/outputs/fits/compact_r1_i15_exp_s127_2026-10-30/attempt_01",
      "/Users/et/Documents/BayesFolio/experiments/2026-11-portfolio-multiobjective-v2/outputs/fits/compact_r1_i15_exp_s227_2026-09-30/attempt_01",
      "/Users/et/Documents/BayesFolio/experiments/2026-11-portfolio-multiobjective-v2/outputs/fits/compact_r1_i15_exp_s227_2026-10-30/attempt_01",
      "/Users/et/Documents/BayesFolio/experiments/2026-11-portfolio-multiobjective-v2/outputs/fits/compact_r1_i15_exp_s27_2026-09-30/attempt_01",
      "/Users/et/Documents/BayesFolio/experiments/2026-11-portfolio-multiobjective-v2/outputs/fits/compact_r1_i15_exp_s27_2026-10-30/attempt_01"
    ],
    "attempts_remaining": 1194,
    "fit_attempts_remaining": 1194,
    "fit_attempts_used": 6,
    "started_attempts": 6
  },
  "policy_settings": {
    "CVaR": false,
    "MGK_zero_enforced_in_eligible_axis": true,
    "covariance_definition": "observed GP covariance including likelihood noise; optional historical risk mix",
    "covariance_disagreement_term": false,
    "eligible_assets": [
      "SPY",
      "VTV",
      "IWM",
      "VEA",
      "VWO",
      "VSS",
      "EWX",
      "BND",
      "IEF",
      "LQD",
      "HYG",
      "BNDX",
      "VWOB",
      "HYEM"
    ],
    "equity_cap": null,
    "fallback": false,
    "fully_invested": true,
    "live_variance_ceiling": 0.0008223728980994785,
    "long_only": true,
    "minimum_effective_assets": 3,
    "monthly_blend_tuning": false,
    "objective": "rank_score @ weights - 1e-8*sum_squared_weights",
    "position_cap": null,
    "postsolve_weight_repairs": false,
    "rank_score": "ascending average eligible rank/14",
    "risk_multiplier": 1.0,
    "risk_reference": "causal expanding available-label LedoitWolf historical equal-weight eligible14 variance"
  },
  "search_counts": {
    "controls": 2,
    "final_confirmation_policies": 10,
    "new_complete_policies": 874,
    "new_expert_specs": 1,
    "rounds": 4
  }
}
```
