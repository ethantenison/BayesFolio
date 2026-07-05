# U.S. Equity ETF Feature Experiments

## Goal

Test whether a coherent U.S. equity ETF multitask GP can produce useful one-month-ahead forecast dispersion without collapsing to the task prior mean.

The first family is:

- `SPY`: broad U.S. large-cap equity
- `MGK`: U.S. mega-cap growth
- `VTV`: U.S. large-cap value
- `IWM`: U.S. small-cap Russell 2000

This family is intentionally narrower than the full BayesFolio universe. It keeps the tasks on the same country, currency, asset class, and return target while preserving meaningful style and size differences.

`IJR` is deferred from the first testing round because it is near-identical to
`IWM` on this target. Add it back after the four-task family has a usable
baseline to test whether the duplicate small-cap task improves calibration or
only slows the search.

## Decision Question

Can feature selection and target/GP design produce forecasts that:

- maintain non-trivial cross-sectional spread across the first-round four-task U.S. equity family,
- improve one-month-ahead ranking versus simple historical/EWMA baselines,
- avoid overfitting one rebalance window or one ETF,
- translate into portfolio choices that survive turnover, drawdown, and benchmark checks?

## Baselines

Minimum baselines before claiming improvement:

- equal-weight U.S. equity family portfolio,
- historical/EWMA return forecast already used by BayesFolio,
- per-ETF trailing mean or EWMA target mean,
- current monthly GP feature set as the incumbent GP baseline.

## Phases

### 1. Feature Expansion

Start from all features BayesFolio can currently generate, not only the current monthly selected columns. Add cheap, literature-motivated U.S. equity features when they can be computed from existing OHLCV inputs without new data dependencies.

Candidate additions in this phase:

- 12-1 momentum (`mom12m_skip1m`)
- short-term reversal (`short_reversal_1m`)
- downside realized volatility (`downside_vol_1m`)
- volatility regime ratio (`vol_ratio_1m_3m`)

### 2. Feature Removal

Exclude features that are structurally aimed at non-U.S. equity tasks or likely redundant for this family. Do not delete them from BayesFolio globally; exclude them from U.S. equity experiment configs.

Removal should be justified by at least one of:

- family mismatch, such as EM FX or global sovereign yields,
- target mismatch, such as credit-only signals for fixed income tasks,
- redundancy with a cleaner feature in the same block,
- empirical zero/near-zero variation or persistent missingness in the U.S. equity family dataset,
- unstable apparent signal across walk-forward windows.

### 3. Input Block Search

Evaluate compact feature blocks before testing broad combinations:

- momentum/trend,
- volatility/drawdown/tail,
- liquidity/flow/market breadth,
- macro/risk regime,
- compact combined U.S. equity set.

Do not drop a block only because its standalone run is weak. Some blocks may
matter primarily through interactions. The first interaction screen should
include:

- momentum/trend + volatility/regime,
- momentum/trend + macro/risk,
- liquidity + volatility/regime,
- target history + macro/risk,
- target history + volatility/regime.

### 4. Anti-Collapse Variants

For promising feature blocks, test GP variants designed to avoid uninformative shrinkage:

- raw target versus EWMA residual target,
- multitask constant mean versus family/task baseline mean,
- current noise floor versus tighter calibrated noise floor,
- current rank versus lower-rank task covariance,
- current feature normalization versus family-specific scaling diagnostics.

## Required Diagnostics

Do not judge by RMSE alone. Each run should report:

- prediction dispersion by rebalance date,
- RMSE, MAE, bias, Gaussian negative log likelihood, predictive coverage, and standardized residual diagnostics by ETF,
- cross-sectional information coefficient,
- top-minus-bottom realized spread,
- calibration/coverage of posterior scenarios,
- task covariance/correlation diagnostics,
- feature lengthscale or relevance summary where available,
- portfolio equity curve, drawdown, turnover, and max weight,
- slice readout by ETF and by stress/risk regime.

## Literature Anchors

- Gu, Kelly, and Xiu, "Empirical Asset Pricing via Machine Learning" (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577): broad stock-return evidence that nonlinear interactions among firm characteristics and macro predictors can improve risk-premium forecasts.
- Pan and Zeng, "ETF Arbitrage, Non-Fundamental Demand, and Return Predictability" (https://academic.oup.com/rof/article/25/4/937/5919085): ETF flows can forecast future ETF and underlying-asset returns through non-fundamental demand pressure.
- ETF return predictability literature using technical predictors (https://www.sciencedirect.com/science/article/pii/S0927539826000447): recent ETF-specific evidence emphasizes momentum and volatility indicators, especially at shorter horizons.
- Ang, Hodrick, Xing, and Zhang, "The Cross-Section of Volatility and Expected Returns" (https://doi.org/10.1111/j.1540-6261.2006.00836.x): volatility and aggregate volatility exposure are relevant equity-return predictors.

These are anchors for feature selection, not proof that this ETF family is forecastable.

## Stop Rules

Stop short of a positive claim if:

- the feature artifact lacks a manifest or exact config,
- signal timing or predictor lagging is unclear,
- forecast dispersion is near zero across most windows,
- improvement is driven by one rebalance date or one ETF,
- portfolio improvement disappears versus EWMA or equal weight,
- turnover or concentration becomes unrealistic,
- diagnostics suggest leakage, stale artifacts, or target/feature drift.
