# Prediction, Uncertainty, and Multitask Gaussian Processes

Status: Active

Audience: Users, modelers, contributors, and coding agents

Canonical topic: Why BayesFolio uses uncertainty-aware multitask forecasting

Last verified: 2026-10-10

## The Decision Problem

BayesFolio does not predict ETF returns for their own sake. The monthly decision is to choose portfolio weights using only information available at the decision date, while controlling downside risk and preserving enough evidence to evaluate the decision later.

Four stages must remain distinct:

1. **Forecasting** estimates a joint distribution of future ETF returns.
2. **Portfolio construction** converts that distribution and explicit constraints into candidate weights.
3. **Execution** turns reviewed weights into trades; this is currently manual and outside BayesFolio.
4. **Measurement** compares subsequent realized performance with benchmarks and the assumptions used at decision time.

A good forecast does not automatically imply a good portfolio, and a profitable realized month does not prove that the forecast was well calibrated.

## Why Monthly ETF Prediction Is Difficult

Expected monthly returns are small relative to ordinary return volatility. Even a long price history contains a limited number of independent monthly observations, while the set of plausible ETF, macroeconomic, and interaction features can be large.

ETFs also share market, rates, credit, inflation, commodity, and liquidity drivers. Those relationships change across regimes. Treating every historical row as exchangeable or every ETF as independent discards information and can produce unstable estimates.

Portfolio optimization amplifies estimation error. A small difference between two expected-return estimates can create a large difference in weights, especially when the optimizer treats those estimates as certain. This is why uncertainty is part of the decision input rather than an optional chart added after modeling.

## What Uncertainty Means Here

For information available through month `t`, the forecast target is a joint predictive distribution for next-period returns rather than a single vector:

```text
p(next-period ETF returns | information available through t)
```

The distribution contains several decision-relevant quantities:

- **Predictive mean**: the model's central return estimate for each ETF.
- **Predictive variance**: the range of plausible outcomes around each estimate.
- **Cross-asset covariance**: how ETF outcomes may move together.
- **Epistemic uncertainty**: uncertainty attributable to limited data or uncertain model structure.
- **Irreducible noise**: return variation that available predictors do not explain.

These values are conditional model outputs, not confidence guarantees. A narrow posterior can still be wrong when the model is misspecified or the market regime changes.

## Why Point Models Are Not Enough

A deterministic point model can be useful as a baseline, but a collection of separate point forecasts does not by itself provide a coherent joint distribution. It can omit cross-ETF dependence, obscure estimation uncertainty, and force the optimizer to work with return estimates that appear more precise than they are.

The issue is not that deterministic models are universally inferior. The issue is that the portfolio decision requires uncertainty and dependence estimates in addition to expected returns. Any alternative model must provide or be paired with credible estimates of those quantities.

## Why a Multitask Gaussian Process

BayesFolio treats each ETF as a related prediction task. A multitask Gaussian process can learn shared feature structure and task covariance while retaining asset-specific predictions.

This design can:

- share statistical strength across related ETFs when individual histories are limited;
- represent market-wide and ETF-specific effects in one model;
- learn which tasks tend to move together;
- produce a joint posterior distribution rather than disconnected point estimates;
- generate posterior scenarios for tail-risk-aware optimization; and
- use helper assets during training without making them eligible for the final portfolio.

Sharing is not automatically beneficial. Poor task structure can cause negative transfer, so cross-task assumptions require out-of-sample validation.

## From Data to Portfolio Weights

The intended decision chain is:

```text
Monthly ETF and macro data
    -> cleaned and lagged feature panel
    -> multitask GP fit
    -> optional model selection and tuning
    -> joint posterior return scenarios
    -> constrained portfolio optimization
    -> walk-forward backtest and review
    -> explicit human approval or rejection
    -> finalized weights when approved
    -> manual brokerage execution
```

BayesFolio uses decimal return units (`0.02 = 2%`). Features associated with a prediction at time `t` must contain only information available before `t`; the target at `t` represents the subsequent outcome. Lagging and release-date handling are defenses against look-ahead bias, not cosmetic preprocessing.

## Model Selection and Tuning

Model selection is optional and separate from the fixed defaults used by the current monthly automation. Candidate kernels, mean functions, task ranks, priors, and feature groups should be evaluated with rolling or expanding time splits.

The accepted monthly model uses double-precision tensors, BoTorch normalization
of non-task inputs, task-stratified outcome standardization, rank-five positive
task covariance without an explicit task prior, and the
`E + M + T + ET + MT + EMT` covariance structure with lengthscale-only time
modulation. These are versioned defaults rather than claims of universal
optimality; replacement requires time-aware evaluation and recorded evidence.

In-sample fit or marginal likelihood alone is not enough. Selection should consider out-of-sample forecast error, ranking behavior, uncertainty calibration, portfolio stability, and decision-level performance. Seeds, dataset cutoffs, and candidate definitions must be recorded so a selected configuration can be reproduced.

Experiment findings remain evidence under `experiments/` until they are promoted into tested defaults and canonical documentation.

## Turning Uncertainty Into Decisions

Posterior samples provide plausible joint ETF return outcomes. The optimizer can use those scenarios to estimate expected return, covariance, and tail risk while applying explicit position and diversification constraints.

Uncertainty does not eliminate the need for guardrails. Position caps, investable-universe rules, risk measures, turnover considerations, and review thresholds protect against model and optimization error. Candidate weights should never be interpreted as certain or automatically executable.

Walk-forward evidence must pair weights fixed on a decision date with returns
that become observable later. BayesFolio records both dates instead of guessing
a shift from dataframe position. Candidate and backtest strategy identifiers
must match, and evidence realized on or after the candidate as-of date cannot
support that decision. Approval also requires at least one benchmark evaluated on the same
decision/realization grid. Even valid comparative evidence does not approve a
portfolio: finalized weights require a timestamped human approval and rejected
candidates emit no final weights.

## Validation and Refactor Parity

A credible evaluation should include:

- rolling or expanding-window out-of-sample tests;
- explicit checks for look-ahead and data leakage;
- deterministic baselines and benchmark portfolios;
- predictive-error, ranking, and uncertainty-calibration diagnostics;
- portfolio return, volatility, drawdown, and tail-risk measures;
- turnover, fees, taxes, and slippage when the evidence supports them; and
- sensitivity to seeds, windows, and reasonable modeling choices.

Structural cleanliness is not sufficient. The refactored pipeline reproduces
the last accepted pre-refactor July 2026 feature artifact, GP predictions and
scenarios, and portfolio weights through the maintained application entry
point. The regression gate allows `1e-12` absolute scenario/prediction error,
`1e-10` per-asset weight error, and `1e-9` total weight L1 error. This parity
evidence protects against refactor drift; it does not establish that the July
configuration remains the best future model.

## Failure Modes and Limitations

- Related-looking ETFs can still exhibit negative transfer.
- Kernel, likelihood, and prior choices can be misspecified.
- Small samples can make hyperparameters unstable.
- Regime changes can invalidate learned relationships.
- A model can be overconfident even when its internal diagnostics look healthy.
- Optimization can magnify small forecast errors.
- Backtests can overstate performance through leakage, repeated selection, or omitted costs.
- Current outputs do not account for every brokerage, tax, liquidity, or execution constraint.
- Forecasts and backtests do not establish future alpha.

Live performance measurement is distinct from backtesting. Brokerage or
performance providers must supply returns that correctly account for external
cash-flow timing; BayesFolio does not infer time-weighted performance from
period-end balances alone. The tracking result reports realized and unrealized
gain/loss changes, fees, benchmark-relative return, tracking error, information
ratio, beta, and a descriptive single-benchmark Jensen alpha estimate. That
estimate is not causal proof of skill and is unidentified when there are fewer
than two periods or the benchmark has no variance.

An approved portfolio decision is still not a trade. Rebalance planning uses a
fresh read-only account snapshot and emits dollar notionals marked as
non-executable. Quantity rounding, fractional shares, tax lots, order types,
price protection, authentication, and explicit order authorization belong to a
future broker adapter and execution workflow.

## Current Implementation Map

- Feature construction: `bayesfolio/engine/features/`
- Multitask GP builders and workflows: `bayesfolio/engine/forecast/gp/`
- Monthly application composition: `bayesfolio/application/monthly_portfolio.py`
- Monthly computation and sequencing: `bayesfolio/engine/automation/`
- Monthly runtime I/O: `bayesfolio/io/monthly_portfolio_runtime.py`
- Portfolio optimization: `bayesfolio/engine/asset_allocation/`
- Backtesting: `bayesfolio/engine/backtest/`
- Boundary contracts: `bayesfolio/contracts/`
- Providers and artifact persistence: `bayesfolio/io/`

Detailed API behavior belongs in module, class, and function docstrings. See the [package map](package_map.md) for preferred public entry points and the [architecture guardrails](copilot_architecture.md) for ownership rules.

Broker-neutral holdings, planning, and realized performance boundaries are
implemented. Direct Schwab authentication, holdings retrieval, and order
execution remain future I/O/application capabilities.
