# Time-Decay Kernel Experiment Plan

Purpose: test kernel-level ways to keep the common-history feature artifact while reducing old-regime influence in the June 2026 ETF multitask GP portfolio workflow.

Constraint from Ethan: do not use the existing GPArchitect time-varying lengthscale wrapper in these tests. Runs for this folder should set:

```text
--time-modulation-mode neither
```

## Data Contract

- Feature artifact: `/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_18etf_common_history_201306_202605.parquet`
- Effective complete-input model panel from prior run: `2013-07-31` to `2026-05-29`
- Construction dates: 11 realized windows from `2025-06-30` through `2026-04-30`, plus live `2026-05-29`
- GP inputs: `t_index`, `lag_y_excess_lead`, 10 ETF features, 16 macro features
- Fit universe: 18 ETFs
- Portfolio universe: 16 ETFs, excluding `MGK` and `BND`
- Transform baseline: BoTorch `Normalize` on non-task features and `StratifiedStandardize` by ETF task
- Portfolio optimizer: Riskfolio hist/CVaR/Sharpe with `upperlng=0.20`, `nea=10`

## Baselines

- Short-history GP, 2021-start artifact, 11 realized + live
- Common-history GP, no recency modification
- Common-history GP with fixed observation recency noise, half-life 36 months
- Historical EWMA2 Riskfolio
- Equal weight

## Candidate 1: Global Multiplicative Time Gate

Construction:

```text
K((x_i, t_i), (x_j, t_j)) =
    K_task(asset_i, asset_j)
    * K_time_decay(t_i, t_j)
    * K_features(x_i, x_j)
```

where `K_time_decay` is a Matérn 1/2 kernel:

```text
K_time_decay(t_i, t_j) = exp(-|t_i - t_j| / ell_time)
```

Rationale: the current additive/hierarchical kernel can let old rows influence current predictions through feature-only ETF or macro similarity. Multiplying all non-task covariance by a short temporal gate removes that back door: an old macro regime can only matter if it is also temporally close enough.

Test values:

- Half-life 24 months
- Half-life 36 months
- Half-life 60 months

Implementation note: if `t_index` is normalized to `[0, 1]` inside the BoTorch input transform, convert month half-life to normalized lengthscale per window:

```text
half_life_norm = half_life_months / training_window_months
ell_time_norm = half_life_norm / log(2)
```

Primary risk: too-short half-life can collapse useful cross-regime pooling and make the GP behave like a noisy rolling-window model.

## Candidate 2: Forecast-Date Recency Amplitude Gate

Construction for each rebalance date `T`:

```text
w_T(t_i) = exp(-(T - t_i) / tau)

K_T(i, j) =
    sqrt(w_T(t_i))
    * K_base(i, j)
    * sqrt(w_T(t_j))
```

Rationale: this is direct covariance-level forgetting. It is asymmetric with respect to the forecast date: old observations have lower covariance amplitude with everything in the current fit. This targets the actual portfolio use case more directly than stationary temporal distance kernels.

Test values:

- Half-life 24 months
- Half-life 36 months
- Half-life 60 months

Implementation note: this is valid positive semidefinite for a fixed rebalance window because it is diagonal scaling of a PSD kernel:

```text
K_T = D_T K_base D_T
```

Primary risk: it is forecast-date-dependent, so it is less elegant than a standard GP kernel and must be rebuilt per rebalance window.

## Candidate 3: Regime/Changepoint Mixture Kernel

Construction:

```text
s(t) = sigmoid((t - c) / width)

K(i, j) =
    s(t_i) s(t_j) K_recent(i, j)
    + (1 - s(t_i))(1 - s(t_j)) K_old(i, j)
```

Optional cross-regime term:

```text
rho * sqrt(s_i(1 - s_i)) sqrt(s_j(1 - s_j)) K_shared(i, j)
```

Rationale: if the 2013-2020 regime and 2021-2026 regime have different relationships, a single global kernel with one set of hyperparameters is under-specified. A changepoint mixture lets the model learn or enforce separation between old and recent regimes while retaining some pooling.

Initial changepoint candidates:

- Fixed `c = 2020-03-31`
- Fixed `c = 2021-03-31`, matching the short-history artifact boundary
- Rolling learned/selected `c` by grid search over candidate dates

Primary risk: extra degrees of freedom with only monthly ETF data. This needs fake-data recovery or at least a small grid-search discipline, not free learning on the final 11-window slice.

## Evaluation Matrix

For each candidate, run:

- `positive_no_prior`
- `signed_no_prior`
- Same 11 realized + live construction dates
- Same posterior scenarios and Riskfolio settings as the common-history runs
- Existing time-varying lengthscale/outputscale wrapper disabled with `--time-modulation-mode neither`

Metrics:

- GP mean IC and median IC
- Cumulative return, CAGR, Sharpe, max drawdown
- Window-level return deltas versus equal weight and historical EWMA2
- Average weights and live top weights
- Turnover and fallback/infeasibility windows
- Recency/kernel diagnostics: effective time half-life, time lengthscale, covariance gate range, and any jitter warnings

Decision rule:

- A kernel is promising only if it improves GP IC and realized portfolio performance versus common-history no-recency, without materially increasing drawdown, turnover, or Riskfolio infeasibility.
- A kernel is not credible if the advantage comes from one or two months only.
- No strong claim without the manifest, window-level decomposition, and plot inspection.

## Follow-up: E/M/T Composition Proposals

These keep the positive task kernel from `positive_no_prior` and disable the
existing time-varying lengthscale/outputscale wrapper with
`--time-modulation-mode neither`.

Block definitions:

- `E`: ETF feature block, Matérn 1/2.
- `M`: macro feature block, additive `Linear + Matérn 1/2 + RQ`.
- `T`: time block, Matérn 1/2.

Requested proposals:

1. `e_plus_mt_plus_t_plus_emt`: `E + M*T + T + E*M*T`
2. `et_plus_mt_plus_emt`: `E*T + M*T + E*M*T`

Additional proposal:

3. `e_plus_et_plus_mt_plus_emt`: `E + E*T + M*T + E*M*T`

Rationale for proposal 3: keep an ETF main effect and explicit ETF-time drift,
while requiring every macro contribution to pass through `T`. This is a middle
ground between proposal 1's stronger standalone anchors and proposal 2's fully
interaction-only structure.

Follow-up TVLS variants around the best observed proposal
`et_plus_mt_plus_emt`:

- `t_plus_et_plus_mt_plus_emt`: `T + E*T + M*T + E*M*T`
- `et_plus_mt15_plus_emt`: `E*T + M*T(nu=1.5) + E*M*T`
- `et_plus_mt_plus_emt_t15`: `E*T + M*T + E*M*T(nu=1.5 on T)`
- `t_plus_et_plus_mt15_plus_emt`: `T + E*T + M*T(nu=1.5) + E*M*T`
- `et_plus_mt_plus_em_plus_emt`: `E*T + M*T + E*M + E*M*T`
- `et_plus_mt_plus_em`: `E*T + M*T + E*M`

These keep `--time-modulation-mode lengthscale_only` for the TVLS comparison.
