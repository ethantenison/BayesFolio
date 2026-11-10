# Family model candidates and the seven asset hybrid comparison

Replacing only the four U.S.-equity means with the five-input U.S.-only model
improved both Sharpe and total return in the approved cached comparison. The
15-ETF portfolio now uses a minimum effective count of seven. Uncertainty
calibration experiments are deferred. No GP was refitted, no covariance was
scaled or blended, and no production model was promoted.

## New matched comparison at effective count seven

All cases use the original historical equal-weight variance budget, normalized
rank allocation, long-only fully invested weights, and the repaired numerical
backend with feasibility checks. There is no ETF cap, CVaR, constraint relaxation
or previous-weight fallback. Rank two, maxiter 15, dates and all three seeds are
unchanged. Returns and conventional Sharpe are gross of unspecified costs.

| Forecast means | Covariance | Sharpe | 24 month total return | Annual volatility | Maximum drawdown |
| --- | --- | ---: | ---: | ---: | ---: |
| Full 14-input joint model | GP | 1.692 | 56.10% | 11.08% | 6.46% |
| Joint except five-input U.S.-equity family means | GP | **1.758** | **57.84%** | **10.98%** | **6.31%** |
| Full 14-input joint model | Historical | 1.506 | 37.26% | 7.92% | 5.42% |
| Joint except five-input U.S.-equity family means | Historical | **1.593** | **38.98%** | **7.88%** | **5.40%** |

With GP covariance, hybrid Sharpe improves by 0.0657 and total return by 1.75
percentage points on average. Both improve in each seed. The three Sharpe
improvements are 0.0388, 0.0630 and 0.0954. The first seed's return improvement
is only 0.022 percentage points, so the return advantage is not equally large
across initializations. These are three fits over the same 24 economic months,
not three independent validation periods.

Pooled rank IC improves slightly from 0.2965 to 0.2993. U.S.-equity rank IC
improves from 0.1333 to 0.1917. International-equity and both fixed-income mean
forecasts remain bit-for-bit unchanged. The improved U.S. forecasts therefore
help the portfolio here, without forcing the U.S. feature mask on other tasks.

The hybrid changes SPY, MGK, VTV and IWM means only. VEA, VWO, VSS, EWX, BND,
IEF, LQD, HYG, BNDX, VWOB and HYEM retain the full joint model's means. The
entire observed-return joint GP covariance is retained, including all cross-family
entries. No family covariance blocks or rows are spliced into it.

## What higher Sharpe establishes about GP covariance

The GP-based allocation rule is more useful than the historical-risk version in
this comparison. This supports retaining it as the preferred research allocator.
It does not establish that its covariance forecasts are more accurate.

The distinction is directly visible in the constraint activity:

| Forecast means | GP risk ceiling binds | Historical risk ceiling binds |
| --- | ---: | ---: |
| Full joint | 0 of 72 allocations | 69 of 72 |
| U.S.-mean hybrid | 3 of 72 | 66 of 72 |

For the full joint baseline, covariance does not change the optimizing weights
when its risk ceiling is inactive. The optimizer selects weights from rank scores
and the effective-count floor. Many different covariance matrices that leave
that solution feasible would produce the same weights and Sharpe. Historical
risk often constrains that ranking-driven solution and changes asset composition.
That can lower realized Sharpe without proving that the GP covariance is more
accurate. More risk by itself does not mechanically improve Sharpe.

The covariance may still be useful. It supplies cross-asset risk estimates and a
ceiling that constrains some hybrid allocations. Full joint covariance also
preserves cross-family relationships absent from independently fitted family
models. The improvement from swapping means, however, occurs under both risk
sources, so it is evidence for the forecast substitution rather than evidence
that GP correlations caused the improvement.

Using means and risk estimates from separate models is mathematically valid.
The resulting Gaussian is not the untouched posterior of one GP, and its error
covariance need not match that of the original joint means. Keeping the whole
positive-definite covariance matrix is preferable to mixing selected covariance
rows, which need not preserve positive definiteness.
The [CVXPY portfolio example](https://www.cvxgrp.org/cvx_short_course/docs/applications/notebooks/portfolio_optimization.html)
shows the separate roles of expected returns and the covariance-based risk model.

## Which family combinations led previously

These are the leading complete family combinations under the **previous floor
of ten, historical covariance and the original risk budget**. They have not yet
been reranked at floor seven. They must not be compared directly against the new
GP-risk hybrid as if all investment settings were identical.

Each family model supplies means only for its own ETFs; the four vectors are
assembled into a single 15-ETF forecast vector. Input counts exclude `t_index`,
which is retained in every model.

| Candidate | U.S. equity | International equity | U.S. fixed income | International fixed income | Sharpe | 24 month return |
| --- | --- | --- | --- | --- | ---: | ---: |
| A | U5 | E10 | F8 local | I19 | **1.678** | **39.35%** |
| B | U6 | E10 | F8 local | I19 | 1.637 | 37.92% |
| C | U5 | E10 | F8 portfolio | I19 | 1.634 | 38.23% |
| D | U5 | E10 | F8 local | I11 | 1.630 | 38.78% |

Candidate A is the observed leader. B adds raw `hy_spread` to the U.S.-equity
mask. C substitutes the other eight-input U.S. fixed-income mask. D reduces
international fixed income from nineteen to eleven non-time inputs, with a
relatively modest performance sacrifice in that old comparison. A and D are
useful next comparators under the new common floor-seven policy.

These are complete combinations, not four independently best standalone
portfolios. Candidate A combines three previously portfolio-selected masks with
the local-ranking U.S. fixed-income mask. The all-local-ranking combination had
pooled IC 0.200 and floor-ten Sharpe 1.507, versus A's IC 0.326 and Sharpe 1.678.
Local ranking leadership did not automatically translate into the best pooled
ranking or portfolio. The same-mask four-family architecture had Sharpe 1.457;
the U.S./international geographic split had Sharpe 1.375, versus 1.463 for the
full joint baseline under those identical historical-risk settings.

## Exact family input masks

### U.S. equities

Targets: SPY, MGK, VTV and IWM.

**U5**, the model used in the new hybrid:
`mom12m_skip1m`, `mom6m`, `cpi_chg_12p`, `hy_spread_z_12p`, `term_spread`.

**U6**: U5 plus raw `hy_spread`.

### International equities

Targets: VEA, VWO, VSS and EWX.

**E10**: `mom12m_skip1m`, `mom6m`, `trend_slope`, `dealer_gamma_proxy`, `dxy`,
`em_fx_ret`, `oil_ret`, `vix`, `vix_slope`, `vix_ts_z_12p`.

The fourteen-input full joint model was better than applying the U.S. five-input
mask to every task; it was not established as the best international-equity model
overall. E10's local international-equity IC is 0.206, versus 0.142 for the full
joint baseline. Better local ranking alone is not sufficient to choose a complete
portfolio, as the assembled-family comparison demonstrates.

### U.S. fixed income

Targets: BND, IEF, LQD and HYG.

**F8 local**: `chmom`, `lag_y_excess_lead`, `ma_signal`, `mom12m`, `oil_ret`,
`schp_ret`, `y10_nominal`, `y10_real_proxy`.

**F8 portfolio**: `chmom`, `ma_signal`, `mom12m`, `ret_autocorr`, `move_proxy`,
`tbill3m`, `term_spread`, `tnote10y`.

### International fixed income

Targets: BNDX, VWOB and HYEM.

**I19**: `chmom`, `cs_mom_rank`, `lag_y_excess_lead`, `ma_signal`, `mom12m`,
`mom36m`, `cpi_chg_12p`, `cpi_chg_1p`, `em_fx_ret`, `erp`, `gold_crude_ratio`,
`hy_spread`, `hy_spread_z_12p`, `oil_ret`, `pct_above_50dma`, `spy_flow_z_12p`,
`vix`, `vix_slope`, `vix_ts_z_12p`.

**I11**: `mom12m`, `cpi_chg_12p`, `cpi_chg_1p`, `em_fx_ret`, `oil_ret`,
`pct_above_50dma`, `spy_flow_z_12p`, `spy_ret`, `vix`, `vix_slope`,
`vix_ts_z_12p`.

These are existing masks, not newly searched inputs or a claim that every listed
variable is causally necessary. No `hy_spread_chg_1p` was added to the joint model.

## Evidence and verification

The matched hybrid results are in `outputs/comparison.json`, with every seed's
monthly weights and performance in `outputs/arms/`. The exact masks and previous
family-combination results come from the sealed approved study contract and
corrected review. Source datasets, means and covariances remain unchanged.

The new policy is recorded in `policy.json`. The four-arm run uses four worker
processes with one numerical thread each. All 288 monthly optimizations completed,
and the complete framework suite passed 114 tests. Ruff and the public-docstring
gate passed. Pylance diagnostics were not exposed in this environment.

```sh
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/minimum_seven_hybrid_v1/run_hybrid.py --verify-only
.venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1
```

The data remain 24 previously reused forecast origins with three fit seeds, not a
new untouched test. The results support a hybrid research candidate, not a promise
of future Sharpe or returns. Rollback is setting aside this new folder; no
production or earlier experiment files were changed.
