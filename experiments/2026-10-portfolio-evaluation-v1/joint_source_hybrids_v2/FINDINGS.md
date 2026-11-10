# Joint model source hybrids and task grouping clarification

Both requested hybrids improve Sharpe and total return over the original full
joint baseline in this cached comparison. The leading new case uses five-input
full-joint U.S.-equity forecasts, fourteen-input full-joint forecasts for the other
eleven ETFs, and the fourteen-input model's complete covariance. Swapping to the
five-input model's complete covariance changes performance only slightly.

## Task grouping and input masks are separate experiments

Task grouping changes which ETF targets train together while keeping their input
mask fixed. Examples are all fifteen ETFs together, U.S. equities alone, U.S.
equities with U.S. fixed income, or the two international families together.

The approved follow-up did test matched task groups using the same fourteen-input
mask: four single families, six pairs, four triples and all four families. Those
results are distinct from the sixteen family-mask assemblies previously reported.
The latter selected between two existing input masks for each family and combined
their forecasts into a portfolio. They were not sixteen task-grouping choices.
Reporting their leaders as the main answer to a task-grouping question obscured
that distinction.

The two new hybrids below both use models trained on all fifteen ETFs. They are
mean/covariance source combinations, not new task-grouping fits or feature searches.

## Exact meaning of the compact joint model

The compact model is the five-input full-joint model from the controlled SPY
comparison, not the separate ten-input joint-NLL leader and not a U.S.-only model.
Its non-time inputs are `mom12m_skip1m`, `mom6m`, `cpi_chg_12p`,
`hy_spread_z_12p` and `term_spread`. Both models retain `t_index`.

For both new cases, SPY, MGK, VTV and IWM means come from this compact joint
model. The other eleven means come from the fourteen-input full joint model:
VEA, VWO, VSS, EWX, BND, IEF, LQD, HYG, BNDX, VWOB and HYEM. Thus U.S.
fixed-income forecasts also remain from the broad joint model, as in the earlier
hybrid comparison.

| Requested case | U.S.-equity means | Other eleven means | Complete covariance |
| --- | --- | --- | --- |
| Broad covariance hybrid | Compact joint | Broad joint | Broad joint |
| Compact covariance hybrid | Compact joint | Broad joint | Compact joint |

The two cases have exactly the same combined mean forecasts. The covariance
source is the only investment-model difference. No rows, variances or covariance
blocks are spliced; each source matrix stays complete and positive definite.
Mean and covariance estimates play separate roles in portfolio optimization.
[CVXPY portfolio formulation](https://www.cvxgrp.org/cvx_short_course/docs/applications/notebooks/portfolio_optimization.html)

## Results under the common seven effective asset policy

All cases use the original variance budget, normalized-rank allocation, rank two,
maxiter fifteen, seeds 27/127/227, 84 initial expanding training months and the
same 24 forecast origins. Returns are conventional total returns and Sharpe uses
arithmetic excess returns. Results average seeds and are gross of unspecified
costs. There is no ETF cap, CVaR, calibration transformation or weight fallback.

| Case | Sharpe | 24 month total return | Annual volatility | Pooled IC | U.S. equity IC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original fourteen-input joint, own covariance | 1.692 | 56.10% | 11.08% | 0.2965 | 0.1333 |
| Previous hybrid using U.S.-only family means, broad covariance | 1.758 | **57.84%** | 10.98% | 0.2993 | 0.1917 |
| **Compact joint U.S. means, broad covariance** | **1.780** | 57.70% | 10.81% | **0.3116** | **0.2583** |
| Compact joint U.S. means, compact covariance | 1.776 | 57.41% | 10.78% | **0.3116** | **0.2583** |
| Compact joint everywhere, own covariance | 1.509 | 48.06% | 10.61% | 0.2153 | 0.2583 |

The original, previous hybrid and compact-everywhere rows are sealed cached
controls under the same investment settings, not new fits. Only the two requested
new cases required additional allocations: 144 monthly solves, all successful.

Relative to the original baseline, the broad-covariance hybrid improves Sharpe by
0.0877 and total return by 1.60 percentage points. Both improve in every seed.
The compact-covariance hybrid improves Sharpe by 0.0842 and total return by 1.32
percentage points; those also improve in every seed.

Relative to the previous U.S.-only-mean hybrid, the new broad-covariance case
has higher Sharpe by 0.0219 but slightly lower total return by 0.15 percentage
points, alongside lower volatility. It therefore does not dominate that previous
hybrid on every metric. The new combination does improve pooled and U.S. ranking.

The compact-everywhere control illustrates why these source hybrids are useful:
keeping compact forecasts only for U.S. equities avoids imposing that compact
mask's weaker other-task forecasts on the entire portfolio.

## The two covariance sources are nearly tied here

The broad-covariance hybrid leads by only 0.0035 Sharpe and 0.28 percentage points
of total return. That is not compelling evidence that its covariance is more
accurate. The GP risk ceiling binds in only two of 72 allocations for each
source. Their active months differ, and the two portfolios differ materially
in only four of the 72 seed/month allocations when a 0.0001 raw-weight threshold
is used. In most months ranking and the diversification floor determine weights.

Keeping the broad joint covariance is a reasonable research control because it
preserves the previously used full covariance while improving the mean forecast
mix. The result does not establish a statistically meaningful covariance winner.
No calibration work was performed, and none is needed to interpret this limited
allocation comparison.

## Remaining limits and evidence

The study uses previously reused months. Three fit seeds are not three independent
market samples. The new hybrid is a research lead, not a production promotion or
a guarantee about future Sharpe or return. The individual kernel or feature
mechanism behind U.S. flattening remains unresolved.

All files are isolated in this new directory. `outputs/prediction_panel.npz`
retains the exact hybrid means, both whole covariance arrays, actual returns,
risk budgets, risk-free returns, ETF/date labels and seed labels. Per-arm JSON
retains every monthly allocation, predictive density score, ETF forecast metric,
fixed-reference distribution score and rank series. These passive diagnostics
remain available without launching calibration experiments or new GP fits.

The complete experimental framework passed 118 tests, plus Ruff and the public
docstring gate. Pylance diagnostics were not exposed. Sealed input/output hashes
verify both source models and the new results. Production and prior completed
experiments remain unchanged; rollback is setting aside this new directory.

```sh
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/joint_source_hybrids_v2/run_joint_sources.py --verify-only
.venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1
```
