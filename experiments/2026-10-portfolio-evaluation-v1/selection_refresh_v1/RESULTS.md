# ETF model selection refresh results

## Summary

- Rescored 501 saved configurations and completed 1,800 fresh monthly fits:
  four shortlisted configurations per family, three additional joint models,
  and six matched-input family/regional controls. Reused the accepted joint
  baseline's 72 verified exact-covariance fits.
- Family-specific feature masks matter. Simply splitting the accepted joint
  model while retaining its input mask did not improve overall performance.
- The portfolio-selected family assembly has the highest observed Sharpe.
  The NLL-selected family assembly offers a stronger within-family ranking
  compromise. Neither is an automatically promoted deployment model.

All results below average the three original seeds over the same 24 months.
Rank stayed 2, maxiter stayed 15, IWM stayed included, and `t_index` stayed in
every mask. No monolithic joint GP used `hy_spread_chg_1p`.

## Portfolio comparison

All rows use the same 15 ETFs, realized returns, RF conversion, and training-only
Ledoit–Wolf covariance. The primary allocator maximizes forecast ranks subject
to a 35% cap, effective asset count at least three, and an equal-weight variance
budget. The no-cap variant retains the variance budget and diversification
constraint; it does not mean unconstrained investing. Returns are gross of costs.

| Architecture | Pooled IC | Equal-family IC | Marginal NLL | Primary Sharpe | No-cap Sharpe | Linear-rank Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accepted joint baseline, 14 inputs | 0.297 | 0.131 | -2.259 | 1.616 | 1.539 | 1.555 |
| Joint ranking/NLL lead, 12 inputs | 0.281 | 0.194 | -2.307 | 1.316 | 1.360 | 1.498 |
| Joint portfolio lead, 12 inputs | 0.269 | 0.158 | -2.250 | 1.745 | 1.712 | 1.528 |
| Joint ranking-diversity candidate, 22 inputs | 0.182 | 0.177 | -2.158 | 1.117 | 1.052 | 1.439 |
| Four families, common joint input mask | 0.264 | 0.103 | -2.163 | 1.567 | 1.480 | 1.524 |
| Two regions, common joint input mask | 0.270 | 0.111 | -2.231 | 1.592 | 1.472 | 1.498 |
| Family-specific ranking leads | 0.200 | 0.343 | -2.119 | 1.192 | 1.196 | 1.498 |
| Family-specific portfolio leads | 0.318 | 0.134 | -2.115 | 2.264 | 2.197 | 1.706 |
| Family-specific marginal-NLL leads | 0.291 | 0.249 | -2.232 | 1.986 | 1.950 | 1.663 |

Higher IC and Sharpe are better; lower NLL is better. Equal-family IC averages
the four within-family monthly rank correlations. Pooled IC ranks all 15 ETFs
together. These measure different skills and should not be substituted for one
another. Marginal NLL averages univariate scores, not a full joint density.

Equal-weight Sharpe is 1.258 and fixed 60/40 Sharpe is 1.284 for every model.
The ranking-selected family assembly falls below both under the primary
allocator despite its strong local IC. The portfolio-selected assembly's primary
Sharpe is 2.323, 2.416, and 2.054 by seed; the NLL-selected assembly's is 1.946,
1.876, and 2.135. Baseline values are 1.537, 1.682, and 1.629.

The portfolio-selected assembly's mean compounded 24-month return is 47.54%,
versus 44.08% for the NLL-selected assembly and 30.27% for the ranking-selected
assembly. These are historical research outcomes, not expected returns.

## What the matched input controls show

| Family | Accepted joint IC | Family GP with same joint mask | Regional GP with same joint mask | Family-specific ranking lead |
| --- | ---: | ---: | ---: | ---: |
| U.S. equity | 0.133 | 0.028 | 0.083 | 0.325 |
| International equity | 0.142 | 0.008 | 0.047 | 0.261 |
| U.S. fixed income | 0.125 | 0.258 | 0.181 | 0.350 |
| International fixed income | 0.125 | 0.118 | 0.132 | 0.438 |

The equity results do not support the simple explanation that task sharing is
always harmful: with inputs held fixed, the joint GP ranks equities better than
the split models. Family-specific masks reverse that comparison. U.S. fixed
income benefits from splitting even with the common mask; international fixed
income is roughly unchanged until its inputs change.

The useful inference is an interaction between feature mask, fitted task
universe, and fitting budget—not a universal rule to join or split everything.
Every fit stopped at the 15-iteration cap. These are comparisons under the
approved compute budget, not comparisons at verified optimizer convergence.

## Predictive density and portfolio calibration

Exact observed predictive covariance includes likelihood noise and retains
cross-ETF dependence. Each family ranking lead has better joint NLL than the
accepted joint GP's marginal distribution on that same ETF subset:

| Same ETF subset | Accepted joint NLL per asset | Family ranking lead NLL per asset |
| --- | ---: | ---: |
| U.S. equity | -1.870 | -2.021 |
| International equity | -2.094 | -2.371 |
| U.S. fixed income | -3.149 | -3.429 |
| International fixed income | -2.997 | -3.056 |

These are same-subset comparisons; do not compare absolute NLL across different
families or confuse these values with marginal NLL in the first table.

For the full joint models, fixed equal-weight portfolio scores expose remaining
under-dispersion:

| Joint GP | Full joint NLL | Equal-weight NLL | Error RMS / predicted SD RMS | 95% coverage |
| --- | ---: | ---: | ---: | ---: |
| Accepted 14-input baseline | -38.873 | -2.017 | 1.524 | 83.3% |
| 12-input ranking/NLL lead | -39.093 | -2.258 | 1.408 | 84.7% |
| 12-input portfolio lead | -38.922 | -2.016 | 1.595 | 83.3% |
| 22-input diversity candidate | -38.572 | -1.587 | 1.823 | 79.2% |

The RMS ratio targets one. The 12-input ranking/NLL lead improves this fixed
portfolio's NLL and CRPS, but still understates forecast error magnitude. The
portfolio-leading joint candidate improves Sharpe without improving this risk
calibration. Lower individual predictive SD would not itself be progress.

Independent family/regional GPs do not supply learned cross-family covariance.
Their assembled portfolios therefore use the common historical risk model;
no block-diagonal or independently sampled covariance is presented as a complete
GP forecast. Component GP calibration, historical-risk calibration, fixed
reference scores, and adaptive-weight diagnostics remain separately labeled.

## Candidate masks and practical tradeoffs

The portfolio-selected assembly uses these non-time input counts:

| Family | Configuration | Inputs |
| --- | --- | ---: |
| U.S. equity | `us_equity_86c90676a157` | 5 |
| International equity | `international_equity_af9e3885f064` | 10 |
| U.S. fixed income | `us_fixed_income_6fcaaf0b83a3` | 8 |
| International fixed income | `international_fixed_income_f0aed11795f3` | 7 |

The NLL-selected assembly uses 9, 12, 7, and 11 non-time inputs, respectively:
`us_equity_e7028305d052`, `international_equity_3c896548ff79`,
`us_fixed_income_3199c073b636`, and
`international_fixed_income_373dc9719ca5`.

Important scope distinction: that historical U.S.-equity NLL mask contains
`hy_spread_chg_1p` **only in its U.S.-equity component**, not in a monolithic
joint GP or other families. The portfolio-selected assembly contains no such
input in any component. No previously selected family mask was silently changed.

There is no shared non-time input in all four selected masks in either assembly.
That does not prove no universal signal exists: these are conditional, selected
configurations. Their feature unions contain 23 inputs for the portfolio assembly
and 26 for the NLL assembly. Lower per-model dimensionality therefore does not
mean fewer maintained data series than a 14-input joint GP.

The two useful compact joint references are `global_bb1b8f9500f2`
(ranking/NLL lead) and `global_d8737126b4e7` (portfolio lead). Complete ordered
masks, commands, and shortlist roles are in
[shortlist.json](outputs/shortlist.json) and [fit_contract.json](outputs/fit_contract.json).

## Impact and proposed next step

Retain the portfolio-selected and NLL-selected family assemblies as research
leads, alongside the accepted baseline and the two 12-input joint alternatives.
The NLL assembly is an interesting compromise: equal-family IC rises from 0.131
to 0.249, pooled IC stays close to baseline, and primary Sharpe rises to 1.986.
It is not the best marginal-NLL model or an approved live replacement.

Do not promote the ranking-only family assembly merely for its local IC. Do
not promote the portfolio-selected assembly solely for its selected Sharpe.
Cross-family comparison, portfolio construction, and predictive risk remain
essential. A future assembly also needs an explicit cross-family risk model
before claiming a complete GP covariance forecast.

The paired three-month block-bootstrap Sharpe-difference intervals versus the
accepted baseline are [-0.275, 1.130] for the portfolio assembly and
[-0.394, 0.854] for the NLL assembly. Both overlap zero. The corresponding
exploratory equal-family IC improvement for the NLL assembly is 0.118 with
interval [0.023, 0.222]. These intervals are conditional, not selection-adjusted.
All 24 months have been reused across extensive feature selection; three seeds
are not 72 independent months. Calibrated optimization and new holdout discipline
remain deferred, as requested. No production model or allocation was promoted.

## Findings and verification

`outputs/cached_configuration_metrics.csv` contains all 501 screens; 497 have
complete strict-allocator results for all seeds, while four have one failed seed
and no aggregate primary Sharpe. The initial preflight stopped at the strict
solver failure; its evidence and source snapshots are preserved. No previous
weights, clipping, renormalization, or constraint relaxation was substituted.
All final architecture allocations solved successfully.

The 1,800 optimizer logs show rank 2, maxiter 15, one optimizer attempt per fit,
and 1,800 `STOPPED` statuses at the cap. Training panels contain exactly 84+k
dates per task. Every replayed forecast matches its original saved run exactly;
the largest legacy-versus-dense difference is 6.87e-16. The numerical libraries
emitted training Cholesky-jitter warnings; original fitting behavior was retained.
Exported predictive covariance was not eigenvalue-clipped or otherwise repaired.

Fit and batch evaluation took 1,053 seconds (17.6 minutes), using 14 workers.
The largest recorded worker peak was 2.95 GiB. Source/data/output hash verification,
an independent weight/return/Sharpe reconstruction, 52 tests, Ruff, and the public
docstring gate passed. Existing study sources/data/results and production code
remain unchanged. Reversing this research requires only setting aside this new
subfolder; there are no production or Git changes to revert.

Exact verification commands, from the repository root:

```sh
PYTHONPATH=experiments/2026-10-portfolio-evaluation-v1 .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/selection_refresh_v1/refresh.py --verify-only
PYTHONPATH=experiments/2026-10-portfolio-evaluation-v1 .venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1
.venv/bin/ruff check experiments/2026-10-portfolio-evaluation-v1/selection_refresh_v1
.venv/bin/python scripts/check_new_public_docstrings.py --files experiments/2026-10-portfolio-evaluation-v1/selection_refresh_v1/refresh.py experiments/2026-10-portfolio-evaluation-v1/selection_refresh_v1/refresh_math.py experiments/2026-10-portfolio-evaluation-v1/selection_refresh_v1/refresh_workers.py
```

Machine-readable results: [architecture summary](outputs/architecture_summary.csv),
[complete architecture scores](outputs/architecture_results.json),
[same-subset family scores](outputs/matched_family_metrics.json),
[exact density comparisons](outputs/exact_density_comparisons.json),
[paired portfolio comparisons](outputs/paired_exploratory_comparisons.json),
and [timing](outputs/timing.json).
