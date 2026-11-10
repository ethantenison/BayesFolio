# ETF model rankings under both portfolio policies

## Main conclusions

The 499 configurations do not produce one winner across ranking, predictive
density, portfolio performance and risk calibration. The accepted 14-input joint
baseline remains the best pooled-ranking joint model. A 12-input joint model
leads historical-covariance portfolio Sharpe without an effective-count floor.
An effective count of ten improves typical joint-model Sharpe and seed stability,
but changes which feature mask wins.

The most consistent diagnostic concern is portfolio uncertainty: every one of
the 35 joint GPs understates errors for the fixed equal-weight portfolio. Nominal
95% predictive coverage ranges from 68.1% to 84.7%, and error RMS divided by
predicted SD RMS ranges from 1.408 to 1.943, rather than one. A good full-vector
NLL does not guarantee good uncertainty for a portfolio direction.

These are exploratory comparisons over 24 reused forecast origins from May 2024
through April 2026, predicting the following month's return. All results average
seeds 27, 127 and 227. Returns are gross of costs; conventional Sharpe annualizes
arithmetic excess returns. Rank remains two, maxiter remains 15, initial training
remains 84 months, and all masks retain `t_index`. No model has been promoted.

## Joint model leaders

Each row is the best observed Sharpe among joint candidates for that exact
covariance source and allocation policy. Inputs exclude `t_index`; effective
count is the average of monthly inverse weight concentration, not a count of
nonzero holdings. The final column averages the largest monthly weight from
each seed, rather than imposing a position cap.

| Covariance | Effective floor | Configuration | Inputs | Sharpe | Baseline Sharpe | Annual volatility | 24 month return | Mean effective count | Largest weight |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical | None | `global_d8737126b4e7` | 12 | 1.783 | 1.451 | 8.25% | 44.87% | 2.67 | 92.44% |
| Historical | 10 | `global_fdb5fb8b385f` | 14 | 1.519 | 1.463 | 8.02% | 37.90% | 10.00 | 17.33% |
| GP predictive | None | `global_4a6348639308` | 13 | 1.729 | 1.606 | 13.85% | 71.60% | 1.93 | 100.00% |
| GP predictive | 10 | `global_d019284686e8` | 11 | 1.627 | 1.611 | 9.03% | 44.62% | 10.00 | 14.45% |

The baseline is `global_e6c41f2020fd`, using 14 non-time inputs. Equal-weight
Sharpe is 1.258 and fixed 60/40 Sharpe is 1.284 over the same months.

The historical/no-floor leader has seed Sharpes with mean 1.783, SD 0.074 and
minimum 1.701. Its maximum drawdown averages 6.02%. Its first and second 12-month
Sharpes are 2.333 and 1.521, so the whole-period lead is not uniform through time.
Relative to the baseline under identical constraints, its Sharpe improvement is
0.332; a paired circular three-month block bootstrap gives [0.054, 0.845]. This
interval is conditional on already selected models and reused months, not
selection-adjusted evidence of future superiority.

The other three leaders' corresponding improvement intervals all include zero:
historical/floor-ten [-0.095, 0.224], GP/no-floor [-0.595, 0.542], and
GP/floor-ten [-0.269, 0.287]. The latter improves baseline Sharpe by only 0.016.
The higher GP/no-floor return comes with substantially higher realized volatility
and concentration, despite using the same numeric variance budget.

The two diversified leaderboard winners contain raw `hy_spread`. That is not
`hy_spread_chg_1p`, which was excluded from all monolithic joint GPs. They are
recorded results, not permission to add another HY input to an accepted model.
Alternatives without raw `hy_spread` are `global_4a6348639308` for historical/floor-ten
(Sharpe 1.502) and the accepted baseline for GP/floor-ten (Sharpe 1.611).

## Paired effects of diversification

These compare the same model under both policies, not just each policy's winner.
Family rows use a floor of three; joint rows use ten. The incomplete historical
joint comparison is excluded, not filled from another seed or month.

| Candidate group | Covariance | Diversified Sharpe higher | Mean Sharpe change | Median Sharpe change |
| --- | --- | ---: | ---: | ---: |
| U.S. equity | Historical | 47 of 48 | +0.096 | +0.082 |
| International equity | Historical | 99 of 186 | -0.006 | +0.013 |
| U.S. fixed income | Historical | 105 of 105 | +0.166 | +0.177 |
| International fixed income | Historical | 125 of 125 | +0.175 | +0.178 |
| Joint | Historical | 20 of 34 | +0.031 | +0.015 |
| Joint | GP predictive | 24 of 35 | +0.112 | +0.060 |

For the accepted joint baseline, the GP-risk floor changes Sharpe only from
1.606 to 1.611, but reduces annual volatility from 13.69% to 9.63%, seed Sharpe
SD from 0.195 to 0.027, and largest weight from 100% to 14.45%. This is a
substantial portfolio-policy change, not a forecast improvement.

GP-based joint-model Sharpe ranks have Spearman correlation 0.647 across the two
policies. For example, `global_d019284686e8` moves from 32nd without the floor
to first with it. Feature selection therefore depends on the allocation policy.

## Family candidates are conditional replacements

Each family candidate replaces only that family's forecasts in the same
15-ETF panel. The other three families retain the frozen ranking-leading anchors.
These are not standalone family portfolios and not an assembly of all four
newly selected masks. NLL is scored on the modeled ETF subset; absolute NLL
must not be compared between different families.

| Family replaced | No floor leader | Inputs | Sharpe | Floor three leader | Inputs | Sharpe |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| U.S. equity | `us_equity_d0a35b28f60b` | 6 | 1.415 | `us_equity_86c90676a157` | 5 | 1.485 |
| International equity | `international_equity_ef2048f76023` | 10 | 2.164 | `international_equity_af9e3885f064` | 10 | 1.988 |
| U.S. fixed income | `us_fixed_income_6fcaaf0b83a3` | 8 | 1.177 | Same | 8 | 1.361 |
| International fixed income | `international_fixed_income_76d45097808c` | 19 | 1.356 | Same | 19 | 1.458 |

The international-equity no-floor leader has Sharpe at least 2.056 in every seed,
but its portfolio reaches 100% weight in one ETF. It is a useful conditional
research lead, not proof that a complete family architecture beats every joint
model. Combining these family winners can change cross-asset ranks and weights;
their individual improvements are not additive.

Neither set of four family portfolio winners shares a non-time input across all
four masks. This does not establish that no universal signal exists: the masks
were selected conditionally, and correlated substitutes were not causally isolated.

## Ranking and uncertainty references

| Joint reference | Inputs | Pooled ranking IC | Joint NLL per asset | Fixed equal-weight NLL | Fixed equal-weight RMS ratio | Fixed equal-weight 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accepted baseline `global_e6c41f2020fd` | 14 | 0.297 | -2.592 | -2.017 | 1.524 | 83.3% |
| Ranking and density compromise `global_bb1b8f9500f2` | 12 | 0.281 | -2.606 | -2.258 | 1.408 | 84.7% |
| Best joint NLL `global_b48732adb186` | 10 | 0.205 | -2.621 | -1.894 | 1.708 | 73.6% |
| Historical no-floor Sharpe leader `global_d8737126b4e7` | 12 | 0.269 | -2.595 | -2.016 | 1.595 | 83.3% |

Lower NLL is better; higher IC is better; the RMS ratio targets one. The best
joint-NLL model sacrifices ranking and has worse equal-weight calibration than
the accepted baseline. `global_bb1b8f9500f2` has the best equal-weight NLL,
closest equal-weight RMS ratio and highest equal-weight coverage among the
35 joint candidates, but weaker realized Sharpe. It remains a useful calibration
reference, not an automatic winner or a calibrated model.

The historical/no-floor Sharpe leader's adaptive historical-risk RMS ratio is
0.892 with 95.8% coverage. The GP/no-floor leader's GP-risk ratio is 1.439 with
88.9% coverage; the GP/floor-ten leader's is 1.634 with 81.9% coverage. These
adaptive diagnostics concern different selected portfolios. Fixed-reference
scores above are the proper comparisons holding evaluated weights constant.

The complete three-objective Pareto fronts are in `outputs/pareto.csv`:
maximize ranking IC, minimize observed joint NLL, maximize portfolio Sharpe,
separately within family, covariance source and policy. Family fronts use local
IC; joint fronts use pooled 15-ETF IC. Input count is recorded, not secretly
weighted into the objectives. There are 9 to 12 nondominated joint configurations
per covariance/policy group, so a single weighted winner is not implied.

## Research shortlist and next comparison

Retain the accepted baseline as the ranking control, `global_d8737126b4e7` as the
historical-risk portfolio lead, and `global_bb1b8f9500f2` as the fixed-portfolio
calibration reference. Keep `global_4a6348639308` as the GP/no-floor comparator.
Do not replace these with the lowest-NLL or smallest-mask candidate alone.

Before choosing a family architecture, evaluate complete assemblies using
selected family masks under the same two 15-ETF portfolio policies as the joint
models: no effective floor versus a floor of ten. Keep historical covariance
common across architectures; independent family GPs do not supply a learned
cross-family GP covariance. The current family floor of three cannot isolate an
architecture effect against joint models constrained to ten.

Before using GP covariance to enforce an economic risk limit, investigate its
portfolio-direction undercoverage against the historical-risk control. Keep
calibration work separate from feature-mask changes so improvements are
attributable. No new fits, calibration transformations, holdout redesign, or
production allocations were launched during this analysis.

## Recorded failures and verification

All 499 configurations and 1,068 covariance/policy rows were recorded; 1,067
rows have all seeds and all months solved. Configuration `global_32cfb52cf8ac`
received `optimal_inaccurate` in June 2024 for seeds 27 and 227 with historical
covariance and floor ten. Those two allocations remain null. The affected
configuration/policy is excluded from aggregate performance and paired ranking;
no prior weights, clipping, renormalization or constraint relaxation was used.

The analysis reads sealed upstream results and writes only this new directory.
Saved input and output hashes passed verification. The framework tests, Ruff
and public docstring gate passed. Pylance diagnostics are not exposed in this
environment and were not verified. Rollback is setting aside this analysis
directory; fitted studies and production code are unchanged.

Run from the repository root:

```sh
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/portfolio_policy_analysis_v1/analyze.py --verify-only
.venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1
.venv/bin/ruff check experiments/2026-10-portfolio-evaluation-v1/portfolio_policy_analysis_v1
```

Machine-readable outputs include [all rankings](outputs/rankings.csv),
[paired policy statistics](outputs/paired_policy_summary.csv),
[Pareto fronts](outputs/pareto.csv), [candidate masks](outputs/selected_masks.json),
[fixed-reference diagnostics](outputs/fixed_reference_scores.csv),
[seed diagnostics](outputs/seed_diagnostics.csv), and
[paired exploratory intervals](outputs/paired_exploratory_bootstrap.json).
