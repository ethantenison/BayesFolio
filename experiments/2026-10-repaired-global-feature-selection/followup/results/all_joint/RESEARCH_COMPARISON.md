# Joint feature selection after both follow-up batches

## Recommendation

Keep the fourteen-input SPY-proxy addition (`global_e6c41f2020fd`) as the primary
research lead. It has the highest pooled IC among all 37 tested joint masks,
substantially higher Sharpe than the old lead, and better NLL. Retain two explicit
alternatives: SPY plus monthly CPI for maximum observed mean Sharpe, and removal
of DXY from the old lead for minimum NLL and fewer inputs. There is no candidate
that wins all three objectives. This is a research recommendation, not production
promotion or evidence that every ETF family benefits.

Both authorized follow-up batches are complete. Batch two's dispatcher exited
zero, all eighteen new seed runs validated, and no dispatcher children remained.
The combined independent audit passed all **37 configurations / 111 seed runs**:
25 original masks plus twelve follow-up masks, with seeds 27, 127 and 227 each.
All share 84 initial training decision months (May 2017–April 2024), then 24
expanding scored decision months (May 2024–April 2026). The scored forward-return
outcomes end in May 2026. No additional fits are planned under this campaign.

## Full three-objective frontier

Numbers are means of the three complete seed runs. Counts exclude `t_index`,
which is retained in every model. Lower marginal Gaussian NLL is better;
negative NLL values are valid for continuous densities in decimal-return units.

| Configuration | Inputs | IC ↑ | NLL ↓ | Conventional Sharpe ↑ | On final front |
| --- | ---: | ---: | ---: | ---: | --- |
| Original all-input control | 27 | 0.108780 | -2.141413 | 1.127881 | No |
| Previous research lead | 13 | 0.264881 | -2.250209 | 1.622642 | No |
| Previous lead + SPY proxy | 14 | **0.296528** | -2.258812 | 1.771496 | Yes |
| Previous lead + SPY proxy + monthly CPI | 15 | 0.286409 | -2.243060 | **1.807445** | Yes |
| Previous lead minus DXY | 12 | 0.281448 | **-2.307496** | 1.390330 | Yes |

The SPY-only addition dominates the previous lead in every objective in every
seed. Its IC/NLL/Sharpe seed SDs are 0.008099 / 0.002709 / 0.033957.
The CPI pair's corresponding SDs are 0.014097 / 0.004830 / 0.058856; DXY removal's
are 0.018554 / 0.004699 / 0.090400. These describe seed variability on the same
market history, not confidence intervals for future performance.

Relative to SPY alone, adding monthly CPI increases mean Sharpe by only 0.035949,
while decreasing IC by 0.010119 and worsening NLL by 0.015752. Sharpe improves in
two seeds, not all three; IC improves in one; NLL worsens in all three. Keep this
as a genuine Pareto alternative, but the extra input is not an unqualified gain.

DXY removal improves NLL in all three seeds versus the old lead and IC in two,
but lowers Sharpe in all three (mean change -0.232312). This is a meaningful
compact density/debt-ranking alternative, not a costless pruning decision. It
also dominates the previous twelve-input density-front candidate
`global_da43a19fc3dd` on all three seed-mean objectives; that older point leaves
the frontier. Dominance of means does not assert dominance in every seed.

Fixed-reference hypervolume, using the unchanged raw reference `(0, 0, 0)` for
maximizing IC, negative NLL and Sharpe, progresses as follows:

| Evidence set | Joint masks | Seed runs | Hypervolume |
| --- | ---: | ---: | ---: |
| Original completed study | 25 | 75 | 1.016778459 |
| After first follow-up batch | 31 | 93 | 1.187618053 |
| After second follow-up batch | 37 | 111 | 1.228693994 |

Batch two adds 3.46% over batch one; both follow-ups together add 20.84% over the
original front. Hypervolume is reference- and scale-dependent, not a return or
a statistical significance measure. Input count is descriptive, not a fourth
hidden objective.

## What all twelve follow-up configurations taught us

All operations below are relative to the original thirteen-input lead, not
silently rebased onto the SPY winner. Full results for all 37 masks, exact inputs
and seed SDs are in [TABLES.md](TABLES.md) and the adjacent CSV/JSON files.

| Follow-up operation | IC | NLL | Sharpe | Interpretation in this context |
| --- | ---: | ---: | ---: | --- |
| Add HY level | 0.263740 | -2.241139 | 1.631714 | Dominated; no transferable global gain |
| Add HY change | 0.277728 | -2.242970 | 1.643740 | IC improves every seed; NLL cost; dominated by SPY |
| Add SPY proxy | 0.296528 | -2.258812 | 1.771496 | Primary research lead; highest IC |
| Add monthly CPI | 0.266667 | -2.238164 | 1.610998 | Weak standalone addition; dominated |
| Add SCHP return | 0.167708 | -2.164351 | 1.475153 | All mean global objectives worse despite U.S. debt IC benefit |
| Add nominal yield | 0.253720 | -2.225665 | 1.432244 | Every global objective worse in every seed |
| Add SPY + monthly CPI | 0.286409 | -2.243060 | 1.807445 | Highest mean Sharpe; real IC/NLL cost versus SPY alone |
| Add nominal yield + SCHP | 0.181746 | -2.203249 | 1.397071 | Every global objective worse in every seed; interaction did not rescue it |
| Remove six-month momentum | 0.269296 | -2.249705 | 1.539813 | IC +0.004415, nearly unchanged NLL, Sharpe -0.082829 |
| Replace six-month with annual momentum | 0.260863 | -2.239131 | 1.500076 | All mean global objectives worse |
| Replace trend with MA signal | 0.265575 | -2.249972 | 1.534868 | Tiny mean IC gain, nearly unchanged NLL, Sharpe -0.087774 |
| Remove DXY | 0.281448 | -2.307496 | 1.390330 | New compact density-front point; important Sharpe cost |

Removing six-month momentum and replacing trend both reduce Sharpe in every
seed. Removing momentum nevertheless raises U.S./international equity IC to
0.252778/0.230556, illustrating why neither pooled IC nor portfolio Sharpe alone
establishes family-level feature importance. The nominal/SCHP pair leaves
international debt IC negative (-0.020833), despite support for those variables
in earlier separate-family studies. Importance is conditional on other inputs,
task sharing, kernel geometry and the fixed finite fitting budget.

## Family outcomes: the remaining limitation

These are cross-sectional ICs within each family, not the fifteen-task pooled
IC. The four families have four, four, four and three ETFs respectively.

| Model | U.S. equity IC | International equity IC | U.S. debt IC | International debt IC |
| --- | ---: | ---: | ---: | ---: |
| All-input control | 0.172222 | 0.041667 | -0.097222 | 0.222222 |
| Previous lead | 0.208333 | 0.183333 | 0.038889 | 0.090278 |
| Add SPY | 0.133333 | 0.141667 | 0.125000 | 0.125000 |
| Add SPY + monthly CPI | 0.080556 | 0.077778 | 0.111111 | 0.131944 |
| Remove DXY | 0.133333 | 0.172222 | 0.261111 | 0.208333 |

SPY's global gains coexist with lower within-equity IC than the previous lead.
Adding CPI lowers both equity-family ICs further. DXY removal is much stronger
for debt ranking, and nearly restores international debt's all-input IC, but
still does not beat that control within international debt. Pooled ranking
can improve through between-family ordering without improving every family's
internal ordering; it must not be interpreted as a macro-average of family ICs.

Family NLL and coverage retain material heterogeneity:

| Frontier model | Family | NLL | 95% coverage | Mean capital allocation |
| --- | --- | ---: | ---: | ---: |
| Add SPY | U.S. equity | -1.742457 | 93.75% | 55.48% |
| Add SPY | International equity | -1.771089 | 91.32% | 37.22% |
| Add SPY | U.S. debt | -2.853372 | 95.49% | 3.25% |
| Add SPY | International debt | -2.804836 | 97.22% | 4.05% |
| Add SPY + CPI | U.S. equity | -1.728288 | 94.44% | 57.62% |
| Add SPY + CPI | International equity | -1.759741 | 91.67% | 37.34% |
| Add SPY + CPI | U.S. debt | -2.826650 | 95.14% | 2.44% |
| Add SPY + CPI | International debt | -2.795728 | 97.22% | 2.60% |
| Remove DXY | U.S. equity | -1.797828 | 96.18% | 54.60% |
| Remove DXY | International equity | -1.853209 | 91.67% | 37.25% |
| Remove DXY | U.S. debt | -2.878768 | 95.49% | 5.10% |
| Remove DXY | International debt | -2.831071 | 95.83% | 3.06% |

DXY removal improves mean NLL in every family versus the previous lead.
Nevertheless, global allocation Sharpe is equity-dominated: equities receive
about 92.7%, 95.0% and 91.8% of mean capital in these three frontier models.
It does not validate a standalone debt allocation strategy. Within-family
allocation diagnostics, including undefined sparse-sleeve Sharpe values, remain
in the CSVs; no missing sleeve Sharpe is filled with zero.

## Forecast uncertainty and diagnostic outcomes

| Model | Mean predictive SD | 50% coverage | 80% coverage | 95% coverage | Cumulative excess-return diagnostic | RMSE diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All-input control | 0.026795 | 47.31% | 76.39% | 90.93% | 34.71% | 0.033576 |
| Previous lead | 0.027480 | 50.56% | 80.83% | 94.07% | 51.37% | 0.031601 |
| Add SPY | 0.027503 | 53.61% | 81.85% | 94.26% | 56.30% | 0.030967 |
| Add SPY + CPI | 0.027508 | 52.69% | 80.46% | 94.44% | 56.69% | 0.031457 |
| Remove DXY | 0.027656 | 55.37% | 82.78% | 94.72% | 41.46% | 0.029803 |

Predictive SD includes observation noise. The improved masks are slightly wider,
not narrower, than the old lead; DXY removal has the lowest RMSE and best NLL
while being the widest here. Better calibration/error tradeoffs, rather than
blanket lower uncertainty, are the supported conclusion. Coverage is descriptive
on the reused sample and not a formal calibration guarantee. Return is the
mean seed compound path of the already-excess targets, not CAGR or total return;
RMSE is calculated over each full forecast panel before averaging seeds. Neither
metric nor turnover participates in Pareto selection.

## Exact inputs and conditional next questions

Primary research lead `global_e6c41f2020fd`:

```text
t_index
lag_y_excess_lead, mom12m_skip1m, mom6m, trend_slope
cpi_chg_12p, dealer_gamma_proxy, dxy, em_fx_ret, gold_crude_ratio
hy_spread_z_12p, oil_ret, spy_ret, term_spread, vix_ts_z_12p
```

The Sharpe alternative `global_4397c0b6828e` adds `cpi_chg_1p` to that exact
list. The compact NLL alternative `global_bb1b8f9500f2` removes both `spy_ret`
and `dxy` from the primary list: its actual experiment was DXY removal from the
old lead, which never contained SPY. All three retain lagged y. These results
do not establish that lagged y can be removed from every family or the joint
model. Exact lists for every tested configuration are in `exact_inputs.json`.

The most informative *future*, not executed, question is whether the SPY branch
can retain its IC/Sharpe while dropping DXY, followed by conditional momentum
or lagged-y pruning. The current DXY and momentum ablations cannot answer that
because they use the old parent. Likewise, a common feature core remains a
useful approximation, not proof of equal importance to all targets. No third
follow-up batch, new holdout, calibrated optimizer or production change was
authorized by this comparison.

## Verification and interpretation boundaries

The comparator checked all registered run signatures, exact inputs, initial and
expanding training lengths, unique complete three-seed coverage, and finite
objectives. It independently reconstructed IC, marginal Gaussian NLL,
conventional Sharpe, return, RMSE and family forecast diagnostics from saved
predictions, weights and actual returns. All 111 passed; no unresolved audit
items remain. The audit tracks 506 source/evidence hashes and rechecks freshness.
Original completed-study final evidence hashes and frozen data passed unchanged.
The GP/portfolio implementation was not edited; no commit or push was performed.

Conventional Sharpe uses arithmetic monthly excess mean divided by sample SD
(`ddof=1`), annualized by `sqrt(12)`, with no second risk-free subtraction.
Fits retain signed LKJ eta 2, task rank 2, maxiter 15 and 256 scenarios, with
the unchanged main-plus-pairwise kernel rule and no time modulation. Feature
count changes alter dimension-dependent GP prior geometry under that same rule;
these are fixed-budget predictive comparisons, not causal relevance estimates.
See the original [model contract audit](../../../results/MODEL_CONTRACT_AUDIT.md)
for the inherited formula and legacy metadata distinctions. `spy_ret` retains
its frozen legacy proxy definition; it was not redefined as monthly total return.

The 24 dates were repeatedly used for adaptive research; this is not untouched
holdout confirmation. Three seeds do not resolve economic sampling uncertainty.
The frozen retrospective adjusted-price/latest-vintage macro limitations remain.
Training coverage did not change between these comparable joint configurations.

Reproduce the read-only comparison and checks from the repository root:

```bash
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/compare_joint_followup.py
.venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection
.venv/bin/ruff check experiments/2026-10-repaired-global-feature-selection
```

Verification: comparison exited zero; 71 tests passed; Ruff passed. The comparison
does not launch fits. `comparison.json`, `independent_audit.json`, `pareto.csv`,
`objectives_by_seed.csv`, `family_mean_diagnostics.csv`, `monthly_diagnostics.csv`
and `conditional_feature_pairs.csv` preserve the machine-readable evidence.
