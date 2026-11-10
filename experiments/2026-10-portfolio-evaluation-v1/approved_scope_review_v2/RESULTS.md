# Corrected ETF experiment findings and checkpoint review

The follow-up study answered several October 22 questions, most importantly the
controlled U.S. equity comparison and the effect of uncertainty calibration. It
does not establish one universally best model or covariance source. Input masks,
diversification and the binding risk constraint interact with model selection.

The approved comparison policy now treats the original variance budget as the
primary analysis, 1.5 times that budget as sensitivity, and half the budget as a
stress test. All effective-count floors remain in the record. The six infeasible
historical-risk stress cases are retained but excluded from matched covariance
comparisons. The corrected data contain 366 complete and six infeasible policies,
with no unresolved numerical allocation failures.
All 228 primary-budget policies and all 72 looser-budget sensitivity policies
are complete. The six infeasible policies occur only in the 72-case stress group.

## What the experiment was trying to establish

The practical goal is to determine which use of multitask GPs produces useful
monthly ETF rankings, portfolios and uncertainty estimates. These are separate
questions: a good rank forecast can coexist with poor covariance calibration,
and portfolio Sharpe can change without any change in forecast quality.

The enhanced evaluation framework and full 499-configuration rerun came first.
They added saved full observed-return predictive covariance, monthly weights,
pooled and within-family rank IC, joint predictive NLL, portfolio-direction
calibration, fixed-portfolio distribution scores, and benchmark comparisons.
Sharpe uses conventional arithmetic excess returns. Returns, volatility, drawdown,
concentration and RMSE provide supporting diagnostics. Input count is a tie-break,
not an objective that overrides ranking or portfolio usefulness.

The latest follow-up reused that framework rather than rerunning all 499 masks.
It evaluated 30 complete-portfolio models and 372 allocation policies across
three seeds and 24 forecast origins. It completed 648 missing monthly GP fits and
four replay checks. A monthly fit is not a distinct feature configuration.

The additions addressed seven approved areas:

| Area | New comparison |
| --- | --- |
| Diversification | Effective-count floors none, 3, 5, 7, 10 and 12, crossed with three variance budgets |
| U.S. forecast collapse | Five versus 14 non-time inputs, each trained on U.S. equities alone and on all 15 ETFs |
| Complete family models | All 16 combinations of existing local-IC and conditional portfolio-selected masks |
| Task grouping | Existing shared inputs fitted to singles, pairs, triples and all four families |
| Uncertainty calibration | Raw GP, strictly chronological scaling, three GP/historical blends, diagonal GP and historical covariance |
| Allocator sensitivity | Rank, concentration-penalized rank and returns shrunk toward historical means |
| Ranking scope | Ranking inside each family versus ranking across the full portfolio universe |

The GP fit is not directly maximizing portfolio Sharpe. The default allocator
maximizes normalized forecast ranks under the declared constraints. IC, NLL,
Sharpe and calibration evaluate different properties of the resulting models and
decisions; they were not collapsed into an undisclosed weighted score.

Controls remain rank 2, maxiter 15, 84 initial expanding training months, seeds
27/127/227 and origins May 2024 through April 2026. These are 24 reused economic
months, not 72 independent months. There are no new feature searches, kernel
redesigns, training-window or holdout studies. There are no CVaR constraints,
position caps, turnover objectives, weight fallbacks or automatic promotions.
Returns and Sharpe are gross because no trading-cost series was supplied.

## U.S. equity deterioration is not explained by joint training alone

The controlled comparison is the most important change from the earlier
interpretation. SPY tracking is averaged across the same three seeds below.
Amplitude is prediction standard deviation divided by actual standard deviation.

| Non-time inputs | Tasks trained | SPY time-series Spearman | SPY amplitude ratio | U.S. cross-sectional IC |
| --- | --- | ---: | ---: | ---: |
| 5 | U.S. equities only | 0.575 | 0.916 | 0.192 |
| 5 | All 15 ETFs | 0.588 | 0.828 | 0.258 |
| 14 | U.S. equities only | 0.333 | 0.367 | 0.028 |
| 14 | All 15 ETFs | 0.235 | 0.334 | 0.133 |

The 14-input model flattens U.S. forecasts even without non-U.S. tasks. The
five-input joint model tracks SPY approximately as well as the five-input U.S.-only
model. IWM time-series correlation also improves from 0.270 to 0.343 when joint
training uses the five-input mask. This does not mean every U.S. ETF improves:
VTV tracking correlation declines in that same comparison.

The results weaken the explanation that joint training itself necessarily ruins
U.S. forecasts. They implicate input configuration and its induced fitted kernel
behavior, with some additional task-sharing effects. They do not isolate a
particular variable, lengthscale, noise estimate or convergence mechanism.

Making the entire joint model use the U.S. mask is not a global solution. Pooled
IC falls from 0.297 for the 14-input baseline to 0.215, while international-equity
IC falls from 0.142 to 0.014. U.S.-specific improvements trade off against other
tasks. The general architecture remains a research choice rather than a proven
failure or winner. BoTorch's multitask model shares a learned task relationship;
this does not imply identical forecast quality across families.
[BoTorch model definitions](https://botorch.org/docs/models)

## GP risk has a conditional portfolio advantage

At the primary budget and effective count at least ten, GP risk beats historical
risk on Sharpe for all six selected joint models. The mean difference is +0.151.
This confirms the earlier floor-ten finding in a smaller selected follow-up set.
It is not six independent market experiments or a new replication period.

| Joint model | GP risk Sharpe | Historical risk Sharpe |
| --- | ---: | ---: |
| 14-input baseline | 1.611 | 1.463 |
| 12-input portfolio reference | 1.585 | 1.465 |
| 12-input ranking and density reference | 1.542 | 1.351 |
| 13-input GP portfolio reference | 1.570 | 1.502 |
| 10-input compact NLL reference | 1.543 | 1.365 |
| 5-input joint U.S. mask control | 1.497 | 1.295 |

The broader budget grid qualifies that conclusion. At the primary budget without
an effective-count floor, GP wins for four of six models. At the 1.5-times budget
without a floor, historical risk wins for all six. Risk-source performance is not
invariant to the investment constraints.

There is a mechanism behind the floor-ten advantage. For the six joint models,
the raw GP risk constraint binds in zero of 432 monthly allocations, while the
historical risk constraint binds in 356 of 432. GP portfolios therefore follow
ranking and diversification without being limited by the risk ceiling. Their mean
realized annual volatility is 9.49%, versus 7.82% under historical risk. These are
the same numeric predicted-risk budgets, not equal realized-risk portfolios.

Removing GP off-diagonal covariance also leaves floor-ten Sharpe essentially
unchanged, while materially damaging portfolio uncertainty scores. Thus higher
floor-ten Sharpe does not establish that superior learned correlations caused
better allocations. In that setting the covariance constraint is inactive.
Portfolio variance and its role in risk-constrained allocation are described in
the [Stanford CVXPY portfolio example](https://www.cvxgrp.org/cvx_short_course/docs/applications/notebooks/portfolio_optimization.html).

## Moderate diversification looks more promising than forcing twelve

For the 14-input baseline with GP risk and the primary variance budget:

| Minimum effective count | Sharpe |
| --- | ---: |
| None | 1.606 |
| 3 | 1.657 |
| 5 | 1.712 |
| 7 | 1.692 |
| 10 | 1.611 |
| 12 | 1.538 |

Across the six selected joint models, floor seven has the highest mean GP-risk
Sharpe among the tested floors: 1.590, versus 1.508 without a floor. This is a
promising region, not a universally optimal floor. Different feature masks favor
different allocations. More diversification lowers concentration and often
realized volatility, but is not a monotonic Sharpe improvement.

At floors ten and twelve, increasing the GP variance budget to 1.5 times barely
changes results because the original ceiling was already inactive. The looser
budget changes historical-risk allocations much more. The remaining infeasible
stress combination asks historical variance to be at most half the original
budget while requiring effective count at least twelve. Its lowest attainable
variance is 58.6 to 59.6% of the original budget, across the 24 months.

## Calibration improves uncertainty without necessarily improving Sharpe

The following fixed equal-weight scores hold the evaluated portfolio constant.
The error RMS to predicted SD RMS ratio targets one. Coverage targets 95%.
The blend fraction is the historical covariance weight.

| Baseline covariance | RMS ratio | 95% coverage | Fixed portfolio NLL | Floor-ten Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Raw GP | 1.524 | 83.3% | -2.017 | 1.611 |
| GP with chronological variance scaling | 1.341 | 87.5% | -2.208 | 1.576 |
| 75% GP and 25% historical | 1.112 | 91.7% | -2.347 | 1.611 |
| 50% GP and 50% historical | 0.918 | 91.7% | -2.370 | 1.573 |
| Historical | 0.718 | 100.0% | -2.296 | 1.463 |
| GP diagonal only | 2.797 | 54.2% | 0.503 | 1.611 |

Both chronological scaling and the tested blends improve the baseline's
equal-weight uncertainty diagnostics. Blending helps across the selected joint
references, but a portfolio-specific improvement is not proof of calibration in
every direction. Historical covariance alone tends to overstate equal-weight
uncertainty here, whereas raw GP covariance understates it.

Calibration is not automatically a Sharpe enhancement. Chronological scaling
reduces no-floor Sharpe for five of the six models. Its scalar adjustment is
estimated using earlier equal-weight forecast errors only, with an explicit prior,
so this is a real adaptive comparison rather than a full-sample fitted correction.
For the baseline it improves fixed equal-weight NLL but worsens full-vector joint
NLL from -2.592 to -2.549 per asset. Different portfolio directions and scores can
favor different covariance adjustments.

The 25%-historical blend is a useful next comparator: it improves the baseline's
uncertainty scores with essentially unchanged floor-ten Sharpe. It is not an
automatically promoted calibrated model. No family architecture was assigned an
invented block-diagonal GP covariance.

## Joint NLL remains a distinct objective

The compact 10-input model still has better raw-GP joint NLL per asset than the
baseline, -2.621 versus -2.592, but worse pooled ranking IC, 0.205 versus 0.297.
Its fixed equal-weight RMS ratio is also worse, 1.708 versus 1.524. These are
the same reference predictions, so their ranking did not change in this follow-up.

This does not make joint NLL invalid. It shows that predicting the joint density,
ranking ETFs and allocating a portfolio are different objectives. The new
calibration and diagonal controls reinforce that distinction: uncertainty can
improve with unchanged IC, or deteriorate with unchanged portfolio weights.

## Selected family assemblies are a stronger lead than arbitrary task splits

Previously each family candidate replaced only that family in an otherwise
frozen 15-ETF forecast panel. The new study tested complete combinations, so it
directly addresses whether conditional family choices can work together.
For architecture comparisons the covariance is historical for every row below.

| Complete forecasting architecture | Pooled IC | No-floor Sharpe | Floor-ten Sharpe |
| --- | ---: | ---: | ---: |
| Joint 14-input baseline | 0.297 | 1.451 | 1.463 |
| Four families with the same shared 14-input mask | 0.264 | 1.516 | 1.457 |
| U.S. equity plus U.S. fixed income, and international equity plus international fixed income | 0.270 | 1.457 | 1.375 |
| Both equity families together, both fixed-income families together | 0.254 | 1.624 | 1.398 |
| Four selected local-IC family masks | 0.200 | 1.038 | 1.507 |
| Best selected family combination | 0.326 | 2.454 | 1.678 |

The best observed combination uses the existing conditional portfolio-selected
masks for international equity, international fixed income and U.S. equity,
plus the local-IC-selected U.S. fixed-income mask. It is `family_assembly_1110`
in the machine-readable evidence. Its no-floor portfolio has mean effective
count only 2.16, so the large no-floor Sharpe is not a broadly diversified result.
Its floor-ten Sharpe is lower but still leads the selected family combinations.

The assembly of four local-IC winners has much stronger within-family ICs than
this portfolio-leading assembly, yet much weaker pooled IC. Selecting the best
model inside each family independently does not guarantee good cross-family
rankings or portfolio performance. The complete 16-combination test answers
that previously open question. Relative forecast levels across families matter.

The matched-input grouping results do not support a universal geographic split.
Task sharing also differs by family: under the shared 14-input mask, U.S.
fixed-income local IC is 0.258 on its own versus 0.125 in the full joint model,
while international-equity local IC rises from 0.008 alone to 0.142 in full joint.
Family-specific masks and task grouping should not be conflated.

## Allocation alternatives provide no universal replacement

The repaired concentration-penalized rank allocator completes every configuration.
Without a count floor it improves Sharpe for three of six models under each risk
source, but the average change is slightly negative: -0.034 for GP risk and
-0.032 for historical risk. At floor ten it changes Sharpe by less than one
millionth in these comparisons, because the hard concentration constraint is
already determining the relevant weight concentration.

The tested 50/50 shrinkage of predicted returns toward training historical means
improves Sharpe for only one of six models in each risk-source/floor comparison.
It does not provide evidence to replace rank allocation. This is one particular
shrinkage rule, not a rejection of every return-magnitude allocator.

## Answers to the October 22 checkpoint

| Open question | Result |
| --- | --- |
| Does diversification help and which level is promising? | Partly answered. Moderate GP-risk floors five to seven are promising; twelve is feasible at the original budget but not a consistent Sharpe winner. |
| Is GP risk universally better than historical risk? | No. It retains a floor-ten Sharpe advantage, but the budget grid reveals reversals and inactive GP risk ceilings. Covariance accuracy is not established by that Sharpe advantage. |
| Can uncertainty calibration help? | Yes for fixed-portfolio uncertainty in this test. Scaling and blending improve diagnostics; Sharpe and full-vector NLL need not improve with them. |
| What causes U.S. equity deterioration? | Substantially narrowed. The broad input mask reproduces flattening in U.S.-only fits, and the compact U.S. mask largely restores SPY tracking even in the full joint model. The individual feature or fitted-hyperparameter mechanism remains unresolved. |
| Can independently selected family models work together? | Answered for the tested candidates. Some complete assemblies beat the joint baseline; combining local-IC winners is not sufficient. |
| Would geographic task grouping solve the problem? | Not as a general rule under matched inputs. |
| Should compact joint-NLL winners replace the ranking baseline? | No support for that conclusion. NLL, ranking and portfolio uncertainty remain distinct. |
| Did this identify a deployment winner? | No. Results use 24 reused exploratory months, fixed short GP fitting budgets and gross returns. No untouched-period or new stability study was performed. |

## Applied changes and verification

`policy.json` records the approved primary, sensitivity and stress roles.
`feasibility.allocate_checked` is the repaired research allocation entrypoint for
future evaluations. It runs training-only feasibility checks first and explicitly
blocks an impossible or numerically undetermined policy. It never increases the
budget, substitutes weights or labels a numerical failure as infeasibility.

`outputs/review.json` retains all 372 policies, all six infeasible cases, fixed
reference scores, controlled tracking metrics and matched comparisons. There are
102 eligible rank/covariance pairs. The six unmatched GP half-budget/floor-twelve
cases remain recorded but do not count as GP wins over an infeasible comparator.
All original experiment and numerical-repair artifacts remain unchanged.

Run from the repository root:

```sh
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/approved_scope_review_v2/review.py --verify-only
.venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/approved_scope_review_v2
.venv/bin/ruff check experiments/2026-10-portfolio-evaluation-v1/approved_scope_review_v2
```

The review is reproducible from sealed JSON/array evidence. It fits no GPs and
runs no new return-optimizing strategy allocations. The original study and the
numerical repair retain separate integrity checks. The report is part of the
review's input hash manifest. Rollback is setting aside this new review folder;
production and completed experiments need no changes.

Verification passed: 111 framework tests, Ruff, the public-docstring gate and
integrity checks for the original study, numerical repair and this review.
Pylance diagnostics are not exposed in this environment and were not verified.
