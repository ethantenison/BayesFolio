# U.S. Equity ETF Feature Experiment Decision Log

## 2026-07-05 - Student-t Predictive Tail Probe Is A Likelihood-Shape Clue, Not A GP Refit

Context: After the expanding-window Stage A input-block screen, `momentum_plus_macro`
had the strongest ranking/portfolio behavior while liquidity and volatility blocks looked
better calibrated. We tested simple heteroskedastic recombinations, fixed-noise probes,
and learned residual-scale calibration from liquidity/volatility features. Those did not
improve NLL versus the `momentum_plus_macro` baseline.

Learning: A post-fit Student-t predictive wrapper did improve NLL on the
`momentum_plus_macro` residual panel, including a 12-month check over 48 forecasts
from `2025-07-31` through `2026-06-30`.

- Gaussian baseline: NLL `-1.6929`, 80% coverage `0.8750`, 95% coverage `0.9375`.
- Student-t df=5 variance-matched: NLL `-1.7609`, 80% coverage `0.7917`, 95% coverage `0.9375`.
- Student-t df=7 variance-matched: NLL `-1.7486`, 80% coverage `0.8125`, 95% coverage `0.9375`.

Interpretation: The improvement is not visibly wider uncertainty bands. The Student-t
versions were variance-matched to the Gaussian predictive variance, so the plotted bands
change only slightly. The useful signal is distribution shape: more density near zero
and in the tails, less in the shoulder region. This helped most around large residual
dates such as `2026-03-31`, especially for SPY/IWM/MGK.

Caveat: This was not a jointly fitted Student-t GP likelihood. It was a predictive
likelihood/tail wrapper applied to existing Gaussian MTGP means and variances. Treat it
as a reason to test robust predictive likelihoods later, not as proof that a true
Student-t MTGP will win after refitting.

Decision: Do not spend the next run on more ad hoc heteroskedastic formulas. Move the
main search back to input-block confirmation with expanding windows. Preserve the
Student-t wrapper as a secondary scoring diagnostic for the shortlist.

## 2026-07-05 - Maxiter 50 Improves Density Metrics But Damages ETF Ranking

Context: Stage B suggested `momentum_plus_macro` was the best ranking/portfolio input
block, and a three-way `time x momentum x macro` interaction improved Gaussian NLL
without materially hurting the 12-month portfolio path at `maxiter=15`.

Confirmation: Stage C reran the same baseline and three-way candidates at `maxiter=50`,
then ran a seed panel for the baseline at seeds `27`, `127`, and `227`.

Learning: Across all three seeds, increasing `maxiter` from 15 to 50 improved RMSE/NLL
but reduced prediction dispersion and damaged cross-sectional ranking:

- Seed 27: return fell from `0.2383` to `0.1828`; mean IC fell from `0.2000` to `-0.3000`.
- Seed 127: return fell from `0.2093` to `0.1544`; mean IC fell from `-0.0167` to `-0.3500`.
- Seed 227: return fell from `0.2334` to `0.1728`; mean IC fell from `0.2667` to `-0.2167`.

Diagnostics: The damage is not isolated to one ETF. SPY is relatively stable, while
ranking errors move across MGK, VTV, and IWM depending on date/seed. Model diagnostics
show the `maxiter=50` fit shrinks several kernel outputscales and increases the
effective time lengthscale, producing smoother lower-dispersion forecasts. Those are
better for density/RMSE but worse for top/bottom ETF ordering.

Decision: Do not promote the Stage B `momentum_plus_macro` portfolio result yet. Before
choosing input blocks, run optimizer/regularization sensitivity focused on preserving
cross-sectional dispersion, or explicitly separate the forecast-density objective from
the portfolio-ranking objective.

## 2026-07-05 - Outputscale Floors Do Not Rescue TVLS Ranking Collapse

Context: After diagnosing the `maxiter=50` collapse, we tested whether lower-bounding
and regularizing `ScaleKernel.outputscale` parameters would preserve useful
cross-sectional dispersion. The sweep used the same 12 monthly scored dates, signed
LKJ rank-2 task kernel, `momentum_plus_macro` main + pairwise kernel, maxiter `50`,
and GP Riskfolio CVaR/MaxRet setup.

Implementation note: Added a narrow `signed_lkj_eta_2_outputscale_floor` experiment
mode so the outputscale-floor sweep keeps the signed LKJ task-covariance family fixed.
The multitask kernel builder already unwraps component `ScaleKernel`s inside product
interactions and applies a single outputscale at the interaction-product level; targeted
tests cover this behavior.

Learning: TVLS plus outputscale floor/prior did not recover ranking. Floors `1e-4`,
`1e-3`, and `1e-2` all landed around cumulative return `0.161` and mean IC `-0.0167`,
well below equal-weight `0.2035`. No-TVLS plus outputscale floors behaved better but
still did not show clean positive ranking signal:

- No-TVLS floor `1e-4`: return `0.1957`, mean IC `-0.1333`.
- No-TVLS floor `1e-3`: return `0.2044`, mean IC `-0.1000`.
- No-TVLS floor `1e-2`: return `0.2002`, mean IC `-0.0833`.

Interpretation: Removing TVLS is helpful for portfolio path stability, but outputscale
floors/priors alone do not solve the ranking problem. The TVLS variants kept weak IC
and low prediction dispersion despite the outputscale constraints. The no-TVLS variants
can get near equal-weight, but their IC remains negative and turnover rises materially.

Decision: Do not promote the regularized `momentum_plus_macro` MTGP as a ranking model.
Keep no-TVLS as the safer sensitivity branch, but treat the current evidence as an
objective mismatch: density-oriented GP fitting is not reliably aligned with
cross-sectional ETF ranking over this small universe.

## 2026-07-05 - Positive No-Prior Index Kernel With No TVLS And Maxiter 15 Is Safer But Still Not A Ranking Win

Context: Ethan asked to rerun the no-TVLS outputscale-prior setup with lower optimizer
budget and without the signed LKJ index kernel. The batch used positive task covariance
with no task-covariance prior, no TVLS, maxiter `15`, `momentum_plus_macro`, 12 scored
monthly windows, GP Riskfolio CVaR/MaxRet, and a no-floor control plus outputscale
floor/prior variants.

Results:

- No floor: return `0.1928`, mean IC `-0.0333`, RMSE `0.0458`, NLL `-1.6476`.
- Floor `1e-4`: return `0.1980`, mean IC `-0.0667`, RMSE `0.0442`, NLL `-1.7025`.
- Floor `1e-3`: return `0.1906`, mean IC `-0.0833`, RMSE `0.0438`, NLL `-1.7076`.
- Floor `1e-2`: return `0.1891`, mean IC `-0.0833`, RMSE `0.0437`, NLL `-1.7066`.

Interpretation: Outputscale priors improve density metrics relative to the no-floor
positive/no-TVLS/maxiter15 control, but they do not improve ranking or beat equal-weight
on the portfolio path. The `1e-4` floor is the best portfolio variant in this batch, but
it still trails equal-weight by about `0.55` percentage points and has negative mean IC.
Turnover remains high around `0.33` to `0.38` for the floor/prior variants.

Decision: Positive no-prior/no-TVLS/maxiter15 is a cleaner and less pathological branch
than signed-LKJ/TVLS/maxiter50, but it is still not promotion-worthy as a ranking model.
Use it as a diagnostic/control branch, not as an incumbent.

## 2026-07-05 - Decomposition Shows Signed Task Covariance, Not TVLS, Carries The Maxiter-15 Ranking Edge

Context: The apparent contradiction between the original maxiter-15 win and the later
positive/no-TVLS underperformance came from changing multiple factors at once. We ran a
targeted decomposition over `momentum_plus_macro`, 12 scored windows, maxiter `15`,
and GP Riskfolio CVaR/MaxRet.

Comparison:

- Signed LKJ + TVLS, maxiter `15`: return `0.2383`, mean IC `0.2000`.
- Signed LKJ + no TVLS, maxiter `15`: return `0.2370`, mean IC `0.2167`.
- Signed LKJ + TVLS, maxiter `50`: return `0.1828`, mean IC `-0.3000`.
- Positive no-prior + no TVLS, maxiter `15`: return `0.1928`, mean IC `-0.0333`.
- Positive no-prior + TVLS, maxiter `15`: return `0.1908`, mean IC `-0.0500`.
- Positive no-prior + TVLS + outputscale floor/prior `1e-4`, maxiter `15`:
  return `0.1848`, mean IC `0.0500`.

Interpretation: TVLS is not required for the original 12-window outperformance. The
signed LKJ task covariance at lower optimizer budget preserves almost the same portfolio
return without TVLS and has slightly higher mean IC. The positive task covariance branch
does not reproduce the ranking/portfolio edge, with or without TVLS. Outputscale
regularization improves density metrics but suppresses prediction dispersion and hurts
portfolio return in the positive/TVLS branch.

Decision: Treat `signed_lkj_eta_2 + no-TVLS + maxiter=15` as the cleaner incumbent
candidate for the next confirmation step. Do not attribute the Stage B win to TVLS.
The next risk to test is whether the signed task-covariance/low-budget edge survives
seed sensitivity and a longer window, not more TVLS/outputscale tweaking.

## 2026-07-06 - Simpler And More Split Matern Groupings Do Not Beat The Current Grouped Kernel

Context: Ethan suggested two kernel-structure checks: split ETF/macro features into more
groups, and test simpler Matérn(0.5) ARD structures. We added experiment-scoped
`--covar-grouping-proposal` options to the U.S. equity wrapper and ran them under the
current cleaner incumbent setup: `signed_lkj_eta_2`, no TVLS, maxiter `15`,
`momentum_plus_macro`, 12 scored monthly windows, GP Riskfolio CVaR/MaxRet.

Candidates:

- `split_groups`: time, ETF momentum, ETF rank/trend, macro market, and macro
  credit/inflation blocks with all pairwise interactions.
- `single_matern`: one Matérn(0.5) ARD block over all selected time, ETF, and macro
  features.
- `time_plus_matern_interaction`: a time Matérn(0.5) block plus one non-time
  Matérn(0.5) ARD feature block and their interaction.

Results:

- Current grouped incumbent: return `0.2370`, mean IC `0.2167`.
- `split_groups`: return `0.1499`, mean IC `-0.1500`.
- `single_matern`: return `0.1908`, mean IC `-0.2667`.
- `time_plus_matern_interaction`: return `0.1840`, mean IC `-0.0333`.

Interpretation: The current grouped structure is carrying useful ranking behavior. The
single Matérn kernel improves density metrics but loses ranking/portfolio utility, again
showing the density-vs-ranking conflict. Splitting into many interacting groups appears
too flexible/noisy for this 4-ETF, 12-window check and materially hurts the portfolio
path.

Decision: Keep the current default grouped kernel for the signed-LKJ/no-TVLS/maxiter15
candidate. Do not promote the split-group or single-Matérn alternatives.

## 2026-07-06 - Removing Macro RQ Improves The Current Grouped Incumbent

Context: Ethan asked to keep the default grouped covariance but remove the RQ component
from the macro block, leaving macro as Matérn(0.5) + Linear. We added an
experiment-scoped `--macro-component-proposal matern_linear` switch and ran the
ablation under the current incumbent setup: `momentum_plus_macro`, `signed_lkj_eta_2`,
no TVLS, maxiter `15`, 12 scored monthly windows, GP Riskfolio CVaR/MaxRet.

Results:

- Default macro Matérn + RQ + Linear: return `0.2370`, mean IC `0.2167`,
  RMSE `0.0458`, NLL `-1.6301`.
- Macro Matérn + Linear, no RQ: return `0.2423`, mean IC `0.2000`,
  RMSE `0.0451`, NLL `-1.6838`.

Interpretation: Removing macro RQ improved portfolio return, Sharpe, RMSE, and Gaussian
NLL while preserving positive IC. IC dipped slightly but remains materially positive.
This is the first kernel simplification in this round that improves the incumbent.

Decision: Promote `signed_lkj_eta_2 + no-TVLS + maxiter=15 + macro Matérn+Linear
without RQ` to the current candidate for the next confirmation step.

## 2026-07-06 - Macro Feature Ablations Confirm Full Macro Matérn+Linear Is Best So Far

Context: Ethan asked to run the full set of macro tests: macro component ablations,
macro feature-family ablations, and an ETF×macro interaction ablation. The fixed setup
was `signed_lkj_eta_2`, no TVLS, maxiter `15`, GP Riskfolio CVaR/MaxRet, and 12 scored
monthly windows.

Results:

- Current all-macro Matérn+Linear: return `0.2423`, mean IC `0.2000`.
- Macro Matérn only: return `0.2082`, mean IC `-0.0833`.
- Macro Linear only: return `0.1890`, mean IC `-0.1167`.
- Macro market/risk only (`vix`, `vix_slope`, `spy_ret`, `pct_above_50dma`):
  return `0.2264`, mean IC `-0.0333`.
- Macro credit/inflation/rates only (`hy_spread_chg_1p`, `hy_spread_z_12p`,
  `cpi_chg_12p`, `term_spread`): return `0.1802`, mean IC `-0.1333`.
- Drop ETF×macro interaction while keeping time×ETF and time×macro:
  return `0.2154`, mean IC `-0.0333`.

Interpretation: The full macro set and the combination of Matérn+Linear are both doing
work. Matérn-only keeps density metrics close but loses ranking, while Linear-only
over-disperses and performs poorly. Market/risk macro features carry more useful
portfolio signal than credit/inflation/rates alone, but the full set is better than
either family. Dropping ETF×macro interaction also hurts, suggesting macro information
is useful partly through ETF-specific regime interaction rather than only broad time
regime.

Decision: Keep the current all-macro Matérn+Linear candidate. Do not drop the linear
or Matérn component, do not split to market-only or credit-only, and keep ETF×macro
interaction.

## 2026-07-06 - Structured Market/Credit Macro Splits Do Not Beat The Full Macro Block

Context: Ethan asked whether preserving the market/risk versus credit/inflation/rates
macro split while using pairwise interactions could work better with a different kernel
configuration. Based on the literature review, we tested asymmetric split-macro kernels:
rougher/nonlinear market/risk blocks and more linear or smoother credit/rates blocks.
The fixed setup was `momentum_plus_macro`, `signed_lkj_eta_2`, no TVLS, maxiter `15`,
12 scored monthly windows, and GP Riskfolio CVaR/MaxRet.

Candidates:

- `split_macro_structured_pairwise`: market/risk Matérn(0.5)+Linear,
  credit/inflation/rates Matérn(1.5)+Linear, with time×ETF, time×market, time×credit,
  ETF×market, and ETF×credit interactions.
- `split_macro_market_nonlinear_credit_linear`: same structure but credit/rates Linear
  only.
- `split_macro_credit_shrunk`: credit/rates Linear only and the ETF×credit interaction
  product is unscaled.
- `split_macro_no_etf_credit`: keep credit as a macro/time regime effect but remove
  ETF×credit.
- `split_macro_cross_family_gate`: remove ETF×credit and add a market×credit interaction
  to test a stress-regime gate.

Results:

- Current all-macro Matérn+Linear: return `0.2423`, mean IC `0.2000`, RMSE `0.0451`,
  NLL `-1.6838`.
- `split_macro_cross_family_gate`: return `0.1955`, mean IC `0.0833`, RMSE `0.0494`,
  NLL `-1.6561`.
- `split_macro_structured_pairwise`: return `0.1954`, mean IC `0.0167`, RMSE `0.0499`,
  NLL `-1.5662`.
- `split_macro_no_etf_credit`: return `0.1867`, mean IC `0.0833`, RMSE `0.0493`,
  NLL `-1.6254`.
- `split_macro_credit_shrunk`: return `0.1702`, mean IC `0.0500`, RMSE `0.0486`,
  NLL `-1.5991`.
- `split_macro_market_nonlinear_credit_linear`: return `0.1689`, mean IC `-0.0500`,
  RMSE `0.0484`, NLL `-1.6055`.
- Equal-weight: return `0.2035`.

Interpretation: The structured split is less disastrous than the earlier unconstrained
all-pairwise split, but it still loses to equal-weight and trails the full macro block
by about 4.7 to 7.3 percentage points. Positive IC on the gate/no-ETF-credit variants
is not enough to rescue portfolio return. The evidence says the two macro families are
not strong enough in this tiny ETF universe to estimate separate block interactions
without diluting the useful full-macro signal.

Decision: Keep the current candidate unchanged: all macro columns in one
Matérn(0.5)+Linear macro block with ETF×macro interaction. Do not promote split
market/credit macro blocks.

## 2026-07-06 - Broad 100-Candidate Kernel Search Finds No Replacement For The Incumbent

Context: Ethan pushed back on constrained hand-designed split variants and asked for
a much broader search over roughly 100 possibilities. We added a broad experiment
runner that generated diverse candidates across feature sets, covariance groupings,
macro components, task-covariance family, time-modulation mode, interaction flags,
optimizer budget, and outputscale-floor sensitivity. The search used a staged budget:
100 completed scout candidates over 3 scored windows and then 12 promoted candidates
over the full 12 scored monthly windows.

Search notes:

- One generated scout candidate was invalid because `single_matern` plus
  `interactions_only` produced no main-effect kernel and no interaction kernel. It was
  retained as an invalid candidate in the manifest and replaced with one additional
  completed scout candidate.
- The scout stage is only a filter, not confirmation. Several candidates tied or looked
  good over the final 3 windows but did not survive the full 12-window rerun.

Promoted full-run results:

- Current incumbent rerun equivalent (`c009`, default grouping, macro Matérn+Linear,
  signed LKJ, no TVLS, maxiter `15`): return `0.2415`, mean IC `0.1667`, RMSE `0.0426`,
  NLL `-1.7797`. This is effectively the incumbent family and is slightly below the
  prior incumbent artifact return `0.2423`.
- `target_plus_macro` default macro Matérn+Linear (`c084`): return `0.2252`,
  mean IC `0.2333`, RMSE `0.0442`, NLL `-1.7571`.
- `compact_pruned` split macro structured pairwise with outputscale-only time
  modulation (`c070`): return `0.2224`, mean IC `-0.0333`, RMSE `0.0455`,
  NLL `-1.4609`.
- `macro_risk` split/no-ETF-credit style (`c055`): return `0.2139`, mean IC `0.1167`,
  RMSE `0.0479`, NLL `-1.6101`.
- `target_plus_macro` market×credit gate (`c051`): return `0.2103`,
  mean IC `0.0833`, RMSE `0.0509`, NLL `-1.6133`.

Interpretation: The broad sweep did not uncover a stronger candidate than the current
all-macro Matérn+Linear/signed-LKJ/no-TVLS/maxiter-15 branch. The closest promoted full
run is the same incumbent family, and the next-best genuinely different candidates
trail by about 1.7 to 3.2 percentage points over the 12-window path. The broader search
also reinforces that 3-window scout wins are fragile: high scout IC and return did not
reliably translate to the 12-window ranking objective.

Decision: Keep the incumbent unchanged for now. Use the broad search leaderboard as
exploration evidence, not as a promotion result. The next useful step is confirmation
of the incumbent family across seeds and/or a longer historical window, not adding more
kernel branches from this search.

## 2026-07-06 - Corrected Broad Search Fixed To Momentum Plus Macro Only

Context: Ethan clarified that the 100-candidate broad search should have held the
feature set fixed at `momentum_plus_macro` and varied only configuration/kernel knobs.
The previous broad search varied feature sets, which answered a different question.
We fixed the search generator to support `--fixed-feature-set` and to dedupe actual
configuration knobs, then reran 100 unique candidates with
`feature_set=momentum_plus_macro` for every row.

Search setup:

- Candidate count: 100 unique `momentum_plus_macro` configurations.
- Scout budget: 3 scored windows, 64 posterior scenarios.
- Promotion budget: top 12 scout candidates rerun over 12 scored monthly windows with
  256 posterior scenarios.
- Varied knobs: covariance grouping proposal, macro component proposal, task-covariance
  family, time-modulation mode, maxiter, interaction flags, time-index exclusion, and
  outputscale-floor sensitivity.

Promoted full-run results:

- `c009`: default grouping, macro Matérn+Linear, signed LKJ, no TVLS, maxiter `15`,
  interactions-only: return `0.2415`, mean IC `0.1667`, RMSE `0.0426`, NLL `-1.7797`.
- `c018`: default macro Matérn+RQ+Linear, signed LKJ, no TVLS, maxiter `8`, drop
  ETF×macro: return `0.2360`, mean IC `0.2333`, RMSE `0.0463`, NLL `-1.6827`.
- `c005`: default grouping, macro Matérn+Linear, signed LKJ, no TVLS, maxiter `15`,
  main effects plus pairwise interactions: return `0.2338`, mean IC `0.2000`,
  RMSE `0.0451`, NLL `-1.6838`.
- `c078`: split macro cross-family gate, macro Matérn-only flag, signed LKJ,
  outputscale-only time modulation, maxiter `8`, no time index: return `0.2108`,
  mean IC `0.2167`, RMSE `0.0455`, NLL `-1.6794`.
- Equal-weight reference from these runs remains near `0.2035`; the prior incumbent
  artifact was `0.2423`.

Interpretation: Holding the feature set fixed changes the search question but not the
practical conclusion. The best promoted full run is an incumbent-family variant and is
slightly below the prior incumbent artifact. The best genuinely different promoted
variants trail materially. Scout-stage winners again do not reliably survive the full
12-window rerun.

Decision: Keep the current `momentum_plus_macro` incumbent family. The only variant
worth a targeted follow-up from this corrected search is the interactions-only
incumbent-family run (`c009`), because it nearly matches the incumbent artifact with
lower RMSE/NLL in this rerun. Treat that as an ablation candidate, not a promotion yet.

## 2026-07-06 - Corrected Broad Search With ETF Block Unsplit And Macro Split Only

Context: Ethan further clarified that the corrected 100-candidate search should not
separate the ETF feature block. The feature set should remain `momentum_plus_macro`,
ETF features should remain one `etf` block, and only the macro block should be split
when testing split structures. We patched the broad-search generator to enforce this
with `--etf-unsplit-macro-only`, added literature-motivated macro-family kernels, fixed
RBF component exposure in the experiment wrapper, and reran a fresh versioned search.

Search setup:

- Run id: `broad_kernel_search_momentum_plus_macro_macro_split_only_20260706T0329Z`.
- Candidate count: 100 unique completed scout candidates.
- Feature set: `momentum_plus_macro` for every candidate.
- ETF structure: one unsplit `etf` block for every split-macro candidate.
- Macro structure: either the default all-macro block or a split into market/risk and
  credit/inflation/rates families.
- Scout budget: 3 scored windows, 64 posterior scenarios.
- Promotion budget: top 12 scout candidates rerun over 12 scored monthly windows with
  256 posterior scenarios.
- Validation: 100 completed scouts, 12 completed full promotions, 0 forbidden
  `split_groups`/`single_matern`/`time_plus_matern_interaction` structures.

Promoted full-run results:

- `c009`: default grouping, macro Matérn+Linear, signed LKJ, no TVLS, maxiter `15`,
  interactions-only: return `0.2415`, mean IC `0.1667`, RMSE `0.0426`, NLL `-1.7797`.
- `c005`: default grouping, macro Matérn+Linear, signed LKJ, no TVLS, maxiter `15`,
  main effects plus pairwise interactions: return `0.2338`, mean IC `0.2000`,
  RMSE `0.0451`, NLL `-1.6838`.
- Best split-macro full run, `c048`: `split_macro_no_etf_credit`, macro Matérn+Linear
  flag, signed LKJ, lengthscale-only time modulation, maxiter `8`: return `0.2071`,
  mean IC `0.1333`, RMSE `0.0509`, NLL `-1.5349`.
- Best literature-kernel split-macro full run, `c094`:
  `macro_split_market_rq_credit_rbf`: return `0.2015`, mean IC `0.0667`,
  RMSE `0.0478`, NLL `-1.5792`.

Interpretation: Once ETF is kept as a single block and the search only splits macro,
the broad search still does not improve on the current all-macro Matérn+Linear
incumbent family. The promoted split-macro candidates trail equal-weight or barely
match it, and none approaches the current all-macro candidate. The repeated survival
of `c009` suggests the interactions-only all-macro incumbent-family ablation remains
worth a targeted confirmation, but macro-family splitting itself is not supported by
this search.

Decision: Keep the current all-macro Matérn+Linear/signed-LKJ/no-TVLS/maxiter-15
candidate. Do not promote macro-family split kernels from this sweep. Treat the
search as evidence that separating macro into market/risk and credit/inflation/rates
does not help under the current 12-window evaluation.
