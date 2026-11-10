# Repaired ETF feature selection research

Completed research scope as of October 18, 2026: 489 universe/input configurations,
1,467 validated seed runs and 35,208 scored GP windows. All four family searches
and the actual fifteen-task search reached their authorized three-round caps.

## Main conclusions

The repaired long-history tests support smaller, different input representations
for the four families, but not a set of variables individually proven useful in
every family. A small actual joint model is more promising than simply taking
the union of family winners. Its pooled gains must be weighed against weaker
within-family ranking in some tasks, particularly international fixed income.

Lag is not universally noise. Its contribution changes with the other predictors
and with joint task sharing. Oil return and the gold/oil ratio also cannot be
collapsed into one universally redundant commodity block: their effects differ
across contexts, and removing either hurt the tested broader joint parent.

The shared research lead uses thirteen non-time inputs instead of twenty-seven:
pooled IC 0.264881 versus 0.108780, NLL -2.250209 versus -2.141413, and conventional
Sharpe 1.622642 versus 1.127881. It improves every objective versus the compute-
matched control in every seed. International-debt ranking still falls, and
predictive spreads are slightly wider, so this is not a universal improvement
or production promotion.

All 35 final family/joint Pareto configurations, exact input lists, seed dispersion,
return and RMSE diagnostics are in [the full tables](/Users/et/Documents/BayesFolio/experiments/2026-10-repaired-global-feature-selection/results/FINAL_PARETO_TABLES.md).
[Structured recommendations](/Users/et/Documents/BayesFolio/experiments/2026-10-repaired-global-feature-selection/results/final_recommendations.json)
also include every frontier's per-seed family calibration and allocation diagnostics.

## Common data and evaluation

The separate frozen experiment panels contain fifteen approved ETFs and 108
monthly observations each. Initial training uses 84 actual usable dates per ETF,
May 2017 through April 2024; evaluation uses the same 24 decision dates, May 2024
through April 2026, with expanding training. The first joint fit has 1,260 labelled
training rows. The runner increases training by one observation per task at each
following window, through 107 months for the last score date.

The target at decision month t is the forward excess return through the next
business month-end. Therefore initial training outcomes end June 2017 through
May 2024, and scored outcomes end June 2024 through May 2026. All 1,620 preserved
labels were independently reconstructed within 0.02 basis points. Predictors
are lagged once; learned input normalization and per-task outcome scaling use
training data only. The evaluated return is already excess return, so Sharpe
does not subtract the risk-free rate a second time.

Seeds are 27, 127 and 227. Primary objectives are mean seed IC, mean seed marginal
Gaussian NLL and mean seed conventional Sharpe. Sharpe uses arithmetic monthly
excess mean divided by sample standard deviation, multiplied by sqrt(12).
Return and RMSE are diagnostic only; turnover is excluded. The saved cumulative
return compounds the monthly excess path, not nominal total investment wealth.
No weighted scalar
score or input-count objective chooses the Pareto front. Each study uses the
same frozen raw hypervolume reference (0, 0, 0), maximizing IC, negative NLL and
Sharpe. Absolute hypervolumes are not compared across different task panels.

### Repairs and provenance

The v1 panel rebuilds validated predictors from frozen trailing source history
beginning January 2013. Skip momentum uses daily adjusted price.shift(21) /
price.shift(252) - 1, not a monthly approximation. The exact BAMLH0A0HYM2 spread
is converted from percentage points to decimals, sampled from the last available
daily observation at month-end, with monthly differences and rolling twelve-month
sample-standard-deviation z-scores. Independent archive and live FRED overlaps
matched exactly on 6,322 and 605 observations respectively.

The v2 feature-pool extension adds eleven eligible historical inputs without
new raw sources or changes to any shared v1 cell. Existing v1 fits retain v1
fingerprints; new masks can use v2. Original experiment caches and results were
not overwritten. Dataset/source hashes, coverage, definitions and repaired-cell
comparisons are recorded in the manifests and dataset audits. ICE raw snapshots
are internal research material and excluded from Git.

Latest-vintage CPI and adjusted prices make this a retrospective forecasting
study, not a vendor-vintage reconstruction. October 2025 CPI is genuinely absent.
The explicitly flagged policy carries the last valid transformed signal for no
more than two months; it does not interpolate or forward-fill CPI levels before
calculating changes. Eight unsupported liquidity/turnover sources remain
quarantined rather than silently filled or removed from purported exact masks.

Definitions also matter: pct_above_50dma is a SPY price/MA proxy, not constituent
breadth; spy_ret is the legacy last-daily-log-return monthly proxy; y10_real_proxy
is a trailing SCHP-return proxy, not an observed real yield. These were preserved
and disclosed rather than silently redefined. The feature-definition audit
distinguishes exact identities, such as the two Treasury-level names, from
empirically tested GP redundancy.

## Fixed model and portfolio settings

Every comparison uses the documented construction rule: scaled time and local
ARD Matern kernels with nu 0.5, an additive macro Matern plus rational-quadratic
plus linear block, main effects and pairwise block interactions, multiplied by
a signed rank-two task kernel with actual LKJ eta-two covariance prior. Means
are task-specific constants. The Hadamard Gaussian likelihood learns per-task
homoskedastic noise with a standardized variance floor of 0.005. Forecast spreads
include observation noise. No time modulation or higher-order kernel tuning is
active; t_index remains in every mask.

The fixed adaptive lengthscale prior/bound rule depends on active dimension.
Thus removing an input changes GP geometry under the same rule; results are not
pure causal information effects. Fits use maxiter 15 and 256 posterior scenarios,
with the saved long cap 0.35 and CVaR/MaxRet portfolio settings. These are bounded
screening fits, not evidence of converged hyperparameter optima.

The 27-input all-input comparator matches the current feature lists under these
research controls. It is not the deployed rank-five, longer-fit, differently
modulated production model. Several new compact models add validated momentum
or macro representations absent from those 27 inputs, so improvements cannot be
attributed only to deleting noisy variables from a fixed basis.

The model-construction audit checked actual objects and adversarial evaluation
isolation in all five universes. Two legacy metadata issues remain documented
in an overlay: manifests mislabel the signed covariance prior as Beta, and legacy
within/between correlation diagnostics label all targets U.S. equity. Neither
field controls fitting or selection; family metrics here use the independent
approved ticker map. Protected fit sources and original manifests were preserved.

## Family results and useful alternatives

All four families completed their historical rebaselines and three adaptive
rounds of twelve new masks each, with all three seeds. Every family round improved
its front. Exploration ended at the authorized cap, not measured convergence.
The corrected totals are 48 U.S. equity, 186 international equity, 105 U.S. fixed
income and 125 international fixed income configurations. Their full fronts
contain nine, eight, six and five points respectively.

| Family and alternative | Non-time inputs | IC | NLL | Conventional Sharpe |
| --- | ---: | ---: | ---: | ---: |
| U.S. equity highest Sharpe | 7 | 0.238889 | -1.779372 | 1.504956 |
| U.S. equity highest IC | 6 | 0.325000 | -1.639829 | 1.456596 |
| U.S. equity best NLL | 9 | 0.155556 | -1.806277 | 1.338789 |
| International equity highest IC and Sharpe | 16 | 0.261111 | -1.645865 | 1.474843 |
| International equity compact higher-IC density branch | 10 | 0.250000 | -1.841178 | 1.369501 |
| International equity best NLL | 12 | 0.225000 | -1.842191 | 1.327043 |
| U.S. fixed income highest Sharpe | 5 | 0.250000 | -2.570066 | 0.597448 |
| U.S. fixed income highest IC | 8 | 0.350000 | -2.609230 | 0.564963 |
| U.S. fixed income best NLL | 7 | 0.255556 | -2.657009 | 0.512473 |
| International fixed income highest Sharpe | 12 | 0.416667 | -2.716427 | 0.879944 |
| International fixed income highest IC | 11 | 0.437500 | -2.735421 | 0.858261 |
| International fixed income best NLL | 11 | 0.361111 | -2.752397 | 0.860687 |

U.S. equity has a concise seven-input Sharpe alternative: skip momentum, six-month
momentum, trend, annual CPI, HY change, HY z-score and the term spread, plus time.
Its six-input highest-IC alternative swaps toward absolute/normalized credit and
drops trend, but has worse NLL even than its own all-input comparator. Lower
dimension alone did not establish lower predictive uncertainty.

International equity retains a real tradeoff: the sixteen-input broad macro
model leads IC and Sharpe, while ten-input risk/oil alternatives have substantially
better NLL. The tiny density advantage of the twelve-input point over the
ten-input points is not evidence of economic significance. A blanket gold/oil
ratio deletion is contradicted by its broader-context tests.

U.S. fixed income has both lagged and lag-free fronts. The five-input highest-
Sharpe model uses lag, MA signal, annual momentum, three-year momentum and HY
z-score. The eight-input highest-IC oil/rate model also includes lag; adding lag
to its exact no-lag parent raises IC in every seed and aggregate Sharpe, with an
NLL cost. The seven-input best-density oil/rate model is lag-free. Removing
genuine nominal/TIPS state from the tested oil branch harmed every objective in
every seed. A universal lag-removal conclusion is not supported.

International fixed income supports eleven-input alternatives, but Sharpe
differences are compressed by three-target allocation constraints. With three
fully invested long targets capped at 0.35, weights have little room to depart
from equal weight; effective NEA is also clamped to two. Small Sharpe differences
are therefore weak prediction evidence. Annual momentum could not be replaced
by skip momentum in the tested final branch without worsening all objectives in
every seed.

The final tables and structured package contain every family frontier, the exact
lists and seed/uncertainty diagnostics, not only these representatives. Their
selected IDs map these descriptive names to exact masks without rounding metrics.

## Actual shared model results

The joint search evaluated the all-input control, six initial hypotheses and
three adaptive rounds of six new masks each, all with three seeds. Its fixed-
reference hypervolume progressed 0.262731421 to 0.822258382 after the initial
hypotheses, then 0.877450650, 0.970672195 and 1.016778459. Adaptive gains were
6.71, 10.62 and 4.75 percent. Stop at the cap, not an asserted convergence point.

| Shared configuration | Non-time inputs | IC | NLL | Conventional Sharpe |
| --- | ---: | ---: | ---: | ---: |
| All-input control | 27 | 0.108780 | -2.141413 | 1.127881 |
| Sharpe-oriented research lead | 13 | 0.264881 | -2.250209 | 1.622642 |
| Compact no-lag alternative | 12 | 0.274901 | -2.234172 | 1.544716 |
| Highest-IC credit alternative | 14 | 0.278026 | -2.232245 | 1.528323 |
| Best-NLL dealer-free alternative | 12 | 0.236359 | -2.262087 | 1.381369 |

These four highlighted alternatives are not the entire front. The fifteen-input
density point and two other thirteen-input points retain small legitimate
tradeoffs. All seven remain in the tables. The highest-Sharpe point is a
disclosed IC/Sharpe-oriented research preference, not a scalar winner replacing
the three objectives. The twelve-input no-lag alternative is attractive if
another input is worth trading for slightly lower mean Sharpe and worse NLL.
Its Sharpe seed SD is 0.170209 versus the lead's 0.094196; neither is an economic
confidence interval.

### Recommended tested shared inputs

Use global_33e57e1c2370 as the primary shared research mask for the fifteen approved
targets. Its thirteen non-time inputs are:

- ETF local: lag_y_excess_lead, mom12m_skip1m, mom6m, trend_slope.
- Macro: cpi_chg_12p, dealer_gamma_proxy, dxy, em_fx_ret, gold_crude_ratio,
  hy_spread_z_12p, oil_ret, term_spread, vix_ts_z_12p.
- Always retain t_index in addition to those thirteen inputs.

This is an actually fitted shared set, not an untested intersection or proof
that each input helps every family. Relative to the current all-input basis,
it removes twenty inputs and adds six validated representations, retaining
seven non-time inputs. Net dimension falls fourteen, approximately 52 percent;
this is feature-representation selection, not only backward deletion.

The compact twelve-input alternative is exactly the same list without lag.
The fourteen-input highest-IC alternative replaces lag with mom12m and adds
hy_spread. The twelve-input best-NLL alternative replaces lag with mom12m and
omits dealer_gamma_proxy. Time remains in each. Exact resolved lists are in the
tables and structured package; never silently combine their preferred changes
into an untested new mask.

Against the lead's twelve-input no-lag parent, adding lag changes mean IC
-0.010020, NLL -0.016037 and Sharpe +0.077926. The favorable seed counts are 1/3,
2/3 and 1/3. In contrast, removing annual momentum from the fourteen-input
lag/no-VIX model improves all aggregate objectives and improves NLL/Sharpe in
every seed. This supports the tested temporal representation, not the claim that
lag itself reliably raises Sharpe in every optimization run.

Removing HY z-score from the tested no-VIX parent worsens all objectives in every
seed. Removing dealer improves NLL but damages IC and Sharpe in every seed.
Earlier ratio and oil removals also worsened all objectives in every seed in
their broader shared contexts. Credit, oil, ratio and dealer therefore cannot
be treated as uniformly redundant proxies.

### Family outcomes inside the joint model

| Family | Control IC | Lead IC | Control NLL | Lead NLL | Lead 95 percent coverage | Lead mean capital weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U.S. equity | 0.172222 | 0.208333 | -1.621425 | -1.742534 | 0.940972 | 0.589373 |
| International equity | 0.041667 | 0.183333 | -1.700185 | -1.771276 | 0.913194 | 0.361605 |
| U.S. fixed income | -0.097222 | 0.038889 | -2.668109 | -2.842665 | 0.951389 | 0.017204 |
| International fixed income | 0.222222 | 0.090278 | -2.720776 | -2.775744 | 0.962963 | 0.031819 |

Pooled IC includes between-family ordering. Three families' within-family ICs
improve, but international fixed income loses 0.131944. All four mean family
NLLs improve. None of the four highlighted joint finalists resolves the foreign-
debt ranking loss. This remains an important unresolved exception; it is not
hidden by the aggregate front or converted into an unrequested veto objective.

Equities receive about 95 percent of the lead's mean allocated capital. Higher
joint Sharpe therefore does not establish better fixed-income portfolio results.
Actual joint Sharpe is reconstructed from all weights and realized excess returns,
never averaged from family Sharpes. Sleeves not invested in every scored month
have null full-path Sharpe diagnostics rather than shortened-path estimates.

## Conditional feature evidence

The final synthesis provides 488 exact one-feature comparisons: 463 family and
25 joint contexts. Each compares
the same model controls, three seeds and monthly panel with and without one
feature. Block substitutions remain block evidence. The full feature-family
matrix marks helpful, harmful, conflicting/tradeoff and untested contexts; it
does not count inclusion frequency or correlated contexts as independent votes.

In the compact summary below, H means helpful across the tested matched contexts,
D means harmful in those contexts, M means tradeoffs or conflicting contexts, and
U means no exact individual add/remove test. These are context summaries, not
universal labels. Block evidence can exist even where the individual axis is U.

| Input | U.S. equity | International equity | U.S. debt | International debt | Joint |
| --- | --- | --- | --- | --- | --- |
| Skip momentum | H | U | U | M | U |
| Six-month momentum | M | M | U | U | U |
| Recent-return lag | M | M | M | M | M |
| HY level | M | M | D | M | M |
| HY z-score | H | M | M | M | H |
| Oil return | M | M | M | M | H |
| Gold/oil ratio | U | M | U | M | H |
| Term spread | H | U | U | U | U |
| Dealer proxy | U | M | U | U | M |
| VIX level | U | H | U | M | M |

The full [feature matrix](/Users/et/Documents/BayesFolio/experiments/2026-10-repaired-global-feature-selection/results/component_synthesis/feature_family_matrix.csv)
contains 275 feature/universe rows. [Every matched context](/Users/et/Documents/BayesFolio/experiments/2026-10-repaired-global-feature-selection/results/component_synthesis/conditional_feature_evidence.csv)
provides the exact with/without IDs, three objective effects and seed consistency.

For example, adding term spread in one compact U.S. equity context improves IC
0.147222, NLL 0.296533 and Sharpe 0.221657, with each objective favorable in every
seed. HY z-score also helps three tested U.S. equity contexts, but has conflicting
or objective-tradeoff effects elsewhere. Oil improves one U.S. fixed-income
lagged rate context by IC 0.141667, NLL 0.108733 and Sharpe 0.116817 in every seed;
other oil contexts show tradeoffs. No one of these effects is a universal ranking.

Family hypotheses generated a compact shared core, a literal union, a reduced
union, rate exceptions and international risk/commodity exceptions. Actual
fifteen-task fits then tested transfer. The literal and reduced unions did not
automatically inherit the separate families' advantages. No zeroed family inputs
or family-masked kernel were used.

## Predictive and performance uncertainty

NLL, 50/80/95 percent coverage and predictive width must be read together. Lower
width without adequate coverage is overconfidence, not success. Predictive
spreads are total return-distribution uncertainty; seed standard deviations
measure optimizer/scenario variation. Both differ from uncertainty in estimated
IC and Sharpe from only twenty-four economically dependent outcome months.

Monthly pooled and family diagnostics expose changing coverage and ranking;
aggregate means are not regime guarantees. This program does not establish
reliable performance confidence intervals or a calibrated winner probability.
Repeatedly inspected score dates make these adaptive research results rather
than untouched confirmation. A new formal holdout and calibrated optimizer
remain deferred as requested.

The lead's pooled mean predictive standard deviation is 0.027480 versus 0.026795
for the control. Its 50/80/95 percent coverage is 0.505556/0.808333/0.940741 versus
0.473148/0.763889/0.909259. Density and coverage improve with slightly wider
forecasts, not lower width. The dealer-free NLL alternative is slightly sharper
(std 0.026683, 95 percent coverage 0.939815) but trades away IC and Sharpe. Marginal
Gaussian NLL does not separately certify calibration of the full task covariance.

The lead's seed SDs are IC 0.003324, NLL 0.002646 and Sharpe 0.094196. Its recorded
compounded excess-return diagnostic is 0.513656 and RMSE 0.031601, versus 0.347077
and 0.033576 for the control. Neither diagnostic selected the front.

## Remaining decisions

Keep the tested thirteen-input lead and its three highlighted alternatives as
research finalists. Remaining family-specific gaps can motivate
future direct global tests of CPI, curve, FX, alternative local volatility and
nominal/TIPS information. They do not authorize another capped round now.

No production defaults, live trading, family-masked architecture or source-prior
changes are included. Production promotion requires a separate decision and
validation appropriate to its different model controls and data availability.
Results apply only to SPY, MGK, VTV, IWM, VEA, VWO, VSS, EWX, BND, IEF, LQD, HYG,
BNDX, VWOB and HYEM. Transfer to additional targets has not been tested. IJR,
REIT and commodity targets were excluded; commodity predictors remained eligible.

## Reproduction and evidence

The one-command resume checks existing outputs against exact source, data,
configuration, date and seed hashes. A final successful execution verified all
27 batches without launching new fits:

```bash
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/workflow.py --execute --workers 15
```

The dispatcher enforces at most fifteen fits, with one Torch/scenario worker
and BLAS thread limits per fit. The one-worker family and global smokes passed.
The independent joint audit passed every one of the 75 joint seeds; execution
coverage validated all 1,467 seeds and all historical required-mask IDs, with
no unresolved registered work. One archived plotting-abort attempt was excluded
and successfully replayed; it is not an extra valid run or a missing result.

The full original adopted scope is GOAL_CONTRACT.md. planning/ contains family
and global hypotheses, explicit operations and interpretations; immutable
pre-round evidence is under planning/evidence/. Cumulative reports and monthly
uncertainty outputs are under results/cumulative/ and results/component_synthesis/.
Old frozen reports remain as-of evidence, including the explicitly corrected
macro-only scope omission. Old and repaired performance are never pooled, and
new-versus-old differences cannot isolate data repair from history/source changes.

[The completion review](/Users/et/Documents/BayesFolio/experiments/2026-10-repaired-global-feature-selection/results/FINAL_COMPLETION_AUDIT.md)
records requirement-by-requirement evidence, verification commands and remaining
scientific limitations. To recompute fits from scratch, use a separate checkout
with the same relative study paths, environment lockfile, frozen internal sources
and registered plans, without copying run/execution outputs. Preserve this study's
original evidence; do not refresh sources or erase completed outputs to manufacture
a fresh run. Exact recovery depends on the frozen source snapshots, not changing
upstream downloads.
