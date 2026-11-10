# Approved ETF follow up experiments

Run approved items 1–6 and 8 from the research discussion. Items 7 and 9 are
excluded entirely: no new feature searches, kernel architecture changes,
alternative training windows, additional stability experiments or holdout studies.
No production model or portfolio is promoted automatically.

## Fixed controls

Rank two, maxiter 15, seeds 27/127/227, the existing 84-month expanding training
initialization, the same 24 forecast origins and all 15 ETFs. Keep IWM and
`t_index`. No IJR, REIT or commodity targets. No CVaR, position caps, turnover
objective, failed-weight fallback, clipping, renormalization or relaxed constraints.
No full joint GP includes `hy_spread_chg_1p`. Existing family masks are not changed.

All values retain discounted-excess decimal return units. Portfolio performance
converts these explicitly to total and arithmetic-excess returns. Sharpe is
conventional and gross of unspecified trading costs. The original 24 months
remain exploratory reused observations, not a new untouched holdout.

## Registered experiments

| Approved area | Comparison | New fitting required |
| --- | --- | --- |
| 1 Diversification | Floors none, 3, 5, 7, 10 and 12; variance budgets 0.5, 1 and 1.5 times the fixed historical equal-weight budget | No |
| 2 US equity collapse | Two existing input masks crossed with US-only versus full-joint training; per-ETF paths, amplitude, correlation, RMSE and local rank IC | One new full-joint mask plus four replay audits |
| 3 Family joint selection | Two existing masks per family, forming all 16 complete assemblies | No |
| 4 Task grouping | Four single families, all six pairs, all four triples and full joint, with the existing joint input mask | Eight missing pair/triple components |
| 5 Risk calibration | GP versus historical, diagonal GP, three fixed covariance blends and strictly prequential variance scaling | No |
| 6 Allocator sensitivity | Existing rank maximization, smoother rank weights and expected returns shrunk toward historical means | No |
| 8 Local versus pooled ranking | Same 15-ETF panel; each family IC versus pooled IC for complete assemblies | No |

Budget multipliers apply to **variance**, not volatility: 0.5× variance means
approximately 0.707× forecast volatility; 1.5× means approximately 1.225×.
The absolute historical-EW variance budget remains common across covariance
sources, models and objectives. Infeasibility is a recorded result, not repaired.

The five initial joint references are the accepted baseline, portfolio lead,
ranking/density compromise, GP-risk no-floor portfolio lead, and compact NLL
lead. The new full-joint component reuses the successful five-input US-equity
mask without adding or removing predictors. No new HY variable is introduced.
Family choices are the existing best local-IC and best conditional historical-risk
floor-three portfolio configurations from the sealed 499 table, frozen before
this study runs. Their combinations are newly evaluated together, not a new
feature-mask search. All complete assemblies use floors none/ten and the same
historical risk model; missing cross-family GP covariance is never invented.

Smoother rank allocation maximizes normalized ranks minus 0.25 times squared
weights under the same risk constraints. Shrunk-return allocation averages GP
means and training-only expanding historical means with equal weights, then
rescales each month's objective by a positive factor. That rescaling does not
change expected-return ordering. Neither method changes the GP forecasts.

Prequential calibration uses equal-weight forecast errors from **earlier forecast
origins only**. Its variance multiplier is `(6 + sum(previous standardized squared
errors)) / (6 + previous_count)`, with initial multiplier one. No evaluated
month's own or future outcome enters its correction. There is no burn-in deletion
or full-period calibration fit. Blends use fixed historical fractions 0.25, 0.5,
0.75; these are explicit experiment arms, not solver or missing-data fallbacks.

## Execution and evidence

The plan contains 30 complete-portfolio models and 372 registered allocation
cases. Every case retains three seeds and all 24 months, including failed solves.
It reuses sealed exact predictive arrays whenever available and runs only 648
missing monthly component fits plus four independent replay checks: 652 new fits
in total. A same-forecast allocation replay must also match protected baseline
weights to 1e-6. The original generic GP fitter is reused read-only; its legacy
portfolio CLI flags are inactive because its portfolio optimizer is never called.

One bounded process pool runs per stage, one numerical thread per process,
at most cores minus two/thirteen workers. Representative late-window smokes set
a lower worker count if needed to remain within a 48 GiB budget and 25% safety
margin. Cached portfolio scoring finishes before the main fresh-fitting pool;
post-fit portfolio statistics are computed in a final batch, not inside fits.

All source inputs and executable files are hashed in `outputs/contract.json`.
Per-fit and per-policy checkpoints have independent signatures and artifact
hashes. Resume does not discard completed fits or change original month indices.
Source datasets and older completed studies are never written. The detached
one-shot launcher installs no recurring jobs and keeps logs/PIDs in outputs.

## Commands

From the repository root:

```sh
.venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/approved_scope_v1
.venv/bin/ruff check experiments/2026-10-portfolio-evaluation-v1/approved_scope_v1
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/approved_scope_v1/run_study.py --prepare
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/approved_scope_v1/run_study.py --smoke-only
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/approved_scope_v1/launch_background.py
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/approved_scope_v1/run_study.py --status
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/approved_scope_v1/run_study.py --verify-only
```

Expected final outputs include `RESULTS.md`, full policy and ranking tables,
complete assembly comparisons, unsmoothed US-only versus joint forecast plots,
per-ETF forecast diagnostics, paired covariance comparisons, fixed-reference risk
scores, every monthly weight, solver failures and parameter snapshots.
Completion requires a passed `COMPLETED.json` and successful verification.
Pylance diagnostics require an exposed language-server tool; local numerical
tests, lint and docstring gates are run separately and are not a substitute claim.
Rollback is stopping this study and setting aside this new directory; nothing
in production or an older experiment needs reverting.
