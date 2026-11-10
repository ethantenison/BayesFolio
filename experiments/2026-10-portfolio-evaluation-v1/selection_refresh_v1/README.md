# ETF model selection refresh

Reassess the saved family and joint configurations using ranking, a common
portfolio allocator, and predictive calibration. Refit a bounded shortlist for
complete observed GP covariance, then compare full-joint, regional, and family
architectures with the same inputs before examining family-specific masks.

## Fixed controls

Rank 2, maxiter 15, seeds 27/127/227, 84 initial training months and expanding
84+k windows. The 24 forecast origins remain May 2024–April 2026. IWM stays;
IJR, REIT and commodity targets remain excluded. Keep `t_index`. The two old
joint masks containing `hy_spread_chg_1p` are visible in cached screening but
ineligible for selection or refitting. Other historical family masks are retained.

One process pool uses at most 14 workers, one numerical thread per process,
with a 48 GiB aggregate research budget and 25% memory safety margin. Months
and seeds fit independently with their original window seeds. All fits complete
before batch portfolio evaluation. No previous weights, constraint relaxation,
or solver fallback is allowed. An iteration cap is not convergence.
Strict allocation failures remain explicit: no Sharpe is reported for an
incomplete three-seed portfolio comparison, and those candidates cannot occupy
the portfolio-performance slot. Their independent forecast scores remain valid.

## Selection before refitting

The cache screen considers all 501 saved configurations, not just the old Pareto
front. Each candidate must pass source/data fingerprints, complete date/task
coverage, training-history and target checks.

Each family receives four distinct shortlist slots: highest mean monthly
within-family Spearman IC, highest common-allocator Sharpe, lowest marginal NLL,
and the fewest inputs among its top five IC candidates. Duplicated roles are
filled from the next highest IC candidates. The top-five compact slot is a
bounded exploration rule, not a claim of equivalent economic performance.

Joint screening uses equal-family average IC for its ranking slot. Its four
slots include the accepted 14-input baseline, whose already-exported exact
covariance is verified and reused. At most three additional joint masks are
refitted. No weighted composite, new hypervolume search, or automatic promotion.

Family candidates are scored by replacing just that family's forecasts in a
15-ETF forecast panel. Other families use fixed highest-IC anchors. Portfolio
effects are therefore conditional on those anchors, not standalone family
returns or globally optimized combinations. Ranking and NLL remain local to
the candidate's actual task universe.

The common allocator maximizes forecast ranks under long-only weights,
35% position cap, effective asset count at least three, and a variance budget
equal to equal-weight variance. Every model uses the same expanding
training-only Ledoit–Wolf covariance. No-cap and linear-rank sensitivities and
fixed equal-weight/60–40 benchmarks accompany the final architecture comparison.
This is a research risk control, not an inferred investor risk tolerance.

## Exact covariance and architecture controls

Shortlisted fits preserve the original kernel, priors, normalization, outcome
transforms and fitting routine. Every replay must match saved means/SDs to
absolute tolerance 1e-8. Dense conditioning exports observed predictive covariance
including likelihood noise, in raw target units, without additional fitting or
Monte Carlo covariance approximation.

Six matched-input components use the accepted joint input mask: four families
and two regions (U.S. equity plus U.S. fixed income, and international equity
plus international fixed income). Compare these to the already-completed full
joint model. Also compare three exploratory family-specific assemblies using
shortlisted ranking, conditional portfolio, and NLL leads.

Joint predictive NLL and equal-weight portfolio distribution scores compare
each component against the full joint GP's marginal covariance on exactly the
same ETF subset. Independent family/regional covariance matrices are never
concatenated into a purported full GP covariance. Full-portfolio allocations
use the common historical risk model; its error calibration is explicitly
distinct from GP calibration. Fixed reference distributions are separated
from adaptive-weight decision diagnostics.

Sharpe uses arithmetic excess returns. The historical target is discounted
excess, `(1+total)/(1+RF)-1`; convert portfolio outcomes with realized RF before
computing Sharpe and NAV. RF is not a future predictor. Return and RMSE remain
recorded; turnover is not an objective. Three seeds are not 72 independent months.
Paired three-month block-bootstrap comparisons are exploratory and not
selection-adjusted. All these repeatedly studied months are reused, not holdout.

## Reproduction and recovery

From the repository root:

```sh
PYTHONPATH=experiments/2026-10-portfolio-evaluation-v1 .venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1
.venv/bin/ruff check experiments/2026-10-portfolio-evaluation-v1/selection_refresh_v1
PYTHONPATH=experiments/2026-10-portfolio-evaluation-v1 .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/selection_refresh_v1/refresh.py --execute
PYTHONPATH=experiments/2026-10-portfolio-evaluation-v1 .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/selection_refresh_v1/refresh.py --verify-only
```

`--screen-only` stops before fresh fits. `outputs/status.json` records progress;
screen and monthly fit checkpoints permit resuming with the same command.
Contracts seal old inputs and all executable sources before dependent work.
Do not edit frozen sources mid-run. Failed jobs stop the run; valid checkpoints
remain. Old studies and production files are read-only. Rollback consists of
stopping this new runner and setting aside this new subfolder; no old results
need restoration.

The first cached-screen preflight stopped at `optimal_inaccurate`. Its three
completed checkpoints and source snapshots are preserved in
`preflight_strict_solver_failure/` and `preflight_sources/`; no fits launched.
The completed screen below records such failures without inventing returns.
