# Track 3: a complementary third GP

Status: preregistered before the new fits. Research only; no production changes.

## Question and fit budget

Can a third model provide useful **different errors**, not just a good standalone
backtest, and thereby improve a broadly applicable static ensemble?

The six configurations in `components.json` require 504 monthly fits: six
components, three seeds (27, 127, 227), and all 28 origins from May 2024 through
August 2026. Rank 2, maxiter 15, the complete 15-ETF universe, the signed LKJ
prior, likelihood, chronological feature construction, and `t_index` are retained.
All fits belong to the parent's single shared worker queue. This track launches
no competing GP process pool.

Four models test rolling 48/60-month history under broad14 and compact5 features.
The two expanding-window feature complements test rate/TIPS information missing
from the two current components, and a fast market-state representation without
slow credit/inflation/curve variables. These are six separate hypotheses, not a
six-way feature-importance experiment. The compact-rates model has eight
non-time inputs; market-state has eight. See the registry for exact columns.

The rationale is grounded in the prior conditional ablations summarized in
`experiments/2026-10-repaired-global-feature-selection/planning/CROSS_FAMILY_SYNTHESIS.md`.
Nominal/TIPS state survived exact U.S.-FI removals; oil/FX/risk axes helped
particular international contexts; lag and credit had family-dependent effects.
None of those findings guarantees incremental value in a 15-task ensemble.
The market-state model deliberately omits gold/oil ratio despite its previously
useful international contexts; it must not be promoted on a simplistic assertion
that the omitted variable is unimportant.

## Chronology and rolling-window requirements

At origin `t`, use only rows with `date < t` and observed training labels. Take
exactly the most recent N complete calendar panels for an N-month rolling fit.
Do not truncate per asset independently, reset `t_index`, renumber the original
global forecast-origin index, or carry a fitted parameter across dates. Preserve
the established deterministic seed convention using the original 0..27 month
indices. Each rolling model rescales using its own permitted training data through
the unchanged existing transform pipeline. This changes its learning sample, not
the future features or outcome definitions.

The expanding complements start with 84 months and reach 111 training months.
The rolling alternatives use exactly 48 or 60 training observations per ETF at
every scored origin. Targets, realized returns, macro vintages, and the dataset
remain those of the sealed monthly extension. No raw HY level or HY change input
is introduced; compact's existing `hy_spread_z_12p` remains eligible. No IJR,
REIT, or commodity target is added.

## Fixed ensemble grid and separation of mean from covariance

Start from the approved two-model Sharpe-leader recipe: compact mean fractions
by U.S. equity / international equity / U.S. FI / international FI are
`[0.5, 0, 0.375, 0.25]`; covariance remains 50% broad plus 50% compact.

For each third model, test global mean fraction gamma in `[0.125, 0.25, 0.5]`:

`mu_new = (1 - gamma) * mu_reference + gamma * mu_third`

Independently test covariance fraction beta in `[0, gamma, 0.5]`:

`Sigma_new = (1 - beta) * Sigma_reference + beta * Sigma_third`

Duplicate recipes when gamma = 0.5 are evaluated only once. Gamma and beta are
fixed across **every month, ETF, and seed**; no monthly switching rule or
outcome-adaptive blending weight is permitted. These are explicit covariance
recipes, not an assertion that arithmetic covariance mixing exactly describes
the error distribution of a linear forecast combination. Cross-model error
covariance is not estimated, and no between-mean variance term is silently added.

The primary comparison uses the existing hard-rank allocator unchanged. Freeze
the NE >=7 constraint and historical expanding equal-weight variance ceiling
across these tests; rolling GPs do not receive a rolling risk budget. Do not add
position caps, CVaR, turnover penalties, fallback weights, or uncertainty
recalibration. Standalone third-model forecasts and unchanged-covariance blends
identify whether gains originate in predictions or covariance. A secondary
integration with Track 2's chosen two-model pair may be reported separately only
after the independent Track 3 report is frozen.

## Selection and evidence

Tune static recipes on the first 16 origins (May 2024–August 2025) using the Pareto
objectives of conventional Sharpe, compounded return, and cross-sectional rank
IC, averaged over the three initialization seeds. Freeze that recipe set and
evaluate the later 12 origins (September 2025–August 2026). Also report all 28
months descriptively and label them exploratory. No month receives a new blend
weight. The market history has already been inspected; this temporal check is
not a pristine holdout, and three seeds are not independent market histories.

For each complement, report forecast/error diversity against the two-model
reference by seed and family, including:

- Raw paired residual correlation, and correlation after demeaning errors within
  each month to remove a common market shock.
- Forecast disagreement RMSE in decimal monthly-return units.
- Directional correction/error alignment: whether moving toward the complement
  actually corrects reference errors rather than merely adding noise.
- Standalone versus blended IC, family IC, RMSE, NLL and portfolio diagnostics.

Lower residual correlation alone is not a promotion rule: a noisy model may be
different but useless. Require incremental portfolio Pareto value relative to
the two-model reference, then inspect temporal and seed consistency, drawdown,
exposure, NE, and allocation feasibility. A candidate that improves standalone
metrics but fails all fixed-blend comparisons is not an ensemble success.
Compare the frozen Schwab aggressive proxy and simple momentum top-7 benchmark
with the exact same months, risk-free convention, total-return conversion,
initial $10,000, and monthly rebalancing.

## Implementation, risk, rollback, and verification

`run_track3.run_track(output_root, workers=14, base_case=None)` executes only
allocation/scoring after the parent has sealed all component forecasts. The
primary stage always uses the approved baseline. Passing a first-16-selected
Track 2 two-model recipe adds an explicitly separate secondary stage. Each stage
evaluates 48 distinct recipes plus a control, with six duplicate grid cells
recorded as aliases. It writes immutable case records and `track3/RESULTS.md`,
`track3/summary.json`, `track3/tables.csv`, and `track3/diversity.json`. Any failed
case prevents a completed-stage winner declaration. The runner never fits a GP
or starts a second pool within another worker pool.

All production boundaries and sealed experiments are unchanged. The principal risks
are adaptive reuse of a short market sample and overinterpreting correlated
errors/seeds; chronological static selection and explicit incremental tests
address but do not eliminate them. Rollback is to stop using this isolated
folder; no existing baseline file needs restoration.

Verification commands:

```sh
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync pytest -q experiments/2026-10-gp-tuning-v1/track3_complements
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache uv run --locked --no-sync ruff check experiments/2026-10-gp-tuning-v1/track3_complements
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache uv run --locked --no-sync python scripts/check_new_public_docstrings.py --files experiments/2026-10-gp-tuning-v1/track3_complements/complementarity_math.py experiments/2026-10-gp-tuning-v1/track3_complements/run_track3.py
```
