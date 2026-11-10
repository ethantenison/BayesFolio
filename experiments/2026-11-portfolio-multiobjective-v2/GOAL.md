# November multiobjective decision v2

Activated October 31, 2026. This implements the user's complete `/goal` in the
chat. This file records its non-negotiable scope; PLAN.md and immutable batch
registrations specify deterministic experiments within that scope.

- Deliver by November 1, 2026 at 18:00 America/Chicago. No new exploratory work
  after 16:30; reserve 90 minutes for finalization. Stop earlier on completion.
- Write only this versioned folder. Preserve prior evidence/data/production and
  dirty worktrees. No branches, commits, pushes, trades, or brokerage changes.
- SAME 15 GP tasks: SPY, MGK, VTV, IWM, VEA, VWO, VSS, EWX, BND, IEF, LQD,
  HYG, BNDX, VWOB, HYEM. MGK stays in all training but receives exactly zero
  allocation. Rank/allocate the other 14, with their principal covariance.
- Reference: compact_r2_i15_exp (5 economic inputs plus t_index),
  broad_r2_i15_exp (14 plus t_index), broad_roll48_r2_i15 (same broad inputs,
  exactly 48 labeled months). All reference ranks 2 and maxiter 15.
- Preserve exact v1 features, transforms, target, lags, windows, t_index, and
  excluded raw HY level/change. GP modifications ONLY compact expanding rank 1
  and preregistered limited kernel/prior hypotheses with development evidence.
  Both broad ranks remain 2; ALL maxiter remain 15. No feature/window/other
  rank/fitting-budget search. Each distinct variant counts toward six specs.
- Stable family prediction blends and covariance recipes, equal moment pooling
  over seeds 27/127/227 BEFORE one allocation. No monthly discretionary blends.
- Long only, fully invested, no ETF cap, effective count search within 3..14,
  registered risk ceilings and optional static equity exposure constraints.
  Reference NE7/50-50 covariance are controls, not universal constraints.
- Primary compounded return/CAGR and drawdown. Sharpe/14-ETF IC secondary;
  NLL/RMSE diagnostic. No turnover objective, CVaR, uncertainty recalibration,
  large new optimizer/forest system, missing-month substitution, previous-weight
  fallback, negative-return floors, or postsolve clipping/renormalization.
- Repeated chronological 12-month EVALUATION blocks, not 12-month training.
  Lock latest 12 for final retrospective confirmation. Freeze shortlist and
  rule before scoring it; no retuning afterward. Prior inspection is disclosed.
- At most four rounds, 256 complete policy tuples each/1,024 total, six new GP
  specs, 1,200 monthly fit attempts including pilots/retries, 96 attempts
  reserved for final forecasts/repairs, one root-owned pool <=14 single-thread
  workers. At most ten confirmation finalists. Ceilings are not targets.
- Only Schwab aggressive and eligible-14 top-seven simple momentum benchmarks.
  Replay unchanged v1 three-model recipe AND older incumbent with matched MGK
  exclusion; original MGK-allowed results are diagnostics only.
- Verify/reuse sealed v1 October extension. As of October 31; market origin
  October 30; forecast November. November training targets remain unknown.
- Preserve failures; fix identical intended numerical cases without relaxing
  their constraints. Infeasible complete candidates are ineligible; no months
  are dropped. Correctness/data blockers must be explicit.
- Deliver DECISION.md, CONFIG.json, all-15 NOVEMBER_WEIGHTS.csv, VALIDATION.json,
  COMPARISON and PARETO tables/plots, reproduction command, hashes, raw moments,
  optimizer evidence, complete ledger and STATUS.json. Completion requires
  passed gates, independent checks and deterministic reproduction.
- Explain Friday-close/Monday-execution gap and vintage/gross-cost limitations.
  No global-optimality, crash-protection or calibrated crash-probability claim.

Agents may implement and inspect independently but cannot launch separate fit
pools. Root registers and owns compute admission. Routine in-scope work and
research choices need no additional approval pauses.
