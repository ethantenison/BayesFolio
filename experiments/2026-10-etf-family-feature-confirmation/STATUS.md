# ETF Family Feature Confirmation Status

Milestone: deterministic U.S. confirmation orchestrator and gated xhigh next-round planning stage implemented; seeds
`127` and `227` have not been launched.

## Summary

- The cached U.S. confirmation panel is frozen at 24 dates from `2024-06-28` through `2026-05-29`; `IJR` is excluded
  and `t_index` is present in every model.
- `U-AC`, `U-C005`, `U-C009`, and `U-ALL-MATCH` completed seed 27 at rank 2, maxiter 15, and 256 scenarios.
- Selection remains Pareto-only on higher IC, lower Gaussian NLL, and higher Sharpe. Return and RMSE are
  diagnostic-only, and turnover is absent from the confirmation evaluation outputs.
- A single controller command now owns the frozen configuration, source/input hashes, four-schema preflight, safe
  resumption, per-run validation, atomic checkpoints, and three-seed Pareto aggregation. Qwen does not interpret or
  control this stage.
- The enclosing workflow now adds exactly one post-results Qwen turn at `xhigh`. It is gated on all 12 runs passing,
  receives a bounded hashed evidence packet, and can only preregister—not execute—the next experiment batch.

## Findings

- The provisional `2024-07-31` through `2026-06-30` panel was not uniformly available because the cached production
  feature artifact ends on `2026-05-29`. The largest recent 24-date common panel therefore starts on `2024-06-28`.
- A versioned cached-only artifact joins the verified U.S. candidate base with only the missing all-input comparator
  columns. It preserves all source artifacts and records source hashes and column provenance.
- All four preflights had identical dates and training histories. All four fits produced 96 finite predictions, 24
  per-date NLL rows, and 96 per-date/ETF NLL rows.
- Seed-27 objectives using conventional annualized Sharpe were:
  - `U-AC`: IC `0.116667`, NLL `-1.653393`, Sharpe `1.205594`;
  - `U-C005`: IC `0.233333`, NLL `-1.683705`, Sharpe `1.141345`;
  - `U-C009`: IC `0.108333`, NLL `-1.703157`, Sharpe `1.014353`;
  - `U-ALL-MATCH`: IC `0.258333`, NLL `-1.636080`, Sharpe `1.287046`.
- All four are nondominated at seed 27. `U-ALL-MATCH` leads IC and Sharpe, while `U-C009` has the lowest NLL. One seed
  is insufficient for promotion.
- Fits emitted small Cholesky-jitter warnings and an empty between-group diagnostic warning. No window failed, and the
  warnings did not produce missing or nonfinite forecast results.
- Earlier local-Qwen planning/execution sessions exhausted or overused their response budget before useful writes.
  The deterministic pilot commands were therefore supervised directly after the user-approved runtime fixes.
- The controller's no-fit integration check passed all four schemas and all 24 dates. It launched no fits and left
  seed `127` and `227` run directories absent.
- The full controller path also completed with seed `27`: it reused all four validated runs, passed source/input
  integrity checks, regenerated the aggregate report, and recorded a completed atomic status checkpoint.
- Hypervolume is now calculated deterministically in `(IC, -NLL, Sharpe)` space against the frozen raw reference
  `(0,0,0)`. The next-round validator requires exploit, bridge, and explore candidates and preserves every fixed control.
- The seed-27 hypervolume preview is `0.558925611`; it is not the stage-3 baseline. The planning gate recalculates and
  binds Qwen to the complete three-seed value after seeds `127` and `227` finish.

## Impact

The U.S. comparison is now executable and auditable on a fair cached panel. The seed-27 result shows a real
multi-objective tradeoff rather than a single winner, so the remaining deterministic seeds are necessary before any
feature-set recommendation.

## Proposed Fix

Run the checked-in one-command workflow. It will validate and reuse the four completed seed-27 runs, execute the eight
remaining runs, aggregate all three seeds, calculate fixed-reference hypervolume, and then run one gated `xhigh` Qwen
planning turn. The proposed next batch will be validated but not executed. Do not start international runs, the
production-budget reference, or later ETF families.

Patch scope in this checkpoint:

- a cached-artifact builder and versioned U.S. artifact/manifest;
- a frozen 24-date scored panel;
- four seed-27 run directories and experiment-scoped NLL/Pareto reports;
- wrapper manifest serialization for the explicit split-macro kernel choices;
- a machine-readable four-configuration specification, one-command controller, multi-seed aggregator, and minimal
  Qwen execution prompt;
- a complete-panel evidence gate, fixed-reference hypervolume calculation, one-turn xhigh Qwen workflow, candidate
  schema, and deterministic next-plan validator;
- updated planning and status documentation.

Risk: the one-seed frontier is unstable and must not be treated as a promotion result. Rollback is removal of this new
experiment directory plus the small wrapper-manifest serialization extension; source feature artifacts are unchanged.

## Verification

Completed:

```bash
uv run --locked ruff check \
  experiments/2026-10-etf-family-feature-confirmation/build_cached_us_confirmation_artifact.py \
  experiments/2026-10-etf-family-feature-confirmation/orchestrate_us_confirmation.py \
  experiments/2026-10-etf-family-feature-confirmation/prepare_next_experiment_turn.py \
  experiments/2026-10-etf-family-feature-confirmation/run_qwen_confirmation_workflow.py \
  experiments/2026-10-etf-family-feature-confirmation/summarize_us_confirmation.py \
  experiments/2026-10-etf-family-feature-confirmation/summarize_us_pilots.py \
  experiments/2026-10-etf-family-feature-confirmation/validate_next_experiment_plan.py \
  experiments/2026-07-us-equity-etf-features/run_first_round_walkforward.py
uv run --locked pytest -q \
  experiments/2026-10-etf-family-feature-confirmation/test_confirmation_workflow.py
uv run --locked python \
  experiments/2026-10-etf-family-feature-confirmation/summarize_us_pilots.py
uv run --locked python \
  experiments/2026-10-etf-family-feature-confirmation/summarize_us_confirmation.py --seeds 27
uv run --locked python \
  experiments/2026-10-etf-family-feature-confirmation/orchestrate_us_confirmation.py --seeds 27,127,227
uv run --locked python \
  experiments/2026-10-etf-family-feature-confirmation/orchestrate_us_confirmation.py \
  --preflight-only --seeds 27 --execution-id orchestrator-preflight-v2-20261013
uv run --locked python \
  experiments/2026-10-etf-family-feature-confirmation/orchestrate_us_confirmation.py \
  --execute --seeds 27 --execution-id orchestrator-seed27-reuse-20261013
uv run --locked python \
  experiments/2026-10-etf-family-feature-confirmation/run_qwen_confirmation_workflow.py
git diff --check -- experiments/2026-10-etf-family-feature-confirmation \
  experiments/2026-07-us-equity-etf-features/run_first_round_walkforward.py
```

The validation report is `results/seed27_validation.json`; all checks passed with a common date hash of
`bea704cb64a42d1d9aeb112ed76ddb28e9487ca8cedbfcccd137f2ee993dde13`.

## Open Questions

None for the workflow setup. Executing the full command will start the eight remaining U.S. fits.

No later phase is implied by approval of the remaining U.S. seed panel.
