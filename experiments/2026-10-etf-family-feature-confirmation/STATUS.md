# ETF Family Feature Confirmation Status

Milestone: the three-seed U.S. confirmation panel is complete and validated. The bounded `xhigh` Qwen planning turn
did not create its required planning artifacts and failed the deterministic output gate.

## Summary

- The cached U.S. panel contains 24 scored dates from `2024-06-28` through `2026-05-29`; `IJR` is excluded and
  `t_index` is present in every model.
- All 12 requested runs completed at seeds `27`, `127`, and `227`, rank 2, maxiter 15, and 256 scenarios. Seed 27 was
  reused because the fitting sources, exact commands, cached inputs, and date panel matched its validated snapshot;
  seeds 127 and 227 were fit fresh.
- Selection is Pareto-only on higher IC, lower Gaussian NLL, and higher conventional annualized Sharpe. Return and
  RMSE are diagnostic-only, and turnover is excluded.
- `U-C005`, `U-C009`, and `U-ALL-MATCH` are nondominated. `U-AC` is dominated by `U-ALL-MATCH` on all three
  objectives.
- The final fixed-reference hypervolume is `0.6102258300032914` in `(IC, -NLL, Sharpe)` space against raw reference
  `(0, 0, 0)`.

## Findings

- Three-seed configuration aggregates are:
  - `U-AC`: IC `0.130556`, NLL `-1.653460`, Sharpe `1.269129`;
  - `U-C005`: IC `0.272222`, NLL `-1.705886`, Sharpe `1.217001`;
  - `U-C009`: IC `0.075000`, NLL `-1.713216`, Sharpe `1.151083`;
  - `U-ALL-MATCH`: IC `0.227778`, NLL `-1.658456`, Sharpe `1.334647`.
- `U-C005` owns the strongest IC, `U-C009` the lowest NLL, and `U-ALL-MATCH` the strongest Sharpe. There is no
  single winner under the three-objective contract.
- All runs use the same date hash, contain 24 scored windows, and passed the manifest, finite-output, feature, and
  optimizer validation checks. No deterministic failure occurred.
- Fits emitted small Cholesky-jitter warnings and an empty between-group diagnostic warning. The warnings did not
  create missing or nonfinite results.
- The Qwen turn verified the evidence hash but repeatedly reread and restated the evidence. It crossed the context
  compaction threshold twice, restarted its checklist after each compaction, and wrote neither
  `NEXT_EXPERIMENT_PLAN.md` nor `next_experiment_candidates.json`.
- The Qwen output validator therefore failed on the two missing artifacts. The planning turn launched zero GP fits
  and did not modify any protected code, configuration, evidence, or result file.

## Impact

The first fair three-seed baseline is now available for later sequential multi-objective experimentation. The result
supports distinct IC/NLL and Sharpe regions of the frontier rather than promoting one configuration. Qwen's planning
failure does not affect the fitted results, but it means no model-generated next batch has been accepted.

## Proposed Fix

Preserve this deterministic panel as the current baseline. Work on a calibrated optimizer, holdout discipline, and a
more compact or externally scaffolded Qwen planning contract is deferred. No proposed Qwen candidate should be run
until both planning artifacts exist and pass `validate_next_experiment_plan.py`.

Patch/result scope at this checkpoint:

- deterministic orchestration, validation, and fixed-reference hypervolume calculation;
- frozen cached input artifact and 24-date panel;
- complete three-seed objective, aggregate, Pareto, validation, and evidence artifacts;
- one attempted `xhigh` Qwen planning turn, recorded as failed at the output gate.

Rollback: revert the result/status commit. Generated run directories and execution snapshots remain ignored local
artifacts.

## Verification

Completed:

```bash
.venv/bin/ruff check <all touched Python files>
.venv/bin/python scripts/check_new_public_docstrings.py --files <touched Python files>
.venv/bin/pytest -q
.venv/bin/python \
  experiments/2026-10-etf-family-feature-confirmation/run_qwen_confirmation_workflow.py --execute
```

Results:

- lint passed;
- public-docstring gate passed;
- `194 passed, 1 skipped` in the full test suite;
- deterministic stage: 12/12 runs passed with common date hash
  `bea704cb64a42d1d9aeb112ed76ddb28e9487ca8cedbfcccd137f2ee993dde13`;
- Qwen stage: failed deterministic validation because both required planning outputs were absent.

## Open Questions

None for the completed three-seed confirmation. The next-batch planning contract should be revised only when work on
the calibrated optimizer and holdout discipline resumes.
