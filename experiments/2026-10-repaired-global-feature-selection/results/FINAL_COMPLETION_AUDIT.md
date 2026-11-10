# Repaired feature selection completion review

The adopted GOAL_CONTRACT.md has been reviewed against executed outputs as of
October 18, 2026. All required research stages and delivery artifacts are complete.
This conclusion combines the execution, metric/source audits and substantive
scientific review below; it is not inferred from file existence or unit tests.

Verified scope: 489 universe/mask configurations, 1,467 complete seed runs and
35,208 scored GP windows. Four families have 48/186/105/125 masks; the actual
joint model has 25. All 27 registered batches are complete. No registered fit is
missing or unresolved. There are 35 final Pareto configurations and 488 exact
one-feature contexts. The independent objective/allocation audit passed all
75 joint seed runs. Final unit tests: 61 passed; Ruff and the public-docstring
gate passed. The complete-only resume exited zero without launching a fit.

## Required review

| Adopted requirement | Executed evidence inspected | Disposition |
| --- | --- | --- |
| Separate versioned data and no old overwrite | v1/v2 dataset and family manifests, source hashes, original tracked-file diff | Reproduced offline exactly |
| Daily skip momentum and exact HY repairs | DATASET_REPAIR_AUDIT.md, raw-source manifests and exact overlap checks | Passed |
| Every eligible predictor audited with warmup and units | Coverage/definition reports, v2 pool manifest, quarantine ledger | Passed with explicit eight-source quarantine |
| Actual 84 initial usable training months, 24 common expanding windows | Every run's window_training_history.csv and verify_training checks | All 1,467 runs passed, including all final seeds |
| Correct one-period feature lag and target availability | 1,620-label independent audit, training-only adversarial transform audit | Passed |
| Historical masks inventoried and rerun with three seeds | October inventory, July scope audit, v2 pool audit, macro-only correction, historical coverage ID equality | 320 family masks plus joint control complete |
| Invalid screens excluded with reasons and no silent local-block constraint | Source-level audit JSONs, two completed macro-only supplements | Corrected explicitly; both supplements dominated |
| Fixed feature-selection controls and actual likelihood/kernel | MODEL_CONTRACT_AUDIT.md/json, study/plan hashes and actual-object checks | Five universes passed; legacy metadata caveats disclosed |
| Three family reasoning rounds at most twelve masks each | Twelve frozen proposals/evidence snapshots, round interpretations and actual comparison outputs | All four families reached round-three cap, twelve new masks each round |
| Paired lag, commodity and redundancy tests with interactions explored | conditional_feature_evidence.csv, round_hypothesis_effects.csv and hosted ledgers | Passed; effects are conditional, not votes |
| Cross-family helpful/harmful/tradeoff/unknown evidence and effect sizes | feature_family_matrix.csv, full matched contexts and exact IDs/seed wins | 275 matrix rows and 488 contexts, including all final joint contexts |
| Actual joint core, union, reduced union and exception hypotheses | global_round0 proposal/evidence and complete actual fifteen-task runs | Six new masks complete plus control |
| Three adaptive global rounds at most six masks each or valid early stop | Frozen proposals, round reports, actual seed outputs, adaptive closure audit | All three complete, six new masks each, all improved; cap closed |
| Both pooled and family forecast/calibration/allocation outcomes | Global family diagnostics and frontier monthly diagnostics | Passed; foreign-debt ranking loss and equity-heavy allocation are prominent |
| Only IC, Gaussian NLL and conventional Sharpe determine fronts/HV | Independent arithmetic audit, fixed reference and selection code | All 75 joint seeds passed; three-objective front/HV unchanged |
| Return and RMSE recorded only, turnover excluded | All aggregate/per-seed tables, three-coordinate Pareto implementation | Passed all final scope; null sleeve diagnostics preserved |
| Uncertainty assessed beyond width, seeds distinguished from performance uncertainty | NLL, 50/80/95 coverage, total predictive std, monthly and seed diagnostics, final limitations | Passed; wider primary forecast and adaptive economic uncertainty explicitly reported |
| No heavy calibrated optimizer or new formal holdout | Study code/decisions and final scope statement | Deferred as requested; no untouched confirmation claim |
| At most fifteen fit workers globally and no nested thread oversubscription | Sequential dispatcher source, execution worker records, one-thread subprocess environment and native process checks | Native final round verified fifteen children; macro-only supplements scheduled at combined maximum nine |
| Preflight, smoke, exact resume and immutable provenance | Per-mask preflight logs, two completed one-worker smoke statuses, source/data/config/date/seed signatures | Passed; all 27 final batches verified without refitting |
| All final fronts and exact lists, compact shared recommendation and unresolved exceptions | FINAL_PARETO_TABLES.md, FINAL_RESEARCH_REPORT.md and final_recommendations.json | All 35 fronts with exact lists; four joint finalists and family alternatives; hashed evidence |
| Requirement-by-requirement review rather than file-existence success | This review, final execution/arithmetic/source audits, exact verification commands | Complete with the scientific limits below, no missing authorized research stage |
| Protected data, production and unrelated changes preserved | Git status/diff, internal raw ignore checks, no deployment/trading/promotion | Verified; no commit or push in this study |

## Failures and qualification checks

One archived U.S. fixed-income seed-27 attempt reached its fitting outputs but
aborted in native plotting. It was not counted as a valid run. A headless replay
reproduced its ninety-six prediction means/spreads and twenty-four portfolio
returns exactly, and the normal run subsequently passed. The final incomplete
run list and execution-audit unresolved list are both empty. This archived
attempt is disclosed, not counted toward the 1,467 successful seed runs.

The macro-only inventory omission was a scope error, not an adaptive fourth
round. Five valid original manifests deduplicated to two masks and were rerun on
the same frozen setup. Neither changes a family frontier. Old frozen round
comparisons remain honest as-of records; final cumulative counts include both.

The misleading legacy prior label and asset-group correlation fields were not
rewritten in frozen sources/manifests. The actual-model audit supplies a source-
hashed interpretation overlay; scientific family diagnostics use the independent
approved ticker map. The five actual constructed models and evaluation-isolation
checks passed with zero optimizer calls.

The report does not attribute new-versus-old metric changes solely to the longer
training window: source revision, feature repair, cross-sectional ranking and
history change together. Old and repaired metrics remain separate.

The shared lead is the tested global_33e57e1c2370 mask, not an untested intersection.
The report shows every family's ranking/density/coverage and allocation beside
pooled gains. Its thirteen inputs include six newly selected representations;
the net reduction from twenty-seven is not mislabelled as pure noise deletion.
All seven joint Pareto masks remain available. The highest-Sharpe reporting
preference does not alter three-objective front membership. Width and count
never determine selection, and the one-versus-three-seed lag tradeoff is disclosed.

International fixed-income IC still falls in every highlighted compact joint
alternative relative to the joint control. The primary model's mean predictive
spread is slightly wider, despite better NLL and coverage. Thus not every desired
economic improvement was achieved. The goal explicitly permits this defensible
mixed result; conclusions remain conditional on the fixed research GP, reused
dates, three optimizer seeds and retrospective source vintages. No untouched
generalization, universal feature necessity or production readiness is claimed.

### Architecture and documentation review

Changes are isolated to the new experiment directory. No contracts, package
exports, engine/IO boundaries or production defaults changed. The workflow reuses
the existing family runner, command builder and validators; small research-only
report/audit adapters do not introduce a competing GP implementation. Ordinary
candidate bytes were tested unchanged; legitimate macro-only masks omit an empty
block naturally instead of zeroing it. All new runnable entry points have purpose,
units and usage docstrings, with commands and phase status documented locally.
No public package map change or broad refactor was needed.

## Successful final verification and reproduction commands

From /Users/et/Documents/BayesFolio:

```bash
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/workflow.py --check
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/workflow.py --execute --workers 15
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/compare_round.py --plan experiments/2026-10-repaired-global-feature-selection/plans/10_global_round3.json
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/analyze_results.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/synthesize_components.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_joint_objectives.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_model_contract.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_execution_coverage.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/package_final_results.py
.venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection
.venv/bin/ruff check experiments/2026-10-repaired-global-feature-selection
.venv/bin/python scripts/check_new_public_docstrings.py --files experiments/2026-10-repaired-global-feature-selection/package_final_results.py experiments/2026-10-repaired-global-feature-selection/test_final_package.py
```

The complete-only resume printed twenty-seven verified batches and launched no
fits. Hashes, exact registered counts and empty unresolved lists were inspected
alongside exit codes. Earlier in this closing turn, repair_dataset.py and
extend_feature_pool.py reproduced frozen data exactly without overwriting them;
audit_target_alignment.py passed all 1,620 labels, audit_repaired_data.py passed
shared cells/targets, and audit_feature_definitions.py passed identities/aliases.
Their reproducible commands are:

```bash
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/repair_dataset.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/extend_feature_pool.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_target_alignment.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_repaired_data.py
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_feature_definitions.py
```

Native Pylance/pyright tooling was not
available in this execution environment; do not claim a live editor diagnostic
check. The new experiment files are outside the repository's default maintained
package type-check scope; the targeted runtime tests and Ruff checks are recorded.
Arrow CPU-cache discovery and Matplotlib cache warnings under the sandbox did not
fail these offline checks. Fit subprocesses used their saved headless/cache setup.

No deployment fix is applied. The reversible next step is to review the four
documented research finalists and separately authorize any broader validation or
production integration. Preserve raw-source internal-use restrictions and the
original frozen plans/results when doing so.
