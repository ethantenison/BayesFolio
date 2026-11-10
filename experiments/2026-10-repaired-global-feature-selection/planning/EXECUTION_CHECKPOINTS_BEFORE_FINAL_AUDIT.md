# Historical execution checkpoints before final audit

These superseded as-of notes preserve execution history, including then-live
process references. Use ../STATUS.md and the final completion audit for current
state. Do not poll or restart a process based on this archived file.

## Latest checkpoint: final global round live

The second global refinement round completed successfully. All 483 configurations,
1,449 seed runs and 34,776 scored GP windows through that round are validated;
the independent joint arithmetic audit passed all 57 joint seed runs. Its global
hypervolume rose from 0.877450650 to 0.970672195 (+10.62%). Read
planning/GLOBAL_ROUND2_INTERPRETATION.md and the immutable round-two checkpoint.

The final third global proposal is registered as plans/10_global_round3.json.
Native inspection verified dispatcher PID 81463. Fifteen final-round seed runs
have completed; the last three global_fdb5fb8b385f seeds are live, at two/three
of twenty-four windows in the latest check. Do not start another dispatcher.
The six final masks are global_9e7175a3ef4a,
global_ac1e9b719b57, global_33e57e1c2370, global_da43a19fc3dd,
global_c86bac72391c and global_fdb5fb8b385f, verified against the registered plan.
Each needs all three seeds. No fourth round is authorized.
The live process/native check takes precedence over the older session notes below.

The final report, structured recommendations, completed global-round-three
interpretation and requirement-by-requirement audit are still outstanding.
Current offline verification passed 61 tests, Ruff, frozen v1/v2 reproduction,
all 1,620 independently checked targets, the v1/v2 equality audit and feature
identity/alias checks. Execution coverage correctly remains incomplete while
the final round is live. No production changes or git writes have been made.
Final reporting guards are in package_final_results.py; it requires complete
fresh audits and explicit hosted decisions before publishing recommendations.
Draft scientific/completion review notes are planning/FINAL_REPORT_NOTES.md and
planning/FINAL_AUDIT_NOTES.md, not completed final deliverables.

## Earlier checkpoints (historical, superseded by the latest note)

The goal remains active and incomplete. Preserve the full family-refinement,
cross-family synthesis, actual joint-model search, and final audit requirements.
All historical rebaselines and all three adaptive rounds for all four families
are complete and validated. The final cross-family synthesis and initial actual
joint comparison and first adaptive global round are complete. Second adaptive
global round is registered/running. GOAL_CONTRACT.md preserves the recovered
original goal with the user-authorized twelve-mask family amendment.

## Verified completed work

- Frozen study: three seeds (27, 127, 227), 84 initial training observation months
  (May 2017–April 2024), 24 scored observation dates (May 2024–April 2026),
  expanding windows, all 15 approved ETFs, t_index always retained.
- Repaired v1 dataset: 1,620 rows. The v2 pool adds 11 eligible historical inputs
  from the same frozen raw sources; every shared v1/v2 cell is bitwise identical.
  Original data, experiments, and production defaults remain untouched.
- All 1,620 preserved forward-return labels were independently reconstructed to
  within 0.02 basis points, using separately frozen final-month audit sources.
  These audit sources do not enter model inputs. See TARGET_ALIGNMENT_AUDIT.md.
- Initial inventory reconciles October and July: 11 U.S. equity, 149 international
  equity, 69 U.S. fixed-income, 89 international fixed-income masks, plus one
  combined 15-task all-input control: 319 configurations and 957 valid seed runs.
  This was the original as-of inventory. The corrected historical scope adds one
  valid macro-only mask to each equity family: 320 family masks plus the control,
  321 configurations/963 seed runs. Both supplemental three-seed batches completed;
  neither changes a family front. Eight liquidity/turnover sources remain quarantined.
- All four first adaptive rounds have 12 genuinely new masks and three seeds:
  another 48 configurations / 144 valid seed runs.
- Three adaptive family rounds add 144 complete masks/432 seed runs. Current
  component-stage total is 463 configurations / 1,389 seed runs / 33,336 scored GP
  windows, including the joint all-input control. The plotting abort is not counted.
- All six initial joint hypotheses and the six-mask first adaptive global round
  are complete. Corrected total: 477 configurations / 1,431 seed runs / 34,344
  scored GP windows, including the two restored historical macro-only baselines.
- Final family counts: 48 U.S. equity, 186 international equity, 105 U.S. FI,
  125 international FI. All reached the three-round cap; none ended by plateau.
- Full offline verification most recently passed: 51 tests and Ruff; the component
  reporter and independent joint-objective audit passed the public-docstring check.
- New audit_joint_objectives.py independently reconstructs IC via scipy ranks,
  marginal Gaussian NLL, actual allocations-times-realized-excess-return portfolio
  paths, conventional Sharpe, and family calibration. All three completed joint
  control and eighteen new initial joint seeds matched published metrics: all 21
  initial-stage seed runs passed. The archived passed audit is in
  results/component_checkpoints/after_global_round0/. The next audit must also
  first-adaptive-round audit also passed all 39 joint seeds. Its archived snapshot
  is results/component_checkpoints/after_global_round1/. The next audit must include
  the eighteen live second-round seeds, for 57 joint seed runs.
- audit_model_contract.py passed five-universe no-fit construction checks and
  adversarial evaluation isolation. Exact training bounds/outcome scaling, actual
  LKJ eta-two prior and Hadamard Gaussian likelihood were verified. See
  results/MODEL_CONTRACT_AUDIT.md for the misleading legacy Beta-prior label and
  all-us_equity correlation-group diagnostics; do not use those legacy fields
  as actual prior/four-family evidence. No GP fit source or run manifest was changed.
- candidate_masks.py admits legitimate macro-only historical masks without
  changing normal candidate bytes or fixed GP sources. Scope correction and
  source hashes are in results/HISTORICAL_MACRO_ONLY_SCOPE_CORRECTION.md.
- Final component synthesis: 463 exact one-feature contexts, explicit unknowns,
  28 full family frontier masks, monthly pooled/within-family coverage/width
  diagnostics and source hashes. Preserved checkpoints after family rounds two
  and three are in results/component_checkpoints/. A post-global-round-zero
  snapshot adds one joint lag context. After global round one, synthesis contains
  470 exact contexts (seven joint) and unchanged family fronts, with full monthly
  uncertainty reports and source hashes.
- Hosted synthesis and concrete global reasoning are saved in
  planning/CROSS_FAMILY_SYNTHESIS.md and planning/GLOBAL_SELECTION_POLICY.md.
- Source-defined identities verified, including identical tnote10y/y10_nominal,
  term_spread = tnote10y - tbill3m, ERP proxy, momentum acceleration and ratios.
  Exact information redundancy is not assumed to imply finite-budget GP invariance.
- An independent full U.S. FI replay after correcting a plotting-only native abort
  reproduced all 96 means/stds and the 24 GP portfolio returns exactly.

## Current process state

Round-two session 18453, round-three session 38870, global-initial session 68332,
and first adaptive global session 87986 completed successfully (exit 0).
Supplemental macro-only family sessions 3745 and 7126 also exited zero; they ran
only after global work fell to three remaining fits, with three workers each
(at most nine combined fits). Do not poll these completed handles as live.

Second adaptive global dispatcher session 30241 is currently running, with at
most fifteen fit workers total. Its six new masks have 15, 13, 13, 13, 13 and 15
non-time inputs, plus t_index. Check the live
session/native process before any restart; never launch duplicate dispatchers.

Native inspection before launch found no prior fit dispatcher or surviving
worker. Recheck actual session/process state before any restart. Run-status
entries remain `queued` until a child finishes; they do not mean it has not started.
Incremental scenario_cache/gp_predictions_<date>.csv files show window progress.
These larger fifteen-task fits take materially longer than family fits; do not
mistake their longer runtime or unchanged `queued` entries for a stalled process.

Second adaptive IDs: combined rank removal/lag global_52d9b7e07091; annual removal
global_c0bc5df949cf; trend removal global_4a6348639308; VIX-level removal
global_a8a6b8394d6e; oil-return removal global_973f696b2e34; HY-level addition
global_d062f4d835aa. Each requires seeds 27, 127 and 227.

First adaptive global IDs: without rank global_15b18df4a34b; without annual
momentum global_20d835629365; without ratio global_67eeea779f2d; plus lag
global_43cd2137e85f; core plus HY level global_d019284686e8; core plus dealer
proxy global_300876dbdd56. Every mask requires seeds 27, 127 and 227.

Initial global IDs: core global_b48732adb186; literal union global_3745593ecc26;
reduced union global_9d4dc5c2a19f; fixed-income exceptions global_2b66b7fc9390;
international exceptions global_389a69848aa0; fixed-income exceptions plus lag
global_1b4304ffe9c1. Each needs seeds 27, 127 and 227.

The initial dispatcher session 69460
was interrupted; native inspection confirmed no surviving worker before resuming.
The resumed baseline session 56869 and first-round session 29325 both terminated
successfully. Do not poll those old handles as if they were live or restart
finished fits. Revalidate current state before launching the next dispatcher.

One command resumes only registered work, revalidating complete batches without
refitting and enforcing at most 15 total fit workers:

```bash
.venv/bin/python experiments/2026-10-repaired-global-feature-selection/workflow.py --execute --workers 15
```

All family plans are complete. Initial global proposal planning/global_round0.json
registered completed plans/05_global_round0.json, with immutable evidence under
planning/evidence/global_round0/. It includes a tested exact-mask contract for
the literal family union and the 14-versus-15-input lag pair. The full initial
interpretation is planning/GLOBAL_INITIAL_INTERPRETATION.md. First adaptive plan
plans/10_global_round1.json completed using frozen complete round-zero evidence.
Second adaptive plan plans/10_global_round2.json uses complete round-one evidence,
the recovered contract and scope/model audits, frozen under planning/evidence/global_round2/.
Only the third adaptive global round remains unregistered. The dispatcher does not invent plans.

## Round-one results (immutable before/after evidence)

| Family | Before masks | After masks | HV before | HV after | Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| U.S. equity | 11 | 23 | 0.664053736 | 0.708220358 | 6.65% |
| International equity | 149 | 161 | 0.515212263 | 0.595004502 | 15.49% |
| U.S. fixed income | 69 | 81 | 0.360575223 | 0.431890306 | 19.78% |
| International fixed income | 89 | 101 | 0.740084338 | 0.945671955 | 27.78% |

All four rounds improved the frontier; none is stalled. These are within-family
fixed-reference comparisons, not comparable absolute economic scores across
families. See results/rounds/*_round1.json and the adjacent Markdown/CSV reports.

Current cumulative complete-three-seed evidence is in results/cumulative/.
First-round reasoning proposals are in planning/*_round1.json. Executable
registrations are plans/10_*_round1.json. Each has frozen, hashed pre-round
evidence under planning/evidence/<family>_round1/.

## Round-two results

| Family | Complete masks | HV before | HV after | Gain |
| --- | ---: | ---: | ---: | ---: |
| U.S. equity | 35 | 0.708220358 | 0.834919079 | 17.89% |
| International equity | 173 | 0.595004502 | 0.620432258 | 4.27% |
| U.S. fixed income | 93 | 0.431890306 | 0.483386142 | 11.92% |
| International fixed income | 113 | 0.945671955 | 1.004573515 | 6.23% |

All improved. See planning/ROUND2_INTERPRETATION.md and immutable round reports.

## Family round-three results (as-of before the two dominated historical supplements)

| Family | Complete masks | Front masks | Final HV | Round-three gain |
| --- | ---: | ---: | ---: | ---: |
| U.S. equity | 47 | 9 | 0.861677628 | 3.21% |
| International equity | 185 | 8 | 0.701799493 | 13.11% |
| U.S. fixed income | 105 | 6 | 0.544561035 | 12.66% |
| International fixed income | 125 | 5 | 1.057468608 | 5.27% |

All improved. Stop at the agreed family cap, not an asserted convergence point.
See planning/ROUND3_INTERPRETATION.md and planning/CROSS_FAMILY_SYNTHESIS.md.

The supplements increase equity cumulative counts to 48 and 186, without changing
any displayed family front or hypervolume. Old frozen comparisons stay as-of records.

## Completed initial global comparison

| Configuration | Non-time inputs | IC | NLL | Sharpe | Front |
| --- | ---: | ---: | ---: | ---: | --- |
| All-input control | 27 | 0.108780 | -2.141413 | 1.127881 | No |
| Shared core | 10 | 0.204514 | -2.238293 | 1.288369 | Yes |
| International exceptions | 15 | 0.258433 | -2.218703 | 1.425040 | Yes |

Hypervolume rose from 0.262731421 to 0.822258382. The fifteen-input mask improved
all three objectives versus the control in every seed, but U.S.-equity and
foreign-debt within-family IC fell. Family NLL improved in aggregate for all four.
These are reused-date research results, not untouched confirmation or universal
feature importance. Literal/reduced unions and the rate-exception masks are
dominated. Lag improves its matched fourteen-input parent in aggregate but the
branch stays dominated. All seven points and exact inputs remain preserved.

## Completed first adaptive global round

Hypervolume: 0.822258382 -> 0.877450650 (+6.71%). New cumulative front:

| Configuration | Non-time inputs | IC | NLL | Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Rank-free international exceptions | 14 | 0.266617 | -2.232702 | 1.469891 |
| International exceptions plus lag | 16 | 0.262054 | -2.239092 | 1.468439 |

Rank removal and lag addition each improve all aggregate objectives against their
original fifteen-input parent. Ratio removal worsens all objectives in every
seed. Annual-momentum removal is a small tradeoff, not on the front; dealer-only
addition does not explain the international block gain. HY-level addition raises
pooled IC but worsens density and does not restore U.S.-equity IC. Family outcomes
remain mixed. Read planning/GLOBAL_ROUND1_INTERPRETATION.md before planning further.

## Next actions: complete the actual research goal

1. Resume/check the live global-refinement dispatcher 30241. Do not restart completed
   family/initial-global fits or add a fourth family round. All six new masks require
   three complete fingerprint-matched seeds and 84+k training observations/task.
2. After completion, run compare_round.py --plan plans/10_global_round2.json,
   then analyze_results.py, synthesize_components.py and audit_joint_objectives.py
   using full study paths. A final joint arithmetic audit must pass for all registered seeds.
   Read actual pooled and within-family results and the exact lag-pair report.
3. Hosted Codex writes planning/global_round3.json from the fresh complete evidence.
   Round one improved, so a single stalled round two is not an early-stop condition.
   Run the third round within the cap, then stop new launches. Never expand
   six new global masks to the family-only twelve-mask amendment.
   Register with register_round.py --proposal <path>; it freezes evidence, applies
   explicit operations and rejects duplicate masks. Never edit registered proposal
   or frozen evidence hashes. For global proposals, the default base lookup must
   use global exact_inputs.json, not component_exact_inputs.json.
4. Run workflow.py --execute --workers 15, compare each completed global round,
   refresh cumulative/component evidence, then reason over the next batch. Up to
   three adaptive rounds, at most six novel masks each; stop after two consecutive
   stalled adaptive rounds or the cap. Initial round zero is not an adaptive round.
5. Preserve tradeoffs, uncertainty calibration, conditional conflicts and untested
   inputs. Do not use inclusion frequency, ARD, correlations or identities alone
   as importance. Do not substitute family unions/intersections for joint results.
6. Deliver results/FINAL_RESEARCH_REPORT.md, results/final_recommendations.json,
   and results/FINAL_COMPLETION_AUDIT.md, with final fronts, exact inputs, uncertainty, family
   breakdowns, recommendations, limitations, and requirement-by-requirement audit.
   COMPLETION_CHECKLIST.md records the unfulfilled requirements. Only then call
   update_goal complete.

## Fixed scientific and safety boundaries

- Objectives only: maximize IC, minimize Gaussian NLL, maximize conventional
  excess-return Sharpe = mean(monthly excess return) / sample_SD * sqrt(12).
  Return/RMSE are diagnostic only. No turnover selection or hidden weighted score.
- Keep t_index. Lag stays eligible; its usefulness is conditional, not prescribed.
  No REIT/commodity targets or IJR. Commodity predictors remain eligible.
- All-input comparator uses actual current production feature lists under the
  fixed research kernel/budget (signed LKJ, no time modulation, rank 2, maxiter 15,
  256 scenarios). It is not the deployed rank-5/maxiter-75 production model.
- Latest-vintage CPI and adjusted-price histories are retrospective, not a
  point-in-time vendor-vintage reconstruction. October 2025 CPI is genuinely absent;
  hold the last valid transformed signal for at most two months, explicitly flagged.
- Dates are decision/observation dates. The target at t is the forward excess
  return through the next month-end; scored outcomes end June 2024–May 2026.
  No second risk-free subtraction and no extra target shift.
- Repeated scored dates are adaptive research, not untouched holdout confirmation.
  Seed repeats are optimization variability, not independent economic observations.
- Calibrated BO and a new formal holdout program remain deferred.
- Do not implement family-masked kernels or zero unused inputs. That is a different
  architecture requiring a separately labelled proposal.
- No production changes, live trading, commits, or pushes have been performed.
  Existing unrelated .learnings/ remains untouched. An unrelated
  output/pdf/etf_ira_monthly_portfolio_strategy_2026-10-18.pdf appeared during
  shared-workspace work; it was not created, read, or changed by this study.
