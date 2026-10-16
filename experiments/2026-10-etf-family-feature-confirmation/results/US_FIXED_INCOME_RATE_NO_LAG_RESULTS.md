# U.S. Fixed-Income Rate-Aware Lag Challenge

## Summary

- Universe: `BND, IEF, LQD, HYG`.
- Cached-only panel: 62 complete months from `2021-03-31` through `2026-04-30`.
- Held-out panel: the same 24 dates from `2024-05-31` through `2026-04-30` used by the earlier fixed-income search.
- Search: 22 paired configurations, three seeds (`27,127,227`), 66 validated fits, rank 2, maxiter 15, and 256 posterior scenarios.
- Every feature design was evaluated both with and without `lag_y_excess_lead`; all other inputs and controls were identical within each pair.

## Result

The expanded experiment found multiple U.S. fixed-income scenarios where `lag_y_excess_lead` is not useful.
Five of the six Pareto configurations omit it. The only with-lag Pareto point is the high-IC local control.

| configuration | IC | Gaussian NLL | Sharpe | return (record only) | RMSE (record only) |
|---|---:|---:|---:|---:|---:|
| USFI-RATE-BALANCED-NO-LAG | 0.108333 | -2.663281 | 0.432755 | 0.034662 | 0.015804 |
| USFI-RATE-HIGH-IC-WITH-LAG | 0.130556 | -2.639765 | 0.380926 | 0.030807 | 0.015576 |
| USFI-RATE-DURATION-CURVE-NO-LAG | 0.077778 | -2.414905 | 0.454503 | 0.038175 | 0.019623 |
| USFI-RATE-REAL-RATE-NO-LAG | 0.130556 | -2.624058 | 0.433566 | 0.035924 | 0.016206 |
| USFI-RATE-BREAKEVEN-NO-LAG | 0.133333 | -2.451504 | 0.449518 | 0.036263 | 0.018858 |
| USFI-RATE-CURVE-RATE-VOL-NO-LAG | 0.141667 | -2.483645 | 0.362938 | 0.029789 | 0.017398 |

## Clean paired evidence

### Balanced local model

Inputs without lag: `t_index, chmom, mom12m, ma_signal, ret_autocorr`.

Removing the lag changed aggregate IC from `0.113889` to `0.108333`, improved NLL from `-2.611770`
to `-2.663281`, and improved Sharpe from `0.365429` to `0.432755`. The no-lag version had better NLL
and Sharpe in all three seeds. It had better IC in one seed and lower IC in two; the aggregate IC difference
was only `-0.005556`.

### Real-rate model

Inputs without lag: `t_index, chmom, mom12m, ma_signal, ret_autocorr, y10_nominal, schp_ret, y10_real_proxy`.

Removing the lag improved aggregate IC from approximately `0.000000` to `0.130556` and Sharpe from
`0.388938` to `0.433566`; NLL changed slightly from `-2.627255` to `-2.624058`. The no-lag version had
better IC and Sharpe in every seed and better NLL in two of three seeds.

## Interpretation

The earlier conclusion that U.S. fixed income required `lag_y_excess_lead` was conditional on the
long-history, generic-feature search. It is not a stable family-level rule. On a recent-history panel with
bond-specific rate state available, no-lag models occupy most of the Pareto front.

The rate-aware result does not mean that every rate block helped. Large bundles were usually harmful, and
`MOVE` alone did not replace the lag. The useful regions were sparse: the balanced local model, a compact
real-rate block, breakeven inflation, or selected curve information.

For a balanced multi-objective challenger, prefer `USFI-RATE-REAL-RATE-NO-LAG`. It preserves the best IC
of the former high-IC with-lag Pareto point while materially improving Sharpe and giving up little NLL.
Retain `USFI-RATE-BALANCED-NO-LAG` as the calibration/parsimony anchor because it has the best NLL among
the new Pareto configurations.

## Limitation

The rate-aware source begins in 2021, so these raw metrics must not be spliced into the earlier 2013-history
front as if training coverage were identical. The within-panel paired lag/no-lag comparisons are fair and
answer the lag question; a production-budget confirmation should compare the two promoted no-lag models
against the long-history incumbent under one common training policy.
