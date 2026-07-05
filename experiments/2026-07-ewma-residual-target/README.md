# EWMA Residual Target GP - next08

Purpose: isolate `next08_predict_residual_over_ewma_baseline` from the parameter-only Pareto waves.

Decision question:

Can a multitask GP escape raw-return mean collapse by modeling residual return over a simple trailing EWMA baseline?

Target contract:

- Raw target: `y_excess_lead`
- Baseline forecast: per-ETF trailing EWMA of prior monthly `y_excess_lead` values only
- Model target: `y_excess_lead - ewma_baseline_pred`
- Prediction output: `y_pred = ewma_baseline_pred + gp_predicted_residual`
- Scenario output: `scenario = ewma_baseline_pred + gp_residual_scenario`

Default candidate:

- Feature artifact: `/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_18etf_common_history_201306_202605.parquet`
- Evaluation: 11 realized windows plus live tail construction window
- Kernel composition: `E*T + M*T + E*M*T`
- Task kernel: positive no-prior
- Time modulation: lengthscale only
- EWMA half-life: 3 months

Minimum rule: compare on reconstructed raw prediction scale and inspect actual-vs-predicted plots before making a promotion claim.
