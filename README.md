# BayesFolio

BayesFolio is a forward-looking portfolio builder that Bayesian machine learning methods. It replaces purely historical optimization with predictive return distributions estimated via uncertainty-aware ML models, enabling scenario-based risk optimization and regime-aware allocation decisions.

------------------------------------------------------------

## Vision

BayesFolio is evolving toward an Agentic AI-powered portfolio optimization platform that:

- Uses Bayesian models (Multitask GPs, Black–Litterman, hierarchical models)
- Incorporates user domain knowledge directly into model structure
- Supports intelligent kernel and feature selection
- Generates forward-looking predictive return distributions
- Optimizes portfolios using scenario-based CVaR and related risk measures
- Produces transparent diagnostic and backtesting reports
- Integrates a future RAG-enhanced conversational interface
- Emphasizes responsible and transparent AI in financial decision systems

**Optimal envisioned workflow**:
1. User specifies ETF tickers
2. The AI asks questions about goals and market views (Bayesian approach)
3. AI generates an optimized portfolio report with data visualizations

------------------------------------------------------------

## Workflow Overview

    Market Data
        ↓
    Feature Engineering
        ↓
    Build Forecasting Model
        ↓
    Generate Posterior Mean & Covariance
        ↓
    Scenario Sampling
        ↓
    Riskfolio CVaR Optimization
        ↓
    Portfolio Weights + Diagnostics

------------------------------------------------------------

## Long-Term Roadmap

Planned future features:

- Agentic model selection and tuning
- Domain-knowledge-aware kernel construction
- Black–Litterman Bayesian integration
- Heavy-tailed likelihood support
- Hyperparameter uncertainty integration
- Online updating
- Transaction cost modeling
- RAG-enhanced portfolio chatbot
- Interactive portfolio report generation
- Transparent model explanation dashboards
- Responsible AI safeguards

------------------------------------------------------------

## Responsible AI Principles

BayesFolio emphasizes:

- Transparent model structure
- Interpretable covariance and task relationships
- Clear separation between historical and predictive risk
- Explicit uncertainty modeling
- Reproducibility through deterministic seeds and experiment logging

This is not a black-box allocation engine. It is designed to make risk structure visible.


------------------------------------------------------------

## Contributor Workflow (Agents)

BayesFolio includes repository-level agent roles in [AGENTS.md](AGENTS.md) to
improve maintainability and prevent architecture/documentation drift.

Recommended PR workflow:

1. Run **Boundary Inspector** checks for package/layer import violations.
2. Run **Documentation Drift Inspector** checks for public docstring and docs sync.
3. If findings exist, use **Patch Proposer** in propose mode for minimal diffs.
4. Apply patches only after explicit approval.
5. Re-run lint/tests and include verification output in PR notes.

Agent details, guardrails, and output format requirements are defined in
[AGENTS.md](AGENTS.md).


------------------------------------------------------------

## License

MIT License

------------------------------------------------------------

## Disclaimer

BayesFolio is for research and educational purposes only.  
It does not constitute financial advice.