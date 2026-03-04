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

## Current Capabilities

### Gaussian Process Modeling

- MultiTask GP implementation
- Heteroskedastic task noise modeling
- Per-task or global scaling options
- Rank-controlled task covariance structure
- Feature-conditioned predictive distributions

### Intelligent Modeling Workflow

- Rolling time-series cross-validation
- Kernel grid search and selection
- Intelligent feature grouping (macro vs ETF-specific)
- Diagnostic evaluation tools

### Posterior Scenario Generation

- Full posterior mean and covariance extraction
- Joint sampling from predictive distribution
- Scenario-based optimization integration

### Portfolio Optimization

- Integration with Riskfolio
- CVaR optimization on GP posterior scenarios
- Comparison against traditional historical portfolios
- Long/short and top-k evaluation workflows

### Evaluation & Diagnostics

- Out-of-sample R² metrics
- Long-short performance tracking
- Task covariance inspection
- Portfolio weight comparison tools


------------------------------------------------------------

## Architecture Overview

    Market Data
        ↓
    Feature Engineering
        ↓
    Multitask GP
        ↓
    Posterior Mean & Covariance
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

## License

MIT License

------------------------------------------------------------

## Disclaimer

BayesFolio is for research and educational purposes only.  
It does not constitute financial advice.