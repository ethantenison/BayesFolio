# BayesFolio

Bayesian Multitask Gaussian Processes for Forward-Looking Portfolio Optimization

BayesFolio is a research-driven Python framework for building forward-looking portfolios using Bayesian machine learning methods. It replaces purely historical optimization with predictive return distributions estimated via Multitask Gaussian Processes (MTGPs), enabling scenario-based risk optimization and regime-aware allocation decisions.

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

- Hadamard multitask GP implementation
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

## Why BayesFolio?

Traditional portfolio optimization relies on backward-looking estimates:

Method        | Mean                  | Risk               | Regime Awareness
--------------|-----------------------|--------------------|-----------------
Historical    | EWMA / average        | Historical CVaR    | No
BayesFolio    | GP posterior mean     | GP posterior CVaR  | Yes

BayesFolio models:

    r_{t+1} ~ N(m(x_t), S(x_t))

Where:

- m(x_t) is the predictive mean conditioned on current features  
- S(x_t) is the predictive covariance including task structure  
- x_t includes macro and asset-level signals  

This enables forward-looking, regime-aware portfolio construction.


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

## Project Structure

    bayesfolio/
        features/
            asset_prices.py
            market_fundamentals.py
            gp_data_prep.py
        models/
            gp.py
            scaling.py
            means.py
            kernels.py
        optimization/
            evaluate.py
            backtesting.py
            portfolio_helpers.py
        schemas/
            common.py
            contracts/
                universe.py
                intent.py
                beliefs.py
                forecast.py
                scenarios.py
                optimize.py
                backtest.py
                report.py
            configs/
                core.py
        io/
            artifacts.py
            parquet_store.py
            mlflow_logger.py
        engine/
            pipeline.py
            data/
            models/
            scenarios/
            optimize/
            backtest/
            report/
            agent/
        visualization/
        mlflow/

    experiments/
        mlflow/
            *.py
            data_snapshots/

------------------------------------------------------------

## Research Direction

BayesFolio sits at the intersection of:

- Bayesian statistics
- Machine learning
- Asset pricing
- Risk management
- Portfolio theory

It aims to move beyond purely empirical optimization toward probabilistic, feature-conditioned portfolio construction.

------------------------------------------------------------

## Status

Current version:
- Research framework
- Modeling + evaluation pipeline
- Scenario-based optimization

Future versions:
- Application layer
- Agentic AI system
- RAG conversational interface

------------------------------------------------------------

## Migration Notes

Recent package reorganization mappings are documented in [docs/migration-map-phase1-3.md](docs/migration-map-phase1-3.md).
Schema-first engine/io migration starts in this version; existing experiment imports remain compatible.
- User-facing portfolio reporting

------------------------------------------------------------

## License

MIT License

------------------------------------------------------------

## Disclaimer

BayesFolio is for research and educational purposes only.  
It does not constitute financial advice.