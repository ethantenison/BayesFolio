# BayesFolio

BayesFolio is an AI-assisted portfolio research prototype that explores how machine learning, conversational interfaces, and risk-aware optimization can improve portfolio construction workflows in financial services.

The project combines natural-language portfolio intake, LLM-assisted intent extraction, optimization pipelines, and transparent diagnostics in a sandbox environment for experimentation. It is designed to test product concepts, modeling workflows, and user experience patterns before productionization.

------------------------------------------------------------

## Project Status

BayesFolio is currently undergoing a major refactor toward a cleaner schema-first architecture with stricter separation between contracts, business logic, and I/O.

This repository should be viewed as an active R&D codebase rather than a polished end-user application. The most relevant artifacts today are the system design, prototype workflows, and architecture decisions that support AI-assisted financial decision tooling.

What this status means in practice:

- The project contains working prototype flows and orchestration patterns
- The repository is being reorganized for clearer boundaries and better maintainability
- The focus is on validating ideas, workflows, and engineering patterns rather than packaging a finished product

------------------------------------------------------------

## Why This Project Exists

BayesFolio started as an exploration of a simple question: how could a portfolio tool move beyond purely historical optimization and toward a more intelligent, more interactive, and more transparent decision-support experience?

The project focuses on three R&D themes:

- AI-assisted workflow design for financial tasks
- Bayesian and uncertainty-aware approaches to portfolio construction
- Explainable reporting that keeps assumptions, constraints, and risk structure visible

This makes BayesFolio less of a static optimizer and more of a research platform for testing how ML and GenAI concepts could fit into future wealth-management experiences.

------------------------------------------------------------

## Current Prototype Capabilities

Today, BayesFolio demonstrates several prototype patterns that are directly relevant to AI/ML R&D work:

- Guided portfolio construction through a conversational step-by-step flow
- Free-form chat-based portfolio intake with rule-based and LLM-assisted parsing
- Extraction of user goals, risk preferences, and portfolio constraints from natural language
- Historical portfolio optimization with diagnostics, warnings, and summary reporting
- A schema-first design direction for keeping AI, domain logic, and external integrations cleanly separated

Representative prototype surfaces in the repository include:

- A guided portfolio builder for structured conversational intake
- A chat-based MVP interface for free-form portfolio requests
- Orchestration modules that connect parsing, data quality checks, optimization, and report assembly

------------------------------------------------------------

## Current Workflow

The current BayesFolio workflow centers on turning ambiguous user intent into a structured portfolio workflow:

    User Request or Guided Input
        ↓
    Intent / Constraint Extraction
        ↓
    Universe Construction and Data Quality Checks
        ↓
    Feature and Forecasting Pipeline Scaffolding
        ↓
    Portfolio Optimization
        ↓
    Diagnostics, Backtest Metrics, and Report Assembly

The longer-term vision extends beyond historical optimization, but the present repository already captures the orchestration and UX patterns required to test those future concepts.

------------------------------------------------------------

## Technical Focus

BayesFolio is intentionally structured around the kinds of concerns that matter in applied AI/ML prototyping:

- Python-first experimentation and orchestration
- Typed contracts and schema-driven interfaces using Pydantic
- Separation between engine logic and external I/O boundaries
- LLM-assisted extraction layered on top of deterministic fallbacks
- Portfolio optimization and backtest evaluation workflows
- Artifact persistence, caching, and reproducible experimentation patterns
- Transparent diagnostic outputs rather than opaque model responses

From an engineering perspective, the repository reflects an emphasis on rapid prototyping without abandoning software design discipline.

------------------------------------------------------------

## Architecture Direction

The refactor is moving BayesFolio toward a clearer architecture with distinct responsibilities across the codebase:

- `bayesfolio/core/` for primitive types and configuration
- `bayesfolio/contracts/` for typed schemas at system boundaries
- `bayesfolio/engine/` for pure business logic and orchestration
- `bayesfolio/io/` for providers, persistence, and external system interaction

That direction matters because the project is intended to support experimentation without allowing prototype code to collapse into tightly coupled application logic.

------------------------------------------------------------

## Relevance To AI/ML R&D

BayesFolio is meant to be credible as a research-and-development prototype, not just as a modeling exercise. The project emphasizes capabilities that are common in exploratory AI engineering work:

- Turning ambiguous ideas into functional prototypes
- Testing user-facing AI concepts in sandbox environments
- Combining traditional ML logic with emerging LLM interaction patterns
- Evaluating whether a prototype is informative, explainable, and directionally useful before hardening it for production
- Designing systems that can be presented, discussed, and iterated on with non-technical stakeholders

In that sense, BayesFolio is as much about workflow invention and product experimentation as it is about forecasting and optimization.

------------------------------------------------------------

## Research Direction

BayesFolio is evolving toward a more forward-looking portfolio intelligence platform that may eventually include:

- Bayesian forecasting models such as multitask Gaussian processes and hierarchical approaches
- Domain-knowledge-aware model structure and kernel design
- Black-Litterman-style view integration
- Scenario-based optimization beyond purely historical estimates
- Richer interactive reporting and model explanation layers
- More agentic conversational workflows for portfolio construction
- Responsible AI safeguards tailored to financial decision support

These are active research directions rather than committed product promises.

------------------------------------------------------------

## Responsible AI Principles

BayesFolio emphasizes:

- Transparent model structure and assumptions
- Visibility into risk diagnostics rather than black-box recommendations
- Clear distinction between historical evidence and predictive estimates
- Explicit treatment of uncertainty where possible
- Reproducibility through deterministic seeds and experiment logging

The project is intended to make portfolio risk structure easier to inspect, question, and communicate.

------------------------------------------------------------

## Contributor Workflow (Agents)

BayesFolio includes repository-level agent roles in [AGENTS.md](AGENTS.md) to improve maintainability and reduce architecture and documentation drift during the refactor.

Recommended PR workflow:

1. Run **Boundary Inspector** checks for package and layer violations.
2. Run **Documentation Drift Inspector** checks for docstring and documentation sync.
3. If findings exist, use **Patch Proposer** in propose mode for minimal diffs.
4. Apply patches only after explicit approval.
5. Re-run linting, targeted tests, and touched-file diagnostics before handoff.

------------------------------------------------------------

## License

MIT License

------------------------------------------------------------

## Disclaimer

BayesFolio is for research and educational purposes only. It does not constitute financial advice.