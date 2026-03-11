"""Guided portfolio builder — BayesFolio wizard chatbot.

A step-by-step conversational UI that walks the user through four questions
(assets, objective, risk, constraints) before running the historical MVP
optimization pipeline.

Usage:
    poetry run streamlit run guided_portfolio_app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from bayesfolio.engine.guided_portfolio_chat import (
    GuidedChatState,
    GuidedChatStep,
    advance_guided_chat,
    build_request_from_state,
    initial_prompt,
)
from bayesfolio.engine.mvp_historical_chat import run_historical_mvp_pipeline

st.set_page_config(page_title="BayesFolio Guided Builder", page_icon="🧭", layout="wide")
st.title("BayesFolio Guided Portfolio Builder")
st.caption("Answer a few questions to build and optimize your portfolio step by step.")

# ── Session state ────────────────────────────────────────────────────────────

if "guided_state" not in st.session_state:
    st.session_state.guided_state = GuidedChatState()

if "guided_messages" not in st.session_state:
    st.session_state.guided_messages = [{"role": "assistant", "content": initial_prompt()}]

if "guided_result" not in st.session_state:
    st.session_state.guided_result = None

# ── Render conversation history ───────────────────────────────────────────────

for msg in st.session_state.guided_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Results panel (shown after pipeline completes) ───────────────────────────

if st.session_state.guided_result is not None:
    result = st.session_state.guided_result

    st.markdown(result.report_markdown)

    st.subheader("Optimized Weights")
    weights_data = result.weights_table.to_dict(orient="records")
    st.dataframe(weights_data, hide_index=True, use_container_width=True)

    if not result.weights_table.empty:
        donut = px.pie(
            result.weights_table,
            names="asset",
            values="weight",
            hole=0.5,
            title="Portfolio Weight Allocation",
        )
        donut.update_traces(textposition="inside", texttemplate="%{label}<br>%{percent}")
        st.plotly_chart(donut, use_container_width=True)

    if result.portfolio_metrics:
        st.subheader("Portfolio Metrics")
        percent_keys = {"cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown"}
        metric_labels = {
            "cumulative_return": "Cumulative Return",
            "annualized_return": "Annualized Return (CAGR)",
            "annualized_volatility": "Annualized Volatility",
            "max_drawdown": "Max Drawdown",
            "sharpe_ratio": "Sharpe Ratio",
            "sortino_ratio": "Sortino Ratio",
            "calmar_ratio": "Calmar Ratio",
        }
        rows: list[dict[str, str]] = []
        for key in metric_labels:
            value = result.portfolio_metrics.get(key)
            if isinstance(value, int | float):
                display = f"{float(value) * 100:.2f}%" if key in percent_keys else f"{float(value):.2f}"
                rows.append({"Metric": metric_labels[key], "Value": display})
        if rows:
            st.table(pd.DataFrame(rows))

    quality = result.data_quality
    st.subheader("Data Quality")
    st.dataframe(
        {
            "Metric": ["Pass Gate", "Periods", "Assets", "Stale Assets", "Insufficient History"],
            "Value": [
                str(quality.pass_gate),
                str(quality.n_periods),
                str(quality.n_assets),
                ", ".join(quality.stale_assets) or "None",
                ", ".join(quality.insufficient_history_assets) or "None",
            ],
        },
        hide_index=True,
        use_container_width=True,
    )

    if result.warnings:
        st.subheader("Warnings")
        for warning in result.warnings:
            st.warning(warning)

    if st.button("Start Over"):
        st.session_state.guided_state = GuidedChatState()
        st.session_state.guided_messages = [{"role": "assistant", "content": initial_prompt()}]
        st.session_state.guided_result = None
        st.rerun()

    st.stop()

# ── Chat input (active while wizard is in progress) ──────────────────────────

prompt = st.chat_input("Your answer...")

if prompt:
    current_state: GuidedChatState = st.session_state.guided_state

    st.session_state.guided_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            new_state, bot_reply = advance_guided_chat(current_state, prompt)
        st.markdown(bot_reply)

    st.session_state.guided_messages.append({"role": "assistant", "content": bot_reply})
    st.session_state.guided_state = new_state

    if new_state.step == GuidedChatStep.COMPLETE:
        request = build_request_from_state(new_state)
        with st.chat_message("assistant"):
            try:
                with st.status("Running portfolio optimization...", expanded=True) as status:
                    result = run_historical_mvp_pipeline(request, progress=status.write)
                    status.update(label="Optimization complete.", state="complete")
                st.session_state.guided_result = result
                st.session_state.guided_messages.append({"role": "assistant", "content": result.report_markdown})
                st.rerun()
            except Exception as exc:
                error_msg = f"Optimization failed: {exc}"
                st.error(error_msg)
                st.session_state.guided_messages.append({"role": "assistant", "content": error_msg})
