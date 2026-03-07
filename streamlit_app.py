"""
To run streamlit app: poetry run streamlit run streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from bayesfolio.engine.mvp_historical_chat import (
    run_historical_mvp_chat_turn,
)

st.set_page_config(page_title="BayesFolio MVP Chat", page_icon="📈", layout="wide")
st.title("BayesFolio Historical MVP Chat")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ask for a portfolio with tickers and optional settings. "
                "Example: 'Build a portfolio for SPY, QQQ, TLT from 2019-01-01 to 2024-12-31 "
                "objective sharpe risk cvar'."
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Describe your portfolio request...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.status("Running MVP agent workflow...", expanded=True) as status:
                turn = run_historical_mvp_chat_turn(prompt, progress=status.write)
                status.update(label="MVP workflow complete.", state="complete")

            if not turn.tool_results:
                raise RuntimeError("No tool results found for chat turn.")

            payload = turn.tool_results[-1].payload
            report_markdown = str(payload.get("report_markdown", "Historical MVP run completed."))
            st.markdown(report_markdown)

            weights = payload.get("weights", [])
            st.subheader("Optimized Weights")
            st.dataframe(weights, hide_index=True, use_container_width=True)

            if isinstance(weights, list) and weights:
                weights_df = pd.DataFrame(weights)
                if {"asset", "weight"}.issubset(weights_df.columns):
                    donut = px.pie(
                        weights_df,
                        names="asset",
                        values="weight",
                        hole=0.5,
                        title="Portfolio Weight Allocation",
                    )
                    donut.update_traces(textposition="inside", texttemplate="%{label}<br>%{percent}")
                    st.plotly_chart(donut, use_container_width=True)

            metrics = payload.get("metrics", {})
            if isinstance(metrics, dict) and metrics:
                st.subheader("Portfolio Metrics")
                percent_keys = {
                    "cumulative_return",
                    "annualized_return",
                    "annualized_volatility",
                    "max_drawdown",
                }
                ratio_keys = {"sharpe_ratio", "sortino_ratio", "calmar_ratio"}
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
                for key in [
                    "cumulative_return",
                    "annualized_return",
                    "annualized_volatility",
                    "max_drawdown",
                    "sharpe_ratio",
                    "sortino_ratio",
                    "calmar_ratio",
                ]:
                    value = metrics.get(key)
                    if isinstance(value, int | float):
                        if key in percent_keys:
                            display_value = f"{float(value) * 100:.2f}%"
                        elif key in ratio_keys:
                            display_value = f"{float(value):.2f}"
                        else:
                            display_value = str(value)
                        rows.append(
                            {
                                "Metric": metric_labels.get(key, key.replace("_", " ").title()),
                                "Value": display_value,
                            }
                        )

                if rows:
                    st.table(pd.DataFrame(rows))

            quality = payload.get("data_quality", {})
            quality_table = {
                "Metric": ["Pass Gate", "Periods", "Assets", "Stale Assets", "Insufficient History Assets"],
                "Value": [
                    str(quality.get("pass_gate", "N/A")),
                    str(quality.get("n_periods", "N/A")),
                    str(quality.get("n_assets", "N/A")),
                    ", ".join(quality.get("stale_assets", [])) or "None",
                    ", ".join(quality.get("insufficient_history_assets", [])) or "None",
                ],
            }
            st.subheader("Data Quality")
            st.dataframe(quality_table, hide_index=True, use_container_width=True)

            feature_artifact = payload.get("feature_artifact")
            if feature_artifact:
                st.subheader("Feature Artifact")
                st.json(feature_artifact)

            warnings = payload.get("warnings", [])
            if warnings:
                st.subheader("Warnings")
                for warning in warnings:
                    st.warning(warning)

            with st.expander("Agent Trace", expanded=False):
                st.markdown("**Turn Diagnostics**")
                st.json(turn.diagnostics)

                st.markdown("**Tool Calls**")
                st.json(
                    [
                        {
                            "call_id": call.call_id,
                            "tool_name": call.tool_name,
                            "arguments": call.arguments,
                        }
                        for call in turn.tool_calls
                    ]
                )

                st.markdown("**Tool Results**")
                st.json(
                    [
                        {
                            "call_id": result_item.call_id,
                            "tool_name": result_item.tool_name,
                            "success": result_item.success,
                            "error_message": result_item.error_message,
                            "payload": result_item.payload,
                        }
                        for result_item in turn.tool_results
                    ]
                )

            assistant_text = turn.assistant_message.content if turn.assistant_message is not None else report_markdown
        except Exception as exc:
            assistant_text = f"Unable to complete workflow: {exc}"
            st.error(assistant_text)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
