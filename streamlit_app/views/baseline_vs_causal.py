import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ui_helpers import section_header, apply_chart_theme, page_header

def render(df: pd.DataFrame) -> None:
    st.markdown(page_header("Model Proof", "Validating causal CATE scores vs naive OLS predictions"), unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available for the current selection.")
        return

    # Safety check for required columns
    if "ols_pred" not in df.columns or "cate" not in df.columns:
        st.error("Missing analytical columns (ols_pred/cate). Ensure pipeline is complete.")
        return

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown('<div class="glass-card"><div style="font-weight: 600; font-family: \'Inter\', sans-serif; font-size: 15px; color: #58a6ff; margin-bottom: 8px;">Naive Prediction (OLS)</div><div style="font-family: \'Inter\', sans-serif; font-size: 13px; color: #8b949e;">Standard OLS models conflate high-return categories (like Fashion) with bad seller behavior. This leads to biased penalties.</div></div>', unsafe_allow_html=True)
    with c_right:
        st.markdown('<div class="glass-card"><div style="font-weight: 600; font-family: \'Inter\', sans-serif; font-size: 15px; color: #3fb950; margin-bottom: 8px;">Causal Engine (Double ML)</div><div style="font-family: \'Inter\', sans-serif; font-size: 13px; color: #8b949e;">ReturnIQ isolates operational effects by netting out categoric confounders. The results show true behavioral impact.</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(section_header("The Discrepancy Map", "Highlighting sellers where OLS got it wrong"), unsafe_allow_html=True)
    
    # Improved scatter with better visibility and fixed range
    sample_size = min(2000, len(df))
    plot_df = df.sample(sample_size, random_state=42)
    
    fig1 = px.scatter(
        plot_df, 
        x="ols_pred", 
        y="cate", 
        color="category",
        hover_data=["seller_id", "proxy_return_rate"],
        opacity=0.6,
        size_max=10
    )
    
    # Add a diagonal 'Equality' line (where OLS == Causal)
    min_val = min(df["ols_pred"].min(), df["cate"].min())
    max_val = max(df["ols_pred"].max(), df["cate"].max())
    fig1.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', name='Naive Bias Baseline',
        line=dict(color='#8b949e', width=1, dash='dot')
    ))

    def _padded_range(series, pad=0.05):
        lo, hi = series.min(), series.max()
        span = hi - lo if hi != lo else 1.0
        return [lo - span * pad, hi + span * pad]

    fig1.update_layout(
        xaxis_title="Naive Prediction (OLS Rate)",
        yaxis_title="Causal Identification (CATE)",
        xaxis=dict(range=_padded_range(df["ols_pred"])),
        yaxis=dict(range=_padded_range(df["cate"]))
    )
    
    fig1, cfg1 = apply_chart_theme(fig1, height=450)
    st.plotly_chart(fig1, config=cfg1, use_container_width=True)

    st.markdown(section_header("Causal Uplift Precision", "Area Under the Uplift Curve (AUUC)") , unsafe_allow_html=True)
    
    # AUUC calculation logic
    def get_auuc_coords(df_in, scol, ocol):
        ds = df_in.sort_values(scol, ascending=False).reset_index(drop=True)
        n = max(1, len(ds))
        # Expected return capture if targeting by score
        target_sum = ds[ocol].sum() if ds[ocol].sum() != 0 else 1
        cum_outcome = np.cumsum(ds[ocol]) / target_sum
        return np.linspace(0, 100, n), cum_outcome * 100

    x_c, y_c = get_auuc_coords(df, "cate", "proxy_return_rate")
    x_o, y_o = get_auuc_coords(df, "ols_pred", "proxy_return_rate")
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=x_c, y=y_c, name="ReturnIQ (Causal)", line=dict(color="#58a6ff", width=3)))
    fig3.add_trace(go.Scatter(x=x_o, y=y_o, name="Naive OLS", line=dict(color="#d29922", width=2, dash="dash")))
    fig3.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Random Choice", line=dict(color="#8b949e", width=1, dash="dot")))
    
    fig3.update_layout(
        xaxis_title="% of Marketplace Targeted", 
        yaxis_title="% of Returns Recovered",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig3, cfg3 = apply_chart_theme(fig3, height=400)
    st.plotly_chart(fig3, config=cfg3, use_container_width=True)
    
    # Calculate real AUUC value
    # Area between the curve and the random diagonal
    # np.trapezoid is the non-deprecated name in numpy >= 2.0 (np.trapz was removed)
    _trapz = getattr(np, "trapezoid", np.trapz)
    auuc_val = _trapz(y_c / 100, x_c / 100) - 0.5
    st.markdown(f'<div class="glass-card" style="text-align: center;"><div style="color: #58a6ff; font-size: 20px; font-weight: 700;">{auuc_val:.3f} AUUC Lift</div><div style="color: #8b949e; font-size: 13px;">Positive lift indicates the model effectively ranks sellers by their true causal impact on losses.</div></div>', unsafe_allow_html=True)
