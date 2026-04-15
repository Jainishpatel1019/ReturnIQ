import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ui_helpers import metric_card, section_header, apply_chart_theme, page_header

def render(df: pd.DataFrame) -> None:
    st.markdown(page_header("Policy Simulator", "Adjust quality thresholds to optimize prevention"), unsafe_allow_html=True)

    if "cate" not in df.columns:
        return

    # User Control
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        threshold = st.slider("Causal Impact Threshold", float(df["cate"].min()), float(df["cate"].max()), float(df["cate"].quantile(0.85)), 0.001)
        st.markdown('</div>', unsafe_allow_html=True)
    
    flagged = df[df["cate"] >= threshold]
    pct_flagged = len(flagged) / len(df) * 100
    
    tot_ret = df["proxy_return_rate"].sum() if "proxy_return_rate" in df.columns else 1
    prev_ret = flagged["proxy_return_rate"].sum() if "proxy_return_rate" in flagged.columns else 0
    pct_prev = (prev_ret / tot_ret * 100) if tot_ret > 0 else 0
    
    med_ret = df.get("proxy_return_rate", pd.Series([0])).median()
    fpr = (flagged.get("proxy_return_rate", pd.Series([0])) < med_ret).mean() * 100 if len(flagged) > 0 else 0

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card("Sellers Flagged", f"{len(flagged):,}", sub=f"Targeting bottom {pct_flagged:.0f}%", color="#f85149"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Returns Preventable", f"{pct_prev:.1f}%", sub="Potential reduction", color="#3fb950"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card("False Positives", f"{fpr:.1f}%", sub="Good sellers flagged", color="#d29922"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card("Target Efficiency", f"{pct_prev/pct_flagged if pct_flagged>0 else 0:.1f}x", sub="Vs random choice"), unsafe_allow_html=True)

    st.markdown(section_header("Prevention vs Precision Tradeoff"), unsafe_allow_html=True)
    
    ts = np.linspace(df["cate"].min(), df["cate"].max(), 60)
    pprev = [df[df["cate"]>=t]["proxy_return_rate"].sum()/tot_ret*100 if tot_ret>0 else 0 for t in ts]
    fp_r = [(df[df["cate"]>=t]["proxy_return_rate"] < med_ret).mean()*100 if len(df[df["cate"]>=t])>0 else 0 for t in ts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fp_r, y=pprev, mode="lines", name="Causal model", line=dict(color="#58a6ff", width=3)))
    fig.add_trace(go.Scatter(x=[fpr], y=[pct_prev], mode="markers", name="Current", marker=dict(size=14, color="#f85149", symbol="star")))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random", line=dict(color="#8b949e", width=1, dash="dash")))
    
    fig.update_layout(xaxis_title="Low-risk sellers caught (%)", yaxis_title="Returns Prevented (%)")
    fig, cfg = apply_chart_theme(fig, height=350)
    st.plotly_chart(fig, config=cfg, use_container_width=True)

    st.markdown(section_header("Flagging Distribution"), unsafe_allow_html=True)
    dpp = df.copy()
    dpp["Status"] = dpp["cate"].apply(lambda x: "Flagged" if x >= threshold else "Safe")
    fig2 = px.histogram(dpp, x="proxy_return_rate", color="Status", nbins=50, barmode="overlay", opacity=0.7, color_discrete_map={"Flagged": "#f85149", "Safe": "#3fb950"})
    fig2, cfg2 = apply_chart_theme(fig2, height=300)
    st.plotly_chart(fig2, config=cfg2, use_container_width=True)
