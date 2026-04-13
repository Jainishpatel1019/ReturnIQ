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
    st.markdown(page_header("Policy simulator", "Drag the slider — see what gets prevented vs who gets caught", status="Interactive", ok=True), unsafe_allow_html=True)

    if "cate" not in df.columns:
        return

    threshold = st.slider("Quality threshold — targeting bottom X% of sellers", float(df["cate"].min()), float(df["cate"].max()), float(df["cate"].quantile(0.85)), 0.001)
    
    flagged = df[df["cate"] >= threshold]
    pct_flagged = len(flagged) / len(df) * 100
    st.caption(f"At this threshold, you're targeting the **bottom {pct_flagged:.0f}%** of sellers by causal impact score.")

    tot_ret = df["proxy_return_rate"].sum() if "proxy_return_rate" in df.columns else 1
    prev_ret = flagged["proxy_return_rate"].sum() if "proxy_return_rate" in flagged.columns else 0
    pct_prev = (prev_ret / tot_ret * 100) if tot_ret > 0 else 0
    
    events = int((pct_prev / 100) * (df.get("proxy_return_rate", pd.Series([0])).mean() * df.get("total_reviews", pd.Series([0])).sum()))
    med_ret = df.get("proxy_return_rate", pd.Series([0])).median()
    fpr = (flagged.get("proxy_return_rate", pd.Series([0])) < med_ret).mean() * 100 if len(flagged) > 0 else 0

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    with r1c1:
        st.markdown(metric_card("Sellers flagged", f"{len(flagged):,}", f"Bottom {pct_flagged:.0f}% by risk", "#f85149"), unsafe_allow_html=True)
    with r1c2:
        st.markdown(metric_card("Returns preventable", f"{pct_prev:.1f}%", f"≈ {events:,} return events", "#3fb950"), unsafe_allow_html=True)
    with r2c1:
        st.markdown(metric_card("Low-risk sellers caught", f"{fpr:.1f}%", "May not need intervention", "#d29922"), unsafe_allow_html=True)
    with r2c2:
        st.markdown(metric_card("Current threshold", f"Top {100-pct_flagged:.0f}% flagged", f"CATE > {threshold:.3f}", "#5b8fff"), unsafe_allow_html=True)

    st.divider()

    st.markdown(section_header("Prevention vs precision tradeoff", "Top-left = ideal: many returns caught, few good sellers flagged"), unsafe_allow_html=True)
    
    ts = np.linspace(df["cate"].min(), df["cate"].max(), 60)
    pprev = [df[df["cate"]>=t]["proxy_return_rate"].sum()/tot_ret*100 if tot_ret>0 else 0 for t in ts]
    fp_r = [(df[df["cate"]>=t]["proxy_return_rate"] < med_ret).mean()*100 if len(df[df["cate"]>=t])>0 else 0 for t in ts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fp_r, y=pprev, mode="lines", name="Causal model", line=dict(color="#5b8fff", width=3)))
    fig.add_trace(go.Scatter(x=[fpr], y=[pct_prev], mode="markers", name="Current threshold", marker=dict(size=14, color="#f85149", symbol="star")))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random guessing", line=dict(color="#8b949e", width=1, dash="dash")))
    fig.add_shape(type="rect", x0=0, y0=70, x1=30, y1=100, fillcolor="#3fb950", opacity=0.1, line_width=0)
    fig.add_annotation(x=15, y=85, text="Ideal zone", showarrow=False, font=dict(size=14, color="#3fb950"))
    fig.add_annotation(x=50, y=10, text="Our model stays above the random line — it targets<br>bad sellers efficiently", showarrow=False, font=dict(size=12, color="#c9d1d9"))
    fig.update_layout(xaxis_title="Low-risk sellers caught (False Positive Rate %)", yaxis_title="Returns Prevented (%)")
    fig, cfg = apply_chart_theme(fig)
    st.plotly_chart(fig, config=cfg, use_container_width=True)

    st.markdown(section_header("Who gets flagged?", "Flagged sellers (red) should cluster right of the median"), unsafe_allow_html=True)
    dpp = df.copy()
    dpp["Status"] = dpp["cate"].apply(lambda x: "Flagged" if x >= threshold else "Not flagged")
    fig2 = px.histogram(dpp, x="proxy_return_rate", color="Status", nbins=50, barmode="overlay", opacity=0.7, color_discrete_map={"Flagged": "#f85149", "Not flagged": "#3fb950"})
    fig2.update_layout(xaxis_title="Estimated return rate (% of reviews that were negative)", yaxis_title="Count")
    fig2, cfg2 = apply_chart_theme(fig2)
    st.plotly_chart(fig2, config=cfg2, use_container_width=True)
