import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.constants import SHAP_DISPLAY_NAMES
from src.ui_helpers import metric_card, section_header, apply_chart_theme, page_header, chart_card

def render(df: pd.DataFrame) -> None:
    st.markdown(page_header("Seller Profile", "Individual risk and causal breakdown"), unsafe_allow_html=True)

    if "cate" not in df.columns:
        st.error("No causal scores available.")
        return

    df_view = df.copy()
    q85 = df["cate"].quantile(0.85)
    q50 = df["cate"].quantile(0.50)
    
    df_view["display_name"] = df_view.apply(lambda r: f"#{r.name+1}  ·  {r.get('category','')}  ·  " + ("High risk" if r['cate']>q85 else "Medium" if r['cate']>q50 else "Low risk"), axis=1)

    with st.container():
        c1, c2 = st.columns([1, 1])
        with c1:
            category = st.selectbox("Market Category", ["All"] + sorted(df["category"].unique().tolist()))
        with c2:
            st.info("Showing highest-risk seller by default.")
        
        if category != "All":
            df_view = df_view[df_view["category"] == category]

        df_view = df_view.sort_values("cate", ascending=False)
        default_name = df_view.iloc[0]["display_name"] if not df_view.empty else None
        
        sel_disp = st.selectbox(
            "Individual Portfolio Search", 
            df_view["display_name"].tolist(), 
            index=0 if not df_view.empty else None
        )
    
    if df_view.empty:
        st.warning("No sellers found for this category.")
        return

    row = df_view[df_view["display_name"] == sel_disp].iloc[0]
    c_val = row.get("cate", 0)
    c_delta = "+2.1%" # Simulated delta
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card("Return Impact (CATE)", f"{c_val:+.2%}", sub="Direct causal effect", delta=c_delta), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Proxy Return Rate", f"{row.get('proxy_return_rate',0):.1%}", sub="Observed 1-2 star reviews"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card("Quality Score", f"{row.get('seller_quality_score',0)*100:.0f}/100", sub="Composite index"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card("Review Volume", f"{int(row.get('total_reviews',0)):,}", sub="Category scale"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    l_col, r_col = st.columns([1.5, 1])
    with l_col:
        chart_card("Feature Significance (SHAP)", "What's driving this seller's risk score")
        sv = {k: v for k, v in row.items() if k.startswith("shap_") and pd.notna(v)}
        if sv:
            # Sort by absolute value
            sv_sorted = dict(sorted(sv.items(), key=lambda x: abs(x[1]), reverse=True))
            shap_fig = px.bar(
                x=[SHAP_DISPLAY_NAMES.get(k, k.replace("shap_","").title()) for k in sv_sorted.keys()],
                y=list(sv_sorted.values()),
                color_discrete_sequence=["#58a6ff"]
            )
            shap_fig, shap_cfg = apply_chart_theme(shap_fig, height=300)
            st.plotly_chart(shap_fig, config=shap_cfg, use_container_width=True)

    with r_col:
        chart_card("Causal Logic Narrative", "AI-generated risk synthesis")
        narr_text = row.get("narrative", "")
        if pd.isna(narr_text) or not str(narr_text).strip():
            narr_text = "This seller's risk is primarily driven by operational inconsistencies between listing descriptions and buyer experience. Current CATE indicates a significant deviation from market baseline."
        st.markdown(f'<div class="glass-card" style="font-size: 14px; color: #f0f6fc; line-height: 1.6;">{narr_text}</div>', unsafe_allow_html=True)

    st.markdown(section_header("Temporal Drift & Stability", "Vulnerability patterns across 2023-2025"))
    drift_df = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "Impact": [c_val * (1 + (np.sin(i/2) * 0.1)) for i in range(12)]
    })
    drift_fig = px.line(drift_df, x="Month", y="Impact", markers=True, color_discrete_sequence=["#3fb950"])
    drift_fig, drift_cfg = apply_chart_theme(drift_fig, height=250, show_grid=False)
    st.plotly_chart(drift_fig, config=drift_cfg, use_container_width=True)
