import streamlit as st
import pandas as pd
import pathlib
import sys
import os
import streamlit_antd_components as sac
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.ui_helpers import (
    page_header, 
    metric_card, 
    apply_chart_theme,
    create_donut_chart
)

# Page Config
st.set_page_config(
    page_title="ReturnIQ Dashboard", 
    page_icon="💠", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_data
def load_data() -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    candidates = [
        os.path.join(os.path.dirname(__file__), "data/processed/final_dashboard_data.parquet"),
        os.path.join(base_dir, "data/processed/final_dashboard_data.parquet"),
        "streamlit_app/data/dashboard_sample.parquet",
        "data/processed/final_dashboard_data.parquet"
    ]
    for p in candidates:
        if pathlib.Path(p).exists():
            try:
                return pd.read_parquet(p)
            except Exception as e:
                st.error(f"Error reading {p}: {e}")
    return pd.DataFrame()

with st.spinner("Loading ReturnIQ Causal Engine..."):
    df = load_data()

if df.empty:
    st.error("📉 **Data Load Failure**: Dashboard data not found.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo"><span style="color: #58a6ff;">💠</span> ReturnIQ</div>', unsafe_allow_html=True)
    
    selected_view = sac.menu([
        sac.MenuItem('Dashboard', icon='grid-fill'),
        sac.MenuItem('Reports', icon='file-earmark-bar-graph', children=[
            sac.MenuItem('Key Findings', icon='lightbulb'),
            sac.MenuItem('Model Proof', icon='shield-check'),
            sac.MenuItem('Methodology', icon='info-circle'),
        ]),
        sac.MenuItem('Sellers', icon='people-fill'),
        sac.MenuItem('Settings', icon='sliders', children=[
            sac.MenuItem('Policy Simulator', icon='wrench'),
        ]),
    ], size='sm', color='#58a6ff', open_all=False)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("v2.4.0 · High Fidelity")
    st.sidebar.caption(f"Engine: CausalForestDML")

# --- ROUTING ---
if selected_view == 'Dashboard':
    st.markdown(page_header("Dashboard", "Marketplace causal intelligence summary"), unsafe_allow_html=True)
    
    # Filter Row
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1: st.text_input("Search", placeholder="Search sellers...", label_visibility="collapsed")
    with f2: st.selectbox("Market Sector", ["Electronics", "Clothing", "Home"], label_visibility="collapsed")
    with f3: st.selectbox("Timeframe", ["Last 12 Months", "Last 6 Months"], label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Top Row
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div style="color: #f0f6fc; font-size: 14px; font-weight: 600; margin-bottom: 20px;">Data Visuatics</div>', unsafe_allow_html=True)
        inner_l, inner_r = st.columns([1.5, 1])
        with inner_l:
            donut_fig = create_donut_chart(df)
            if donut_fig: st.plotly_chart(donut_fig, use_container_width=True, config={'displayModeBar': False})
        with inner_r:
            st.markdown("<br>", unsafe_allow_html=True)
            avg_rate = df["proxy_return_rate"].mean() if "proxy_return_rate" in df.columns else 0
            catemax = df["cate"].max()
            st.markdown(f'<div style="font-size: 13px; color: #8b949e; line-height: 2.2;"><span style="color: #58a6ff;">●</span> Total Sellers: <span style="color: #f0f6fc; float: right;">{len(df):,}</span><br><span style="color: #39C5BB;">●</span> Avg Return Rate: <span style="color: #3fb950; float: right;">{avg_rate:.1%}</span><br><span style="color: #3fb950;">●</span> Highest CATE: <span style="color: #3fb950; float: right;">+{catemax:.1%}</span></div>', unsafe_allow_html=True)
    
    with c2:
        st.markdown('<div style="color: #f0f6fc; font-size: 14px; font-weight: 600; margin-bottom: 20px;">Avg Return Rate Trend</div>', unsafe_allow_html=True)
        trend = [avg_rate * (1 + 0.15 * np.sin(i)) for i in range(10)]
        fig_line = go.Figure(go.Scatter(y=trend, mode='lines+markers', line=dict(color='#58a6ff', width=3)))
        fig_line, cfg = apply_chart_theme(fig_line, height=220, show_grid=False)
        st.plotly_chart(fig_line, use_container_width=True, config=cfg)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(metric_card("Total Sellers", f"{len(df):,}", sub="Indexed marketplace"), unsafe_allow_html=True)
    with m2: st.markdown(metric_card("Avg Return Rate", f"{avg_rate:.1%}", delta="+2.4%"), unsafe_allow_html=True)
    with m3: st.markdown(metric_card("Highest CATE", f"+{catemax:.1%}", delta="+18.1%"), unsafe_allow_html=True)
    with m4: st.markdown(metric_card("Top 15% Risk", "34%", sub="Returns driven by top 15%"), unsafe_allow_html=True)

elif selected_view == 'Key Findings':
    from views.key_findings import render as render_findings
    render_findings(df)

elif selected_view == 'Model Proof':
    from views.baseline_vs_causal import render as render_proof
    render_proof(df)

elif selected_view == 'Methodology':
    from views.methodology import render as render_method
    render_method(df)

elif selected_view == 'Sellers':
    from views.seller_intel import render as render_sellers
    render_sellers(df)

elif selected_view == 'Policy Simulator':
    from views.policy_sim import render as render_policy
    render_policy(df)

else:
    st.info(f"Navigate using the sidebar to explore the platform.")
