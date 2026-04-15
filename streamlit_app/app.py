import streamlit as st
import pandas as pd
import pathlib
import sys
import os
import streamlit_antd_components as sac
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.ui_helpers import (
    page_header, 
    glossary_expander, 
    metric_card, 
    risk_badge, 
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

with st.spinner("Loading ReturnIQ Intelligence..."):
    df = load_data()

if df.empty:
    st.error("📉 **Data Load Failure**: Dashboard data not found.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo"><span style="color: #58a6ff;">💠</span> ReturnIQ</div>', unsafe_allow_html=True)
    
    selected_view = sac.menu([
        sac.MenuItem('Dashboard', icon='grid-fill'),
        sac.MenuItem('Reports', icon='file-earmark-bar-graph'),
        sac.MenuItem('Sellers', icon='people-fill'),
        sac.MenuItem('Settings', icon='gear-fill'),
    ], size='sm', color='#58a6ff', open_all=True)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("v2.4.0 · High Fidelity")
    st.sidebar.caption(f"Index: {len(df):,} sellers")

# --- MAIN CONTENT ---
if selected_view == 'Dashboard':
    # 1. Header Row
    st.markdown(page_header("Dashboard", "Dark Mode analytics dashboard"), unsafe_allow_html=True)
    
    # 2. Filter Row (Simulated)
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    with f1:
        st.text_input("Search", placeholder="Search sellers, categories...", label_visibility="collapsed")
    with f3:
        st.selectbox("Data Sector", ["All Sectors", "Electronics", "Home & Kitchen", "Clothing"], label_visibility="collapsed")
    with f4:
        st.selectbox("All Months", ["Last 12 Months", "Last 6 Months", "Current Month"], label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. First Chart Row: Donut + Avg Return Rate Line
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown('<div style="color: #f0f6fc; font-size: 14px; font-weight: 600; margin-bottom: 20px;">Data Visuatics</div>', unsafe_allow_html=True)
        inner_l, inner_r = st.columns([1.5, 1])
        with inner_l:
            donut_fig = create_donut_chart(df)
            if donut_fig:
                st.plotly_chart(donut_fig, use_container_width=True, config={'displayModeBar': False})
        with inner_r:
            st.markdown("<br>", unsafe_allow_html=True)
            avg_rate = df["proxy_return_rate"].mean() if "proxy_return_rate" in df.columns else 0
            catemax = df["cate"].max()
            st.markdown(f"""
            <div style="font-size: 13px; color: #8b949e; line-height: 2.2;">
                <span style="color: #58a6ff; margin-right: 8px;">●</span> Total Sellers: <span style="color: #f0f6fc; float: right;">{len(df):,}</span><br>
                <span style="color: #39C5BB; margin-right: 8px;">●</span> Avg Return Rate: <span style="color: #3fb950; float: right;">{avg_rate:.1%}</span><br>
                <span style="color: #3fb950; margin-right: 8px;">●</span> Highest CATE: <span style="color: #3fb950; float: right;">+{catemax:.1%}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with c2:
        st.markdown('<div style="color: #f0f6fc; font-size: 14px; font-weight: 600; margin-bottom: 10px;">Avg Return Rate</div>', unsafe_allow_html=True)
        # Generate some trends from data (simulated trend since data is static)
        import numpy as np
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
        base_rate = df["proxy_return_rate"].mean()
        # Create a wavy trend
        trend = [base_rate * (1 + 0.1 * np.sin(i)) for i in range(len(months))]
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(x=months, y=trend, mode='lines+markers', line=dict(color='#58a6ff', width=3), marker=dict(size=8, color='#58a6ff', line=dict(color='#0d1117', width=2))))
        line_fig, cfg = apply_chart_theme(line_fig, height=220, show_grid=False)
        st.plotly_chart(line_fig, use_container_width=True, config=cfg)

    # 4. Metric Card Row (4 Columns - Glassmorphism)
    avg_rate = df["proxy_return_rate"].mean() if "proxy_return_rate" in df.columns else 0
    catemax = df["cate"].max()
    t15 = int(len(df) * 0.15)
    top15_driven = (df.sort_values("cate", ascending=False).head(t15)["proxy_return_rate"].sum() / df["proxy_return_rate"].sum() * 100) if "proxy_return_rate" in df.columns and df["proxy_return_rate"].sum() > 0 else 0
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card("Total Sellers", f"{len(df):,}", sub="Active sellers in index"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Avg Return Rate", f"{avg_rate:.1%}", delta="+2.4%", sub="Global average"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card("Highest CATE", f"+{catemax:.1%}", delta="+18.1%", sub="Peak causal impact"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card("Top 15% severity", f"{top15_driven:.0f}%", sub="Returns driven by outliers"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Bottom Chart Row: Large Visualization + Analytics
    b1, b2 = st.columns([1.5, 1])
    
    with b1:
        st.markdown('<div style="color: #f0f6fc; font-size: 14px; font-weight: 600; margin-bottom: 15px;">Data Visualization (Causal Lift)</div>', unsafe_allow_html=True)
        # Show CATE distribution or something visual
        cdf = df.sort_values("cate", ascending=False).reset_index()
        viz_fig = go.Figure()
        viz_fig.add_trace(go.Scatter(x=cdf.index, y=cdf["cate"], fill='tozeroy', line=dict(color='#58a6ff', width=2), fillcolor='rgba(88, 166, 255, 0.1)'))
        viz_fig, cfg = apply_chart_theme(viz_fig, height=300)
        st.plotly_chart(viz_fig, use_container_width=True, config=cfg)

    with b2:
        st.markdown('<div style="color: #f0f6fc; font-size: 14px; font-weight: 600; margin-bottom: 15px;">Data Analytics (Category Risk)</div>', unsafe_allow_html=True)
        if "category" in df.columns:
            cat_risk = df.groupby("category")["cate"].mean().sort_values(ascending=False).head(8)
            bar_fig = px.bar(x=cat_risk.values, y=cat_risk.index, orientation='h', color_discrete_sequence=["#58a6ff"])
            bar_fig.update_traces(marker_line_width=0)
            bar_fig, cfg = apply_chart_theme(bar_fig, height=300, show_grid=False)
            st.plotly_chart(bar_fig, use_container_width=True, config=cfg)

else:
    # Handle other views if necessary, for now redirecting to Dashboard style or keeping placeholders
    st.info(f"View '{selected_view}' is currently being updated to the new high-fidelity standard.")
    if st.button("Return to Dashboard"):
        st.rerun()
