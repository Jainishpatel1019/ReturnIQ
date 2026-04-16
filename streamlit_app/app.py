import streamlit as st
import pandas as pd
import pathlib
import sys
import os
import streamlit_antd_components as sac
import plotly.graph_objects as go
import numpy as np

# Add project root to path (works for local subdir and HF root)
curr_dir = os.path.dirname(__file__)
proj_root = os.path.dirname(curr_dir) if os.path.basename(curr_dir) == "streamlit_app" else curr_dir
if proj_root not in sys.path:
    sys.path.append(proj_root)
from src.ui_helpers import (
    page_header, 
    metric_card, 
    apply_chart_theme,
    create_donut_chart
)

# Page Config
st.set_page_config(
    page_title="ReturnIQ | Causal Marketplace Intelligence", 
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
    # Robust path resolution: handle both local (subdir) and HF (root) environments
    curr_dir = os.path.dirname(__file__)
    proj_root = os.path.dirname(curr_dir) if os.path.basename(curr_dir) == "streamlit_app" else curr_dir
    
    candidates = [
        # Check relative to project root (works in both local and HF if structured correctly)
        os.path.join(proj_root, "data", "processed", "final_dashboard_data.parquet"),
        os.path.join(proj_root, "data", "processed", "dashboard_sample.parquet"),
        os.path.join(proj_root, "data", "sample", "dashboard_sample.parquet"),
        # Last resort: check immediate subfolder (unlikely but safe)
        os.path.join(curr_dir, "data", "processed", "final_dashboard_data.parquet"),
    ]
    for p in candidates:
        if pathlib.Path(p).exists():
            try:
                return pd.read_parquet(p)
            except Exception as e:
                st.error(f"Error reading {p}: {e}")
    return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
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
    st.sidebar.caption("v2.4.0 · Production Ready")
    st.sidebar.caption("Engine: Pre-computed DML (EconML)")
    st.sidebar.caption("Scope: 3.7M Customer Reviews")

# --- GLOBAL FILTERS (Stateful) ---
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = "All Sectors"
if 'selected_seller' not in st.session_state:
    st.session_state.selected_seller = "All Sellers"

# --- ROUTING ---
if selected_view == 'Dashboard':
    st.markdown(page_header("Dashboard", "Marketplace causal intelligence summary"), unsafe_allow_html=True)
    
    # Filter Row
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        seller_list = ["All Sellers"] + sorted(df_raw["seller_id"].unique().tolist())
        sel_seller = st.selectbox("Search Sellers", seller_list, index=seller_list.index(st.session_state.selected_seller) if st.session_state.selected_seller in seller_list else 0, label_visibility="collapsed")
        st.session_state.selected_seller = sel_seller
    with f2:
        sectors = ["All Sectors"] + sorted(df_raw["category"].unique().tolist())
        sel_sector = st.selectbox("Market Sector", sectors, index=sectors.index(st.session_state.selected_sector) if st.session_state.selected_sector in sectors else 0, label_visibility="collapsed")
        st.session_state.selected_sector = sel_sector
    with f3:
        st.selectbox("Timeframe", ["Full History [Finalized]"], label_visibility="collapsed")
    
    # Apply Filtering
    df = df_raw.copy()
    if st.session_state.selected_sector != "All Sectors":
        df = df[df["category"] == st.session_state.selected_sector]

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Global Stats for deltas
    global_avg = df_raw["proxy_return_rate"].mean()
    global_catemax = df_raw["cate"].max()

    # Top Row
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div style="color: #f0f6fc; font-size: 14px; font-weight: 600; margin-bottom: 20px;">Return Intensity Profile</div>', unsafe_allow_html=True)
        inner_l, inner_r = st.columns([1.5, 1])
        with inner_l:
            donut_fig = create_donut_chart(df)
            if donut_fig:
                st.plotly_chart(donut_fig, use_container_width=True, config={'displayModeBar': False})
        with inner_r:
            st.markdown("<br>", unsafe_allow_html=True)
            avg_rate = df["proxy_return_rate"].mean() if not df.empty else 0
            catemax = df["cate"].max() if not df.empty else 0
            st.markdown(f'<div style="font-size: 13px; color: #8b949e; line-height: 2.2;"><span style="color: #58a6ff;">●</span> Total Sellers: <span style="color: #f0f6fc; float: right;">{len(df):,}</span><br><span style="color: #39C5BB;">●</span> Proxy Return Rate (1-2★): <span style="color: #3fb950; float: right;">{avg_rate:.1%}</span><br><span style="color: #3fb950;">●</span> Peaks (CATE): <span style="color: #3fb950; float: right;">{catemax:+.1%}</span></div>', unsafe_allow_html=True)
    
    with c2:
        st.markdown('<div style="color: #f0f6fc; font-size: 14px; font-weight: 600; margin-bottom: 20px;">Risk Density (CATE Distribution)</div>', unsafe_allow_html=True)
        if not df.empty:
            hist_data = df["cate"].sample(min(2000, len(df)), random_state=42)
            clo, chi = float(hist_data.min()), float(hist_data.max())
            pad = (chi - clo) * 0.05
            counts, bins = np.histogram(hist_data, bins=25)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            fig_dist = go.Figure(go.Scatter(
                x=bin_centers.tolist(), y=counts.tolist(), fill='tozeroy',
                line=dict(color='#58a6ff', width=2)
            ))
            fig_dist.update_layout(
                xaxis_title="Causal Impact Score (CATE)",
                yaxis_title="Seller Frequency",
                xaxis=dict(range=[clo - pad, chi + pad]),
                yaxis=dict(range=[0, int(counts.max() * 1.15)]),
            )
            fig_dist, cfg = apply_chart_theme(fig_dist, height=220, show_grid=False)
            st.plotly_chart(fig_dist, use_container_width=True, config=cfg)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card("Total Sellers", f"{len(df):,}", sub="Segment scale"), unsafe_allow_html=True)
    with m2:
        # Calculate real delta vs global avg
        delta_val = (avg_rate - global_avg) / global_avg if global_avg > 0 else 0
        st.markdown(metric_card("Proxy Return Rate", f"{avg_rate:.1%}", delta=f"{delta_val:+.1%}", sub="1-2★ review signal"), unsafe_allow_html=True)
    with m3:
        # Calculate real delta vs global cate max
        cmax_delta = (catemax - global_catemax) / abs(global_catemax) if global_catemax != 0 else 0
        st.markdown(metric_card("Max Operational Impact", f"{catemax:+.1%}", delta=f"{cmax_delta:+.1%}", sub="Top causal driver"), unsafe_allow_html=True)
    
    # Calculate returns preventable for filtered set
    t15 = int(len(df) * 0.15) if len(df) > 0 else 0
    tot_ret = df["proxy_return_rate"].sum() if not df.empty else 1
    # Using real sorted data here
    top_15_impact = df.sort_values("cate", ascending=False).head(t15)["proxy_return_rate"].sum()
    pct_prev = (top_15_impact / tot_ret * 100) if tot_ret > 0 else 0
    with m4:
        # Calculate efficiency: How much better than random? (Random would be 15%)
        efficiency = pct_prev / 15 if pct_prev > 0 else 0
        st.markdown(metric_card("Prevention Lift", f"{pct_prev:.1f}%", sub=f"{efficiency:.1f}x targeting efficiency"), unsafe_allow_html=True)

    if st.session_state.selected_seller != "All Sellers":
        st.markdown("---")
        st.markdown(f'<div style="color: #58a6ff; font-size: 16px; font-weight: 600; margin-bottom: 10px;">Quick Insight: {st.session_state.selected_seller}</div>', unsafe_allow_html=True)
        s_row = df_raw[df_raw["seller_id"] == st.session_state.selected_seller].iloc[0]
        i1, i2, i3 = st.columns(3)
        with i1:
            st.markdown(metric_card("Seller CATE", f"{s_row['cate']:+.2%}", sub="Direct causal influence"), unsafe_allow_html=True)
        with i2:
            st.write(f"Category: **{s_row['category']}**")
        with i3: 
            if st.button("Go to full profile"): 
                st.info("Switch to 'Sellers' view for full breakdown.")

elif selected_view == 'Key Findings':
    from views.key_findings import render as render_findings
    render_findings(df_raw[df_raw["category"] == st.session_state.selected_sector] if st.session_state.selected_sector != "All Sectors" else df_raw)

elif selected_view == 'Model Proof':
    from views.baseline_vs_causal import render as render_proof
    render_proof(df_raw[df_raw["category"] == st.session_state.selected_sector] if st.session_state.selected_sector != "All Sectors" else df_raw)

elif selected_view == 'Methodology':
    from views.methodology import render as render_method
    render_method(df_raw)

elif selected_view == 'Sellers':
    from views.seller_intel import render as render_sellers
    render_sellers(df_raw)

elif selected_view == 'Policy Simulator':
    from views.policy_sim import render as render_policy
    render_policy(df_raw[df_raw["category"] == st.session_state.selected_sector] if st.session_state.selected_sector != "All Sectors" else df_raw)

else:
    st.info("Navigate using the sidebar to explore the platform.")
