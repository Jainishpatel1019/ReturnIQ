import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add project root to path
curr_dir = os.path.dirname(__file__)
proj_root = os.path.dirname(curr_dir) if os.path.basename(curr_dir) == "views" else curr_dir
if proj_root not in sys.path:
    sys.path.append(proj_root)

from src.ui_helpers import metric_card, section_header, apply_chart_theme, page_header

def render(df: pd.DataFrame) -> None:
    st.markdown(page_header("Policy Simulator", "Adjust quality thresholds to optimize prevention"), unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available for the current filters.")
        return

    if "cate" not in df.columns:
        return

    # User Control
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        # Handle cases with no variance
        min_c = float(df["cate"].min())
        max_c = float(df["cate"].max())
        if min_c == max_c:
            max_c += 0.001
        
        threshold = st.slider("Causal Impact Threshold", min_c, max_c, float(df["cate"].quantile(0.85)), 0.001)
        st.markdown('</div>', unsafe_allow_html=True)
    
    flagged = df[df["cate"] >= threshold]
    pct_flagged = len(flagged) / len(df) * 100 if len(df) > 0 else 0
    
    tot_ret = df["proxy_return_rate"].sum() if not df.empty else 1
    prev_ret = flagged["proxy_return_rate"].sum() if not flagged.empty else 0
    pct_prev = (prev_ret / tot_ret * 100) if tot_ret > 0 else 0
    
    med_ret = df.get("proxy_return_rate", pd.Series([0])).median()
    fpr = (flagged.get("proxy_return_rate", pd.Series([0])) < med_ret).mean() * 100 if len(flagged) > 0 else 0

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card("Sellers Flagged", f"{len(flagged):,}", sub=f"Targeting bottom {pct_flagged:.0f}%", color="#f85149"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Returns Prevented", f"{pct_prev:.1f}%", sub="Potential reduction", color="#3fb950"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card("False Positives", f"{fpr:.1f}%", sub="Good sellers flagged", color="#d29922"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card("Target Efficiency", f"{pct_prev/pct_flagged if pct_flagged>0 else 0:.1f}x", sub="Vs random choice"), unsafe_allow_html=True)

    st.markdown(section_header("Prevention vs Precision Tradeoff"), unsafe_allow_html=True)
    
    # Sweep high→low so flagged count grows monotonically → smooth ROC-style curve
    ts = np.linspace(df["cate"].max(), df["cate"].min(), 80)
    pprev_curve, fp_curve = [], []
    for t in ts:
        flagged_t = df[df["cate"] >= t]
        if len(flagged_t) == 0:
            pprev_curve.append(0.0)
            fp_curve.append(0.0)
        else:
            pprev_curve.append(flagged_t["proxy_return_rate"].sum() / tot_ret * 100 if tot_ret > 0 else 0)
            fp_curve.append((flagged_t["proxy_return_rate"] < med_ret).mean() * 100)

    pairs  = sorted(zip(fp_curve, pprev_curve))
    fp_x   = [p[0] for p in pairs]
    prev_y = [p[1] for p in pairs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fp_x, y=prev_y, mode="lines", name="Causal model",
                             line=dict(color="#58a6ff", width=3)))
    fig.add_trace(go.Scatter(x=[fpr], y=[pct_prev], mode="markers", name="Current threshold",
                             marker=dict(size=14, color="#f85149", symbol="star")))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random baseline",
                             line=dict(color="#8b949e", width=1, dash="dash")))

    fig.update_layout(
        xaxis_title="Low-risk sellers caught (%)",
        yaxis_title="Returns Prevented (%)",
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 105]),
    )
    fig, cfg = apply_chart_theme(fig, height=380)
    st.plotly_chart(fig, config=cfg, use_container_width=True)
