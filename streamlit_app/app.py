import streamlit as st
import pandas as pd
import pathlib
import sys
import os
import streamlit_antd_components as sac

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.ui_helpers import page_header, glossary_expander, metric_card, risk_badge

st.set_page_config(page_title="ReturnIQ", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">', unsafe_allow_html=True)

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

with st.spinner("Initializing ReturnIQ Causal Engine..."):
    df = load_data()

if df.empty:
    st.error("📉 **Data Load Failure**: Dashboard data not found.")
    st.info("Please ensure `data/processed/final_dashboard_data.parquet` is present.")
    st.stop()

st.sidebar.markdown('<div style="font-family: \'Inter\', sans-serif; font-size: 16px; font-weight: 600; color: #f0f6fc; letter-spacing: -0.2px; margin-bottom: 12px;">ReturnIQ</div>', unsafe_allow_html=True)
st.sidebar.caption(f"{len(df):,} sellers indexed")

selected_view = sac.menu([
    sac.MenuItem('Overview', icon='house-fill'),
    sac.MenuItem('Key findings', icon='lightbulb-fill'),
    sac.MenuItem('Seller profile', icon='person-badge'),
    sac.MenuItem('Policy sim', icon='sliders'),
    sac.MenuItem('Model proof', icon='bar-chart-line-fill'),
    sac.MenuItem('How it works', icon='journal-code'),
], size='sm', color='#5b8fff', open_all=True)

st.sidebar.markdown("---")
st.sidebar.caption("Method: Causal Inference (Double ML)")
st.sidebar.caption("Data: 3.7M Amazon reviews · 2023")
st.sidebar.caption("Model confidence: high")

if selected_view == 'Overview':
    st.markdown(page_header("Seller Return Rate Intelligence", "Causal analysis of what drives returns — not just who has them", status="Live · 1,000 sellers", ok=True), unsafe_allow_html=True)
    if not df.empty and "cate" in df.columns:
        col1, col2, col3, col4 = st.columns(4)
        avg_rate = df["proxy_return_rate"].mean() if "proxy_return_rate" in df.columns else 0
        catemax = df["cate"].max()
        t15 = int(len(df) * 0.15)
        top15pct = (df.sort_values("cate", ascending=False).head(t15)["proxy_return_rate"].sum() / df["proxy_return_rate"].sum() * 100) if "proxy_return_rate" in df.columns and df["proxy_return_rate"].sum() > 0 else 0
        
        col1.markdown(metric_card("Total sellers", f"{len(df):,}"), unsafe_allow_html=True)
        col2.markdown(metric_card("Avg return rate", f"{avg_rate:.1%}"), unsafe_allow_html=True)
        col3.markdown(metric_card("Highest CATE", f"{catemax:+.3%}"), unsafe_allow_html=True)
        col4.markdown(metric_card("Top 15% severity", f"{top15pct:.0f}%", sub="Returns driven by top 15%"), unsafe_allow_html=True)
        
        st.markdown("<p style='font-family: \"Inter\", sans-serif; font-size: 14px; color: #c9d1d9; line-height: 1.6;'>This tool answers one question: when return rates are high, is it the seller's fault or the customer's? We use a technique called <strong>causal inference</strong> — which controls for product category, price, and buyer region before measuring seller impact. The result is a score per seller that shows their true causal effect on returns, not just their correlation with them.</p>", unsafe_allow_html=True)
        glossary_expander()
        st.divider()
        
        cl, cr = st.columns(2)
        with cl:
            st.markdown("<div style='font-family: \"Inter\", sans-serif; font-size: 15px; font-weight: 600; color: #e6edf3; margin-bottom: 10px;'>Top 5 Highest-Risk Sellers</div>", unsafe_allow_html=True)
            top5 = df.sort_values("cate", ascending=False).head(5)
            table_html = "<div style='background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px;'><table style='width: 100%; border-collapse: collapse;'><tr style='border-bottom: 1px solid #30363d; color: #8b949e; font-size: 11px; text-transform: uppercase;'><th>Seller</th><th>Category</th><th>CATE</th><th>Risk</th></tr>"
            for i, (_, ro) in enumerate(top5.iterrows()):
                table_html += f"<tr style='border-bottom: 1px solid #21262d; font-size: 13px; color: #c9d1d9;'><td style='padding: 8px 0;'>Seller #{i+1}</td><td>{ro.get('category','')}</td><td style='color: #f0f6fc; font-weight: 500;'>{ro.get('cate',0):+.3%}</td><td>{risk_badge(ro.get('cate',0), df['cate'].quantile(0.5), df['cate'].quantile(0.85))}</td></tr>"
            st.markdown(table_html + "</table></div>", unsafe_allow_html=True)
        with cr:
            st.markdown("<div style='font-family: \"Inter\", sans-serif; font-size: 15px; font-weight: 600; color: #e6edf3; margin-bottom: 10px;'>Return Rate by Category</div>", unsafe_allow_html=True)
            if "category" in df.columns and "proxy_return_rate" in df.columns:
                cdf = df.groupby("category")["proxy_return_rate"].mean().reset_index()
                import plotly.express as px
                from src.ui_helpers import apply_chart_theme
                fg = px.bar(cdf, x="category", y="proxy_return_rate", color_discrete_sequence=["#58a6ff"])
                fg.update_layout(xaxis_title="", yaxis_title="Return Rate")
                fg, cfg = apply_chart_theme(fg, 250)
                st.plotly_chart(fg, config=cfg, use_container_width=True)

elif selected_view == 'Key findings':
    from views.key_findings import render
    render(df)
elif selected_view == 'Seller profile':
    from views.seller_intel import render
    render(df)
elif selected_view == 'Policy sim':
    from views.policy_sim import render
    render(df)
elif selected_view == 'Model proof':
    from views.baseline_vs_causal import render
    render(df)
elif selected_view == 'How it works':
    from views.methodology import render
    render(df)
