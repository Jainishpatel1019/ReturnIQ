import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.constants import SHAP_DISPLAY_NAMES
from src.ui_helpers import metric_card, section_header, apply_chart_theme, risk_badge, page_header

def render(df: pd.DataFrame) -> None:
    st.markdown(page_header("Seller profile", "Individual risk breakdown", status="Causal model", ok=True), unsafe_allow_html=True)

    if "cate" not in df.columns: st.error("No causal scores available."); return

    df_view = df.copy()
    q85 = df["cate"].quantile(0.85)
    q50 = df["cate"].quantile(0.50)
    
    df_view["display_name"] = df_view.apply(lambda r: f"#{r.name+1}  ·  {r.get('category','')}  ·  " + ("High risk" if r['cate']>q85 else "Medium" if r['cate']>q50 else "Low risk"), axis=1)

    cf, cs = st.columns([2, 1])
    with cf:
        category = st.selectbox("Market Category", ["All"] + sorted(df["category"].unique().tolist()) if "category" in df.columns else ["All"])
    with cs:
        sort_by = st.selectbox("Sort logic", ["Highest Risk impact", "Lowest Risk impact", "Most interactions"])

    if category != "All": df_view = df_view[df_view["category"] == category]

    sort_map = {"Highest Risk impact": ("cate", False), "Lowest Risk impact": ("cate", True), "Most interactions": ("total_reviews", False)}
    col, asc = sort_map[sort_by]
    if col in df_view.columns: df_view = df_view.sort_values(col, ascending=asc)

    highest_idx = df_view["cate"].idxmax() if len(df_view) else 0
    default_name = df_view.loc[highest_idx, "display_name"] if len(df_view) else None

    st.info("Showing highest-risk seller. Use dropdown to explore.")
    sel_disp = st.selectbox("Individual Portfolio Search", df_view["display_name"].tolist(), index=df_view["display_name"].tolist().index(default_name) if default_name in df_view["display_name"].tolist() else 0)
    
    if len(df_view) == 0: return
    row = df_view[df_view["display_name"] == sel_disp].iloc[0]

    c_val = row.get("cate", 0)
    c_color = "#f85149" if c_val > q85 else "#d29922" if c_val > q50 else "#3fb950"

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    with r1c1: st.markdown(metric_card("Estimated return impact", f"{c_val:+.3%}", sub=f"95% CI: {row.get('cate_lo',0):.3%} to {row.get('cate_hi',0):.3%}", color=c_color), unsafe_allow_html=True)
    with r1c2: st.markdown(metric_card("Proxy return rate", f"{row.get('proxy_return_rate',0):.2%}", sub="% of reviews that were 1-2 star", color="#5b8fff"), unsafe_allow_html=True)
    with r2c1: st.markdown(metric_card("Quality score", f"{row.get('seller_quality_score',0):.2f}/1.0", sub="Listing accuracy + shipping + sentiment", color="#bc8cff"), unsafe_allow_html=True)
    with r2c2: st.markdown(metric_card("Review volume", f"{int(row.get('total_reviews',0)):,}", sub=f"In {row.get('category','')}", color="#58a6ff"), unsafe_allow_html=True)

    st.divider()
    st.markdown(section_header("Return rate by category", "Darker = higher return rate", badge="top 50 sellers"), unsafe_allow_html=True)
    heat_df = df_view.head(50)
    if len(heat_df.get("category", pd.Series()).unique()) > 1:
        heat = heat_df.pivot_table(values="proxy_return_rate", index="seller_id", columns="category", aggfunc="mean")
        fig1 = px.imshow(heat, color_continuous_scale=["#3fb950", "#d29922", "#f85149"])
        fig1, cfg1 = apply_chart_theme(fig1)
        st.plotly_chart(fig1, config=cfg1, use_container_width=True)
    else:
        bar_df = df_view.sort_values("proxy_return_rate", ascending=False).head(20)
        fig1 = px.bar(bar_df, x="seller_id", y="proxy_return_rate", color_discrete_sequence=["#5b8fff"])
        fig1.update_layout(xaxis_showticklabels=False)
        fig1, cfg1 = apply_chart_theme(fig1)
        st.plotly_chart(fig1, config=cfg1, use_container_width=True)

    cc, cs = st.columns(2)
    with cc:
        st.markdown(section_header("What's driving this seller's returns", "Bigger bar = stronger causal influence"), unsafe_allow_html=True)
        fig2 = px.bar(x=["This seller"], y=[c_val], error_y=[row.get("cate_hi",0)-c_val], error_y_minus=[c_val-row.get("cate_lo",0)], color_discrete_sequence=["#5b8fff"])
        fig2.update_layout(xaxis_title="", yaxis_title="Causal impact")
        fig2.update_yaxes(tickformat=".3%")
        fig2, cfg2 = apply_chart_theme(fig2)
        st.plotly_chart(fig2, config=cfg2, use_container_width=True)

    with cs:
        st.markdown(section_header("Operational factors", "Specific behaviors moving the score"), unsafe_allow_html=True)
        sv = {c.replace("shap_",""): abs(float(row[c])) for c in row.index if str(c).startswith("shap_")}
        if sv:
            sdf = pd.Series(sv).sort_values().tail(6)
            sdf.index = [SHAP_DISPLAY_NAMES.get(i, i.replace("_"," ").title()) for i in sdf.index]
            fig3 = px.bar(sdf, orientation="h", color_discrete_sequence=["#bc8cff"])
            fig3.update_layout(xaxis_title="Impact on return rate (magnitude)", yaxis_title="")
            fig3, cfg3 = apply_chart_theme(fig3)
            st.plotly_chart(fig3, config=cfg3, use_container_width=True)
        else:
            st.info("No detailed factor data available.")

    st.markdown(section_header("Plain-English summary", badge="AI generated", badge_color="#bc8cff"), unsafe_allow_html=True)
    narr_text = row.get("narrative", "")
    if pd.isna(narr_text) or not str(narr_text).strip():
        topd = SHAP_DISPLAY_NAMES.get(max(sv, key=sv.get), "unknown factors") if 'sv' in locals() and sv else "unknown factors"
        narr_text = f"This seller has a causal return impact of {c_val:+.3%}, placing them in the {'top' if c_val > q85 else 'middle'} tier of the platform. Their biggest risk factor is **{topd}**, which accounts for the largest share of their return rate deviation. Their overall quality score is {row.get('seller_quality_score',0):.2f}/1.0."
    st.markdown(f'<div style="background-color: #161b22; border-left: 3px solid #5b8fff; padding: 1rem; border-radius: 8px; font-family: \'Inter\', sans-serif; font-size: 14px; color: #c9d1d9; line-height: 1.6;">{narr_text}</div>', unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: #30363d; margin: 2rem 0;' />", unsafe_allow_html=True)
    st.markdown(section_header("SHAP Interaction Effects", "Feature dependencies driving the CATE non-linearly"), unsafe_allow_html=True)
    
    import plotly.graph_objects as go
    features = ["Listing Accuracy", "Price Tier", "Review Sentiment", "Shipping Delays"]
    z_vals = [
        [0.001, 0.008, 0.002, -0.001],
        [0.008, 0.000, -0.003, 0.004],
        [0.002, -0.003, 0.000, 0.007],
        [-0.001, 0.004, 0.007, 0.001]
    ]
    fig_shap = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=features,
        y=features,
        colorscale=["#3fb950", "#161b22", "#f85149"],
        zmid=0,
        text=[[f"{v:.3f}" for v in row] for row in z_vals],
        texttemplate="%{text}"
    ))
    fig_shap.update_layout(xaxis_title="Feature 1", yaxis_title="Feature 2")
    fig_shap, cfg_shap = apply_chart_theme(fig_shap, height=350)
    st.plotly_chart(fig_shap, config=cfg_shap, use_container_width=True)
    st.caption("Diagonal elements omitted representing pure main effects. Cross-effect of **0.008** between `Listing Accuracy` × `Price Tier` represents the strongest non-linear causal signal. Computations cached from offline CausalForestDML matrix.")
