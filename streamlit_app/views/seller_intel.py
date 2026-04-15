import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ui_helpers import metric_card, section_header, apply_chart_theme, page_header, chart_card

def render(df: pd.DataFrame) -> None:
    st.markdown(page_header("Seller Profile", "Individual risk and causal breakdown"), unsafe_allow_html=True)

    if "cate" not in df.columns:
        st.error("No causal scores available.")
        return

    df_view = df.copy()
    q85 = df["cate"].quantile(0.85)
    q50 = df["cate"].quantile(0.50)
    
    # Pre-select based on global state
    global_seller = st.session_state.get('selected_seller', "All Sellers")
    global_sector = st.session_state.get('selected_sector', "All Sectors")

    with st.container():
        c1, c2 = st.columns([1, 1])
        with c1:
            cat_list = ["All Sectors"] + sorted(df["category"].unique().tolist())
            def_cat_idx = cat_list.index(global_sector) if global_sector in cat_list else 0
            category = st.selectbox("Market Category", cat_list, index=def_cat_idx)
        with c2:
            st.info("💡 Pro Tip: Select 'All Sectors' to see the market-wide risk chart below.")

        if category != "All Sectors":
            df_view = df_view[df_view["category"] == category]

        df_view = df_view.sort_values("cate", ascending=False)
        df_view["display_name"] = df_view.apply(lambda r: f"{r['seller_id']} | {r['category']} | " + ("High Risk" if r['cate']>q85 else "Medium" if r['cate']>q50 else "Low Risk"), axis=1)
        
        disp_list = df_view["display_name"].tolist()
        def_idx = 0
        if global_seller != "All Sellers":
            for i, d in enumerate(disp_list):
                if d.startswith(global_seller):
                    def_idx = i
                    break
        
        sel_disp = st.selectbox("Select Seller to Analyze", disp_list, index=def_idx if disp_list else None)
    
    if not disp_list:
        st.warning("No sellers match the current filters.")
        return

    row = df_view[df_view["display_name"] == sel_disp].iloc[0]
    c_val = row.get("cate", 0)
    st.session_state.selected_seller = row['seller_id']

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card("Return Impact (CATE)", f"{c_val:+.2%}", sub="Direct causal effect"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Proxy Return Rate", f"{row.get('proxy_return_rate',0):.1%}", sub="Observed metrics"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card("Quality Score", f"{row.get('seller_quality_score',0)*100:.0f}/100", sub="Composite index"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card("Review Volume", f"{int(row.get('total_reviews',0)):,}", sub="Category scale"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    l_col, r_col = st.columns([1.5, 1])
    with l_col:
        chart_card("Feature Significance (SHAP)", f"Drivers for {row['seller_id']}")
        sv = {k: v for k, v in row.items() if k.startswith("shap_") and pd.notna(v)}
        if sv:
            sv_sorted = dict(sorted(sv.items(), key=lambda x: abs(x[1]), reverse=True))
            shap_fig = px.bar(
                x=[k.replace("shap_","").title().replace("_"," ") for k in sv_sorted.keys()],
                y=list(sv_sorted.values()),
                color_discrete_sequence=["#58a6ff"]
            )
            shap_fig, shap_cfg = apply_chart_theme(shap_fig, height=300)
            st.plotly_chart(shap_fig, config=shap_cfg, use_container_width=True)

    with r_col:
        chart_card("Causal Logic Narrative", "AI-generated risk synthesis")
        narr_text = row.get("narrative", "")
        if pd.isna(narr_text) or len(str(narr_text).strip()) < 10:
             narr_text = f"This seller has a causal return impact of {c_val:+.2%}. Their operational profile suggests that their return rate is {'higher than expected' if c_val > 0 else 'lower than expected'} given their product category."
        st.markdown(f'<div class="glass-card" style="font-size: 14px; color: #f0f6fc; line-height: 1.6;">{narr_text}</div>', unsafe_allow_html=True)

    # --- MARKET COMPARISON ---
    st.markdown(section_header("Market Comparison", "Ranking vs peers"), unsafe_allow_html=True)
    
    if category == "All Sectors":
        # Show top 20 sellers per category as a grouped horizontal bar chart
        # (pivot heatmap was 66.7% NaN since each seller belongs to exactly one category)
        top_per_cat = (
            df.sort_values("cate", ascending=False)
              .groupby("category", group_keys=False)
              .head(20)
              .sort_values("cate", ascending=True)
        )
        top_per_cat["highlight"] = top_per_cat["seller_id"].apply(
            lambda s: "Selected" if s == row["seller_id"] else s.split("_")[0]
        )
        color_map = {"Selected": "#f85149"}
        st.markdown('<div class="chart-card"><div style="font-size: 13px; color: #8b949e; margin-bottom: 10px;">Top 20 Sellers by CATE — All Categories</div>', unsafe_allow_html=True)
        fig_heat = px.bar(
            top_per_cat,
            x="cate",
            y="seller_id",
            color="category",
            orientation="h",
            hover_data=["proxy_return_rate", "seller_quality_score"],
            color_discrete_sequence=["#58a6ff", "#39C5BB", "#3fb950"],
        )
        # Highlight the selected seller
        if row["seller_id"] in top_per_cat["seller_id"].values:
            sel_cate = top_per_cat.loc[top_per_cat["seller_id"] == row["seller_id"], "cate"].values[0]
            fig_heat.add_vline(x=sel_cate, line_width=2, line_dash="dot", line_color="#f85149",
                               annotation_text="Selected", annotation_position="top right",
                               annotation_font_color="#f85149")
        fig_heat, cfg_heat = apply_chart_theme(fig_heat, height=450)
        st.plotly_chart(fig_heat, config=cfg_heat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chart-card"><div style="font-size: 13px; color: #8b949e; margin-bottom: 10px;">Impact Ranking: {category} (Top 15 Sellers)</div>', unsafe_allow_html=True)
        top_15 = df[df["category"] == category].sort_values("cate", ascending=False).head(15)
        top_15["color"] = ["#39C5BB" if sid == row["seller_id"] else "#58a6ff" for sid in top_15["seller_id"]]
        fig_comp = px.bar(top_15, x="seller_id", y="cate", color="color", color_discrete_map="identity")
        fig_comp.update_layout(showlegend=False)
        fig_comp, cfg_comp = apply_chart_theme(fig_comp, height=300)
        st.plotly_chart(fig_comp, config=cfg_comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # REPLACING RANDOM DRIFT WITH ACTUAL STATISTICAL STABILITY
    st.markdown(section_header("Operational Stability Analysis", "Behavioral anchoring vs noise"), unsafe_allow_html=True)
    
    # Calculate a real stability index for this seller: 1 - relative error
    stability_idx = 1 - abs(row["cate_hi"] - row["cate_lo"]) / abs(row["cate"]) if row["cate"] != 0 else 0
    stability_idx = min(max(stability_idx, 0.1), 0.95)
    
    st.markdown(f'<div class="glass-card" style="text-align: center;">'
                f'<div style="color: #8b949e; font-size: 13px;">Confidence Stability Index</div>'
                f'<div style="color: #3fb950; font-size: 32px; font-weight: 700;">{stability_idx:.2f}</div>'
                f'<div style="color: #8b949e; font-size: 12px;">Based on 200-iteration bootstrap variance</div>'
                f'</div>', unsafe_allow_html=True)
