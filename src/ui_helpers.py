import streamlit as st
import pandas as pd

def metric_card(label, value, sub="", color="#5b8fff", icon=""):
    """Returns self-contained HTML div for a styled metric card."""
    icon_html = f'<div style="float: right; font-size: 20px; color: {color};">{icon}</div>' if icon else ""
    return f"""
<div style="background-color: #161b22; border: 1px solid #21262d; border-left: 3px solid {color}; border-radius: 10px; padding: 1rem 1.25rem;">
{icon_html}
<div style="font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 500; text-transform: uppercase; color: #8b949e; letter-spacing: 0.05em;">{label}</div>
<div style="font-family: 'Inter', sans-serif; font-size: 28px; font-weight: 600; color: #f0f6fc; margin: 4px 0;">{value}</div>
<div style="font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 400; color: #6e7681;">{sub}</div>
</div>
""".strip()

def section_header(title, subtitle="", badge="", badge_color="#5b8fff"):
    badge_html = f'<span style="float: right; font-size: 10px; font-weight: 500; border-radius: 20px; padding: 2px 10px; background-color: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}44;">{badge}</span>' if badge else ""
    return f"""
<div style="border-bottom: 1px solid #21262d; padding-bottom: 10px; margin-bottom: 16px;">
{badge_html}
<div style="font-family: 'Inter', sans-serif; font-size: 15px; font-weight: 600; color: #e6edf3;">{title}</div>
<div style="font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 400; color: #8b949e; margin-top: 2px;">{subtitle}</div>
</div>
""".strip()

def apply_chart_theme(fig, height=380):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#c9d1d9", size=12),
        xaxis=dict(showgrid=False, linecolor="#30363d", zeroline=False, tickfont=dict(color="#8b949e")),
        yaxis=dict(showgrid=True, gridcolor="#21262d", gridwidth=0.5, linecolor="#30363d", zeroline=False, tickfont=dict(color="#8b949e")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e"), borderwidth=0),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height
    )
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title_font=dict(color="#e6edf3"))
    return fig, {"displayModeBar": False, "responsive": True}

def page_header(title, subtitle, status="Live", ok=True):
    if ok:
        pill_style = "background-color: #3fb95022; color: #3fb950; border: 1px solid #3fb95044;"
    else:
        pill_style = "background-color: #d2992222; color: #d29922; border: 1px solid #d2992244;"
    return f"""
<div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid #21262d; padding-bottom: 20px; margin-bottom: 24px;">
<div>
<div style="font-family: 'Inter', sans-serif; font-size: 22px; font-weight: 600; color: #f0f6fc; letter-spacing: -0.3px;">{title}</div>
<div style="font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 400; color: #8b949e; margin-top: 4px;">{subtitle}</div>
</div>
<div style="font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 500; border-radius: 20px; padding: 3px 10px; {pill_style}">{status}</div>
</div>
""".strip()

def risk_badge(cate, low=0.01, high=0.04):
    if pd.isna(cate):
        return ""
    color = "#f85149" if cate >= high else "#d29922" if cate >= low else "#3fb950"
    text = "High risk" if cate >= high else "Medium" if cate >= low else "Low risk"
    return f'<span style="font-family: \'Inter\', sans-serif; font-size: 10px; font-weight: 500; border-radius: 20px; padding: 2px 8px; border: 1px solid {color}44; background-color: {color}22; color: {color};">{text}</span>'

def glossary_expander():
    with st.expander("What do these terms mean?"):
        terms = [
            ("CATE", "How much THIS seller's behavior is moving the return rate, isolated from everything else."),
            ("Double ML", "A technique that separates cause from coincidence — controlling for price, category, region before measuring seller impact."),
            ("Proxy Return Rate", "Fraction of reviews with 1-2 stars — our best signal for actual returns."),
            ("AUUC", "If you ranked sellers by our model and acted on the worst ones first, how many returns would you catch vs acting randomly."),
            ("SHAP", "A method to explain which specific operational factors drove a seller's risk score."),
            ("Causal Inference", "Statistical analysis used to determine what actually causes returns, rather than just what correlates with them."),
            ("OLS Baseline", "A standard naive prediction model that gets confused by market noise. We use it to show why our causal model is better."),
            ("Seller Quality Score", "A composite score checking listing accuracy, fulfillment consistency, and sentiment.")
        ]
        html_rows = "".join([f'<div style="display: flex; padding: 8px 0; border-bottom: 1px solid #21262d;"><div style="flex: 1; color: #5b8fff; font-weight: 500; font-size: 13px;">{t}</div><div style="flex: 2; color: #8b949e; font-size: 13px;">{d}</div></div>' for t, d in terms])
        st.markdown(f'<div style="font-family: \'Inter\', sans-serif;">{html_rows}</div>', unsafe_allow_html=True)
