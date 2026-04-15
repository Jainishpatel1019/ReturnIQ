import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def metric_card(label, value, sub="", color="#5b8fff", delta=""):
    """Returns a glassmorphism metric card."""
    delta_html = f'<div class="metric-delta-pos">{delta}</div>' if delta else ""
    return f"""
<div class="glass-card">
    <div class="metric-label">{label}</div>
    <div style="display: flex; align-items: baseline; gap: 10px;">
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    <div style="font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 400; color: #6e7681; margin-top: 4px;">{sub}</div>
</div>
""".strip()

def apply_chart_theme(fig, height=350, show_grid=True):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#8b949e", size=12),
        xaxis=dict(
            showgrid=False, 
            linecolor="rgba(48, 54, 61, 0.5)", 
            zeroline=False, 
            tickfont=dict(color="#8b949e"),
            gridcolor="rgba(48, 54, 61, 0.2)"
        ),
        yaxis=dict(
            showgrid=show_grid, 
            gridcolor="rgba(48, 54, 61, 0.2)", 
            gridwidth=0.5, 
            linecolor="rgba(48, 54, 61, 0.5)", 
            zeroline=False, 
            tickfont=dict(color="#8b949e")
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", 
            font=dict(color="#8b949e", size=11), 
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#161b22", font_size=12, font_family="Inter")
    )
    return fig, {"displayModeBar": False, "responsive": True}

def page_header(title, subtitle):
    return f"""
<div style="margin-bottom: 32px;">
    <div style="font-family: 'Inter', sans-serif; font-size: 32px; font-weight: 700; color: #f0f6fc; letter-spacing: -0.5px;">{title}</div>
    <div style="font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 400; color: #8b949e; margin-top: 4px;">{subtitle}</div>
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

def create_donut_chart(df):
    """Creates a donut chart matching 'Data Visuatics' in the mockup."""
    if df.empty or "category" not in df.columns:
        return None
    
    counts = df["category"].value_counts().head(5)
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, 
        values=counts.values, 
        hole=.7,
        marker=dict(colors=["#58a6ff", "#39C5BB", "#3fb950", "#79c0ff", "#1f6feb"]),
        textinfo='none'
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=220,
        font=dict(family="Inter, sans-serif", color="#8b949e", size=11),
    )
    return fig
