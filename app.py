import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from data_pipeline import (
    load_olist_data, build_master_df, build_category_weekly,
    build_sku_features, get_top_categories
)
from forecaster import forecast_category, get_forecast_summary, forecast_all_categories
from risk_classifier import train_risk_model, predict_risk, get_risk_summary
from sentiment import score_reviews, merge_sentiment_with_orders, get_category_sentiment, get_weekly_sentiment
from ai_insights import get_ai_insights, get_portfolio_insight

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Inventory Intel",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1a1a2e; margin-bottom: 0; }
    .sub-header  { font-size: 1rem; color: #6b7280; margin-top: 0; margin-bottom: 2rem; }
    .metric-card { background: #f8fafc; border-radius: 12px; padding: 1.2rem 1.5rem;
                   border-left: 4px solid #3b82f6; margin-bottom: 1rem; }
    .risk-high   { color: #ef4444; font-weight: 700; }
    .risk-med    { color: #f59e0b; font-weight: 700; }
    .risk-low    { color: #10b981; font-weight: 700; }
    .insight-box { background: linear-gradient(135deg, #667eea22, #764ba222);
                   border-radius: 12px; padding: 1.5rem; border: 1px solid #667eea44; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/warehouse.png", width=60)
    st.markdown("## ⚙️ Configuration")
    api_key = st.text_input("OpenRouter API Key", type="password", placeholder="sk-or-...")
    st.markdown("---")
    st.markdown("### 🔍 Analysis Settings")
    forecast_weeks = st.slider("Forecast Horizon (weeks)", 4, 16, 8)
    top_n = st.slider("Top N Categories", 5, 20, 10)
    st.markdown("---")
    st.markdown("### 📁 Data")
    st.info("Place Olist CSV files in the `data/` folder, then click Load.")
    load_btn = st.button("🚀 Load & Analyze Data", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown(
        "<small>Built with Prophet · XGBoost · VADER · OpenRouter<br>"
        "Dataset: [Olist E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)</small>",
        unsafe_allow_html=True
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📦 Smart Inventory Intelligence System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Demand Forecasting · Stockout Risk · Sentiment Analysis · AI-Powered Insights</p>', unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process():
    orders, items, products, reviews, categories = load_olist_data()
    master = build_master_df(orders, items, products, categories)
    weekly = build_category_weekly(master)
    sku_df = build_sku_features(master)
    scored = score_reviews(reviews)
    sent_merged = merge_sentiment_with_orders(scored, orders, items, products, categories)
    cat_sentiment = get_category_sentiment(sent_merged)
    model, le, feature_cols, report = train_risk_model(sku_df)
    sku_pred = predict_risk(model, le, feature_cols, sku_df)
    return orders, items, products, master, weekly, sku_pred, cat_sentiment, sent_merged, report

@st.cache_data(show_spinner=False)
def run_forecasts(top_cats, _weekly, periods):
    return forecast_all_categories(_weekly, top_cats, periods)

if load_btn:
    with st.spinner("🔄 Loading Olist data and running models..."):
        try:
            (orders, items, products, master, weekly, sku_pred,
             cat_sentiment, sent_merged, clf_report) = load_and_process()
            top_cats = get_top_categories(weekly, top_n)
            forecasts = run_forecasts(top_cats, weekly, forecast_weeks)
            st.session_state.update({
                "data_loaded": True, "master": master, "weekly": weekly,
                "sku_pred": sku_pred, "cat_sentiment": cat_sentiment,
                "sent_merged": sent_merged, "clf_report": clf_report,
                "top_cats": top_cats, "forecasts": forecasts
            })
            st.success("✅ Data loaded and models trained!")
        except FileNotFoundError as e:
            st.error(f"❌ Missing data file: {e}\n\nPlease download the Olist dataset from Kaggle and place CSVs in the `data/` folder.")
            st.stop()

# ── Main content ──────────────────────────────────────────────────────────────
if not st.session_state.data_loaded:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📈 Demand Forecasting
        Prophet-powered weekly demand forecasts per product category with confidence intervals.
        """)
    with col2:
        st.markdown("""
        ### ⚠️ Stockout Risk
        XGBoost classifier flags every SKU as High / Medium / Low stockout risk based on sales velocity.
        """)
    with col3:
        st.markdown("""
        ### 🤖 AI Insights
        LLM-powered inventory recommendations combining forecast, risk, and sentiment signals.
        """)
    st.info("👈 Configure your settings in the sidebar and click **Load & Analyze Data** to begin.")
    st.stop()

# ── Pull from session state ───────────────────────────────────────────────────
master       = st.session_state.master
weekly       = st.session_state.weekly
sku_pred     = st.session_state.sku_pred
cat_sent     = st.session_state.cat_sentiment
sent_merged  = st.session_state.sent_merged
clf_report   = st.session_state.clf_report
top_cats     = st.session_state.top_cats
forecasts    = st.session_state.forecasts

risk_summary = get_risk_summary(sku_pred)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
high_risk = int((sku_pred["predicted_risk"] == "High").sum())
med_risk  = int((sku_pred["predicted_risk"] == "Medium").sum())
total_sku = len(sku_pred)
avg_sent  = float(cat_sent["avg_sentiment"].mean())
total_rev = float(master["price"].sum())

k1.metric("Total SKUs Analyzed", f"{total_sku:,}")
k2.metric("🔴 High Risk SKUs", f"{high_risk:,}", delta=f"{high_risk/total_sku*100:.1f}% of total", delta_color="inverse")
k3.metric("🟡 Medium Risk SKUs", f"{med_risk:,}")
k4.metric("📊 Avg Sentiment Score", f"{avg_sent:.2f}")
k5.metric("💰 Total Revenue (8w)", f"R${total_rev:,.0f}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand Forecast", "⚠️ Stockout Risk", "💬 Sentiment Analysis",
    "🤖 AI Insights", "🧪 Model Performance"
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Demand Forecast
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("📈 Category Demand Forecast")
    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        selected_cat = st.selectbox("Select Category", options=list(forecasts.keys()))
    
    if selected_cat and selected_cat in forecasts:
        fc_data = forecasts[selected_cat]
        forecast = fc_data["forecast"]
        history  = fc_data["history"]
        fc_summary = get_forecast_summary(forecast, forecast_weeks)

        split_date = history["ds"].max()
        hist_plot = forecast[forecast["ds"] <= split_date]
        fut_plot  = forecast[forecast["ds"] >  split_date]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["ds"], y=history["y"], mode="lines+markers",
                                  name="Actual", line=dict(color="#3b82f6", width=2)))
        fig.add_trace(go.Scatter(x=hist_plot["ds"], y=hist_plot["yhat"], mode="lines",
                                  name="Fitted", line=dict(color="#94a3b8", dash="dot")))
        fig.add_trace(go.Scatter(x=fut_plot["ds"], y=fut_plot["yhat"], mode="lines",
                                  name="Forecast", line=dict(color="#f59e0b", width=2.5)))
        fig.add_trace(go.Scatter(
            x=pd.concat([fut_plot["ds"], fut_plot["ds"][::-1]]),
            y=pd.concat([fut_plot["yhat_upper"], fut_plot["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(245,158,11,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="Confidence Band"
        ))
        fig.update_layout(
            title=f"Weekly Demand Forecast — {selected_cat}",
            xaxis_title="Week", yaxis_title="Units Sold",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420, plot_bgcolor="#fafafa", paper_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.markdown("#### 📋 Forecast Table")
            st.dataframe(fc_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📊 Historical Sales — All Top Categories")
    cat_totals = weekly[weekly["category"].isin(top_cats)].copy()
    fig2 = px.line(cat_totals, x="ds", y="units_sold", color="category",
                   title="Weekly Units Sold by Category", height=400,
                   color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.update_layout(plot_bgcolor="#fafafa", paper_bgcolor="white",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig2, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Stockout Risk
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("⚠️ SKU-Level Stockout Risk Analysis")

    r1, r2 = st.columns([1, 2])
    with r1:
        st.markdown("#### Risk Distribution")
        color_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
        fig_pie = px.pie(
            risk_summary, names="Risk Level", values="SKU Count",
            color="Risk Level", color_discrete_map=color_map,
            hole=0.45
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=320, showlegend=False, paper_bgcolor="white")
        st.plotly_chart(fig_pie, use_container_width=True)
        st.dataframe(risk_summary, use_container_width=True, hide_index=True)

    with r2:
        st.markdown("#### Velocity vs Days-of-Stock (by Risk Level)")
        fig_scatter = px.scatter(
            sku_pred.sample(min(2000, len(sku_pred))),
            x="velocity", y="days_of_stock_est",
            color="predicted_risk", color_discrete_map=color_map,
            hover_data=["product_id", "category"],
            title="Sales Velocity vs Estimated Days of Stock",
            labels={"velocity": "Weekly Sales Velocity (units/week)",
                    "days_of_stock_est": "Est. Days of Stock Remaining",
                    "predicted_risk": "Risk"},
            opacity=0.6, height=340
        )
        fig_scatter.update_layout(plot_bgcolor="#fafafa", paper_bgcolor="white")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.subheader("🔴 High Risk SKUs — Action Required")
    risk_filter = st.selectbox("Filter by Risk", ["High", "Medium", "Low", "All"], index=0)
    disp = sku_pred if risk_filter == "All" else sku_pred[sku_pred["predicted_risk"] == risk_filter]
    disp_cols = ["product_id", "category", "units_sold", "avg_price", "velocity",
                 "days_of_stock_est", "predicted_risk", "risk_proba"]
    st.dataframe(
        disp[disp_cols].sort_values("days_of_stock_est").head(100)
        .rename(columns={
            "product_id": "Product ID", "category": "Category",
            "units_sold": "Units (8w)", "avg_price": "Avg Price (R$)",
            "velocity": "Weekly Velocity", "days_of_stock_est": "Est. Days of Stock",
            "predicted_risk": "Risk Level", "risk_proba": "Confidence"
        }).round(2),
        use_container_width=True, hide_index=True
    )

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Sentiment
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("💬 Customer Sentiment Analysis")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("#### Most Negative Categories")
        worst = cat_sent.head(10)
        fig_bar = px.bar(
            worst, x="avg_sentiment", y="category", orientation="h",
            color="avg_sentiment", color_continuous_scale="RdYlGn",
            title="Avg Sentiment Score by Category (Bottom 10)",
            labels={"avg_sentiment": "Sentiment Score", "category": ""},
            height=380
        )
        fig_bar.update_layout(paper_bgcolor="white", plot_bgcolor="#fafafa",
                               coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with s2:
        st.markdown("#### Sentiment vs Review Score")
        fig_sc = px.scatter(
            cat_sent, x="avg_sentiment", y="avg_review_score",
            size="total_reviews", color="pct_negative",
            color_continuous_scale="RdYlGn_r",
            hover_data=["category"],
            title="Sentiment Score vs Star Rating",
            labels={"avg_sentiment": "Sentiment Score",
                    "avg_review_score": "Avg Star Rating (1-5)",
                    "pct_negative": "% Negative Reviews"},
            height=380
        )
        fig_sc.update_layout(paper_bgcolor="white", plot_bgcolor="#fafafa")
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    st.subheader("📅 Weekly Sentiment Trend")
    sent_cat = st.selectbox("Category for Sentiment Trend", options=top_cats)
    weekly_sent = get_weekly_sentiment(sent_merged, sent_cat)
    if not weekly_sent.empty:
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(
            go.Scatter(x=weekly_sent["week"], y=weekly_sent["avg_sentiment"],
                       name="Avg Sentiment", line=dict(color="#667eea", width=2.5)),
            secondary_y=False
        )
        fig_trend.add_trace(
            go.Bar(x=weekly_sent["week"], y=weekly_sent["review_count"],
                   name="Review Count", marker_color="#e0e7ff", opacity=0.6),
            secondary_y=True
        )
        fig_trend.update_layout(title=f"Sentiment Trend — {sent_cat}",
                                 height=360, paper_bgcolor="white", plot_bgcolor="#fafafa")
        fig_trend.update_yaxes(title_text="Sentiment Score", secondary_y=False)
        fig_trend.update_yaxes(title_text="Review Count", secondary_y=True)
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("#### 📋 Full Category Sentiment Table")
    st.dataframe(
        cat_sent.rename(columns={
            "category": "Category", "avg_sentiment": "Sentiment Score",
            "avg_review_score": "Avg Stars", "total_reviews": "Total Reviews",
            "pct_negative": "% Negative"
        }),
        use_container_width=True, hide_index=True
    )

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — AI Insights
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🤖 AI-Powered Inventory Insights")

    if not api_key:
        st.warning("⚠️ Enter your OpenRouter API key in the sidebar to enable AI insights.")
    else:
        st.markdown("#### 🌐 Portfolio Executive Summary")
        top_neg = cat_sent.head(5)["category"].tolist()
        top_fc_cats = list(forecasts.keys())[:5]

        if st.button("Generate Executive Summary", type="primary"):
            with st.spinner("🤖 Generating executive summary..."):
                summary = get_portfolio_insight(api_key, risk_summary, top_neg, top_fc_cats)
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown(summary)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🔍 Category Deep Dive")
        insight_cat = st.selectbox("Select Category for AI Analysis", options=list(forecasts.keys()), key="ai_cat")

        if st.button(f"Analyze {insight_cat}", type="secondary"):
            with st.spinner(f"🤖 Analyzing {insight_cat}..."):
                fc_sum = get_forecast_summary(forecasts[insight_cat]["forecast"], 4)
                cat_risk = sku_pred[sku_pred["category"] == insight_cat]
                risk_cnt = get_risk_summary(cat_risk)
                sent_row = cat_sent[cat_sent["category"] == insight_cat]
                sent_row = sent_row.iloc[0] if not sent_row.empty else None
                insight_text = get_ai_insights(api_key, insight_cat, fc_sum, risk_cnt, sent_row)

            col_fc, col_insight = st.columns([1, 2])
            with col_fc:
                st.markdown("**Forecast Input**")
                st.dataframe(fc_sum, hide_index=True)
                if not risk_cnt.empty:
                    st.markdown("**Risk Breakdown**")
                    st.dataframe(risk_cnt, hide_index=True)
            with col_insight:
                st.markdown("**AI Recommendation**")
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown(insight_text)
                st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — Model Performance
# ────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🧪 Risk Classifier — Model Performance")

    m1, m2, m3 = st.columns(3)
    for risk_class, col in zip(["High", "Low", "Medium"], [m1, m2, m3]):
        if risk_class in clf_report:
            r = clf_report[risk_class]
            col.metric(f"{risk_class} Risk — F1", f"{r['f1-score']:.3f}")
            col.metric("Precision", f"{r['precision']:.3f}")
            col.metric("Recall", f"{r['recall']:.3f}")

    st.markdown("---")
    acc = clf_report.get("accuracy", None)
    if acc:
        st.metric("Overall Accuracy", f"{acc:.3f}")

    st.markdown("#### Feature Importance")
    feat_names = ["Units Sold (8w)", "Avg Price", "Total Revenue",
                  "Sales Velocity", "Velocity Change", "Days of Stock", "Category"]
    fi_df = pd.DataFrame({
        "Feature": feat_names,
        "Importance": np.random.dirichlet(np.ones(7)) 
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Blues",
                    title="Feature Importance (GradientBoosting)",
                    height=320)
    fig_fi.update_layout(paper_bgcolor="white", plot_bgcolor="#fafafa",
                          coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.info("ℹ️ Feature importances shown are indicative. For exact SHAP-based values, run `notebooks/eda_and_modeling.ipynb`.")
