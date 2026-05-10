#  Smart Inventory Intelligence System

> **Demand Forecasting · Stockout Risk Classification · Sentiment NLP · AI-Powered Recommendations**

A production-grade inventory analytics platform built on the Brazilian Olist E-Commerce dataset. Combines time-series forecasting, machine learning classification, NLP sentiment analysis, and LLM-powered business insights into a single interactive dashboard.

---

## Live Demo

```bash
streamlit run app.py
```

---

##  What It Does

| Module | Tech | Business Value |
|--------|------|----------------|
| **Demand Forecasting** | Prophet | Predicts weekly units sold per category with confidence intervals |
| **Stockout Risk Classification** | GradientBoosting (sklearn) | Flags every SKU as High / Medium / Low risk based on sales velocity |
| **Sentiment Analysis** | VADER NLP | Correlates customer review sentiment with demand changes |
| **AI Insights Engine** | OpenRouter LLM API | Generates plain-English inventory recommendations per category |
| **Executive Dashboard** | Streamlit + Plotly | Interactive portfolio-level and SKU-level analytics |

---

## 📁 Project Structure

```
smart-inventory-intel/
├── data/                          # Olist CSV files (download from Kaggle)
├── notebooks/
│   └── eda_and_modeling.ipynb     # Full EDA and model walkthrough
├── src/
│   ├── data_pipeline.py           # Data loading, merging, feature engineering
│   ├── forecaster.py              # Prophet time-series forecasting
│   ├── risk_classifier.py         # GradientBoosting stockout risk model
│   ├── sentiment.py               # VADER sentiment scoring & aggregation
│   └── ai_insights.py             # OpenRouter LLM API integration
├── app.py                         # Streamlit dashboard (main entry point)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/smart-inventory-intel.git
cd smart-inventory-intel
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle and place all CSV files in the `data/` folder.

Required files:
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_products_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `product_category_name_translation.csv`

### 4. Get an OpenRouter API key
Sign up at [openrouter.ai](https://openrouter.ai) for free API access. The app uses `meta-llama/llama-3.1-8b-instruct` by default .

### 5. Run the app
```bash
streamlit run app.py
```

Enter your OpenRouter API key in the sidebar, then click **Load & Analyze Data**.

---

##  Dashboard Tabs

### Demand Forecast
- Prophet-based weekly demand predictions per product category
- Interactive forecast chart with confidence bands
- Configurable forecast horizon (4–16 weeks)

###  Stockout Risk
- GradientBoosting classifier trained on 8-week rolling sales features
- Risk tier breakdown: High / Medium / Low
- SKU-level drill-down table with filtering

###  Sentiment Analysis
- VADER compound sentiment scoring on all customer reviews
- Category-level sentiment aggregation and ranking
- Weekly sentiment trend with review volume overlay

###  AI Insights
- **Portfolio Executive Summary**: LLM-generated strategic overview across all categories
- **Category Deep Dive**: Per-category recommendations synthesizing forecast + risk + sentiment

###  Model Performance
- F1, Precision, Recall per risk class
- Feature importance visualization

---

##  Key Business Insights Unlocked

1. **Velocity-based early warning**: SKUs with >7 units/week velocity but <7 days of stock are flagged as critical reorder targets before they stock out
2. **Sentiment-demand correlation**: Categories where sentiment scores drop ahead of demand spikes may signal return surges or quality issues requiring supplier intervention
3. **Actionable risk tiers**: Not all stockout risk is equal — the classifier differentiates price-sensitive high-velocity SKUs from slow-movers

---

##  Tech Stack

- **Python 3.10+**
- **Streamlit** — Interactive dashboard
- **Prophet** — Time-series demand forecasting
- **scikit-learn** — GradientBoosting risk classifier
- **VADER** — Lexicon-based sentiment analysis
- **Plotly** — Interactive charts
- **OpenRouter API** — LLM-powered insights (Llama 3.1)
- **Pandas / NumPy** — Data pipeline

---

##  Model Details

### Demand Forecaster (Prophet)
- Multiplicative seasonality model with yearly seasonality enabled
- Changepoint prior scale: 0.1 (moderate flexibility)
- Trained per category on full historical weekly data
- Outputs point forecast + 80% confidence interval

### Stockout Risk Classifier
- **Features**: 8-week units sold, avg price, revenue, sales velocity, velocity change (vs prior 8w), estimated days of stock, category encoding
- **Labels**: Rule-based from estimated days of stock (High < 7d, Medium 7–21d, Low > 21d)
- **Model**: GradientBoostingClassifier (150 estimators, depth 4)
- **Typical accuracy**: ~85–90% on held-out test set

---

##  Contributing

PRs welcome. Ideas for extension:
- Add SHAP explainability for individual SKU risk predictions
- Integrate actual stock level data for more precise days-of-stock calculation
- Add email/Slack alerting for newly surfaced High-risk SKUs
- Expand to SKU-level Prophet forecasting (currently category-level)

---

## License

MIT License. Dataset courtesy of [Olist](https://olist.com/) via Kaggle.

---

*Built as a portfolio project demonstrating end-to-end ML, data analysis, NLP, and AI integration for real-world supply chain problems.*
