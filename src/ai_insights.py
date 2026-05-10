import requests
import json


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-oss-120b:free"


def build_inventory_prompt(category, forecast_summary, risk_counts, sentiment_row):
    risk_text = ", ".join([f"{row['Risk Level']}: {row['SKU Count']} SKUs" for _, row in risk_counts.iterrows()])
    sentiment_text = (
        f"Avg sentiment score: {sentiment_row['avg_sentiment']:.2f}, "
        f"Avg review rating: {sentiment_row['avg_review_score']:.1f}/5, "
        f"Negative review %: {sentiment_row['pct_negative']:.1f}%"
        if sentiment_row is not None else "No sentiment data available."
    )

    forecast_text = forecast_summary.head(4).to_string(index=False)

    prompt = f"""You are an expert inventory and supply chain analyst. Analyze the following data for the product category: **{category}**

## Demand Forecast (next 4 weeks):
{forecast_text}

## Stockout Risk Distribution:
{risk_text}

## Customer Sentiment:
{sentiment_text}

Provide a concise, actionable inventory management report with:
1. **Demand Outlook** — Is demand growing, stable, or declining? How confident are you?
2. **Stockout Risk Assessment** — Which risk level dominates and what does it mean operationally?
3. **Sentiment Signal** — What does customer sentiment tell us about this category's health?
4. **Top 3 Actionable Recommendations** — Specific, prioritized actions the inventory team should take this week.

Be direct, business-focused, and avoid generic advice. Max 250 words."""

    return prompt


def get_ai_insights(api_key, category, forecast_summary, risk_counts, sentiment_row):
    prompt = build_inventory_prompt(category, forecast_summary, risk_counts, sentiment_row)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/smart-inventory-intel",
        "X-Title": "Smart Inventory Intel"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.4
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"⚠️ Could not fetch AI insights: {str(e)}"
    except (KeyError, IndexError):
        return "⚠️ Unexpected response format from OpenRouter API."


def get_portfolio_insight(api_key, risk_summary_df, top_negative_categories, top_forecast_categories):
    risk_text = risk_summary_df.to_string(index=False)
    neg_cats = ", ".join(top_negative_categories[:5])
    top_cats = ", ".join(top_forecast_categories[:5])

    prompt = f"""You are a senior supply chain strategist. Here is a portfolio-level view of an e-commerce inventory:

## Risk Distribution Across All SKUs:
{risk_text}

## Categories with Highest Negative Sentiment:
{neg_cats}

## Highest Demand Growth Categories (next 4 weeks):
{top_cats}

Provide a 3-paragraph executive summary:
1. Overall inventory health and biggest operational risk right now
2. Which categories need immediate attention and why
3. Strategic recommendation for the next 30 days

Be specific, data-driven, and executive-level. Max 200 words."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/smart-inventory-intel",
        "X-Title": "Smart Inventory Intel"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.3
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Could not fetch portfolio insight: {str(e)}"
