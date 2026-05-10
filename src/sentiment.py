import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")

analyzer = SentimentIntensityAnalyzer()


def score_reviews(reviews_df):
    df = reviews_df.copy()
    df["review_comment_message"] = df["review_comment_message"].fillna("")
    df["sentiment_score"] = df["review_comment_message"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"]
    )
    df["sentiment_label"] = pd.cut(
        df["sentiment_score"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["Negative", "Neutral", "Positive"]
    )
    return df


def merge_sentiment_with_orders(scored_reviews, orders, items, products, categories):
    merged = scored_reviews[["order_id", "sentiment_score", "sentiment_label", "review_creation_date", "review_score"]].merge(
        items[["order_id", "product_id"]], on="order_id", how="left"
    ).merge(
        products[["product_id", "product_category_name"]], on="product_id", how="left"
    ).merge(
        categories, on="product_category_name", how="left"
    )
    merged["product_category_name_english"] = merged["product_category_name_english"].fillna("uncategorized")
    return merged


def get_category_sentiment(merged_df):
    cat_sent = (
        merged_df.groupby("product_category_name_english")
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            avg_review_score=("review_score", "mean"),
            total_reviews=("order_id", "count"),
            pct_negative=("sentiment_label", lambda x: (x == "Negative").mean() * 100)
        )
        .reset_index()
        .rename(columns={"product_category_name_english": "category"})
        .round(3)
    )
    return cat_sent.sort_values("avg_sentiment", ascending=True)


def get_weekly_sentiment(merged_df, category):
    df = merged_df[merged_df["product_category_name_english"] == category].copy()
    df["week"] = pd.to_datetime(df["review_creation_date"]).dt.to_period("W").dt.start_time
    weekly = (
        df.groupby("week")
        .agg(avg_sentiment=("sentiment_score", "mean"), review_count=("order_id", "count"))
        .reset_index()
    )
    return weekly
