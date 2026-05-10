import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_olist_data():
    orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ])
    items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
    products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
    reviews = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv", parse_dates=["review_creation_date"])
    categories = pd.read_csv(DATA_DIR / "product_category_name_translation.csv")
    return orders, items, products, reviews, categories


def build_master_df(orders, items, products, categories):
    df = items.merge(orders[["order_id", "order_purchase_timestamp", "order_status"]], on="order_id", how="left")
    df = df.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    df = df.merge(categories, on="product_category_name", how="left")
    df = df[df["order_status"] == "delivered"].copy()
    df["order_date"] = df["order_purchase_timestamp"].dt.to_period("W").dt.start_time
    df["product_category_name_english"] = df["product_category_name_english"].fillna("uncategorized")
    return df


def build_category_weekly(df):
    weekly = (
        df.groupby(["product_category_name_english", "order_date"])
        .agg(units_sold=("order_item_id", "count"), revenue=("price", "sum"))
        .reset_index()
        .rename(columns={"order_date": "ds", "product_category_name_english": "category"})
    )
    weekly["ds"] = pd.to_datetime(weekly["ds"])
    return weekly


def build_sku_features(df):
    latest_date = df["order_purchase_timestamp"].max()
    cutoff = latest_date - pd.Timedelta(weeks=8)
    recent = df[df["order_purchase_timestamp"] >= cutoff]

    sku = (
        recent.groupby("product_id")
        .agg(
            category=("product_category_name_english", "first"),
            units_sold=("order_item_id", "count"),
            avg_price=("price", "mean"),
            total_revenue=("price", "sum"),
        )
        .reset_index()
    )

    prev = df[df["order_purchase_timestamp"] < cutoff]
    prev_sku = prev.groupby("product_id").agg(prev_units=("order_item_id", "count")).reset_index()
    sku = sku.merge(prev_sku, on="product_id", how="left")
    sku["prev_units"] = sku["prev_units"].fillna(0)

    sku["velocity"] = sku["units_sold"] / 8
    sku["velocity_change"] = (sku["units_sold"] - sku["prev_units"]) / (sku["prev_units"] + 1)
    sku["days_of_stock_est"] = np.where(sku["velocity"] > 0, 30 / sku["velocity"], 999)

    sku["stockout_risk"] = pd.cut(
        sku["days_of_stock_est"],
        bins=[-np.inf, 7, 21, np.inf],
        labels=["High", "Medium", "Low"]
    )
    return sku


def get_top_categories(weekly, n=10):
    return (
        weekly.groupby("category")["units_sold"]
        .sum()
        .nlargest(n)
        .index.tolist()
    )
