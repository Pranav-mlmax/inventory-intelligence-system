import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


def prepare_features(sku_df):
    df = sku_df.copy()
    le = LabelEncoder()
    df["category_enc"] = le.fit_transform(df["category"].astype(str))

    feature_cols = ["units_sold", "avg_price", "total_revenue", "velocity",
                    "velocity_change", "days_of_stock_est", "category_enc"]
    X = df[feature_cols].fillna(0)
    return X, le, feature_cols


def train_risk_model(sku_df):
    X, le, feature_cols = prepare_features(sku_df)
    y = sku_df["stockout_risk"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    return model, le, feature_cols, report


def predict_risk(model, le, feature_cols, sku_df):
    df = sku_df.copy()
    df["category_enc"] = le.transform(df["category"].astype(str).apply(
        lambda x: x if x in le.classes_ else le.classes_[0]
    ))
    X = df[feature_cols].fillna(0)
    df["predicted_risk"] = model.predict(X)
    df["risk_proba"] = model.predict_proba(X).max(axis=1).round(3)
    return df


def get_risk_summary(pred_df):
    return (
        pred_df.groupby("predicted_risk")
        .agg(sku_count=("product_id", "count"), avg_velocity=("velocity", "mean"))
        .reset_index()
        .rename(columns={"predicted_risk": "Risk Level", "sku_count": "SKU Count", "avg_velocity": "Avg Weekly Sales"})
        .round(2)
    )
