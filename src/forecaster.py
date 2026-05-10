import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


def forecast_category(weekly_df, category, periods=8):
    cat_df = weekly_df[weekly_df["category"] == category][["ds", "units_sold"]].copy()
    cat_df = cat_df.rename(columns={"units_sold": "y"})
    cat_df = cat_df.sort_values("ds").dropna()

    if len(cat_df) < 10:
        return None, None

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_mode="multiplicative",
    )
    m.fit(cat_df)

    future = m.make_future_dataframe(periods=periods, freq="W")
    forecast = m.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    return forecast, cat_df


def get_forecast_summary(forecast, periods=8):
    future_fc = forecast.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    future_fc.columns = ["Week", "Predicted Units", "Lower Bound", "Upper Bound"]
    future_fc["Week"] = future_fc["Week"].dt.strftime("%Y-%m-%d")
    future_fc = future_fc.round(1)
    return future_fc


def forecast_all_categories(weekly_df, categories, periods=8):
    results = {}
    for cat in categories:
        fc, hist = forecast_category(weekly_df, cat, periods)
        if fc is not None:
            results[cat] = {"forecast": fc, "history": hist}
    return results
