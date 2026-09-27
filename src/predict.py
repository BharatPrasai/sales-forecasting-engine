import numpy as np
import pandas as pd
import joblib
import os

def recursive_forecast(horizon_days=30, promo_days=None):
    """
    Performs recursive multi-step forecasting:
    Predicts step t+1, updates historical lag/rolling windows, and repeats.
    """
    model_path = "data/rf_forecast_model.joblib"
    data_path = "data/daily_sales.csv"

    if not os.path.exists(model_path):
        raise FileNotFoundError("[!] Trained model artifact not found. Run train_models.py first.")

    # 1. Load trained model & historical observations
    rf_model = joblib.load(model_path)
    df = pd.read_csv(data_path, parse_dates=["date"], index_col="date")

    # Working buffer of historical sales
    sales_history = df["sales"].copy()
    last_date = sales_history.index[-1]

    if promo_days is None:
        promo_days = []

    print("=" * 60)
    print(f"[*] Starting Recursive Forecast for {horizon_days} Days Forward")
    print(f"[*] Historical Horizon Cutoff: {last_date.strftime('%Y-%m-%d')}")
    print("=" * 60)

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    forecast_results = []

    # 2. Sequential day-by-day rollout
    for current_date in future_dates:
        # Determine exogenous promotional flag
        is_promo = 1 if current_date.strftime("%Y-%m-%d") in promo_days else 0

        # Dynamic lag lookup from rolling buffer
        lags = {}
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            lags[f"lag_{lag}"] = sales_history.iloc[-lag]

        # Dynamic rolling windows
        rolling_7 = sales_history.iloc[-7:]
        rolling_30 = sales_history.iloc[-30:]

        # Cyclical calendar features
        dow = current_date.dayofweek
        month = current_date.month

        # Feature vector assembled identically to training time
        features = {
            "is_promo": is_promo,
            "is_weekend": 1 if dow in [5, 6] else 0,
            "sin_dow": np.sin(2 * np.pi * dow / 7),
            "cos_dow": np.cos(2 * np.pi * dow / 7),
            "sin_month": np.sin(2 * np.pi * month / 12),
            "cos_month": np.cos(2 * np.pi * month / 12),
            **lags,
            "rolling_mean_7": rolling_7.mean(),
            "rolling_std_7": rolling_7.std(),
            "rolling_mean_30": rolling_30.mean(),
            "rolling_std_30": rolling_30.std()
        }

        feature_df = pd.DataFrame([features])
        pred_sales = float(rf_model.predict(feature_df)[0])

        # Append prediction back into historical series for subsequent lag steps
        sales_history.loc[current_date] = pred_sales

        forecast_results.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "day_of_week": current_date.strftime("%A"),
            "is_promo": is_promo,
            "forecasted_sales": round(pred_sales, 2)
        })

    result_df = pd.DataFrame(forecast_results)
    
    # 3. Supply Chain Buffer Summary
    total_projected_demand = result_df["forecasted_sales"].sum()
    avg_daily_demand = result_df["forecasted_sales"].mean()
    daily_std = result_df["forecasted_sales"].std()

    # Lead time = 7 days, 95% service level (Z = 1.645)
    lead_time_days = 7
    safety_stock = 1.645 * np.sqrt(lead_time_days) * daily_std
    reorder_point = (avg_daily_demand * lead_time_days) + safety_stock

    print("\n--- FORECAST PREVIEW (FIRST 7 DAYS) ---")
    print(result_df.head(7).to_string(index=False))

    print("\n--- INVENTORY REPLENISHMENT DIRECTIVES (7-Day Lead Time, 95% Service Level) ---")
    print(f"Total Projected Demand ({horizon_days} Days): {total_projected_demand:,.2f} units")
    print(f"Average Daily Demand:               {avg_daily_demand:.2f} units/day")
    print(f"Safety Stock Requirement:           {safety_stock:.2f} units")
    print(f"Recommended Reorder Point (ROP):    {reorder_point:.2f} units")
    print("=" * 60)

    # Save to disk
    output_path = "data/future_forecast_30d.csv"
    result_df.to_csv(output_path, index=False)
    print(f"[*] Saved full 30-day forecast to {output_path}")

if __name__ == "__main__":
    recursive_forecast(horizon_days=30, promo_days=["2024-01-15", "2024-01-26"])