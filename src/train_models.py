import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

def create_time_series_features(df):
    data = df.copy()
    
    # 1. Calendar & Cyclical features
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["day_of_year"] = data.index.dayofyear
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    
    # Sin/Cos encodings for smooth periodic signals
    data["sin_dow"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["cos_dow"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    data["sin_month"] = np.sin(2 * np.pi * data["month"] / 12)
    data["cos_month"] = np.cos(2 * np.pi * data["month"] / 12)

    # 2. Historical Lag Features (strict shifts to prevent target leakage)
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        data[f"lag_{lag}"] = data["sales"].shift(lag)

    # 3. Rolling Window Aggregations
    data["rolling_mean_7"] = data["sales"].shift(1).rolling(window=7).mean()
    data["rolling_std_7"] = data["sales"].shift(1).rolling(window=7).std()
    data["rolling_mean_30"] = data["sales"].shift(1).rolling(window=30).mean()
    data["rolling_std_30"] = data["sales"].shift(1).rolling(window=30).std()

    return data.dropna()

def calculate_wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100

def run_forecasting_pipeline():
    print("[*] Loading time series dataset...")
    df = pd.read_csv("data/daily_sales.csv", parse_dates=["date"], index_col="date")
    
    # Hold out final 60 days for out-of-time evaluation
    test_horizon_days = 60
    split_date = df.index[-test_horizon_days]
    print(f"[*] Splitting at {split_date.strftime('%Y-%m-%d')} (Test set: {test_horizon_days} days)")

    # 1. Classical Statistical Benchmark: SARIMAX
    print("\n[*] Training SARIMAX(1, 1, 1)x(1, 1, 1, 7) with promotional regressors...")
    train_df = df.iloc[:-test_horizon_days]
    test_df = df.iloc[-test_horizon_days:]

    sarimax_model = SARIMAX(
        train_df["sales"],
        exog=train_df[["is_promo"]],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    sarimax_pred = sarimax_model.forecast(
        steps=test_horizon_days,
        exog=test_df[["is_promo"]]
    )

    # 2. Machine Learning Pipeline: Random Forest
    print("[*] Engineering lag/rolling features for ML Regressor...")
    ml_data = create_time_series_features(df)
    
    feature_cols = [c for c in ml_data.columns if c not in ["sales", "day_of_week", "month", "day_of_year"]]
    X = ml_data[feature_cols]
    y = ml_data["sales"]

    X_train = X.loc[X.index < split_date]
    y_train = y.loc[y.index < split_date]
    X_test = X.loc[X.index >= split_date]
    y_test = y.loc[y.index >= split_date]

    print(f"[*] Training features: {X_train.shape[1]} variables | Training instances: {len(X_train)}")
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Align evaluation indices
    common_idx = test_df.index.intersection(X_test.index)
    actuals = df.loc[common_idx, "sales"]
    sarimax_aligned = sarimax_pred.loc[common_idx]
    rf_aligned = pd.Series(rf_pred, index=X_test.index).loc[common_idx]

    # 3. Model Benchmark Evaluation
    metrics = {
        "Model": ["SARIMAX (Statistical)", "Random Forest (Feature Engineered)"],
        "MAE": [
            mean_absolute_error(actuals, sarimax_aligned),
            mean_absolute_error(actuals, rf_aligned)
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(actuals, sarimax_aligned)),
            np.sqrt(mean_squared_error(actuals, rf_aligned))
        ],
        "WAPE (%)": [
            calculate_wape(actuals, sarimax_aligned),
            calculate_wape(actuals, rf_aligned)
        ]
    }

    res_df = pd.DataFrame(metrics)
    print("\n" + "=" * 60)
    print("           OUT-OF-TIME EVALUATION BENCHMARK (60-DAY)")
    print("=" * 60)
    print(res_df.to_string(index=False, justify="center"))
    print("=" * 60)

    # Serialize champion artifact
    joblib.dump(rf, "data/rf_forecast_model.joblib")
    print("[*] Random Forest model exported to data/rf_forecast_model.joblib")

    # 4. Forecast Visualizations
    plt.figure(figsize=(15, 6))
    plt.plot(actuals.index, actuals, label="Actual Daily Sales", color="black", linewidth=1.8)
    plt.plot(actuals.index, sarimax_aligned, label="SARIMAX Forecast", color="#ff7f0e", linestyle="--")
    plt.plot(actuals.index, rf_aligned, label="Random Forest Forecast", color="#2ca02c", linestyle="-.")
    plt.title("60-Day Out-of-Time Demand Forecast vs Actuals", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sales Volume", fontsize=12)
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/forecast_vs_actuals.png", dpi=300)
    print("[*] Saved forecast chart: data/forecast_vs_actuals.png")

if __name__ == "__main__":
    run_forecasting_pipeline()
