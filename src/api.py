from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Retail Demand Forecasting & Replenishment API",
    description="Production scoring service serving recursive multi-step retail sales predictions and inventory buffers.",
    version="1.0.0"
)

# Load model artifact into application state
MODEL_PATH = "data/rf_forecast_model.joblib"
DATA_PATH = "data/daily_sales.csv"

if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    raise RuntimeError("Required model artifacts or historical data not found.")

model = joblib.load(MODEL_PATH)
history_df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")

class ForecastRequest(BaseModel):
    horizon_days: int = Field(default=14, ge=1, le=90, description="Forecast horizon length in days (1 to 90)")
    lead_time_days: int = Field(default=7, ge=1, le=30, description="Supplier lead time in days for replenishment calculations")
    service_level_z: float = Field(default=1.645, description="Z-score for target service level (1.645 = 95%, 2.326 = 99%)")
    planned_promotions: Optional[List[str]] = Field(default=[], description="List of target dates (YYYY-MM-DD) with active marketing promotions")

class DailyForecastItem(BaseModel):
    date: str
    day_of_week: str
    is_promo: int
    forecasted_sales: float

class InventoryDirectives(BaseModel):
    total_projected_demand: float
    avg_daily_demand: float
    safety_stock: float
    reorder_point: float

class ForecastResponse(BaseModel):
    horizon_days: int
    historical_cutoff: str
    directives: InventoryDirectives
    daily_breakdown: List[DailyForecastItem]

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "historical_records": len(history_df),
        "cutoff_date": history_df.index[-1].strftime("%Y-%m-%d")
    }

@app.post("/forecast", response_model=ForecastResponse)
def generate_forecast(payload: ForecastRequest):
    try:
        sales_history = history_df["sales"].copy()
        last_date = sales_history.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=payload.horizon_days, freq="D")
        
        forecast_items = []
        
        # Recursive roll-forward inference
        for current_date in future_dates:
            date_str = current_date.strftime("%Y-%m-%d")
            is_promo = 1 if date_str in payload.planned_promotions else 0
            
            lags = {f"lag_{l}": sales_history.iloc[-l] for l in [1, 2, 3, 7, 14, 21, 28]}
            rolling_7 = sales_history.iloc[-7:]
            rolling_30 = sales_history.iloc[-30:]
            
            dow = current_date.dayofweek
            month = current_date.month

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

            pred_val = float(model.predict(pd.DataFrame([features]))[0])
            sales_history.loc[current_date] = pred_val

            forecast_items.append(
                DailyForecastItem(
                    date=date_str,
                    day_of_week=current_date.strftime("%A"),
                    is_promo=is_promo,
                    forecasted_sales=round(pred_val, 2)
                )
            )

        # Compute replenishment parameters
        forecast_series = pd.Series([item.forecasted_sales for item in forecast_items])
        avg_demand = float(forecast_series.mean())
        daily_std = float(forecast_series.std())
        
        safety_stock = payload.service_level_z * np.sqrt(payload.lead_time_days) * daily_std
        reorder_point = (avg_demand * payload.lead_time_days) + safety_stock

        return ForecastResponse(
            horizon_days=payload.horizon_days,
            historical_cutoff=last_date.strftime("%Y-%m-%d"),
            directives=InventoryDirectives(
                total_projected_demand=round(float(forecast_series.sum()), 2),
                avg_daily_demand=round(avg_demand, 2),
                safety_stock=round(safety_stock, 2),
                reorder_point=round(reorder_point, 2)
            ),
            daily_breakdown=forecast_items
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))