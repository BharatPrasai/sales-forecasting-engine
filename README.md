# ?? Retail Sales Forecasting & Inventory Demand Planning Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end time series forecasting and demand planning engine designed to optimize supply chain inventory levels.

Compares classical statistical econometric formulations (SARIMAX) against machine learning regressors (Random Forest) with multi-lag rolling-window engineering, evaluated against strict out-of-time temporal partitions.

---

## ?? Benchmark Performance (60-Day Out-of-Time Test)
![Demand Planning Forecast](data/demand_planning_forecast.png)

| Model Architecture | Features / Inputs | MAE | RMSE | WAPE (%) | Working Capital Impact (Safety Stock)* |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SARIMAX(1, 1, 1)x(1, 1, 1)7** | Promotional exogenous regressors | 33.00 | 36.66 | 10.84% | ~160 units buffer |
| **Random Forest (Champion)** | Lags (t-1 ... t-28), 7d/30d rolling stats, cyclical calendar | **22.23** | **29.33** | **7.30%** | **~128 units buffer (20% reduction)** |

*\*Assumes 95% service level (Z = 1.645) and a 7-day replenishment lead time (L = 7).*

---

## ?? Technical Highlights

1. **Temporal Integrity (Zero Data Leakage):** Strict chronological out-of-time test horizon (60 days). All rolling aggregations use shift(1) to prevent target leakage.
2. **Feature Engineering Engine:** Autoregressive lags, 7d/30d rolling means and standard deviations, plus sin/cos cyclical encodings.
3. **Supply Chain Valuation:** Incorporates Weighted Absolute Percentage Error (WAPE) to evaluate real inventory and safety stock reduction.

---

## ?? Repository Structure

- **data/**: Daily transactions and generated artifacts
- **src/generate_data.py**: Trend, seasonality, and promotional event generator
- **src/diagnostics.py**: ADF stationarity test and classical seasonal decomposition
- **src/train_models.py**: Feature engineering, SARIMAX benchmark, and Random Forest training
