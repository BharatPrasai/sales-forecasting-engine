import numpy as np
import pandas as pd

def generate_retail_sales_data(
    start_date="2021-01-01",
    end_date="2023-12-31",
    output_path="data/daily_sales.csv",
    seed=42
):
    np.random.seed(seed)
    
    # 1. Continuous daily date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)
    
    # 2. Components
    # Base level + Linear upward trend
    base_sales = 200.0
    trend = np.linspace(0, 80, n_days)
    
    # Day-of-week seasonality (Mon=0 ... Sun=6)
    # Weekend lift (Fri: +25, Sat: +50, Sun: +35)
    dow_effects = {0: -10, 1: -15, 2: -5, 3: 0, 4: 25, 5: 50, 6: 35}
    weekly_seasonality = np.array([dow_effects[d.dayofweek] for d in dates])
    
    # Yearly seasonality (Fourier wave peaking in Q4 / late November-December)
    day_of_year = np.array([d.dayofyear for d in dates])
    yearly_seasonality = 40 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    # Add extra Thanksgiving / Christmas spike in Q4
    q4_boost = np.where((day_of_year >= 320) & (day_of_year <= 360), 65, 0)
    
    # Exogenous promotional campaigns (~8% random promo days)
    is_promo = np.random.binomial(1, 0.08, size=n_days)
    promo_effect = is_promo * np.random.uniform(40, 90, size=n_days)
    
    # Stochastic noise (random customer variance)
    noise = np.random.normal(loc=0, scale=12, size=n_days)
    
    # 3. Aggregate total daily sales
    sales = base_sales + trend + weekly_seasonality + yearly_seasonality + q4_boost + promo_effect + noise
    sales = np.maximum(sales, 10.0) # Ensure strictly non-negative demand
    
    # 4. Construct DataFrame
    df = pd.DataFrame({
        "date": dates,
        "sales": np.round(sales, 2),
        "is_promo": is_promo,
        "day_of_week": [d.strftime("%A") for d in dates],
        "is_weekend": [1 if d.dayofweek in [5, 6] else 0 for d in dates]
    })
    
    df.to_csv(output_path, index=False)
    print(f"[*] Generated {n_days} daily records: {start_date} to {end_date}")
    print(f"[*] Saved dataset successfully to: {output_path}")
    print("\nDataset Preview:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df[["sales", "is_promo"]].describe())

if __name__ == "__main__":
    generate_retail_sales_data()