import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_inventory_forecast_plot():
    # 1. Load historical and predicted data
    history_df = pd.read_csv("data/daily_sales.csv", parse_dates=["date"], index_col="date")
    future_df = pd.read_csv("data/future_forecast_30d.csv", parse_dates=["date"], index_col="date")

    # Focus on the last 90 days of history for clear visual context
    recent_history = history_df.iloc[-90:]

    # 2. Key Supply Chain Parameters (matching predict.py)
    avg_daily_demand = future_df["forecasted_sales"].mean()
    daily_std = future_df["forecasted_sales"].std()
    lead_time_days = 7
    safety_stock = 1.645 * np.sqrt(lead_time_days) * daily_std
    reorder_point = (avg_daily_demand * lead_time_days) + safety_stock

    # 3. Plotting
    plt.figure(figsize=(16, 7), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Historical demand line
    plt.plot(recent_history.index, recent_history["sales"], label="Observed Daily Sales (History)", color="#1f2937", linewidth=1.5)

    # Future recursive forecast line
    plt.plot(future_df.index, future_df["forecasted_sales"], label="30-Day Recursive ML Forecast", color="#2563eb", linewidth=2.2, linestyle="-")

    # Highlight promotional spikes in forecast
    promo_days = future_df[future_df["is_promo"] == 1]
    if not promo_days.empty:
        plt.scatter(promo_days.index, promo_days["forecasted_sales"], color="#dc2626", s=90, zorder=5, label="Planned Promotional Uplift", edgecolors="black")

    # Historical / Forecast Cutoff line
    cutoff_date = recent_history.index[-1]
    plt.axvline(x=cutoff_date, color="#6b7280", linestyle="--", linewidth=1.8, label="Forecast Horizon Cutoff")

    # Annotate Forecast Zone
    plt.axvspan(cutoff_date, future_df.index[-1], color="#eff6ff", alpha=0.6, label="Inference Window (Out-of-Sample)")

    # Inventory Metrics Callout Box
    textstr = (
        f"INVENTORY POLICIES (L=7d, 95% SL)\n"
        f"------------------------------------\n"
        f"30-Day Projected Demand: {future_df['forecasted_sales'].sum():,.0f} units\n"
        f"Average Daily Demand:    {avg_daily_demand:.1f} units/day\n"
        f"Calculated Safety Stock: {safety_stock:.1f} units\n"
        f"Reorder Point (ROP):     {reorder_point:.1f} units"
    )
    plt.gca().text(
        0.02, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#d1d5db", alpha=0.95)
    )

    # Formatting
    plt.title("Retail Demand Forecasting Engine: 30-Day Forward Trajectory & Supply Replenishment", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=11, fontweight="medium")
    plt.ylabel("Sales Volume (Units)", fontsize=11, fontweight="medium")
    plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path = "data/demand_planning_forecast.png"
    plt.savefig(output_path)
    print(f"[*] Generated presentation-quality chart: {output_path}")

if __name__ == "__main__":
    generate_inventory_forecast_plot()