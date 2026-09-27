import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def run_diagnostics(file_path="data/daily_sales.csv"):
    df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
    ts = df["sales"]
    
    # 1. Augmented Dickey-Fuller Test
    print("=" * 50)
    print("      AUGMENTED DICKEY-FULLER (ADF) TEST")
    print("=" * 50)
    result = adfuller(ts.dropna())
    print(f"ADF Statistic:       {result[0]:.4f}")
    print(f"p-value:             {result[1]:.4f}")
    print(f"Used Lags:           {result[2]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
        
    if result[1] <= 0.05:
        print("[+] Verdict: Series is STATIONARY (Reject H0 at 5% alpha).")
    else:
        print("[-] Verdict: Series is NON-STATIONARY (Fail to reject H0). Differencing required (d >= 1).")
    print("=" * 50)

    # 2. Classical Seasonal Decomposition (Weekly period = 7 days)
    decomposition = seasonal_decompose(ts, model="additive", period=7)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    decomposition.observed.plot(ax=axes[0], color="#1f77b4", title="Observed Daily Sales")
    decomposition.trend.plot(ax=axes[1], color="#ff7f0e", title="Estimated Trend")
    decomposition.seasonal.plot(ax=axes[2], color="#2ca02c", title="Weekly Seasonality (Period = 7)")
    decomposition.resid.plot(ax=axes[3], color="#d62728", style=".", title="Residuals / Stochastic Noise")
    plt.tight_layout()
    plt.savefig("data/seasonal_decomposition.png", dpi=300)
    print("[*] Saved decomposition plot: data/seasonal_decomposition.png")

    # 3. Autocorrelation (ACF) & Partial Autocorrelation (PACF)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(ts.diff().dropna(), lags=35, ax=ax1, title="Differenced Series ACF (Lag Autocorrelation)")
    plot_pacf(ts.diff().dropna(), lags=35, ax=ax2, method="ywm", title="Differenced Series PACF (Partial Lag Correlation)")
    plt.tight_layout()
    plt.savefig("data/acf_pacf_plots.png", dpi=300)
    print("[*] Saved ACF/PACF analysis plot: data/acf_pacf_plots.png")

if __name__ == "__main__":
    run_diagnostics()