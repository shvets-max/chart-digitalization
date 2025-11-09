import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.scale import LogScale
from matplotlib.ticker import ScalarFormatter

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEP = ";"


def simulate_time_series(
    start_date, end_date, initial_value=100, volatility=0.01, avg_daily_return=0
):
    dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
    returns = np.random.normal(loc=avg_daily_return, scale=volatility, size=len(dates))
    prices = initial_value * np.exp(np.cumsum(returns))
    return pd.DataFrame({"date": dates, "value": prices})


def generate_linear_scaled(
    start_date,
    end_date,
    start_value=100,
    output_csv="simulated_close_prices.csv",
    output_image="simulated_close_prices.png",
):
    trend = np.random.choice([-1, 1])  # Randomly choose upward or downward trend
    ts = simulate_time_series(
        start_date, end_date, start_value, avg_daily_return=1e-3 * trend
    )
    ts.to_csv(output_csv, index=False, sep=SEP)

    plt.figure(figsize=(12, 6))
    plt.plot(ts["date"], ts["value"], label="Value")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Simulated Values Over Time")
    plt.legend()
    # plt.grid()
    plt.savefig(output_image)


def generate_log_scaled(
    start_date,
    end_date,
    start_value=100,
    output_csv="simulated_close_prices_log.csv",
    output_image="simulated_close_prices_log.png",
):
    trend = np.random.choice([-1, 1])  # Randomly choose upward or downward trend
    ts = simulate_time_series(
        start_date,
        end_date,
        start_value,
        volatility=0.05,
        avg_daily_return=5e-3 * trend,
    )
    log_base = (ts["value"].max() / ts["value"].min()) ** (1 / 6)
    scale = LogScale(1, base=log_base)
    ts.to_csv(output_csv, index=False, sep=SEP)

    plt.figure(figsize=(12, 6))
    plt.plot(ts["date"], ts["value"], label="Log-Scaled Value")
    plt.xlabel("Date")
    plt.ylabel("Log-Scaled Value")
    plt.title("Simulated Log-Scaled Values Over Time")
    plt.yscale(scale)
    # Force integer tick labels
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().yaxis.get_major_formatter().set_scientific(False)

    plt.legend()
    # plt.grid()
    plt.savefig(output_image)


if __name__ == "__main__":
    sample_size = 5
    linear_path, log_path = (
        os.path.join(TEST_DATA_DIR, "linear_scaled"),
        os.path.join(TEST_DATA_DIR, "log_scaled"),
    )
    os.makedirs(linear_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    for i in range(sample_size):
        generate_linear_scaled(
            start_date="2023-01-01",
            end_date="2025-03-31",
            output_csv=os.path.join(linear_path, f"linear_scaled_{i}.csv"),
            output_image=os.path.join(linear_path, f"linear_scaled_{i}.png"),
        )

        generate_log_scaled(
            start_date="2023-01-01",
            end_date="2025-03-31",
            output_csv=os.path.join(log_path, f"log_scaled_{i}.csv"),
            output_image=os.path.join(log_path, f"log_scaled_{i}.png"),
        )
