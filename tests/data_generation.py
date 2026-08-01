import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.scale import LogScale
from matplotlib.ticker import FuncFormatter

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
    plt.grid()
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
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x))}"))

    plt.grid()
    plt.savefig(output_image)


def generate_with_in_area_text(
    start_date,
    end_date,
    start_value=100,
    variant="legend",
    output_csv="simulated_in_area_text.csv",
    output_image="simulated_in_area_text.png",
):
    """
    Linear-scaled chart with non-series ink drawn inside the plotting area, the way
    real screenshots carry legends, watermarks and value badges (see
    data/ebit-margin.png). `variant` is one of "legend", "watermark", "badge",
    "marker" or "corners".
    """
    trend = np.random.choice([-1, 1])
    ts = simulate_time_series(
        start_date, end_date, start_value, avg_daily_return=1e-3 * trend
    )
    ts.to_csv(output_csv, index=False, sep=SEP)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.plot(ts["date"], ts["value"])
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Simulated Values Over Time")
    plt.grid()

    common = dict(transform=ax.transAxes, color="black")
    if variant == "legend":
        # Header strip in the top-left, like a chart widget's title bar.
        ax.text(0.01, 0.96, "ACME: Revenue (TTM)   11.75%", fontsize=11, **common)
    elif variant == "watermark":
        # Large faint text across the middle, overlapping the series itself.
        ax.text(
            0.5,
            0.5,
            "SAMPLE",
            fontsize=48,
            alpha=0.35,
            ha="center",
            va="center",
            **common,
        )
    elif variant == "badge":
        # Value badge pinned to the right edge, next to the last data point.
        ax.text(
            0.995,
            0.5,
            "123.45",
            fontsize=11,
            ha="right",
            va="center",
            bbox=dict(facecolor="0.5", edgecolor="none"),
            **common,
        )
    elif variant == "marker":
        # Isolated annotation dot away from the series.
        ax.plot(
            [0.45],
            [0.75],
            marker="o",
            markersize=4,
            linestyle="",
            **common,
        )
    elif variant == "corners":
        # Text in every corner, so both the leading and trailing columns are
        # ambiguous and cannot rely on an already-resolved neighbour.
        for x, y, ha, va in (
            (0.005, 0.97, "left", "top"),
            (0.995, 0.97, "right", "top"),
            (0.005, 0.03, "left", "bottom"),
            (0.995, 0.03, "right", "bottom"),
        ):
            ax.text(x, y, "note", fontsize=10, ha=ha, va=va, **common)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    plt.savefig(output_image)
    plt.close()


IN_AREA_TEXT_VARIANTS = ("legend", "watermark", "badge", "marker", "corners")


if __name__ == "__main__":
    sample_size = 15
    linear_path, log_path, in_area_path = (
        os.path.join(TEST_DATA_DIR, "linear_scaled"),
        os.path.join(TEST_DATA_DIR, "log_scaled"),
        os.path.join(TEST_DATA_DIR, "in_area_text"),
    )
    os.makedirs(linear_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(in_area_path, exist_ok=True)

    np.random.seed(20260801)
    for variant in IN_AREA_TEXT_VARIANTS:
        generate_with_in_area_text(
            start_date="2023-01-01",
            end_date="2025-03-31",
            variant=variant,
            output_csv=os.path.join(in_area_path, f"in_area_{variant}.csv"),
            output_image=os.path.join(in_area_path, f"in_area_{variant}.png"),
        )

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
