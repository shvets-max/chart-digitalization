import os
from typing import Optional

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


def simulate_mean_reverting_series(dates, mean_value, volatility=0.02, reversion=0.03):
    """
    An Ornstein-Uhlenbeck-style walk that oscillates around `mean_value` for its
    whole length, rather than a plain random walk's drift (whose variance grows
    without bound over a multi-year date range, eventually carrying it away from
    any fixed level for good). Used for dense_crossing charts in
    generate_scrab_style_chart, which need the noisy line to keep crossing the
    step lines throughout the date range, not just near the start.
    """
    values = np.empty(len(dates))
    values[0] = mean_value
    noise = np.random.normal(scale=volatility * mean_value, size=len(dates))
    for t in range(1, len(dates)):
        values[t] = values[t - 1] + reversion * (mean_value - values[t - 1]) + noise[t]
    return pd.DataFrame({"date": dates, "value": values})


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


def generate_multiline_chart(
    start_date,
    end_date,
    start_values,
    names=None,
    colors=None,
    output_csv="simulated_multiline.csv",
    output_image="simulated_multiline.png",
):
    """
    Chart with one distinctly colored line per entry in `start_values`, sharing one
    date index. When `names` is given, each line gets a legend entry whose label
    text is recolored to match its line -- mirroring the dashboard screenshots in
    data/multiline/ (colored label text, no separate swatch icon), not
    matplotlib's default black-text-plus-icon legend.
    """
    series = []
    for start_value in start_values:
        trend = np.random.choice([-1, 1])
        series.append(
            simulate_time_series(
                start_date, end_date, start_value, avg_daily_return=1e-3 * trend
            )
        )

    combined = series[0][["date"]].copy()
    for i, ts in enumerate(series):
        combined[f"value{i + 1}"] = ts["value"].values
    combined.to_csv(output_csv, index=False, sep=SEP)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    lines = []
    for i, ts in enumerate(series):
        label = names[i] if names else None
        color = colors[i] if colors else None
        (line,) = ax.plot(ts["date"], ts["value"], label=label, color=color)
        lines.append(line)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Simulated Multi-Series Values Over Time")
    plt.grid()
    if names:
        # handlelength=0 and hiding the handles removes the swatch icon entirely,
        # leaving only the colored text -- matching the dashboard screenshots in
        # data/multiline/, which have no icon at all.
        legend = ax.legend(loc="upper left", handlelength=0, handletextpad=0)
        for text, line, handle in zip(legend.get_texts(), lines, legend.legend_handles):
            text.set_color(line.get_color())
            handle.set_visible(False)
    plt.savefig(output_image)
    plt.close()


def _quarterly_step_series(dates, start_value, volatility, avg_quarterly_return):
    """A value that only changes once per calendar quarter, held flat between
    revisions -- like an analyst estimate, which is what draws the step-shaped
    lines in data/scrab/ (e.g. anet-rev.png, wm-eps.png)."""
    quarters = pd.PeriodIndex(dates, freq="Q")
    unique_quarters = quarters.unique().sort_values()
    returns = np.random.normal(
        loc=avg_quarterly_return, scale=volatility, size=len(unique_quarters)
    )
    quarterly_values = start_value * np.exp(np.cumsum(returns))
    return pd.Series(quarterly_values, index=unique_quarters).reindex(quarters).values


def _crossing_actual_series(dates, level, amplitude, n_crossings, noise_scale):
    """
    A continuous line that sweeps a fixed-count sine around `level`, crossing it
    close to `n_crossings` times over `dates` -- an explicit, comparable
    alternative to `simulate_mean_reverting_series`'s random walk (whose crossing
    count is an emergent, unmeasured property of noise), for stress-testing
    extraction as a function of crossing density (docs/series-gaps-diagnosis.md).
    """
    t = np.linspace(0, 1, len(dates))
    swing = amplitude * np.sin(np.pi * n_crossings * t)
    noise = np.random.normal(scale=noise_scale, size=len(dates))
    return level + swing + noise


def generate_scrab_style_chart(
    start_date,
    end_date,
    start_value=100,
    n_step_series=2,
    log_scale=False,
    dense_crossing=False,
    n_crossings: Optional[int] = None,
    output_csv="simulated_scrab.csv",
    output_image="simulated_scrab.png",
):
    """
    Mimics the analyst-estimate dashboard screenshots in data/scrab/: a pale
    lavender background, gridlines on the y-axis only, a right-hand y-axis, no
    title, a top-left legend with a colored bullet + name + current value per
    line, a "Log"/"Lin" scale-toggle label sharing the legend's corner, one
    smooth "actual" line, and `n_step_series` quarterly-revised (step-shaped)
    estimate lines above it.

    By default the step lines sit comfortably above the actual line (as in
    data/scrab/anet-rev.png), so the lines barely touch. With `dense_crossing`,
    the step levels are pinned close to the actual line's own level and its
    volatility is raised, so the noisy line repeatedly crosses every step line
    over the whole date range -- the shape of data/scrab/nvo-pt.png, avgo-pt.png
    and panw-pt.png, which is what exposed the gaps and wrong values these tests
    guard against (see chart_extraction._is_same_trajectory and
    geometry.cut_chart_area). `n_crossings` selects a third mode: the actual line
    sweeps a fixed-count sine around the step band instead of a random walk, so
    the crossing count is an explicit, comparable parameter across fixtures
    rather than an emergent property of noise (see _crossing_actual_series).
    """
    trend = np.random.choice([-1, 1])
    if n_crossings is not None:
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        actual = pd.DataFrame(
            {
                "date": dates,
                "value": _crossing_actual_series(
                    dates,
                    level=start_value,
                    amplitude=0.15 * start_value,
                    n_crossings=n_crossings,
                    noise_scale=0.0008 * start_value,
                ),
            }
        )
    elif dense_crossing:
        # A plain random walk's variance grows without bound over a multi-year
        # range and would eventually carry it away from the steps for good (only
        # crossing near the start) -- mean-reverting around the step band's
        # center keeps it crossing throughout instead.
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        actual = simulate_mean_reverting_series(dates, mean_value=start_value)
    else:
        actual = simulate_time_series(
            start_date, end_date, start_value, avg_daily_return=1e-3 * trend
        )
        dates = actual["date"]

    palette = ["tab:red", "tab:green", "tab:purple", "tab:orange"]
    colors = [palette[i % len(palette)] for i in range(n_step_series)]
    # Avoid a bare trailing digit (e.g. "Estimate 1"): real dashboard names never
    # end that way, and it can coincide in x-position with an unrelated tick
    # label, corrupting axis-tick-group selection (see select_axis_tick_group).
    ordinals = ["First", "Second", "Third", "Fourth"]
    names = [f"Estimate {ordinals[i % len(ordinals)]}" for i in range(n_step_series)]

    combined = actual[["date"]].copy()
    combined["actual"] = actual["value"].values

    # A tight band (dense_crossing or n_crossings) keeps the step levels close to
    # the actual line's own level throughout, so crossings actually happen; the
    # default band sits comfortably above it instead.
    tight_band = dense_crossing or n_crossings is not None
    step_values = []
    for i in range(n_step_series):
        if n_crossings is not None:
            # Centered near, and barely drifting from, start_value -- the sine
            # sweep's own center -- so it stays inside the swing's amplitude for
            # the whole range and the requested crossing count is actually
            # realized, rather than the step wandering (via drift/volatility)
            # out of the swing's reach the way dense_crossing's random walk can.
            # A little quarter-to-quarter wiggle (not zero) keeps the step from
            # sitting at the exact same pixel row for the WHOLE chart: a
            # perfectly flat line has ink at every column of that one row, which
            # extract_chart's own grid-row heuristic (>50% of columns inked)
            # then misreads as a background gridline and erases outright.
            multiplier, volatility, avg_quarterly_return = 1.025, 0.01, 0.0
        elif tight_band:
            multiplier = 0.85 + 0.1 * i
            volatility, avg_quarterly_return = 0.02, 2e-3 * trend
        else:
            multiplier = 1.3 + 0.4 * i
            volatility, avg_quarterly_return = 0.03, 3e-2 * trend
        level = _quarterly_step_series(
            dates,
            start_value * multiplier,
            volatility=volatility,
            avg_quarterly_return=avg_quarterly_return,
        )
        combined[f"estimate{i + 1}"] = level
        step_values.append(level)
    combined.to_csv(output_csv, index=False, sep=SEP)

    # Must stay barely off-white: chart_extraction thresholds ink at gray < 250,
    # so anything darker would have its own background mistaken for ink (see
    # data/scrab/*.png, whose real backgrounds are gray ~251).
    background = "#FBFAFF"
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    all_names = ["Actual"] + names
    all_colors = ["tab:blue"] + colors
    all_values = [actual["value"].values] + step_values
    for values, color, drawstyle in zip(
        all_values, all_colors, ["default"] + ["steps-post"] * n_step_series
    ):
        ax.plot(dates, values, color=color, linewidth=1.2, drawstyle=drawstyle)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.grid(axis="y", color="0.85")
    if log_scale:
        ax.set_yscale("log")
        # Real dashboards label round numbers on a log axis ("17.4B", not
        # "$2\times10^2$"), which is also what OCR can actually read -- force
        # plain decimal labels instead of matplotlib's default scientific ones.
        plain_formatter = FuncFormatter(lambda x, _: f"{x:.4g}")
        ax.yaxis.set_major_formatter(plain_formatter)
        ax.yaxis.set_minor_formatter(plain_formatter)
    ax.set_xlabel("")
    ax.set_ylabel("")

    for i, (name, color, values) in enumerate(zip(all_names, all_colors, all_values)):
        ax.text(
            0.01,
            0.97 - i * 0.045,
            f"● {name}   {values[-1]:.2f}",
            color=color,
            fontsize=9,
            transform=ax.transAxes,
            va="top",
        )
    ax.text(
        0.97,
        0.97,
        "Log" if log_scale else "Lin",
        fontsize=9,
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    plt.savefig(output_image, facecolor=fig.get_facecolor())
    plt.close()


if __name__ == "__main__":
    sample_size = 15
    (
        linear_path,
        log_path,
        in_area_path,
        multiline_path,
        scrab_style_path,
        dense_crossing_path,
        crossing_multiline_path,
    ) = (
        os.path.join(TEST_DATA_DIR, "linear_scaled"),
        os.path.join(TEST_DATA_DIR, "log_scaled"),
        os.path.join(TEST_DATA_DIR, "in_area_text"),
        os.path.join(TEST_DATA_DIR, "multiline"),
        os.path.join(TEST_DATA_DIR, "scrab_style"),
        os.path.join(TEST_DATA_DIR, "dense_crossing"),
        os.path.join(TEST_DATA_DIR, "crossing_multiline"),
    )
    os.makedirs(linear_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(in_area_path, exist_ok=True)
    os.makedirs(multiline_path, exist_ok=True)
    os.makedirs(scrab_style_path, exist_ok=True)
    os.makedirs(dense_crossing_path, exist_ok=True)
    os.makedirs(crossing_multiline_path, exist_ok=True)

    np.random.seed(20260801)
    generate_multiline_chart(
        start_date="2023-01-01",
        end_date="2025-03-31",
        start_values=[100, 160],
        names=["Revenue Growth", "Operating Margin"],
        colors=["tab:red", "tab:blue"],
        output_csv=os.path.join(multiline_path, "multiline_legend.csv"),
        output_image=os.path.join(multiline_path, "multiline_legend.png"),
    )
    generate_multiline_chart(
        start_date="2023-01-01",
        end_date="2025-03-31",
        start_values=[100, 160, 220],
        names=None,
        colors=["tab:red", "tab:blue", "tab:green"],
        output_csv=os.path.join(multiline_path, "multiline_colors.csv"),
        output_image=os.path.join(multiline_path, "multiline_colors.png"),
    )

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

    for i, (n_step_series, log_scale) in enumerate(
        [(1, False), (2, True), (3, True), (2, False)]
    ):
        generate_scrab_style_chart(
            start_date="2022-01-01",
            end_date="2025-06-30",
            n_step_series=n_step_series,
            log_scale=log_scale,
            output_csv=os.path.join(scrab_style_path, f"scrab_style_{i}.csv"),
            output_image=os.path.join(scrab_style_path, f"scrab_style_{i}.png"),
        )

    # Charts shaped like data/scrab/nvo-pt.png, avgo-pt.png and panw-pt.png: a
    # noisy actual line that crosses every step line over and over, which is what
    # exposed the x-axis offset and duplicate-shade-merge bugs these fixtures
    # regression-test (see TestDenseCrossingExtraction).
    for i, n_step_series in enumerate([2, 3, 3]):
        generate_scrab_style_chart(
            start_date="2022-01-01",
            end_date="2025-06-30",
            n_step_series=n_step_series,
            dense_crossing=True,
            output_csv=os.path.join(dense_crossing_path, f"dense_crossing_{i}.csv"),
            output_image=os.path.join(dense_crossing_path, f"dense_crossing_{i}.png"),
        )

    # Same legend + step-line shape as dense_crossing, but with the crossing
    # count as an explicit, comparable parameter (see _crossing_actual_series)
    # instead of an emergent property of a random walk -- lets extraction quality
    # be measured as a function of crossing density (docs/series-gaps-diagnosis.md).
    for i, n_crossings in enumerate([2, 5, 10, 20]):
        generate_scrab_style_chart(
            start_date="2022-01-01",
            end_date="2025-06-30",
            n_step_series=1,
            n_crossings=n_crossings,
            output_csv=os.path.join(
                crossing_multiline_path, f"crossing_multiline_{i}.csv"
            ),
            output_image=os.path.join(
                crossing_multiline_path, f"crossing_multiline_{i}.png"
            ),
        )
