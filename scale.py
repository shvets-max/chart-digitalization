from typing import Callable, Optional

import numpy as np

from data_integrity import ensure_linear_continuity
from function import Linear, LinearDatetime, Logarithmic


def estimate_log_base(numbers: np.ndarray) -> float:
    numbers = np.array(numbers)
    valid = numbers > 0
    y = np.arange(len(numbers))[valid]
    log_vals = np.log(numbers[valid])
    # Linear regression: log_vals = intercept + slope * y
    slope, intercept = np.polyfit(y, log_vals, 1)
    base = np.exp(slope)
    return base


def is_log_scale(numbers: np.ndarray, tolerance: float = 0.1) -> bool:
    diffs = np.diff(numbers)
    if max(diffs) - min(diffs) < tolerance:
        return False

    # is decreasing?
    if all(np.diff(numbers) >= 0):
        sorted_numbers = np.array(numbers.copy())
    else:
        sorted_numbers = np.sort(numbers)

    sorted_numbers = sorted_numbers[sorted_numbers > 0]
    if len(sorted_numbers) < 3:
        return False

    pct_diff = (sorted_numbers[1:] - sorted_numbers[:-1]) / sorted_numbers[1:]
    if max(pct_diff) - min(pct_diff) < tolerance:
        return True
    return False


def nice_ticks(vmin: float, vmax: float, count: int = 6) -> list[float]:
    """
    Round tick values covering [vmin, vmax], spaced by a 1/2/5 x 10^n step.
    All returned ticks lie inside the range.
    """
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    span = vmax - vmin
    if span <= 0 or count < 2:
        return [vmin]

    raw_step = span / (count - 1)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    candidates = [multiple * magnitude for multiple in (1, 2, 2.5, 5, 10)]
    step = min(candidates, key=lambda s: abs(np.log(s / raw_step)))

    start = np.ceil(vmin / step) * step
    ticks = np.arange(start, vmax + step * 1e-6, step)
    decimals = max(0, int(-np.floor(np.log10(step))) + 1)
    return [round(float(t), decimals) for t in ticks if vmin <= t <= vmax]


def log_ticks(vmin: float, vmax: float, count: int = 6) -> list[float]:
    """Ticks at round 1/2/5 x 10^n values covering [vmin, vmax] on a log axis."""
    vmin, vmax = max(min(vmin, vmax), 1e-12), max(vmin, vmax)
    if vmax <= vmin:
        return [vmin]

    candidates = []
    exponents = range(int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax))) + 1)
    for exponent in exponents:
        for mantissa in (1, 2, 5):
            value = mantissa * 10.0**exponent
            if vmin <= value <= vmax:
                candidates.append(value)

    if len(candidates) > count:  # thin out evenly, keeping both endpoints
        keep = np.linspace(0, len(candidates) - 1, count).round().astype(int)
        candidates = [candidates[i] for i in dict.fromkeys(keep.tolist())]
    return candidates or nice_ticks(vmin, vmax, count)


def create_y_scale(values, knots: np.ndarray) -> Optional[Callable]:
    """

    :param values:
    :param knots:
    :return: function mapping y-coordinate to value. Function should be reversible.
    """
    if len(knots) != len(values):
        raise ValueError("Number of bounding boxes and numbers must match")

    if len(values) < 2:
        return None

    if is_log_scale(values):
        print("Using logarithmic scale for y-axis")
        arg_sorted = np.argsort(knots)
        y_sorted = knots[arg_sorted]
        n_sorted = np.array(values)[arg_sorted]
        return Logarithmic(knots=y_sorted, values=n_sorted)
    else:
        print("Using linear scale for y-axis")
        values, knots = ensure_linear_continuity(
            x1=np.array(values), x2=np.array(knots)
        )
        arg_sorted = np.argsort(knots)
        y_sorted = knots[arg_sorted]
        n_sorted = np.array(values)[arg_sorted]
        return Linear(knots=y_sorted, values=n_sorted)


def create_x_scale(row_index, knots: np.ndarray) -> Optional[Callable]:
    if len(knots) != len(row_index):
        raise ValueError("Number of bounding boxes and index values must match")

    if len(row_index) < 2:
        return None

    argsort = np.argsort(knots)
    x_sorted = knots[argsort]
    idx_sorted = np.array(row_index)[argsort]

    if hasattr(idx_sorted[0], "year") and hasattr(idx_sorted[-1], "year"):
        return LinearDatetime(knots=x_sorted, datetimes=idx_sorted)
    else:
        return Linear(knots=x_sorted, values=idx_sorted)
