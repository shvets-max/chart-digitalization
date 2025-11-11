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
