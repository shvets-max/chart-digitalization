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


def is_log_scale(numbers: np.ndarray, tolerance: float = 0.98) -> bool:
    # Remove zeros and negatives for log
    if len({round(d) for d in np.diff(numbers)}) == 1:
        return False

    numbers = np.array(numbers)
    log_base = estimate_log_base(numbers)
    if log_base <= 0.8 or log_base >= 1.2:
        return False
    valid = numbers > 0
    if np.sum(valid) < 2:
        return False
    y = np.arange(len(numbers))[valid]
    log_vals = np.log(numbers[valid])
    corr = np.corrcoef(y, log_vals)[0, 1]
    return abs(corr) > tolerance


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

    values, knots = ensure_linear_continuity(x1=np.array(values), x2=np.array(knots))

    arg_sorted = np.argsort(knots)
    y_sorted = knots[arg_sorted]
    n_sorted = np.array(values)[arg_sorted]

    if is_log_scale(values):
        print("Using logarithmic scale for y-axis")
        return Logarithmic(y_sorted[0], y_sorted[-1], n_sorted[0], n_sorted[-1])
    else:
        print("Using linear scale for y-axis")
        return Linear(y_sorted[0], y_sorted[-1], n_sorted[0], n_sorted[-1])


def create_x_scale(row_index, knots: np.ndarray) -> Optional[Callable]:
    if len(knots) != len(row_index):
        raise ValueError("Number of bounding boxes and index values must match")

    if len(row_index) < 2:
        return None

    argsort = np.argsort(knots)
    x_sorted = knots[argsort]
    idx_sorted = np.array(row_index)[argsort]

    if hasattr(idx_sorted[0], "year") and hasattr(idx_sorted[-1], "year"):
        return LinearDatetime(x_sorted[0], x_sorted[-1], idx_sorted[0], idx_sorted[-1])
    else:
        return Linear(x_sorted[0], x_sorted[-1], 0, len(row_index) - 1)
