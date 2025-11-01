from typing import Callable, Optional

import numpy as np

from data_integrity import ensure_numeric_consistency
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


def create_y_scale(column_numbers, column_bboxes) -> Optional[Callable]:
    """

    :param column_numbers:
    :param column_bboxes:
    :return: function mapping y-coordinate to value. Function should be reversible.
    """
    if len(column_bboxes) != len(column_numbers):
        raise ValueError("Number of bounding boxes and numbers must match")

    if len(column_numbers) < 2:
        return None

    bboxes_y_centers = np.array([(box[1] + box[3]) / 2 for box in column_bboxes])

    # Ensure numeric consistency
    column_numbers, bboxes_y_centers = ensure_numeric_consistency(
        column_numbers, bboxes_y_centers
    )
    argsort = np.argsort(bboxes_y_centers)
    y_sorted = bboxes_y_centers[argsort]
    n_sorted = np.array(column_numbers)[argsort]

    if is_log_scale(column_numbers):
        return Logarithmic(y_sorted[0], y_sorted[-1], n_sorted[0], n_sorted[-1])
    else:
        return Linear(y_sorted[0], y_sorted[-1], n_sorted[0], n_sorted[-1])


def create_x_scale(row_index, row_bboxes) -> Optional[Callable]:
    if len(row_bboxes) != len(row_index):
        raise ValueError("Number of bounding boxes and index values must match")

    if len(row_index) < 2:
        return None

    bboxes_x_centers = np.array([(box[0] + box[2]) / 2 for box in row_bboxes])

    argsort = np.argsort(bboxes_x_centers)
    x_sorted = bboxes_x_centers[argsort]
    idx_sorted = np.array(row_index)[argsort]

    if hasattr(idx_sorted[0], "year") and hasattr(idx_sorted[-1], "year"):
        return LinearDatetime(x_sorted[0], x_sorted[-1], idx_sorted[0], idx_sorted[-1])
    else:
        return Linear(x_sorted[0], x_sorted[-1], 0, len(row_index) - 1)
