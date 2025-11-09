from typing import Iterable

import numpy as np


def find_minimal_powers_of_10(float_series: Iterable[float]) -> list[int]:
    """
    Find the minimal power of 10 needed to convert the fractional parts
    of the float numbers in `float_series` to integers.

    :param float_series: Iterable of float numbers
    :return: List of minimal powers of 10 for each float number
    """
    float_series_strings = (str(f) for f in float_series)
    return [
        len(s.split(".")[-1].rstrip("0")) if "." in s else 0
        for s in float_series_strings
    ]


def ensure_linear_continuity(
    x1: np.ndarray,
    x2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure that the numeric values in `column_numbers` are consistent and linear,
    filling in any missing points based on the `bboxes_y_centers`.

    :param x1:
    :param x2:
    :return:
    """
    min_powers_of_10 = find_minimal_powers_of_10(x1)
    min_power = max(min_powers_of_10)
    multiplier = 10**min_power

    diffs = np.diff(x1 * multiplier).round().astype(int)
    abs_diffs = np.abs(diffs)
    counts = np.bincount(abs(diffs))
    vals = np.nonzero(counts)[0]
    counts = counts[vals]
    base_diff = vals[np.argmax(counts)]
    abs_diffs = np.concatenate([[base_diff], abs_diffs])

    interpolation_points = (
        (np.array(min_powers_of_10) == 0) & (abs_diffs != base_diff) & (x1 != 0)
    )
    for idx in np.flatnonzero(interpolation_points):
        if idx == 0 or idx == len(x1) - 1:
            continue
        lp, rp = x1[idx - 1], x1[idx + 1]
        x1[idx] = (lp + rp) / 2

    # Find missing points in bboxes_y_centers
    missing_points = find_missing_points(x2)
    n_points = len(x1) + len(missing_points)

    # ensure numbers are in descending order
    final_x1 = np.array(
        [round(x1[0] - i * base_diff / multiplier, min_power) for i in range(n_points)]
    )

    final_x2 = x2.copy()
    if not missing_points:
        return final_x1, final_x2

    final_x2 = np.sort(np.concatenate([final_x2, missing_points]))
    return final_x1, final_x2


def find_missing_points(arr: np.ndarray):
    """
    Find missing points in a sorted array based on expected step size.

    :param arr:
    :return:
    """
    arr_sorted = np.sort(arr)
    diffs = np.diff(arr_sorted)

    # Use the most common diff as the expected step
    step = np.bincount(diffs.round().astype(int)).argmax()
    missing_points = []
    for i, d in enumerate(diffs):
        if d > step * 1.5:  # Allow some tolerance
            num_missing = int(round(d / step)) - 1
            for j in range(num_missing):
                missing = arr_sorted[i] + (j + 1) * step
                missing_points.append(missing)
    return missing_points


def assign_numbers_to_missing_points(
    bboxes_y_centers: np.ndarray,
    column_numbers: np.ndarray,
    missing_points: list,
):
    # Sort by y-center (ascending)
    sorted_indices = np.argsort(bboxes_y_centers)
    y_sorted = np.array(bboxes_y_centers)[sorted_indices]
    n_sorted = np.array(column_numbers)[sorted_indices]

    # Estimate step (assume linear)
    # steps = np.diff(y_sorted)
    # num_steps = np.diff(n_sorted)
    # avg_step = np.mean(num_steps / steps)

    assigned = []
    for mp in missing_points:
        # Find where the missing point fits
        idx = np.searchsorted(y_sorted, mp)
        if 0 < idx < len(y_sorted):
            y0, y1 = y_sorted[idx - 1], y_sorted[idx]
            n0, n1 = n_sorted[idx - 1], n_sorted[idx]
            # Linear interpolation
            n_mp = n0 + (mp - y0) * (n1 - n0) / (y1 - y0)
            assigned.append(n_mp)
        else:
            assigned.append(None)  # Out of bounds
    return assigned
