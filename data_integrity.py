import numpy as np


def minimal_power_of_10(float_series):
    decimals = (
        len(str(f).split(".")[-1].rstrip("0")) if "." in str(f) else 0
        for f in float_series
    )
    return max(decimals)


def ensure_numeric_consistency(column_numbers, bboxes_y_centers):
    min_power = minimal_power_of_10(column_numbers)
    multiplier = 10**min_power

    diffs = np.diff(np.array(column_numbers) * multiplier).astype(int)
    counts = np.bincount(abs(diffs))
    vals = np.nonzero(counts)[0]
    counts = counts[vals]
    is_linear = counts.size < diffs.size

    # ensure numbers are in descending order
    if not (diffs < 0).all():
        if is_linear:
            base = vals[np.argmax(counts)]
            column_numbers = [
                column_numbers[0] - i * base / multiplier
                for i in range(len(column_numbers))
            ]

    missing_points = find_missing_points(bboxes_y_centers)
    if missing_points:
        assigned = assign_numbers_to_missing_points(
            bboxes_y_centers, column_numbers, missing_points
        )
        for mp, val in zip(missing_points, assigned):
            if val is not None:
                bboxes_y_centers = np.append(bboxes_y_centers, mp)
                column_numbers.append(val)

        argsort = np.argsort(bboxes_y_centers)
        bboxes_y_centers = bboxes_y_centers[argsort]
        column_numbers = np.array(column_numbers)[argsort]

    return list(column_numbers), bboxes_y_centers


def find_missing_points(arr):
    bboxes_y_centers = np.sort(np.array(arr))
    diffs = np.diff(bboxes_y_centers)
    # Use the most common diff as the expected step
    step = np.bincount(diffs.astype(int)).argmax()
    missing_points = []
    for i, d in enumerate(diffs):
        if d > step * 1.5:  # Allow some tolerance
            num_missing = int(round(d / step)) - 1
            for j in range(num_missing):
                missing = bboxes_y_centers[i] + (j + 1) * step
                missing_points.append(missing)
    return missing_points


def assign_numbers_to_missing_points(bboxes_y_centers, column_numbers, missing_points):
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
