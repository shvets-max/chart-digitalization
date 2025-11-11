import os

import cv2
import numpy as np

from geometry import cluster_data, cut_chart_area, get_column_bboxes, get_row_bboxes
from ocr_utils import ocr, texts_to_datetimes, texts_to_numbers
from scale import create_x_scale, create_y_scale


def adjust_knots_to_grid(knots, grid_centers, min_dist=1, max_dist=10):
    """
    Adjusts each knot to the closest grid center if the distance is within
    (min_dist, max_dist].
    Returns a new numpy array of adjusted knots.
    """
    knots = np.copy(knots)
    for i in range(len(knots)):
        diffs = [abs(knots[i] - g) for g in grid_centers]
        closest_grid = grid_centers[np.argmin(diffs)]
        if min_dist < abs(knots[i] - closest_grid) <= max_dist:
            knots[i] = closest_grid
    return knots


def fill_gaps_in_time_series(time_series, window_size=5):
    """
    Fill gaps (None values) in the time_series by averaging the nearest previous and
    next non-None values within a window.
    Modifies the time_series in place.
    """
    for x in range(window_size, len(time_series) - window_size):
        if time_series[x][1][0] is None:
            prev_vals = [v[1][0] for v in time_series[x - window_size : x - 1]]
            next_vals = [v[1][0] for v in time_series[x + 1 : x + window_size]]
            prev_val = next(
                (val for val in reversed(prev_vals) if val is not None), None
            )
            next_val = next((val for val in next_vals if val is not None), None)
            if prev_val is not None and next_val is not None:
                time_series[x] = (time_series[x][0], [(prev_val + next_val) / 2])
    return time_series


def extract_time_series(image_path):
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texts, bboxes = ocr(gray)

    # Threshold to get the line (assuming black line on white background)
    thresh = (gray < 250).astype(np.uint8)

    # Get Y-axis components
    ids, columns_bboxes = get_column_bboxes(bboxes)
    max_len_id = np.argmax([len(id_group) for id_group in ids]) if ids else 0
    ids, columns_bboxes = ids[max_len_id], columns_bboxes[max_len_id]
    column_texts = [texts[i] for i in ids]
    column_numbers = texts_to_numbers(column_texts)

    # Get X-axis components
    ids, rows_bboxes = get_row_bboxes(bboxes)
    max_len_id = np.argmax([len(id_group) for id_group in ids]) if ids else 0
    ids, rows_bboxes = ids[max_len_id], rows_bboxes[max_len_id]
    rows_texts = [texts[i] for i in ids]
    row_index = texts_to_datetimes(rows_texts)

    cut_area, location, grid_l = cut_chart_area(thresh, rows_bboxes, columns_bboxes)
    x_offset, y_offset, x2, y2 = location

    # reconstruct grid components
    grid_y_component_map = cut_area.mean(axis=1) > 0.5
    grid_x_component_map = cut_area.mean(axis=0) > 0.5

    grid_x_component = np.nonzero(grid_x_component_map)[0]
    grid_y_component = np.nonzero(grid_y_component_map)[0]

    # Adjust knots to create scales
    y_knots = np.array([(box[1] + box[3]) / 2 for box in columns_bboxes])
    x_knots = np.array([(box[2] + box[0]) / 2 for box in rows_bboxes])

    grid_y_component_clusters = cluster_data(grid_y_component + y_offset, margin=5)
    grid_x_component_clusters = cluster_data(grid_x_component + x_offset, margin=5)

    grid_y_component_clusters_centers = [
        round(np.mean(cluster)) for cluster in grid_y_component_clusters
    ]
    grid_x_component_clusters_centers = [
        round(np.mean(cluster)) for cluster in grid_x_component_clusters
    ]

    # find the closest grid line to each knot and adjust
    y_knots = adjust_knots_to_grid(y_knots, grid_y_component_clusters_centers)
    x_knots = adjust_knots_to_grid(x_knots, grid_x_component_clusters_centers)

    y_scale = create_y_scale(column_numbers, y_knots)
    x_scale = create_x_scale(row_index, x_knots)

    # Remove grid lines from chart area
    chart_area = thresh[y_offset:y2, x_offset:x2]
    chart_area[grid_y_component, :] = 0
    chart_area[:, grid_x_component] = 0

    # Find the y-coordinate of the line for each x
    time_series = []
    height, width = chart_area.shape
    allowed_margin = 5

    time_series = extract_time_series_from_chart_area(
        chart_area,
        x_scale,
        y_scale,
        grid_x_component,
        grid_y_component_map,
        grid_l,
        x_offset,
        y_offset,
        allowed_margin=allowed_margin,
        reversed=False,
    )
    time_series = fill_gaps_in_time_series(time_series, window_size=5)
    return time_series


def extract_time_series_from_chart_area(
    chart_area,
    x_scale,
    y_scale,
    grid_x_component,
    grid_y_component_map,
    grid_l,
    x_offset,
    y_offset,
    allowed_margin=5,
    reversed=False,
):
    time_series = []
    height, width = chart_area.shape
    x_range = range(width - 1, -1, -1) if reversed else range(width)

    for x in x_range:
        x_date = x_scale(x + x_offset - grid_l)
        if x in grid_x_component:
            time_series.append((x_date, [None]))
            continue

        ys = np.nonzero(chart_area[~grid_y_component_map, x])[0]
        if ys.size > 0:
            unique_diffs = np.unique(np.diff(ys))
            if unique_diffs.size > 1 and any(
                d > allowed_margin for d in unique_diffs[1:]
            ):
                recent_points = [
                    pt
                    for _, pt_list in time_series[-5:]
                    for pt in pt_list
                    if pt is not None
                ]
                if recent_points:
                    clusters = cluster_data(ys, allowed_margin)
                    inverted = y_scale.invert(np.mean(recent_points)) - y_offset
                    if clusters:
                        closest_cluster = min(
                            clusters, key=lambda c: abs(np.mean(c) - inverted)
                        )
                        y = np.mean(closest_cluster)
                    else:
                        y = np.mean(ys)
                else:
                    y = np.mean(ys)
            else:
                y = np.mean(ys)

            scaled = y_scale(y + y_offset)
            time_series.append((x_date, [scaled]))
        else:
            time_series.append((x_date, [None]))  # No data at this x

    return time_series if not reversed else time_series[::-1]


# from PIL import Image
# im = Image.fromarray(c2*255)
# im.show()
