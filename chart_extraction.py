import os

import cv2
import numpy as np

from geometry import cut_chart_area, get_column_bboxes, get_row_bboxes
from ocr_utils import ocr, texts_to_datetimes, texts_to_numbers
from scale import create_x_scale, create_y_scale

offset = 3
EDGE_KERNEL = (
    np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=np.float32)
) / 12.0


def cluster_data(points, margin):
    points = sorted(points)
    clusters = []
    current_cluster = [points[0]]
    for p in points[1:]:
        if p - current_cluster[-1] <= margin:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]
    clusters.append(current_cluster)
    return clusters


def extract_time_series(image_path):
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = cv2.imread(image_path)
    img_area = img.shape[0] * img.shape[1]
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

    # mask out the text areas in the chart area
    for bbox in bboxes:
        left, top, right, bottom = bbox
        if abs(right - left) * (bottom - top) > img_area * 0.025:
            continue
        thresh[top:bottom, left:right] = 0

    chart_area, location = cut_chart_area(thresh, rows_bboxes, columns_bboxes)
    x_offset, y_offset, _, _ = location

    # Grid detection
    grid_y_component_map = chart_area.mean(axis=1) > 0.5
    grid_x_component_map = chart_area.mean(axis=0) > 0.5

    grid_x_component = np.nonzero(grid_x_component_map)[0]
    grid_y_component = np.nonzero(grid_y_component_map)[0]

    # cut edges of grid lines
    grid_l, grid_r = grid_x_component[0], grid_x_component[-1]
    grid_t, grid_b = grid_y_component[0], grid_y_component[-1]
    if grid_l < 50:
        grid_x_component_map[:grid_l] = True
    if chart_area.shape[1] - grid_r < 50:
        grid_x_component_map[grid_r:] = True
    if grid_t < 50:
        grid_y_component_map[:grid_t] = True
    if chart_area.shape[0] - grid_b < 50:
        grid_y_component_map[grid_b:] = True

    # reconstruct grid components
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
    y_knots_copy = np.copy(y_knots)
    for i in range(len(y_knots_copy)):
        diffs = [abs(y_knots_copy[i] - g) for g in grid_y_component_clusters_centers]
        closest_grid = grid_y_component_clusters_centers[np.argmin(diffs)]
        if 0 < abs(y_knots_copy[i] - closest_grid) <= 5:
            y_knots_copy[i] = closest_grid
    y_knots = y_knots_copy

    x_knots_copy = np.copy(x_knots)
    for i in range(len(x_knots_copy)):
        diffs = [abs(x_knots_copy[i] - g) for g in grid_x_component_clusters_centers]
        closest_grid = grid_x_component_clusters_centers[np.argmin(diffs)]
        if 0 < abs(x_knots_copy[i] - closest_grid) <= 5:
            x_knots_copy[i] = closest_grid
    x_knots = x_knots_copy

    y_scale = create_y_scale(column_numbers, y_knots)
    x_scale = create_x_scale(row_index, x_knots)

    # Remove grid lines from chart area
    chart_area_copy = chart_area.copy()
    chart_area_copy[grid_y_component, :] = 0
    chart_area_copy[:, grid_x_component] = 0
    edges = cv2.filter2D(chart_area_copy, -1, EDGE_KERNEL)

    # Find the y-coordinate of the line for each x
    time_series = []
    height, width = edges.shape
    allowed_margin = 5

    for x in range(width):
        if x in grid_x_component:
            continue
        x_date = x_scale(x + x_offset - grid_l)
        ys = np.nonzero(edges[~grid_y_component_map, x])[0]
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

            scaled = y_scale(y + y_offset + grid_t)
            time_series.append((x_date, [scaled]))
        else:
            time_series.append((x_date, [None]))  # No data at this x
    return time_series


# from PIL import Image
# c2 = chart_area.copy()
# c2[grid_y_component, :] = 0
# c2[:, grid_x_component] = 0
# im = Image.fromarray(c2*255)
# im.show()
