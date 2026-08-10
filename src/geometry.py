from collections.abc import Sequence

import cv2
import numpy as np


def cluster_data(points: Sequence[float], margin: float) -> list[list[float]]:
    """Group sorted `points` into clusters where consecutive points differ by at most `margin`."""
    points = sorted(points)
    if not points:
        return []
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


def cut_chart_area(
    img: np.ndarray,
    rows_bboxes: list,
    columns_bboxes: list,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    0. Cut chart area - stage 1: locate x and y axes to exclude them from chart area
    1. Cut chart area - stage 2: cut empty edges

    `area_loc`'s x1 is the chart area's left edge in the ORIGINAL image's pixel
    coordinates: since axis-tick pixel positions (used to fit the scales) are also
    read from the original image, column `i` of the returned `chart_area` always
    corresponds to absolute image column `x1 + i` -- no further offset needed.
    :param img:
    :param rows_bboxes:
    :param columns_bboxes:
    :return: chart_area, area_loc (x1, y1, x2, y2)
    """
    h, w = img.shape

    # 1. Locate x and y axes to exclude them from chart area
    y1_min = np.min([b[1] for b in rows_bboxes])
    y2_max = np.max([b[3] for b in rows_bboxes])
    x1_min = np.min([b[0] for b in columns_bboxes])
    x2_max = np.max([b[2] for b in columns_bboxes])

    is_bottom = y2_max > h * 0.5
    is_right = x2_max > w * 0.5

    if is_bottom:
        y = 0
        h = y1_min
    else:
        y = h - (h - y2_max)
        h = h - y

    if is_right:
        x = 0
        w = x1_min
    else:
        x = w - (w - x2_max)
        w = w - x

    x1, y1, x2, y2 = x, y, x + w, y + h
    cut_area_1 = img[y1:y2, x1:x2]

    # 2. Cut empty edges
    sum_over_x = cut_area_1.sum(axis=1)
    sum_over_y = cut_area_1.sum(axis=0)

    non_zero_xs = np.nonzero(sum_over_x)[0]
    non_zero_ys = np.nonzero(sum_over_y)[0]

    non_zero_x_range = (non_zero_xs[0], non_zero_xs[-1])
    non_zero_y_range = (non_zero_ys[0], non_zero_ys[-1])

    new_x1 = non_zero_y_range[0]
    new_y1 = non_zero_x_range[0]
    new_w = non_zero_y_range[1] - non_zero_y_range[0]
    new_h = non_zero_x_range[1] - non_zero_x_range[0]

    y1 += new_y1
    x1 += new_x1
    x2 = x1 + new_w
    y2 = y1 + new_h

    cut_area_2 = img[y1:y2, x1:x2]

    # 3. Grid-edges cut
    grid_y_component_map = cut_area_2.mean(axis=1) > 0.5
    grid_x_component_map = cut_area_2.mean(axis=0) > 0.5

    grid_x_component = np.nonzero(grid_x_component_map)[0]
    grid_y_component = np.nonzero(grid_y_component_map)[0]

    grid_x_component_clusters = cluster_data(grid_x_component, margin=5)
    grid_y_component_clusters = cluster_data(grid_y_component, margin=5)

    # cut edges of grid lines
    grid_l = max(grid_x_component_clusters[0])
    grid_r = min(grid_x_component_clusters[-1])
    grid_t = max(grid_y_component_clusters[0])
    grid_b = min(grid_y_component_clusters[-1])

    if grid_l < 50:
        x1 += grid_l
    if cut_area_2.shape[1] - grid_r < 50:
        x2 -= cut_area_2.shape[1] - grid_r
    if grid_t < 50:
        y1 += grid_t
    if cut_area_2.shape[0] - grid_b < 50:
        y2 -= cut_area_2.shape[0] - grid_b

    # update chart area
    chart_area = img[y1:y2, x1:x2]
    area_loc = (x1, y1, x2, y2)

    return chart_area, area_loc


def find_largest_empty_rectangle(
    img_shape: tuple[int, int], bboxes: list[tuple[int, int, int, int]]
) -> tuple[int, int, int, int]:
    """Largest axis-aligned rectangle not overlapping any of `bboxes`, as (x, y, w, h)."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for left, top, right, bottom in bboxes:
        cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
    inv_mask = cv2.bitwise_not(mask)
    binary = (inv_mask > 0).astype(np.uint8)

    # Dynamic programming to find largest rectangle of 1s
    height, width = binary.shape
    dp = np.zeros((height, width), dtype=int)
    max_area = 0
    max_rect = (0, 0, 0, 0)  # x, y, w, h

    for i in range(height):
        for j in range(width):
            if binary[i, j]:
                dp[i, j] = dp[i - 1, j] + 1 if i > 0 else 1
            else:
                dp[i, j] = 0

        # For each row, use histogram approach to find max rectangle
        stack = []
        j = 0
        while j <= width:
            cur_height = dp[i, j] if j < width else 0
            if not stack or cur_height >= dp[i, stack[-1]]:
                stack.append(j)
                j += 1
            else:
                h = dp[i, stack.pop()]
                w = j if not stack else j - stack[-1] - 1
                area = h * w
                if area > max_area:
                    max_area = area
                    x = stack[-1] + 1 if stack else 0
                    y = i - h + 1
                    max_rect = (x, y, w, h)
    return max_rect  # (x, y, w, h)


def get_lines(img: np.ndarray) -> np.ndarray:
    """Straight line segments detected in `img` via Canny edges + probabilistic Hough transform."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
    )

    if lines is not None:
        return np.array([line[0] for line in lines])

    return []


def find_largest_rectangle(img: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding rectangle of the chart's near-full-width/height axis lines, as (x, y, w, h)."""
    lines = get_lines(img)
    h_lines = lines[lines[:, 1] == lines[:, 3]]
    v_lines = lines[lines[:, 0] == lines[:, 2]]

    h_lengths = abs(h_lines[:, 2] - h_lines[:, 0])
    v_lengths = abs(v_lines[:, 3] - v_lines[:, 1])

    h_lines = h_lines[h_lengths / img.shape[1] > 0.8, :]
    v_lines = v_lines[v_lengths / img.shape[0] > 0.8, :]

    x1 = min(v_lines[:, 0], default=0)
    x2 = max(v_lines[:, 0], default=img.shape[1])
    y1 = min(h_lines[:, 1], default=0)
    y2 = max(h_lines[:, 1], default=img.shape[0])

    if x1 > 0.1 * img.shape[1]:
        x1 = 0
    if x2 < 0.9 * img.shape[1]:
        x2 = img.shape[1]
    if y1 > 0.1 * img.shape[0]:
        y1 = 0
    if y2 < 0.9 * img.shape[0]:
        y2 = img.shape[0]

    return x1, y1, x2 - x1, y2 - y1  # (x, y, w, h)


def _group_bboxes_by_overlap(
    bboxes: list, axis_start: int, axis_end: int, overlap_thresh: float
) -> tuple[list[list[int]], list[list]]:
    """
    Group bounding boxes sharing at least `overlap_thresh` overlap along one axis.

    :param bboxes: list of (left, top, right, bottom) boxes
    :param axis_start: index of the axis' start coordinate (0 for x, 1 for y)
    :param axis_end: index of the axis' end coordinate (2 for x, 3 for y)
    :param overlap_thresh: minimum fraction of the smaller box's span that must overlap
    :return: (index groups, box groups), one entry per group of 2+ overlapping boxes
    """
    ids, groups = [], []
    used = set()
    for i, box in enumerate(bboxes):
        if i in used:
            continue
        group, group_ids = [box], [i]
        used.add(i)
        for j, other in enumerate(bboxes):
            if j in used:
                continue
            start = max(box[axis_start], other[axis_start])
            end = min(box[axis_end], other[axis_end])
            overlap = max(0, end - start)
            span = min(
                box[axis_end] - box[axis_start], other[axis_end] - other[axis_start]
            )
            if span > 0 and overlap / span > overlap_thresh:
                group.append(other)
                group_ids.append(j)
                used.add(j)
        if len(group) > 1:
            groups.append(group)
            ids.append(group_ids)
    return ids, groups


def get_column_bboxes(
    bboxes: list, x_overlap_thresh: float = 0.7
) -> tuple[list[list[int]], list[list]]:
    """Group bounding boxes into columns based on horizontal overlap."""
    return _group_bboxes_by_overlap(bboxes, 0, 2, x_overlap_thresh)


def get_row_bboxes(
    bboxes: list, y_overlap_thresh: float = 0.7
) -> tuple[list[list[int]], list[list]]:
    """Group bounding boxes into rows based on vertical overlap."""
    return _group_bboxes_by_overlap(bboxes, 1, 3, y_overlap_thresh)
