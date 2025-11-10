import cv2
import numpy as np


def cut_chart_area(
    img: np.ndarray,
    rows_bboxes: list,
    columns_bboxes: list,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    0. Cut chart area - stage 1: locate x and y axes to exclude them from chart area
    1. Cut chart area - stage 2: cut empty edges
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

    # 2. Cut empty edges
    cut_area = img[y1:y2, x1:x2]
    sum_over_x = cut_area.sum(axis=1)
    sum_over_y = cut_area.sum(axis=0)

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

    chart_area = img[y1:y2, x1:x2]
    area_loc = (x1, y1, x2, y2)

    return chart_area, area_loc


def find_largest_empty_rectangle(img_shape, bboxes):
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


def get_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
    )

    if lines is not None:
        return np.array([line[0] for line in lines])

    return []


def find_largest_rectangle(img):
    # Convert to grayscale and threshold to binary
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


def get_column_bboxes(bboxes: list, x_overlap_thresh: float = 0.7):
    """
    Group bounding boxes into columns based on horizontal overlap.

    :param bboxes:
    :param x_overlap_thresh:
    :return:
    """
    ids, columns = [], []
    used = set()
    for i, box in enumerate(bboxes):
        if i in used:
            continue
        col, col_ids = [box], [i]
        used.add(i)
        for j, other in enumerate(bboxes):
            if j in used:
                continue
            # Compute horizontal overlap
            left = max(box[0], other[0])
            right = min(box[2], other[2])
            overlap = max(0, right - left)
            width = min(box[2] - box[0], other[2] - other[0])
            if width > 0 and overlap / width > x_overlap_thresh:
                col.append(other)
                col_ids.append(j)
                used.add(j)
        if len(col) > 1:
            columns.append(col)
            ids.append(col_ids)
    return ids, columns


def get_row_bboxes(bboxes: list, y_overlap_thresh: float = 0.7):
    """
    Group bounding boxes into rows based on vertical overlap.

    :param bboxes:
    :param y_overlap_thresh:
    :return:
    """
    ids, rows = [], []
    used = set()
    for i, box in enumerate(bboxes):
        if i in used:
            continue
        row, row_ids = [box], [i]
        used.add(i)
        for j, other in enumerate(bboxes):
            if j in used:
                continue
            # Compute vertical overlap
            top = max(box[1], other[1])
            bottom = min(box[3], other[3])
            overlap = max(0, bottom - top)
            height = min(box[3] - box[1], other[3] - other[1])
            if height > 0 and overlap / height > y_overlap_thresh:
                row.append(other)
                row_ids.append(j)
                used.add(j)
        if len(row) > 1:
            rows.append(row)
            ids.append(row_ids)
    return ids, rows
