import os

import cv2
import numpy as np

from geometry import cut_chart_area, get_column_bboxes, get_row_bboxes
from ocr_utils import ocr, texts_to_datetimes, texts_to_numbers
from scale import create_x_scale, create_y_scale


def extract_time_series(image_path):
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texts, bboxes = ocr(gray)

    # Threshold to get the line (assuming black line on white background)
    thresh = (gray < 250).astype(np.uint8)

    # Get Y-scale
    ids, columns_bboxes = get_column_bboxes(bboxes)
    max_len_id = np.argmax([len(id_group) for id_group in ids]) if ids else 0
    ids, columns_bboxes = ids[max_len_id], columns_bboxes[max_len_id]
    column_texts = [texts[i] for i in ids]
    column_numbers = texts_to_numbers(column_texts)
    y_scale = create_y_scale(column_numbers, columns_bboxes)

    # Get X-scale
    ids, rows_bboxes = get_row_bboxes(bboxes)
    max_len_id = np.argmax([len(id_group) for id_group in ids]) if ids else 0
    ids, rows_bboxes = ids[max_len_id], rows_bboxes[max_len_id]
    rows_texts = [texts[i] for i in ids]
    row_index = texts_to_datetimes(rows_texts)
    x_scale = create_x_scale(row_index, rows_bboxes)

    chart_area, offset = cut_chart_area(thresh, rows_bboxes, columns_bboxes)

    # Grid detection
    grid_y_component_map = chart_area.mean(axis=1) > 0.5
    grid_x_component = np.nonzero(chart_area.mean(axis=0) > 0.5)[0]

    # Find the y-coordinate of the line for each x
    time_series = []
    height, width = chart_area.shape

    for x in range(width):
        if x in grid_x_component:
            continue
        x_date = x_scale(x + offset[0])
        ys = np.nonzero(chart_area[~grid_y_component_map, x])[0]
        if ys.size > 0:
            y = np.mean(ys)
            scaled = y_scale(y + offset[1])
            time_series.append((x_date, [scaled]))
        else:
            time_series.append((x_date, [None]))  # No data at this x
    return time_series
