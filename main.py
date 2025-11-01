import cv2
import matplotlib.pyplot as plt
import numpy as np

from geometry import (
    find_largest_empty_rectangle,
    find_largest_rectangle,
    get_column_bboxes,
    get_row_bboxes,
)
from ocr_utils import ocr, texts_to_datetimes, texts_to_numbers
from scale import create_x_scale, create_y_scale


def extract_time_series(image_path):
    # Load image
    img = cv2.imread(image_path)
    texts, bboxes = ocr(img)

    ids, columns_bboxes = get_column_bboxes(bboxes)
    max_len_id = np.argmax([len(id_group) for id_group in ids]) if ids else 0
    ids, columns_bboxes = ids[max_len_id], columns_bboxes[max_len_id]
    column_texts = [texts[i] for i in ids]
    column_numbers = texts_to_numbers(column_texts)
    y_scale = create_y_scale(column_numbers, columns_bboxes)

    ids, rows_bboxes = get_row_bboxes(bboxes)
    max_len_id = np.argmax([len(id_group) for id_group in ids]) if ids else 0
    ids, rows_bboxes = ids[max_len_id], rows_bboxes[max_len_id]
    rows_texts = [texts[i] for i in ids]
    row_index = texts_to_datetimes(rows_texts)
    x_scale = create_x_scale(row_index, rows_bboxes)

    offset = (0, 0)

    # get chart area
    x, y, w, h = find_largest_empty_rectangle(img.shape, bboxes)
    chart_area = img[y : y + h, x : x + w]
    offset = (x, y)

    x, y, w, h = find_largest_rectangle(chart_area)
    chart_area = chart_area[y : y + h, x : x + w]
    offset = (offset[0] + x, offset[1] + y)

    gray = cv2.cvtColor(chart_area, cv2.COLOR_BGR2GRAY)
    # Threshold to get the line (assuming black line on white background)
    thresh = gray < 250

    # Find the y-coordinate of the line for each x
    time_series = []
    height, width = thresh.shape

    h_lines_map = thresh.mean(axis=1) > 0.95
    for x in range(width):
        x_date = x_scale(x + offset[0])
        ys = np.nonzero(thresh[:, x] & ~h_lines_map)[0]
        if len(ys) > 0:
            y = int(np.mean(ys))
            scaled = y_scale(y + offset[1])
            time_series.append((x_date, scaled))
        else:
            time_series.append(None)  # No data at this x
    return time_series


if __name__ == "__main__":
    ts = extract_time_series("data/img_8.png")
    # plot time series
    dates = [t[0] for t in ts if t is not None]
    values = [t[1] for t in ts if t is not None]
    plt.plot(dates, values)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Extracted Time Series")
    plt.show()
