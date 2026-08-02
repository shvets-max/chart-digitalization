import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from function import FunctionBase, Logarithmic
from geometry import cluster_data, cut_chart_area, get_column_bboxes, get_row_bboxes
from ocr_utils import ocr, texts_to_datetimes, texts_to_numbers
from scale import create_x_scale, create_y_scale, log_ticks, nice_ticks


def adjust_knots_to_grid(knots, grid_centers, min_dist=1, max_dist=10):
    """
    Adjusts each knot to the closest grid center if the distance is within
    (min_dist, max_dist].
    Returns a new numpy array of adjusted knots.
    """
    knots = np.copy(knots)
    if not len(grid_centers):
        return knots
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


@dataclass
class ChartExtraction:
    """
    Full result of digitalizing one chart image: the extracted series plus everything
    needed to draw it back on top of the original picture (pixel coordinates, the
    plotting area, detected grid lines and the fitted axis scales).
    """

    image_size: tuple[int, int]  # (width, height) of the source image
    chart_area: tuple[int, int, int, int]  # (x1, y1, x2, y2) of the plotting area
    time_series: list  # [(x_value, [y_value, ...]), ...]
    x_pixels: list[int]  # image x coordinate of every time_series entry
    detected_grid_x: list[int] = field(default_factory=list)  # vertical grid lines (px)
    detected_grid_y: list[int] = field(default_factory=list)  # horizontal grid lines
    x_scale: Optional[FunctionBase] = None
    y_scale: Optional[FunctionBase] = None
    x_pixel_offset: int = 0  # shift between image x and the x_scale domain
    x_is_datetime: bool = False
    y_is_log: bool = False

    def x_value_at(self, x_pixel: float):
        """Axis value (datetime or number) at an image x coordinate."""
        return self.x_scale(x_pixel - self.x_pixel_offset)

    def x_pixel_at(self, x_value) -> float:
        """Image x coordinate of an axis value."""
        return self.x_scale.invert(x_value) + self.x_pixel_offset

    def y_value_at(self, y_pixel: float) -> float:
        """Axis value at an image y coordinate."""
        return self.y_scale(y_pixel)

    def y_pixel_at(self, y_value: float) -> float:
        """Image y coordinate of an axis value."""
        return self.y_scale.invert(y_value)

    def y_pixels(self) -> list[list[Optional[float]]]:
        """Image y coordinate of every value; None where the series has a gap."""
        return [
            [None if v is None else self.y_pixel_at(v) for v in values]
            for _, values in self.time_series
        ]

    def value_range(self) -> tuple[float, float]:
        """(min, max) of all extracted values."""
        values = [v for _, vals in self.time_series for v in vals if v is not None]
        return (min(values), max(values)) if values else (0.0, 0.0)


def format_value(value, is_datetime: bool = False) -> str:
    """Short human-readable label for an axis value."""
    if value is None:
        return ""
    if is_datetime or isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    value = float(value)
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= threshold:
            return f"{value / threshold:.4g}{suffix}"
    return f"{value:.4g}"


def build_axis_ticks(
    extraction: "ChartExtraction",
    source: str = "generated",
    x_count: int = 8,
    y_count: int = 6,
) -> dict:
    """
    Tick positions for the overlay grid and numeric scale.

    `source` is either "detected" (grid lines found in the image) or "generated"
    (round values computed from the fitted scales). Each tick is
    {"pixel": float, "value": float | iso-string, "label": str}.
    """
    x1, y1, x2, y2 = extraction.chart_area
    x_ticks, y_ticks = [], []

    def x_tick(pixel):
        value = extraction.x_value_at(pixel)
        return {
            "pixel": float(pixel),
            "value": value.isoformat() if isinstance(value, datetime) else value,
            "label": format_value(value, extraction.x_is_datetime),
        }

    def y_tick(pixel, value=None):
        value = extraction.y_value_at(pixel) if value is None else value
        return {"pixel": float(pixel), "value": value, "label": format_value(value)}

    if source == "detected":
        if extraction.x_scale is not None:
            x_ticks = [x_tick(px) for px in extraction.detected_grid_x]
        if extraction.y_scale is not None:
            y_ticks = [y_tick(py) for py in extraction.detected_grid_y]
        return {"x": x_ticks, "y": y_ticks}

    if extraction.x_scale is not None and x_count >= 2:
        x_ticks = [x_tick(round(px)) for px in np.linspace(x1, x2, x_count)]

    if extraction.y_scale is not None and y_count >= 2:
        bottom, top = extraction.y_value_at(y2), extraction.y_value_at(y1)
        ticker = log_ticks if extraction.y_is_log else nice_ticks
        for value in ticker(min(bottom, top), max(bottom, top), y_count):
            pixel = extraction.y_pixel_at(value)
            if y1 <= pixel <= y2:
                y_ticks.append(y_tick(round(pixel), value))

    return {"x": x_ticks, "y": y_ticks}


def extract_chart(image_path) -> ChartExtraction:
    """Digitalize a chart image: the series plus the pixel geometry to draw it."""
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

    # Drop axis labels OCR could not parse into a number, keeping bboxes aligned
    valid = [n is not None for n in column_numbers]
    columns_bboxes = [box for box, ok in zip(columns_bboxes, valid) if ok]
    column_numbers = [n for n, ok in zip(column_numbers, valid) if ok]

    # Get X-axis components
    ids, rows_bboxes = get_row_bboxes(bboxes)
    max_len_id = np.argmax([len(id_group) for id_group in ids]) if ids else 0
    ids, rows_bboxes = ids[max_len_id], rows_bboxes[max_len_id]
    rows_texts = [texts[i] for i in ids]
    row_index = texts_to_datetimes(rows_texts)

    # Drop axis labels OCR could not parse into a date, keeping bboxes aligned
    valid = [dt is not None for dt in row_index]
    rows_bboxes = [box for box, ok in zip(rows_bboxes, valid) if ok]
    row_index = [dt for dt, ok in zip(row_index, valid) if ok]

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
    time_series = extract_time_series_from_chart_area(
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
    )
    time_series = fill_gaps_in_time_series(time_series, window_size=5)

    return ChartExtraction(
        image_size=(int(img.shape[1]), int(img.shape[0])),
        chart_area=(int(x_offset), int(y_offset), int(x2), int(y2)),
        time_series=time_series,
        x_pixels=[int(x_offset) + x for x in range(len(time_series))],
        detected_grid_x=[int(px) for px in grid_x_component_clusters_centers],
        detected_grid_y=[int(py) for py in grid_y_component_clusters_centers],
        x_scale=x_scale,
        y_scale=y_scale,
        x_pixel_offset=int(grid_l),
        x_is_datetime=bool(time_series) and isinstance(time_series[0][0], datetime),
        y_is_log=isinstance(y_scale, Logarithmic),
    )


def extract_time_series(image_path):
    """Digitalize a chart image into [(x_value, [y_value, ...]), ...]."""
    return extract_chart(image_path).time_series


def ink_rows_in_column(chart_area, grid_y_component_map, rows_kept, x):
    """Absolute image rows of the non-grid ink in column `x` of the chart area."""
    return rows_kept[np.nonzero(chart_area[~grid_y_component_map, x])[0]]


def column_candidate_rows(
    chart_area, grid_y_component_map, rows_kept, grid_x_lookup, allowed_margin
):
    """
    Candidate series rows per chart-area column: the mean row of each ink cluster.
    Columns on a vertical grid line, or with no ink, get an empty list. More than
    one candidate means something other than the series is drawn in that column.
    """
    candidates = []
    for x in range(chart_area.shape[1]):
        if x in grid_x_lookup:
            candidates.append([])
            continue
        ys = ink_rows_in_column(chart_area, grid_y_component_map, rows_kept, x)
        if ys.size == 0:
            candidates.append([])
            continue
        candidates.append(
            [float(np.mean(cluster)) for cluster in cluster_data(ys, allowed_margin)]
        )
    return candidates


def _longest_unambiguous_run(candidates) -> Optional[int]:
    """Index at the middle of the longest run of columns holding a single candidate."""
    best_length = best_end = run = 0
    for i, rows in enumerate(candidates):
        run = run + 1 if len(rows) == 1 else 0
        if run > best_length:
            best_length, best_end = run, i
    if best_length == 0:
        return None
    return best_end - best_length // 2


def resolve_series_rows(candidates, max_step: float) -> list[Optional[float]]:
    """
    Pick one row per column, following the series through columns that also contain
    other ink.

    A line series is the only ink spanning the full width, so the longest run of
    unambiguous columns is taken as the series and resolution spreads outwards from
    it, each column keeping the candidate closest to the last accepted row. Starting
    from the left instead would let a legend drawn before the series begins capture
    the whole traversal. Candidates further than `max_step` from the running row are
    treated as foreign ink and yield None.
    """
    resolved: list[Optional[float]] = [None] * len(candidates)
    seed = _longest_unambiguous_run(candidates)
    if seed is None:
        # Every column is ambiguous, so there is no series to lock on to.
        return [rows[0] if len(rows) == 1 else None for rows in candidates]

    resolved[seed] = candidates[seed][0]
    for indices in (range(seed + 1, len(candidates)), range(seed - 1, -1, -1)):
        last = resolved[seed]
        for i in indices:
            rows = candidates[i]
            if not rows:
                continue
            closest = min(rows, key=lambda row: abs(row - last))
            if abs(closest - last) <= max_step:
                resolved[i] = closest
                last = closest
    return resolved


def estimate_max_step(candidates, height: int) -> float:
    """
    Largest plausible row change between neighbouring columns, from how fast the
    series actually moves where it is unambiguous. Keeps the interference cut-off
    adaptive instead of a fixed pixel budget.
    """
    unambiguous = [rows[0] if len(rows) == 1 else None for rows in candidates]
    steps = [
        abs(b - a)
        for a, b in zip(unambiguous, unambiguous[1:])
        if a is not None and b is not None
    ]
    if not steps:
        return height / 2
    return max(float(np.percentile(steps, 99)) * 4, 8.0)


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
    height, width = chart_area.shape

    # Masking out grid rows renumbers what is left, so keep a lookup from the
    # masked column's index back to the absolute image row.
    rows_kept = np.nonzero(~grid_y_component_map)[0] + y_offset
    grid_x_lookup = set(np.asarray(grid_x_component).tolist())

    candidates = column_candidate_rows(
        chart_area, grid_y_component_map, rows_kept, grid_x_lookup, allowed_margin
    )
    rows = resolve_series_rows(candidates, estimate_max_step(candidates, height))

    time_series = [
        (
            x_scale(x + x_offset - grid_l),
            [None if rows[x] is None else y_scale(rows[x])],
        )
        for x in range(width)
    ]
    return time_series[::-1] if reversed else time_series


# from PIL import Image
# im = Image.fromarray(c2*255)
# im.show()
