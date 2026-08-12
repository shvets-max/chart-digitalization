import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from src.function import FunctionBase, Logarithmic
from src.geometry import cluster_data, cut_chart_area, get_column_bboxes, get_row_bboxes
from src.multiline import (
    cluster_ink_colors,
    default_legend_search_area,
    find_legend_entries,
    match_series_to_legend,
)
from src.ocr_utils import ocr, texts_to_datetimes, texts_to_numbers
from src.scale import (
    create_x_scale,
    create_y_scale,
    drop_monotonicity_outliers,
    log_ticks,
    nice_ticks,
)

# Series-cluster selection (see select_series_clusters)
DEFAULT_MIN_RESOLVED_FRACTION = (
    0.5  # of columns a color must resolve to count as a series
)
DEFAULT_MIN_CANDIDATE_COLUMN_COVERAGE = 0.15  # loose pre-filter before resolving
NEAR_GRAY_SATURATION_THRESHOLD = 25  # max-min channel spread below this is "gray"
NEAR_GRAY_MIN_BRIGHTNESS = 150  # mean channel above this is "light" (a real black
# line is dark; a gridline's anti-aliased edge is desaturated AND light)

# Duplicate-trajectory merging (see _is_same_trajectory)
DEFAULT_DUPLICATE_MEDIAN_DISTANCE = 12.0  # px: trajectories this close are one line
DEFAULT_DUPLICATE_MIN_OVERLAP_FRACTION = 0.15  # of the SMALLER trajectory's own
# resolved columns, to trust the comparison at all
DEFAULT_DUPLICATE_MIN_OVERLAP_SPREAD = 0.3  # of width: shared columns must not all
# sit in one narrow stretch
MIN_DUPLICATE_OVERLAP_COLUMNS = 15  # floor so a handful of columns can't pass on
# fraction alone

# Disjoint-shade merging (see _is_disjoint_shade) -- second path for a candidate
# that _is_same_trajectory rejects for lack of shared columns, not disagreement
# (docs/series-gaps-diagnosis.md)
DEFAULT_DISJOINT_MAX_OVERLAP_FRACTION = 0.1  # of the SMALLER trajectory's own
# resolved columns: above this there's enough shared evidence that
# _is_same_trajectory's distance/spread judgement should be trusted instead, so
# this path only ever fires where that one is structurally unable to decide
DEFAULT_DISJOINT_COLOR_MERGE_DISTANCE = 65.0  # looser than cluster_ink_colors'
# own discovery-time merge_distance (40): a complementary shade rejected by
# discovery for sitting just past that bound is still the case this path exists
# to catch


def select_axis_tick_group(
    ids: list[list[int]], texts: list[str], parser: Callable[[list[str]], list]
) -> int:
    """
    Index of the OCR bbox group most likely to be the axis' tick labels: the one
    with the most entries `parser` (texts_to_numbers or texts_to_datetimes) can
    actually parse, tie-broken by group size.

    Picking by raw group size alone is ambiguous whenever another group -- e.g.
    legend text, wrapped title lines -- happens to have the same number of
    entries; a real tick-label group is defined by parsing as numbers or dates,
    not merely by how many OCR tokens it contains.
    """
    if not ids:
        return 0
    scores = []
    for id_group in ids:
        try:
            parsed = parser([texts[i] for i in id_group])
            valid_count = sum(v is not None for v in parsed)
        except Exception:
            # `parser` is only designed for the real tick-label group; probing
            # groups it was never meant to handle (legend text, titles) can throw.
            valid_count = 0
        scores.append((valid_count, len(id_group)))
    return max(range(len(ids)), key=lambda i: scores[i])


def adjust_knots_to_grid(
    knots: np.ndarray,
    grid_centers: list[float],
    min_dist: float = 1,
    max_dist: float = 10,
) -> np.ndarray:
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


def fill_gaps_in_time_series(time_series: list, window_size: int = 5) -> list:
    """
    Fill gaps (None values) in each series by averaging the nearest previous and
    next non-None values within a window.
    Modifies the time_series in place.
    """
    if not time_series:
        return time_series
    n_series = len(time_series[0][1])
    for series_idx in range(n_series):
        for x in range(window_size, len(time_series) - window_size):
            if time_series[x][1][series_idx] is None:
                prev_vals = [
                    v[1][series_idx] for v in time_series[x - window_size : x - 1]
                ]
                next_vals = [
                    v[1][series_idx] for v in time_series[x + 1 : x + window_size]
                ]
                prev_val = next(
                    (val for val in reversed(prev_vals) if val is not None), None
                )
                next_val = next((val for val in next_vals if val is not None), None)
                if prev_val is not None and next_val is not None:
                    time_series[x][1][series_idx] = (prev_val + next_val) / 2
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
    x_is_datetime: bool = False
    y_is_log: bool = False
    series_names: list = field(default_factory=list)  # legend name per series, or None
    legend_area: Optional[tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) actually
    # searched for a legend -- the user-highlighted area if one was given, else the
    # default top-left corner (see multiline.default_legend_search_area)

    def x_value_at(self, x_pixel: float):
        """Axis value (datetime or number) at an image x coordinate."""
        return self.x_scale(x_pixel)

    def x_pixel_at(self, x_value) -> float:
        """Image x coordinate of an axis value."""
        return self.x_scale.invert(x_value)

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


def extract_chart(
    image_path: str, legend_area: Optional[tuple[int, int, int, int]] = None
) -> ChartExtraction:
    """
    Digitalize a chart image: the series plus the pixel geometry to draw it.

    `legend_area` (left, top, right, bottom) restricts legend detection to a
    user-highlighted region, e.g. when the legend sits somewhere other than the
    default top-left corner. Omit it to search the default corner.
    """
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
    best_id = select_axis_tick_group(ids, texts, texts_to_numbers)
    ids, columns_bboxes = ids[best_id], columns_bboxes[best_id]
    column_texts = [texts[i] for i in ids]
    column_numbers = texts_to_numbers(column_texts)

    # Drop axis labels OCR could not parse into a number, keeping bboxes aligned
    valid = [n is not None for n in column_numbers]
    columns_bboxes = [box for box, ok in zip(columns_bboxes, valid) if ok]
    column_numbers = [n for n, ok in zip(column_numbers, valid) if ok]

    # Get X-axis components
    ids, rows_bboxes = get_row_bboxes(bboxes)
    best_id = select_axis_tick_group(ids, texts, texts_to_datetimes)
    ids, rows_bboxes = ids[best_id], rows_bboxes[best_id]
    rows_texts = [texts[i] for i in ids]
    row_index = texts_to_datetimes(rows_texts)

    # Drop axis labels OCR could not parse into a date, keeping bboxes aligned
    valid = [dt is not None for dt in row_index]
    rows_bboxes = [box for box, ok in zip(rows_bboxes, valid) if ok]
    row_index = [dt for dt, ok in zip(row_index, valid) if ok]

    cut_area, location = cut_chart_area(thresh, rows_bboxes, columns_bboxes)
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

    column_numbers, y_knots = drop_monotonicity_outliers(
        np.array(column_numbers, dtype=float), y_knots
    )
    y_scale = create_y_scale(column_numbers, y_knots)
    x_scale = create_x_scale(row_index, x_knots)

    # Remove grid lines from chart area
    chart_area = thresh[y_offset:y2, x_offset:x2]
    chart_area[grid_y_component, :] = 0
    chart_area[:, grid_x_component] = 0

    # A legend (if any) must be detected before color separation, not after: small
    # text is mostly anti-aliased blur, and its scattered pale pixels can otherwise
    # piece together into a spurious "series" of their own (see
    # multiline.find_legend_entries). Its bbox is excluded from the ink mask used
    # for color clustering below.
    chart_bounds = (x_offset, y_offset, x2, y2)
    legend_search_area = (
        legend_area
        if legend_area is not None
        else default_legend_search_area(chart_bounds)
    )
    legend_entries = find_legend_entries(
        texts, bboxes, chart_bounds, img, search_area=legend_search_area
    )
    ink_mask = chart_area.astype(bool)
    for entry in legend_entries:
        left, top, right, bottom = entry["bbox"]
        ink_mask[
            max(0, top - y_offset - 2) : bottom - y_offset + 2,
            max(0, left - x_offset - 2) : right - x_offset + 2,
        ] = False

    # Separate ink by color: a chart may hold several distinctly colored lines, but
    # value badges, watermarks and stray markers are also colored ink (see
    # multiline.cluster_ink_colors and select_series_clusters below for how
    # genuine series lines are told apart from those).
    color_chart_area = img[y_offset:y2, x_offset:x2]
    ink_clusters = cluster_ink_colors(color_chart_area, ink_mask)
    series_clusters = select_series_clusters(
        ink_clusters,
        grid_y_component_map,
        grid_x_component,
        y_offset,
        allowed_margin=5,
    )

    if series_clusters:
        width = chart_area.shape[1]
        time_series = [
            (
                x_scale(x + x_offset),
                [
                    None if cluster["rows"][x] is None else y_scale(cluster["rows"][x])
                    for cluster in series_clusters
                ],
            )
            for x in range(width)
        ]
        series_colors = [cluster["color"] for cluster in series_clusters]
    else:
        # Nothing resolved via color separation (e.g. an unusually faint or broken
        # line): fall back to reading the whole ink mask (legend excluded) as a
        # single series.
        time_series = extract_time_series_from_chart_area(
            ink_mask,
            x_scale,
            y_scale,
            grid_x_component,
            grid_y_component_map,
            x_offset,
            y_offset,
            allowed_margin=5,
            reversed=False,
        )
        series_colors = [ink_clusters[0]["color"]] if ink_clusters else [(0, 0, 0)]
    time_series = fill_gaps_in_time_series(time_series, window_size=5)

    series_names = match_series_to_legend(series_colors, legend_entries)

    return ChartExtraction(
        image_size=(int(img.shape[1]), int(img.shape[0])),
        chart_area=(int(x_offset), int(y_offset), int(x2), int(y2)),
        time_series=time_series,
        x_pixels=[int(x_offset) + x for x in range(len(time_series))],
        detected_grid_x=[int(px) for px in grid_x_component_clusters_centers],
        detected_grid_y=[int(py) for py in grid_y_component_clusters_centers],
        x_scale=x_scale,
        y_scale=y_scale,
        x_is_datetime=bool(time_series) and isinstance(time_series[0][0], datetime),
        y_is_log=isinstance(y_scale, Logarithmic),
        series_names=series_names,
        legend_area=tuple(int(round(v)) for v in legend_search_area),
    )


def extract_time_series(image_path: str) -> list:
    """Digitalize a chart image into [(x_value, [y_value, ...]), ...]."""
    return extract_chart(image_path).time_series


def ink_rows_in_column(chart_area, grid_y_component_map, rows_kept, x):
    """Absolute image rows of the non-grid ink in column `x` of the chart area."""
    return rows_kept[np.nonzero(chart_area[~grid_y_component_map, x])[0]]


def column_candidate_rows(
    chart_area, grid_y_component_map, rows_kept, grid_x_lookup, allowed_margin
):
    """
    Candidate series rows per chart-area column: (mean, min, max) of each ink
    cluster's rows. Columns on a vertical grid line, or with no ink, get an empty
    list. More than one candidate means something other than the series is drawn
    in that column. The min/max span is kept alongside the mean so a genuine
    vertical step-transition (one tall, contiguous run of ink) can be told apart
    from an isolated, unrelated blob that merely happens to average out nearby.
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
            [
                (float(np.mean(cluster)), float(min(cluster)), float(max(cluster)))
                for cluster in cluster_data(ys, allowed_margin)
            ]
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
    treated as foreign ink and yield None -- unless the candidate's own ink span
    reaches back to the running row: a step-shaped series (e.g. an analyst estimate
    revised once a quarter) draws its jump as one tall, unbroken vertical run whose
    span covers every row between the old and new level, so the running row falls
    inside it even though the cluster's mean is far away. An isolated, unrelated
    blob (a stray marker, anti-aliasing debris) has no such connecting span and is
    still rejected.
    """
    resolved: list[Optional[float]] = [None] * len(candidates)
    seed = _longest_unambiguous_run(candidates)
    if seed is None:
        # Every column is ambiguous, so there is no series to lock on to.
        return [rows[0][0] if len(rows) == 1 else None for rows in candidates]

    resolved[seed] = candidates[seed][0][0]
    for indices in (range(seed + 1, len(candidates)), range(seed - 1, -1, -1)):
        last = resolved[seed]
        for i in indices:
            rows = candidates[i]
            if not rows:
                continue
            closest = min(rows, key=lambda row: abs(row[0] - last))
            mean = closest[0]
            if len(rows) == 1 or abs(mean - last) <= max_step:
                resolved[i] = mean
                last = mean
    return resolved


def estimate_max_step(candidates, height: int) -> float:
    """
    Largest plausible row change between neighbouring columns, from how fast the
    series actually moves where it is unambiguous. Keeps the interference cut-off
    adaptive instead of a fixed pixel budget.
    """
    unambiguous = [rows[0][0] if len(rows) == 1 else None for rows in candidates]
    steps = [
        abs(b - a)
        for a, b in zip(unambiguous, unambiguous[1:])
        if a is not None and b is not None
    ]
    if not steps:
        return height / 2
    return max(float(np.percentile(steps, 99)) * 4, 8.0)


def resolve_series_pixel_rows(
    ink_mask, grid_y_component_map, grid_x_component, y_offset, allowed_margin=5
):
    """
    Resolve one row-per-column trajectory from a single ink mask, rejecting ink
    that doesn't belong to the dominant continuous run (see resolve_series_rows).
    Rows are absolute image rows, not yet mapped through the y scale.
    """
    height = ink_mask.shape[0]
    rows_kept = np.nonzero(~grid_y_component_map)[0] + y_offset
    grid_x_lookup = set(np.asarray(grid_x_component).tolist())
    candidates = column_candidate_rows(
        ink_mask, grid_y_component_map, rows_kept, grid_x_lookup, allowed_margin
    )
    return resolve_series_rows(candidates, estimate_max_step(candidates, height))


def _color_spread(color) -> int:
    return max(color) - min(color)


def _is_same_trajectory(
    rows_a,
    rows_b,
    max_median_distance,
    min_overlap_fraction,
    min_overlap_spread=DEFAULT_DUPLICATE_MIN_OVERLAP_SPREAD,
):
    """
    Whether two resolved row trajectories track the same physical line, judged by
    the MEDIAN row distance over their shared columns.

    A single anti-aliased line commonly splits into several color shades (edge vs.
    core), and at steep segments a shade can legitimately sit tens of pixels from
    another shade of the very same line, purely because the line's slope spreads
    ink across many rows within one column. The median is robust to those
    slope-driven outliers, while two genuinely different lines differ by a large
    margin at most columns, not just a few.

    Dense line crossings can split ONE physical line's ink into shades whose color
    distance clears the discovery-time merge threshold, each then resolving on a
    different, largely disjoint subset of columns (whichever shade happens to sit
    on that segment): comparing overlap against the full image width would demand
    more shared columns than either trajectory even has, rejecting real duplicates.
    Comparing it against the SMALLER trajectory's own resolved count instead asks
    the right question -- "does the shared evidence corroborate it" -- but that
    alone would also pass two genuinely different lines that merely cross paths
    for a while, since a crossing produces a run of near-zero distance too. The
    spread check rules that out: shared columns from anti-aliasing shades of one
    line are scattered across the whole trajectory, while a crossing's shared
    columns cluster in the narrow stretch where the lines actually meet.
    """
    overlap_idx = [
        i
        for i, (a, b) in enumerate(zip(rows_a, rows_b))
        if a is not None and b is not None
    ]
    if len(overlap_idx) < MIN_DUPLICATE_OVERLAP_COLUMNS:
        return False
    smaller_count = min(
        sum(r is not None for r in rows_a), sum(r is not None for r in rows_b)
    )
    if len(overlap_idx) < min_overlap_fraction * smaller_count:
        return False
    if overlap_idx[-1] - overlap_idx[0] < min_overlap_spread * len(rows_a):
        return False
    diffs = [abs(rows_a[i] - rows_b[i]) for i in overlap_idx]
    return float(np.median(diffs)) <= max_median_distance


def _is_disjoint_shade(
    rows_a,
    colors_a,
    rows_b,
    colors_b,
    max_color_distance,
    max_overlap_fraction,
    max_median_distance,
):
    """
    Second path for merging a color candidate into an accepted group, for when
    they share too few columns for `_is_same_trajectory`'s overlap-then-distance
    test to judge at all (docs/series-gaps-diagnosis.md): a real complementary
    anti-aliasing shade of the same line can resolve on columns almost entirely
    disjoint from the main shade's -- e.g. a flat, fixed-row segment that renders
    as one evenly-blended shade with no core-color pixel anywhere nearby -- which
    is exactly the case `_is_same_trajectory`'s overlap-fraction gate structurally
    can't tell apart from "not enough evidence".

    Here color, not row agreement, is the discriminator: two truly unrelated
    lines that cross would still leave a spread of overlapping columns near the
    crossing (see `_is_same_trajectory`), so near-total column disjointness plus
    a color close to an already-accepted shade is treated as one more shade of
    that line rather than coincidence. `max_overlap_fraction` is deliberately
    lower than `_is_same_trajectory`'s own overlap floor, so this path only fires
    where that one is structurally unable to decide, not wherever it happens to
    reject a real crossing between two different lines.
    """
    color_distance = float(
        np.linalg.norm(
            np.array(colors_a, dtype=float) - np.array(colors_b, dtype=float)
        )
    )
    if color_distance > max_color_distance:
        return False

    overlap_idx = [
        i
        for i, (a, b) in enumerate(zip(rows_a, rows_b))
        if a is not None and b is not None
    ]
    smaller_count = min(
        sum(r is not None for r in rows_a), sum(r is not None for r in rows_b)
    )
    if smaller_count == 0:
        return False
    if len(overlap_idx) > max_overlap_fraction * smaller_count:
        return False
    if not overlap_idx:
        return True  # no shared evidence to contradict a color-based merge
    diffs = [abs(rows_a[i] - rows_b[i]) for i in overlap_idx]
    return float(np.median(diffs)) <= max_median_distance


def select_series_clusters(
    ink_clusters,
    grid_y_component_map,
    grid_x_component,
    y_offset,
    allowed_margin=5,
    min_resolved_fraction=DEFAULT_MIN_RESOLVED_FRACTION,
    min_candidate_column_coverage=DEFAULT_MIN_CANDIDATE_COLUMN_COVERAGE,
    duplicate_median_distance=DEFAULT_DUPLICATE_MEDIAN_DISTANCE,
    duplicate_min_overlap_fraction=DEFAULT_DUPLICATE_MIN_OVERLAP_FRACTION,
    disjoint_max_overlap_fraction=DEFAULT_DISJOINT_MAX_OVERLAP_FRACTION,
    disjoint_color_merge_distance=DEFAULT_DISJOINT_COLOR_MERGE_DISTANCE,
    near_gray_saturation_threshold=NEAR_GRAY_SATURATION_THRESHOLD,
    near_gray_min_brightness=NEAR_GRAY_MIN_BRIGHTNESS,
):
    """
    Decide which color clusters from cluster_ink_colors are genuine series lines.

    A raw column-coverage check on ink alone is fooled by anti-aliasing: the blend
    between a line's color and a nearby gridline forms its own "color" that can
    span much of the width without being a real line (it appears only in short,
    scattered runs near each crossing). Instead, every color candidate is run
    through the same continuity-tracking resolver used for the final extraction:
    a real line resolves at most columns, while blend artifacts and decorations
    resolve at few, since neither forms one continuous path.

    That still lets through one specific case: a gridline's own anti-aliased edge,
    which is desaturated (grid lines are gray) and light (they blend toward the
    white background, never toward black), yet is crossed by data lines at enough
    columns to span nearly the whole width. Real series lines are either a
    saturated color or genuinely dark (a "black" line), never both desaturated and
    light, so that combination is rejected outright.

    Candidates whose resolved trajectory nearly matches an already-accepted one
    (e.g. a second anti-aliasing shade of the same line) are merged into it rather
    than discarded, so no real ink is lost. `min_resolved_fraction` is checked only
    on the FINAL, post-merge groups: dense line crossings can split one physical
    line's ink into shades that individually resolve well under that bar (each on
    whatever columns that particular shade happens to win), so gating candidacy on
    it before merging would throw those shades away before they ever got a chance
    to reunite.

    A candidate that `_is_same_trajectory` rejects only for lack of shared columns
    (not disagreement) gets a second chance via `_is_disjoint_shade`: a real
    complementary shade of an already-accepted line can land on columns almost
    entirely disjoint from it (see docs/series-gaps-diagnosis.md), which is the
    one case `_is_same_trajectory`'s overlap gate can't tell apart from "two
    coincidentally-similar-colored but different lines" -- so this path leans on
    color closeness instead of row agreement to decide.

    Returns a list of {"color", "mask", "rows", ...} ordered by descending
    resolved-column count.
    """
    if not ink_clusters:
        return []
    width = ink_clusters[0]["mask"].shape[1]

    resolved = []
    for cluster in ink_clusters:
        color = cluster["color"]
        if (
            max(color) - min(color) < near_gray_saturation_threshold
            and sum(color) / 3 > near_gray_min_brightness
        ):
            continue
        if (
            np.count_nonzero(cluster["mask"].any(axis=0))
            < min_candidate_column_coverage * width
        ):
            continue
        rows = resolve_series_pixel_rows(
            cluster["mask"],
            grid_y_component_map,
            grid_x_component,
            y_offset,
            allowed_margin,
        )
        resolved_count = sum(r is not None for r in rows)
        resolved.append({**cluster, "rows": rows, "resolved_count": resolved_count})
    resolved.sort(key=lambda c: -c["resolved_count"])

    groups = []
    for candidate in resolved:
        match = next(
            (
                g
                for g in groups
                if _is_same_trajectory(
                    candidate["rows"],
                    g["rows"],
                    duplicate_median_distance,
                    duplicate_min_overlap_fraction,
                )
                or _is_disjoint_shade(
                    candidate["rows"],
                    candidate["color"],
                    g["rows"],
                    g["color"],
                    disjoint_color_merge_distance,
                    disjoint_max_overlap_fraction,
                    duplicate_median_distance,
                )
            ),
            None,
        )
        if match is None:
            groups.append(dict(candidate))
            continue
        merged_mask = match["mask"] | candidate["mask"]
        merged_rows = resolve_series_pixel_rows(
            merged_mask,
            grid_y_component_map,
            grid_x_component,
            y_offset,
            allowed_margin,
        )
        match["mask"] = merged_mask
        match["rows"] = merged_rows
        match["resolved_count"] = sum(r is not None for r in merged_rows)
        match["count"] = match["count"] + candidate["count"]
        # A thin line's anti-aliased blend toward the background can outnumber its
        # solid core in pixel count, so the more prominent shade isn't necessarily
        # the truer color. The least diluted (most saturated) shade seen for this
        # physical line is: dilution moves every channel toward the background,
        # which only ever narrows the spread between them.
        if _color_spread(candidate["color"]) > _color_spread(match["color"]):
            match["color"] = candidate["color"]

    return [g for g in groups if g["resolved_count"] >= min_resolved_fraction * width]


def extract_time_series_from_chart_area(
    chart_area,
    x_scale,
    y_scale,
    grid_x_component,
    grid_y_component_map,
    x_offset,
    y_offset,
    allowed_margin=5,
    reversed=False,
):
    width = chart_area.shape[1]
    rows = resolve_series_pixel_rows(
        chart_area, grid_y_component_map, grid_x_component, y_offset, allowed_margin
    )
    time_series = [
        (
            x_scale(x + x_offset),
            [None if rows[x] is None else y_scale(rows[x])],
        )
        for x in range(width)
    ]
    return time_series[::-1] if reversed else time_series


def extract_multi_series_from_chart_area(
    chart_area,
    ink_masks,
    x_scale,
    y_scale,
    grid_x_component,
    grid_y_component_map,
    x_offset,
    y_offset,
    allowed_margin=5,
    reversed=False,
):
    """
    Same as extract_time_series_from_chart_area, but resolves one row-per-column
    trajectory per mask in `ink_masks` and packs them into a single time series, in
    mask order.
    """
    width = chart_area.shape[1]
    all_rows = [
        resolve_series_pixel_rows(
            mask, grid_y_component_map, grid_x_component, y_offset, allowed_margin
        )
        for mask in ink_masks
    ]
    time_series = [
        (
            x_scale(x + x_offset),
            [None if rows[x] is None else y_scale(rows[x]) for rows in all_rows],
        )
        for x in range(width)
    ]
    return time_series[::-1] if reversed else time_series
