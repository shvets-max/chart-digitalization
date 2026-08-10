from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.geometry import get_row_bboxes
from src.ocr_utils import texts_to_numbers

Color = tuple[int, int, int]
Box = tuple[int, int, int, int]

DEFAULT_COLOR_MERGE_DISTANCE = 40.0
DEFAULT_MAX_LEGEND_MATCH_DISTANCE = 80.0
CORE_INK_MAX_GRAY = 200  # stricter than the general ink threshold (gray < 250)

LEGEND_TOP_FRACTION = 0.35  # legend rows must start within this fraction of height
LEGEND_LEFT_FRACTION = 0.6  # ...and within this fraction of width, from the corner
MIN_LEGEND_TOKEN_HEIGHT = 5  # px; smaller is an OCR sliver, not real text
SATURATION_THRESHOLD = (
    25  # min max-min channel spread in BGR to call a color "not gray"
)
SWATCH_WIDTH = 40
SWATCH_GAP = 2


def cluster_ink_colors(
    color_area: np.ndarray,
    ink_mask: np.ndarray,
    merge_distance: float = DEFAULT_COLOR_MERGE_DISTANCE,
) -> list[dict]:
    """
    Group the ink pixels of `color_area` (BGR) by color.

    Cluster centers are discovered from "core" pixels only (gray < CORE_INK_MAX_GRAY,
    stricter than the general ink threshold). Anti-aliased pixels form a continuum
    from a line's color toward the background color; including them in discovery
    would let a single anti-aliased line masquerade as several width-spanning
    colors, each a different shade along that continuum. Once centers are known,
    every ink pixel (including faint anti-aliased ones) is assigned to its nearest
    center, so thin edges still end up in the right series' mask.

    Returns a list of {"color": (b, g, r), "mask": bool array, "count": int} sorted
    by descending pixel count.
    """
    gray = color_area.mean(axis=2)
    core_ys, core_xs = np.nonzero(ink_mask & (gray < CORE_INK_MAX_GRAY))
    if core_ys.size == 0:
        # Nothing is confidently colored (e.g. a very light line): fall back to
        # using all ink for discovery too.
        core_ys, core_xs = np.nonzero(ink_mask)
        if core_ys.size == 0:
            return []

    core_colors = color_area[core_ys, core_xs].astype(np.int32)
    quantized = core_colors // 8 * 8
    unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(-counts)

    cluster_centers = []  # running mean color per cluster (float)
    cluster_counts = []
    for idx in order:
        color = unique_colors[idx].astype(np.float64)
        count = counts[idx]
        best, best_dist = None, merge_distance
        for c_idx, center in enumerate(cluster_centers):
            dist = float(np.linalg.norm(color - center))
            if dist <= best_dist:
                best, best_dist = c_idx, dist
        if best is None:
            cluster_centers.append(color)
            cluster_counts.append(count)
        else:
            total = cluster_counts[best] + count
            cluster_centers[best] = (
                cluster_centers[best] * cluster_counts[best] + color * count
            ) / total
            cluster_counts[best] = total

    if not cluster_centers:
        return []

    # Assign every ink pixel, including faint anti-aliased ones, to its nearest
    # discovered center.
    all_ys, all_xs = np.nonzero(ink_mask)
    all_colors = color_area[all_ys, all_xs].astype(np.float64)
    centers = np.array(cluster_centers)
    distances = np.linalg.norm(all_colors[:, None, :] - centers[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)

    clusters = []
    for c_idx, center in enumerate(cluster_centers):
        member = nearest == c_idx
        mask = np.zeros(ink_mask.shape, dtype=bool)
        mask[all_ys[member], all_xs[member]] = True
        clusters.append(
            {
                "color": tuple(int(round(v)) for v in center),
                "mask": mask,
                "count": int(np.count_nonzero(mask)),
            }
        )
    clusters.sort(key=lambda c: -c["count"])
    return clusters


def _is_numeric_token(text: str) -> bool:
    return texts_to_numbers([text])[0] is not None


def _union_bbox(boxes: list[Box]) -> Box:
    lefts, tops, rights, bottoms = zip(*boxes)
    return min(lefts), min(tops), max(rights), max(bottoms)


def _dominant_ink_color(color_img: np.ndarray, box: Box) -> Optional[Color]:
    """Modal color of the non-background pixels inside `box` (left, top, right, bottom)."""
    left, top, right, bottom = box
    left, top = max(0, left), max(0, top)
    window = color_img[top:bottom, left:right]
    if window.size == 0:
        return None
    gray = window.mean(axis=2)
    # Small text is mostly anti-aliased blur (thin strokes have few solid-color
    # pixels), so the loose ink threshold alone would make the modal color come
    # out washed out. Prefer confidently-inked pixels; fall back to the loose
    # threshold only if the window has none (e.g. a very light color).
    ink = gray < CORE_INK_MAX_GRAY
    if not ink.any():
        ink = gray < 250
        if not ink.any():
            return None
    pixels = window[ink].astype(np.int32)
    quantized = pixels // 8 * 8
    unique, counts = np.unique(quantized, axis=0, return_counts=True)
    return tuple(int(v) for v in unique[np.argmax(counts)])


def _is_saturated(color: Color) -> bool:
    return (max(color) - min(color)) >= SATURATION_THRESHOLD


def _swatch_color(color_img: np.ndarray, first_token_box: Box) -> Optional[Color]:
    """Dominant color immediately left of a legend row's first token, matplotlib-style."""
    left, top, right, bottom = first_token_box
    x_end = max(0, left - SWATCH_GAP)
    x_start = max(0, x_end - SWATCH_WIDTH)
    if x_end <= x_start:
        return None
    return _dominant_ink_color(color_img, (x_start, top, x_end, bottom))


def find_legend_entries(
    texts: list[str], bboxes: list[Box], chart_area: Box, color_img: np.ndarray
) -> list[dict]:
    """
    Detect legend rows in the top-left of the chart area: each row is a
    left-to-right run of OCR tokens forming a series name, optionally followed by
    a trailing value (e.g. "NOW: EBITDA (TTM)   2.16B"). A row's color is the
    label text's own pixel color when it is saturated (dashboard-style legends,
    where the label is colored to match its line), else a swatch sampled
    immediately to its left (matplotlib's default black-text-plus-icon legend).

    Returns a list of {"name": str, "color": (b, g, r), "bbox": (left, top, right,
    bottom)} ordered top to bottom. `bbox` is the whole row (name and any trailing
    value), useful for excluding legend ink from series color detection.
    """
    x1, y1, x2, y2 = chart_area
    width, height = x2 - x1, y2 - y1

    # Restrict to the corner where a legend conventionally sits BEFORE grouping
    # into rows. get_row_bboxes groups purely by vertical overlap with no
    # horizontal-proximity check, so a distant token at the same height (a scale
    # toggle button, a y-axis tick label on the far side of the chart) would
    # otherwise merge into the same "row" as a real legend entry it has nothing
    # to do with, producing a row that spans most of the chart's width.
    region_right = x1 + LEGEND_LEFT_FRACTION * width
    region_bottom = y1 + LEGEND_TOP_FRACTION * height
    candidate_idx = [
        i
        for i, box in enumerate(bboxes)
        if x1 <= box[0]
        and box[2] <= region_right
        and y1 <= box[1]
        and box[3] <= region_bottom
        and box[3] - box[1] >= MIN_LEGEND_TOKEN_HEIGHT  # drop OCR slivers (e.g. a
        # 1px-tall misread of the axis spine at the chart's edge), not real text
    ]
    if not candidate_idx:
        return []
    candidate_boxes = [bboxes[i] for i in candidate_idx]

    local_ids, rows = get_row_bboxes(candidate_boxes)
    entries = []
    for local_id_group, box_group in zip(local_ids, rows):
        order = sorted(range(len(local_id_group)), key=lambda k: box_group[k][0])
        id_group = [candidate_idx[local_id_group[k]] for k in order]
        box_group = [box_group[k] for k in order]
        row_bbox = _union_bbox(box_group)
        top = row_bbox[1]

        row_texts = [texts[i] for i in id_group]

        # Drop a leading legend bullet/marker icon: dashboard legends draw a small
        # colored dot before the name, which OCR reads as garbage with no
        # alphanumeric content.
        while row_texts and not any(ch.isalnum() for ch in row_texts[0]):
            row_texts, box_group = row_texts[1:], box_group[1:]
        if not row_texts:
            continue

        numeric_flags = [_is_numeric_token(t) for t in row_texts]
        if sum(numeric_flags) > 0.5 * len(row_texts):
            continue  # looks like a tick-label row, not a legend

        # Drop the trailing current-value badge (e.g. "2.16B") and anything after
        # it in the row: a "Lin"/"Log" scale-toggle control can share the
        # legend's top row and would otherwise be read as part of the name.
        name_texts, name_boxes = list(row_texts), list(box_group)
        last_numeric = next(
            (i for i in range(len(numeric_flags) - 1, -1, -1) if numeric_flags[i]),
            None,
        )
        if last_numeric is not None:
            name_texts, name_boxes = (
                name_texts[:last_numeric],
                name_boxes[:last_numeric],
            )
        if not name_texts:
            continue
        name = " ".join(name_texts).strip()

        color = _dominant_ink_color(color_img, _union_bbox(name_boxes))
        if color is None or not _is_saturated(color):
            swatch = _swatch_color(color_img, name_boxes[0])
            if swatch is not None:
                color = swatch
        if color is None:
            continue

        entries.append({"name": name, "color": color, "bbox": row_bbox, "top": top})

    entries.sort(key=lambda e: e["top"])
    return [
        {"name": e["name"], "color": e["color"], "bbox": e["bbox"]} for e in entries
    ]


def match_series_to_legend(
    series_colors: list[Color],
    legend_entries: list[dict],
    max_distance: float = DEFAULT_MAX_LEGEND_MATCH_DISTANCE,
) -> list[Optional[str]]:
    """
    Pair each series color with a legend entry color (Euclidean, BGR) so that the
    total distance across all pairs is minimal; each legend entry is used at most
    once. If there are at least as many legend entries as series, every series is
    assigned one (even past `max_distance`); otherwise a series is left unmatched
    (None) when its best remaining legend entry is farther than `max_distance`.
    """
    names = [None] * len(series_colors)
    if not series_colors or not legend_entries:
        return names

    distances = np.linalg.norm(
        np.array(series_colors, dtype=float)[:, None, :]
        - np.array([entry["color"] for entry in legend_entries], dtype=float)[
            None, :, :
        ],
        axis=2,
    )
    row_idx, col_idx = linear_sum_assignment(distances)

    guaranteed_match = len(legend_entries) >= len(series_colors)
    for s_idx, l_idx in zip(row_idx, col_idx):
        if guaranteed_match or distances[s_idx, l_idx] <= max_distance:
            names[s_idx] = legend_entries[l_idx]["name"]
    return names
