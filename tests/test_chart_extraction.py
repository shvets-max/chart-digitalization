import csv
import os
from datetime import date, datetime
from typing import Optional
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from src.chart_extraction import (
    ChartGridData,
    ChartImageData,
    adjust_knots_to_grid,
    extract_chart,
    extract_chart_series,
    extract_time_series,
    extract_time_series_from_chart_area,
    select_axis_tick_group,
    select_series_clusters,
)
from src.function import Linear
from src.multiline import default_legend_search_area
from src.ocr_utils import texts_to_numbers
from tests.test_data import adjust_knots_to_grid_data, extract_series_interference_data

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LINEAR_SCALE_DIR = os.path.join(TEST_DATA_DIR, "linear_scaled")
LOG_SCALE_DIR = os.path.join(TEST_DATA_DIR, "log_scaled")
IN_AREA_TEXT_DIR = os.path.join(TEST_DATA_DIR, "in_area_text")
MULTILINE_DIR = os.path.join(TEST_DATA_DIR, "multiline")
SCRAB_STYLE_DIR = os.path.join(TEST_DATA_DIR, "scrab_style")
DENSE_CROSSING_DIR = os.path.join(TEST_DATA_DIR, "dense_crossing")
CROSSING_MULTILINE_DIR = os.path.join(TEST_DATA_DIR, "crossing_multiline")
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
REAL_MULTILINE_DIR = os.path.join(REPO_ROOT, "data", "multiline")

SEP = ";"

ExpectedSeries = dict[date, list[float]]


def _load_expected_results(data_dir: str) -> dict[str, ExpectedSeries]:
    """Load ground-truth time series from every CSV in `data_dir`, keyed by file stem."""
    expected_results = {}
    for file in sorted(os.listdir(data_dir)):
        if not file.endswith(".csv"):
            continue
        idx = os.path.splitext(file)[0]
        with open(os.path.join(data_dir, file), newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=SEP)
            next(reader)  # skip header
            expected_results[idx] = {
                datetime.strptime(row[0], "%Y-%m-%d").date(): [
                    float(val.replace(",", ".")) for val in row[1:]
                ]
                for row in reader
            }
    return expected_results


def _compute_errors(
    expected: ExpectedSeries, extracted: ExpectedSeries, n_series: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (abs_errors, rel_errors) arrays of shape (n_dates, n_series) for expected vs extracted values."""
    abs_errors = []
    rel_errors = []
    for key, expected_values in expected.items():
        extracted_values = extracted.get(key, [None] * n_series)
        extracted_values = [
            round(np.float32(v or 0), 4).item() for v in extracted_values
        ]
        abs_errors.append(
            [abs(ev - dv) for ev, dv in zip(expected_values, extracted_values)]
        )
        rel_errors.append(
            [
                abs(ev - dv) / ev if ev != 0 else 0.0
                for ev, dv in zip(expected_values, extracted_values)
            ]
        )
    return np.array(abs_errors), np.array(rel_errors)


def _log_error_stats(
    label: str, abs_errors: np.ndarray, rel_errors: np.ndarray
) -> None:
    """Print mean/std of absolute and relative error for `label`."""
    print(
        f"{label}: "
        f"mean_abs={abs_errors.mean():.4f} std_abs={abs_errors.std():.4f} "
        f"mean_rel={rel_errors.mean():.4f} std_rel={rel_errors.std():.4f}"
    )


class TestChartExtraction(TestCase):
    """Runs chart extraction against generated charts and checks it recovers the right shape of data."""

    def setUp(self):
        self.linear_expected = _load_expected_results(LINEAR_SCALE_DIR)
        self.log_expected = _load_expected_results(LOG_SCALE_DIR)

    def _check_scale(
        self, data_dir: str, expected_results: dict[str, ExpectedSeries], label: str
    ) -> None:
        """Extract every chart in `data_dir`, asserting series count matches and reporting error stats."""
        overall_abs_errors = []
        overall_rel_errors = []
        for idx, expected in expected_results.items():
            with self.subTest(chart=idx):
                image_path = os.path.join(data_dir, f"{idx}.png")
                extracted_data = extract_time_series(image_path)
                extracted_data = {dt.date(): values for dt, values in extracted_data}

                n_series_expected = len(next(iter(expected.values())))
                n_series_extracted = len(next(iter(extracted_data.values())))
                self.assertEqual(
                    n_series_extracted,
                    n_series_expected,
                    f"Number of series mismatch for chart {idx}",
                )

                abs_errors, rel_errors = _compute_errors(
                    expected, extracted_data, n_series_expected
                )
                overall_abs_errors.append(abs_errors)
                overall_rel_errors.append(rel_errors)
                _log_error_stats(f"{label}/{idx}", abs_errors, rel_errors)

        _log_error_stats(
            f"{label}/overall",
            np.concatenate(overall_abs_errors),
            np.concatenate(overall_rel_errors),
        )

    def test_linear_scale_extraction(self):
        """Extraction should recover the correct series count from linear-scaled charts."""
        self._check_scale(LINEAR_SCALE_DIR, self.linear_expected, "linear")

    def test_log_scale_extraction(self):
        """Extraction should recover the correct series count from log-scaled charts."""
        self._check_scale(LOG_SCALE_DIR, self.log_expected, "log")


class TestAdjustKnotsToGrid(TestCase):
    """Unit tests for adjust_knots_to_grid, isolated from full chart extraction."""

    def test_adjust_knots_to_grid(self):
        for i, (knots, grid_centers, min_dist, max_dist, expected) in enumerate(
            adjust_knots_to_grid_data
        ):
            result = adjust_knots_to_grid(
                np.array(knots), grid_centers, min_dist, max_dist
            )
            np.testing.assert_array_equal(
                result, expected, err_msg=f"Failed for input {i}: {knots}"
            )


def _read_ascii_chart(ascii_rows: list[str]) -> list[Optional[float]]:
    """
    Run extract_time_series_from_chart_area over an ASCII ink pattern ('#' is ink)
    with identity scales, so each returned value is the pixel row it was read from.
    """
    chart_area = np.array(
        [[1 if ch == "#" else 0 for ch in row] for row in ascii_rows], dtype=np.uint8
    )
    height, width = chart_area.shape
    time_series = extract_time_series_from_chart_area(
        chart_area,
        x_scale=Linear(knots=[0, width - 1], values=[0, width - 1]),
        y_scale=Linear(knots=[0, height - 1], values=[0, height - 1]),
        grid_x_component=np.array([], dtype=int),
        grid_y_component_map=np.zeros(height, dtype=bool),
        x_offset=0,
        y_offset=0,
    )
    return [None if v[0] is None else round(v[0], 1) for _, v in time_series]


class TestExtractTimeSeriesFromChartArea(TestCase):
    """
    Regression test for a real x-axis mislabeling bug: extract_chart used to
    compute each point's x-value from `local_column + x_offset - grid_l`, where
    `grid_l` (from geometry.cut_chart_area) was meant to correct for how far the
    chart area's left edge sits from image column 0, but actually came out equal
    to `x_offset` whenever cut_chart_area trimmed no further than its own
    axis-bbox pass -- silently cancelling `x_offset` out and feeding the scale a
    raw LOCAL column instead of the point's true ABSOLUTE image position. Every
    tick position used to fit `x_scale` is read in absolute image coordinates
    (see extract_chart), so a local column's value must come from its absolute
    position, `local_column + x_offset` -- nothing else.
    """

    def test_x_value_uses_absolute_column_not_local_index(self):
        chart_area = np.array([[1], [0]], dtype=np.uint8)  # single column, ink at row 0
        x_offset = 500  # far from image column 0, unlike the local column index (0)
        x_scale = Linear(knots=[0, 1000], values=[0, 1000])

        time_series = extract_time_series_from_chart_area(
            chart_area,
            x_scale=x_scale,
            y_scale=Linear(knots=[0, 1], values=[0, 1]),
            grid_x_component=np.array([], dtype=int),
            grid_y_component_map=np.zeros(2, dtype=bool),
            x_offset=x_offset,
            y_offset=0,
        )

        self.assertEqual(time_series[0][0], x_scale(x_offset))


class TestChartAreaInterference(TestCase):
    """
    Non-series ink inside the plotting area (legend text, value badges, stray
    markers) must be rejected rather than averaged into the series.
    """

    def test_interfering_ink_is_rejected(self):
        for description, ascii_rows, expected in extract_series_interference_data:
            with self.subTest(case=description):
                self.assertEqual(_read_ascii_chart(ascii_rows), expected, description)


def _build_synthetic_chart(
    line_row: int,
    width: int = 60,
    height: int = 40,
    x_offset: int = 20,
    y_offset: int = 15,
) -> tuple[ChartImageData, ChartGridData]:
    """
    A minimal ChartImageData/ChartGridData pair with one horizontal ink line at
    `line_row` (chart_bounds-relative) spanning the full width, in light gray --
    a color select_series_clusters' near-gray/light filter rejects outright, so
    extract_chart_series falls back to reading the ink mask directly instead of
    going through color-cluster resolution. That makes the fallback's ink mask
    (legend bbox excluded) the only thing standing between "line resolves" and
    "line don't" in the tests below: exactly what legend-bbox masking touches.

    Identity x/y scales, so a resolved value equals its pixel row/column.
    """
    canvas_h, canvas_w = y_offset + height + 20, x_offset + width + 20
    thresh = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    thresh[y_offset + line_row, x_offset : x_offset + width] = 1

    img = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    img[y_offset + line_row, x_offset : x_offset + width] = (200, 200, 200)

    image_data = ChartImageData(
        img=img,
        thresh=thresh,
        texts=[],
        bboxes=[],
        columns_bboxes=[],
        column_numbers=[],
        rows_bboxes=[],
        row_index=[],
    )
    x2, y2 = x_offset + width, y_offset + height
    bounds = (x_offset, y_offset, x2, y2)
    grid_data = ChartGridData(
        default_bounds=bounds,
        chart_bounds=bounds,
        grid_x_lines=np.array([], dtype=int),
        grid_y_lines=np.array([], dtype=int),
        detected_grid_x=[],
        detected_grid_y=[],
        x_scale=Linear(knots=[0, canvas_w - 1], values=[0, canvas_w - 1]),
        y_scale=Linear(knots=[0, canvas_h - 1], values=[0, canvas_h - 1]),
    )
    return image_data, grid_data


class TestExtractChartSeriesLegendMasking(TestCase):
    """
    extract_chart_series excludes each legend entry's bbox from the ink mask
    before resolving series (see extract_chart_series), converting it from
    absolute image coordinates to chart_bounds-relative array indices. That
    conversion must clip correctly whether the legend sits fully inside,
    partially inside, or entirely outside chart_bounds -- especially the last
    case, where an unclamped slice *stop* going negative (Python slicing reads
    a negative stop as "N from the end", not "empty") used to zero out nearly
    the whole mask instead of doing nothing.
    """

    WIDTH, HEIGHT, X_OFFSET, Y_OFFSET = 60, 40, 20, 15
    LINE_ROW = 5

    def _extract(self, legend_bbox):
        image_data, grid_data = _build_synthetic_chart(
            self.LINE_ROW, self.WIDTH, self.HEIGHT, self.X_OFFSET, self.Y_OFFSET
        )
        with patch("src.chart_extraction.find_legend_entries") as mocked:
            mocked.return_value = [
                {
                    "bbox": legend_bbox,
                    "text": "Legend",
                    "name": "Legend",
                    "color": (0, 0, 0),
                }
            ]
            return extract_chart_series(image_data, grid_data, legend_area=None)

    def _values_by_local_column(self, extraction):
        """{local column: resolved value or None}, columns numbered from x_offset."""
        return {
            x - self.X_OFFSET: values[0]
            for x, (_, values) in zip(extraction.x_pixels, extraction.time_series)
        }

    def test_legend_fully_inside_chart_area_masks_only_that_region(self):
        # Absolute bbox: columns [x_offset+10, x_offset+25], rows spanning line_row.
        legend_bbox = (
            self.X_OFFSET + 10,
            self.Y_OFFSET + self.LINE_ROW - 2,
            self.X_OFFSET + 25,
            self.Y_OFFSET + self.LINE_ROW + 2,
        )
        by_col = self._values_by_local_column(self._extract(legend_bbox))

        for col in (0, 5, 40, self.WIDTH - 1):
            self.assertEqual(
                by_col[col], float(self.Y_OFFSET + self.LINE_ROW), f"column {col}"
            )
        for col in range(12, 24):  # comfortably inside the masked span
            self.assertIsNone(by_col[col], f"column {col} should be masked out")

    def test_legend_straddling_top_edge_masks_only_the_interior_part(self):
        # Top is above chart_bounds, bottom is inside past line_row: exercises
        # the pre-existing max(0, ...) clamp on the slice START.
        legend_bbox = (
            self.X_OFFSET + 10,
            self.Y_OFFSET - 10,
            self.X_OFFSET + 25,
            self.Y_OFFSET + self.LINE_ROW + 2,
        )
        by_col = self._values_by_local_column(self._extract(legend_bbox))

        for col in (0, 5, 40, self.WIDTH - 1):
            self.assertEqual(
                by_col[col], float(self.Y_OFFSET + self.LINE_ROW), f"column {col}"
            )
        for col in range(12, 24):
            self.assertIsNone(by_col[col], f"column {col} should be masked out")

    def test_legend_entirely_above_chart_area_does_not_corrupt_series(self):
        # Regression: bottom is above y_offset, so (bottom - y_offset + 2) is
        # negative. Unclamped, that slice stop wraps to "near the end" and
        # zeroes almost the entire mask instead of nothing.
        legend_bbox = (self.X_OFFSET + 10, 0, self.X_OFFSET + 25, self.Y_OFFSET - 5)
        by_col = self._values_by_local_column(self._extract(legend_bbox))

        for col in range(self.WIDTH):
            self.assertEqual(
                by_col[col], float(self.Y_OFFSET + self.LINE_ROW), f"column {col}"
            )

    def test_legend_entirely_left_of_chart_area_does_not_corrupt_series(self):
        # Regression: right is left of x_offset, so (right - x_offset + 2) is
        # negative -- the horizontal counterpart of the above. Rows deliberately
        # straddle line_row so a wrongly-zeroed column range would actually be
        # caught: the row bounds alone are valid/positive here (0 <= 8 <= 40),
        # so only the column clamp is under test.
        legend_bbox = (
            0,
            self.Y_OFFSET + self.LINE_ROW - 2,
            self.X_OFFSET - 5,
            self.Y_OFFSET + self.LINE_ROW + 2,
        )
        by_col = self._values_by_local_column(self._extract(legend_bbox))

        for col in range(self.WIDTH):
            self.assertEqual(
                by_col[col], float(self.Y_OFFSET + self.LINE_ROW), f"column {col}"
            )

    def test_legend_entirely_below_and_right_of_chart_area_is_ignored(self):
        # Sanity check: a legend past the bottom/right edge was never buggy
        # (positive slice bounds beyond the array just clip to empty), but a
        # regression here would mean the clamp fix broke the normal case.
        legend_bbox = (
            self.X_OFFSET + self.WIDTH + 5,
            self.Y_OFFSET + self.HEIGHT + 5,
            self.X_OFFSET + self.WIDTH + 15,
            self.Y_OFFSET + self.HEIGHT + 15,
        )
        by_col = self._values_by_local_column(self._extract(legend_bbox))

        for col in range(self.WIDTH):
            self.assertEqual(
                by_col[col], float(self.Y_OFFSET + self.LINE_ROW), f"column {col}"
            )


class TestStepShapedSeries(TestCase):
    """
    A step-shaped series (e.g. an analyst estimate revised once a quarter, as in
    data/scrab/anet-rev.png) draws its jump as one tall, unbroken vertical run in
    a single column -- a legitimate large row delta, not foreign ink.
    """

    def test_large_vertical_jump_is_followed(self):
        width, jump_col, low_row, high_row = 10, 4, 2, 15
        ascii_rows = []
        for row in range(20):
            cells = [" "] * width
            if row == low_row:
                for col in range(jump_col):
                    cells[col] = "#"
            if low_row <= row <= high_row:
                cells[jump_col] = "#"
            if row == high_row:
                for col in range(jump_col + 1, width):
                    cells[col] = "#"
            ascii_rows.append("".join(cells))

        result = _read_ascii_chart(ascii_rows)
        self.assertEqual(result[:jump_col], [float(low_row)] * jump_col)
        self.assertEqual(
            result[jump_col + 1 :], [float(high_row)] * (width - jump_col - 1)
        )


class TestInAreaTextExtraction(TestCase):
    """
    End-to-end extraction of charts that carry non-series ink inside the plotting
    area (legend strip, watermark, value badge, stray marker, text in every
    corner), as real screenshots do. See data/ebit-margin.png for the original.
    """

    # Generous enough to absorb the pixel-quantisation error the clean corpus also
    # has, tight enough to fail if the interfering ink is tracked or averaged in.
    MAX_MEDIAN_REL_ERROR = 0.05
    MAX_REL_ERROR = 0.15
    MAX_RANGE_REL_ERROR = 0.10

    def test_interfering_ink_does_not_capture_the_series(self):
        expected_results = _load_expected_results(IN_AREA_TEXT_DIR)
        self.assertTrue(expected_results, f"no fixtures in {IN_AREA_TEXT_DIR}")

        for idx, expected in expected_results.items():
            with self.subTest(chart=idx):
                extracted = extract_time_series(
                    os.path.join(IN_AREA_TEXT_DIR, f"{idx}.png")
                )
                by_date = {}
                for dt, values in extracted:
                    by_date.setdefault(dt.date(), values[0])

                rel_errors = np.array(
                    [
                        abs(by_date[key] - values[0]) / values[0]
                        for key, values in expected.items()
                        if by_date.get(key) is not None and values[0] != 0
                    ]
                )
                self.assertGreater(rel_errors.size, 0, f"nothing matched for {idx}")
                self.assertLess(
                    np.median(rel_errors),
                    self.MAX_MEDIAN_REL_ERROR,
                    f"median relative error too high for {idx}",
                )
                self.assertLess(
                    rel_errors.max(),
                    self.MAX_REL_ERROR,
                    f"worst relative error too high for {idx}",
                )

                # Following the interfering ink instead of the series shows up as a
                # value range well outside the real one, even if most points are fine.
                values = [v for _, vals in extracted for v in vals if v is not None]
                true_values = [v[0] for v in expected.values()]
                for got, true, edge in (
                    (min(values), min(true_values), "minimum"),
                    (max(values), max(true_values), "maximum"),
                ):
                    self.assertLess(
                        abs(got - true) / abs(true),
                        self.MAX_RANGE_REL_ERROR,
                        f"extracted {edge} {got:.4g} is far from {true:.4g} for {idx}",
                    )


class TestTextsToNumbers(TestCase):
    """Unit tests for texts_to_numbers, isolated from full chart extraction."""

    def test_plain_and_suffixed_numbers(self):
        self.assertEqual(
            texts_to_numbers(["100", "1.5k", "2m", "3b", "50%"]),
            [100.0, 1500.0, 2e6, 3e9, 0.5],
        )

    def test_unparsable_text_returns_none(self):
        self.assertEqual(texts_to_numbers(["100", "n/a", "200"]), [100.0, None, 200.0])

    def test_comma_as_decimal_separator(self):
        self.assertEqual(texts_to_numbers(["1,5", "2,0"]), [1.5, 2.0])

    def test_recovers_b_suffix_misread_as_trailing_8(self):
        # Real OCR misreadings from a column axis: "B" (billions) rendered as
        # "8" in most tokens ("298" -> "29B", "8.458" -> "8.45B"), while a few
        # tokens keep the correct "B" ("9.2B"). Series-aware recovery should
        # use the surviving "B" tokens to fix the "8"-suffixed ones instead of
        # parsing them as wildly smaller plain numbers.
        texts = [
            "298",
            "268",
            "238",
            "218",
            "198",
            "15.5B",
            "12.88",
            "10,.4B",
            "9.2B",
            "8.458",
            "7.658",
            "7.058",
            "6.45B",
            "5.858",
            "5.25B",
            "4.758",
            "4.358",
            "3.95B",
            "3.55B",
            "3.258",
            "2.978",
            "2.738",
            "2.518",
            "2.31B",
        ]
        expected = [
            2.9e10,
            2.6e10,
            2.3e10,
            2.1e10,
            1.9e10,
            1.55e10,
            1.28e10,
            1.04e10,
            9.2e9,
            8.45e9,
            7.65e9,
            7.05e9,
            6.45e9,
            5.85e9,
            5.25e9,
            4.75e9,
            4.35e9,
            3.95e9,
            3.55e9,
            3.25e9,
            2.97e9,
            2.73e9,
            2.51e9,
            2.31e9,
        ]
        numbers = texts_to_numbers(texts)
        for actual, want in zip(numbers, expected):
            self.assertAlmostEqual(actual, want, delta=1e6)
        # the recovered series should be monotonically decreasing, matching a
        # real axis -- confirms the "8"->"B" fix, not just individual values
        self.assertTrue(all(a > b for a, b in zip(numbers, numbers[1:])))

    def test_lone_stray_suffix_does_not_force_correction(self):
        # Only one token in the whole series ends in "B" -- not enough signal
        # to treat "B" as the series' dominant unit, so a plain "8"-ending
        # number should be left alone rather than reinterpreted as billions.
        self.assertEqual(texts_to_numbers(["100", "108", "1B"]), [100.0, 108.0, 1e9])


class TestSelectAxisTickGroup(TestCase):
    """
    Unit tests for select_axis_tick_group, isolated from full chart extraction.
    """

    def test_prefers_parseable_group_over_first_on_a_length_tie(self):
        # Reproduces a real failure: adding legend text to a chart can produce an
        # OCR bbox group that ties the true tick-label group in size. Picking by
        # raw length alone (argmax) breaks the tie in favour of whichever group
        # comes first, which is wrong whenever that's the non-numeric one.
        texts = ["Growth", "Margin", "Costs", "100", "200", "300"]
        ids = [[0, 1, 2], [3, 4, 5]]  # same size; only group 1 is numeric
        self.assertEqual(select_axis_tick_group(ids, texts, texts_to_numbers), 1)

    def test_empty_ids_returns_zero(self):
        self.assertEqual(select_axis_tick_group([], [], texts_to_numbers), 0)


class TestSelectSeriesClusters(TestCase):
    """Unit tests for select_series_clusters, isolated from full chart extraction."""

    def test_merged_duplicate_shades_keep_the_saturated_color(self):
        # A thin anti-aliased line commonly splits into a solid "core" shade and a
        # lighter "edge" shade blended toward the background. The edge shade can
        # have MORE ink than the core (thin lines are mostly edge), so picking
        # whichever shade is more prominent is not enough: the representative
        # color must be the least background-diluted (most saturated) one.
        height, width = 20, 40
        core_mask = np.zeros((height, width), dtype=bool)
        core_mask[10, :] = True
        edge_mask = np.zeros((height, width), dtype=bool)
        edge_mask[10:13, :] = True  # thicker -> more ink, but a lighter shade

        ink_clusters = [
            {
                "color": (240, 235, 245),
                "mask": edge_mask,
                "count": int(edge_mask.sum()),
            },
            {"color": (20, 15, 200), "mask": core_mask, "count": int(core_mask.sum())},
        ]
        grid_y_component_map = np.zeros(height, dtype=bool)
        grid_x_component = np.array([], dtype=int)

        groups = select_series_clusters(
            ink_clusters, grid_y_component_map, grid_x_component, y_offset=0
        )

        self.assertEqual(len(groups), 1, "the two shades should merge into one line")
        self.assertEqual(groups[0]["color"], (20, 15, 200))

    def test_distinct_lines_are_not_merged(self):
        height, width = 20, 40
        mask_a = np.zeros((height, width), dtype=bool)
        mask_a[2, :] = True
        mask_b = np.zeros((height, width), dtype=bool)
        mask_b[17, :] = True  # far from mask_a: a genuinely different line

        ink_clusters = [
            {"color": (0, 0, 200), "mask": mask_a, "count": int(mask_a.sum())},
            {"color": (200, 0, 0), "mask": mask_b, "count": int(mask_b.sum())},
        ]
        grid_y_component_map = np.zeros(height, dtype=bool)
        grid_x_component = np.array([], dtype=int)

        groups = select_series_clusters(
            ink_clusters, grid_y_component_map, grid_x_component, y_offset=0
        )

        self.assertEqual(len(groups), 2)

    @staticmethod
    def _mask_at_columns(height, width, columns, row):
        mask = np.zeros((height, width), dtype=bool)
        mask[row, list(columns)] = True
        return mask

    def test_shades_each_below_threshold_still_merge_into_one_series(self):
        # Dense line crossings can split one physical line's ink into shades that
        # each resolve on a different, largely disjoint subset of columns
        # (whichever shade happens to win that segment) -- so neither alone
        # clears the default 50% resolved-fraction bar, yet together they cover
        # most of the width and clearly trace the SAME line (rows agree by
        # construction: both are read off the same diagonal here).
        height, width = 130, 120
        shared = set(range(0, width, 3))  # ~40 cols, spread across the full width
        a_only = set(range(1, 55, 3))  # ~18 cols, left half only
        b_only = set(range(65, 119, 3))  # ~18 cols, right half only
        a_cols, b_cols = shared | a_only, shared | b_only
        self.assertLess(len(a_cols), 0.5 * width, "test setup: a must fail alone")
        self.assertLess(len(b_cols), 0.5 * width, "test setup: b must fail alone")
        self.assertGreaterEqual(
            len(a_cols | b_cols), 0.5 * width, "test setup: union must clear the bar"
        )

        mask_a = self._mask_at_columns(height, width, a_cols, row=20)
        mask_b = self._mask_at_columns(height, width, b_cols, row=21)
        ink_clusters = [
            {"color": (142, 134, 38), "mask": mask_a, "count": int(mask_a.sum())},
            {"color": (124, 116, 6), "mask": mask_b, "count": int(mask_b.sum())},
        ]
        grid_y_component_map = np.zeros(height, dtype=bool)
        grid_x_component = np.array([], dtype=int)

        groups = select_series_clusters(
            ink_clusters, grid_y_component_map, grid_x_component, y_offset=0
        )

        self.assertEqual(len(groups), 1, "the two shades should merge into one line")
        self.assertEqual(groups[0]["resolved_count"], len(a_cols | b_cols))

    def test_narrow_crossing_overlap_does_not_merge_distinct_lines(self):
        # Two genuinely different lines that only happen to sit close together
        # while briefly crossing must NOT be merged just because that crossing
        # gave them enough overlapping, closely-matching columns to otherwise
        # pass: unlike a split shade's overlap (scattered across the whole
        # trajectory), a real crossing's shared columns cluster in one narrow
        # stretch, which is exactly what should disqualify it.
        height, width = 130, 120
        mask_a = self._mask_at_columns(height, width, range(0, 100), row=20)
        mask_b = self._mask_at_columns(height, width, range(80, 120), row=21)
        ink_clusters = [
            {"color": (0, 0, 200), "mask": mask_a, "count": int(mask_a.sum())},
            {"color": (0, 200, 0), "mask": mask_b, "count": int(mask_b.sum())},
        ]
        grid_y_component_map = np.zeros(height, dtype=bool)
        grid_x_component = np.array([], dtype=int)

        groups = select_series_clusters(
            ink_clusters, grid_y_component_map, grid_x_component, y_offset=0
        )

        self.assertEqual(len(groups), 1, "only cluster a clears the bar on its own")
        self.assertEqual(
            groups[0]["resolved_count"],
            100,
            "cluster b's unrelated tail must not have been merged in",
        )


class TestMultilineExtraction(TestCase):
    """
    End-to-end extraction of charts with more than one line: legend-based naming
    and pure color separation with no legend at all. See data/multiline/ for the
    real dashboard screenshots that motivated this (colored legend text, no
    swatch icon; several distinctly colored lines with no legend at all).
    """

    MAX_MEAN_REL_ERROR = 0.05

    @staticmethod
    def _series_by_date(image_path):
        extraction = extract_chart(image_path)
        n_series = len(extraction.time_series[0][1]) if extraction.time_series else 0
        by_series = [dict() for _ in range(n_series)]
        for dt, values in extraction.time_series:
            for series_idx, value in enumerate(values):
                by_series[series_idx].setdefault(dt.date(), value)
        return by_series, extraction.series_names

    @staticmethod
    def _mean_rel_error(expected_col, got_col):
        common = [d for d in expected_col if got_col.get(d) is not None]
        if not common:
            return None
        return float(
            np.mean(
                [abs(got_col[d] - expected_col[d]) / expected_col[d] for d in common]
            )
        )

    def test_legend_names_match_each_line(self):
        """Scenario 1: a legend must name each line correctly, not just detect them."""
        expected = _load_expected_results(MULTILINE_DIR)["multiline_legend"]
        n_expected = len(next(iter(expected.values())))
        expected_by_col = [
            {d: v[i] for d, v in expected.items()} for i in range(n_expected)
        ]
        # CSV column order, fixed by the generator call in tests/data_generation.py
        expected_names = ["Revenue Growth", "Operating Margin"]

        by_series, names = self._series_by_date(
            os.path.join(MULTILINE_DIR, "multiline_legend.png")
        )
        self.assertEqual(len(by_series), n_expected)
        self.assertEqual(set(names), set(expected_names))

        for col_idx, name in enumerate(expected_names):
            series_idx = names.index(name)
            rel_error = self._mean_rel_error(
                expected_by_col[col_idx], by_series[series_idx]
            )
            self.assertIsNotNone(rel_error, f"no overlapping dates for {name!r}")
            self.assertLess(
                rel_error,
                self.MAX_MEAN_REL_ERROR,
                f"{name!r} was matched to the wrong line's values",
            )

    def test_distinct_colors_separate_without_legend(self):
        """Scenario 2: 3 differently-colored, crossing lines with no legend at all."""
        expected = _load_expected_results(MULTILINE_DIR)["multiline_colors"]
        n_expected = len(next(iter(expected.values())))
        expected_by_col = [
            {d: v[i] for d, v in expected.items()} for i in range(n_expected)
        ]

        by_series, names = self._series_by_date(
            os.path.join(MULTILINE_DIR, "multiline_colors.png")
        )
        self.assertEqual(len(by_series), n_expected)
        self.assertEqual(names, [None] * n_expected, "this chart has no legend")

        used_series = set()
        for col_idx, expected_col in enumerate(expected_by_col):
            best_idx, best_error = None, None
            for series_idx, got_col in enumerate(by_series):
                error = self._mean_rel_error(expected_col, got_col)
                if error is not None and (best_error is None or error < best_error):
                    best_idx, best_error = series_idx, error
            self.assertIsNotNone(best_idx, f"no series matched csv column {col_idx}")
            self.assertNotIn(
                best_idx,
                used_series,
                f"csv column {col_idx} matched an already-claimed series -- "
                "colors were not separated into distinct series",
            )
            used_series.add(best_idx)
            self.assertLess(
                best_error,
                self.MAX_MEAN_REL_ERROR,
                f"csv column {col_idx}: closest matching series is still far off",
            )

    def test_real_screenshots_detect_log_scale_and_legend(self):
        """
        Regression test for data/multiline/img.png and img_3.png: both have a
        "small base" log y-axis (ratio close to 1 between consecutive ticks)
        that the previous ratio-based is_log_scale misclassified as linear, and
        img_3.png has a genuine top-left text legend that must be matched to its
        lines. These screenshots have no ground-truth CSV (see
        docs/multiline-extraction.md), so this only checks scale type and
        legend names, not extracted values.
        """
        no_legend = extract_chart(os.path.join(REAL_MULTILINE_DIR, "img.png"))
        self.assertTrue(no_legend.y_is_log, "img.png's log y-axis was not detected")
        self.assertEqual(
            no_legend.series_names,
            [None] * len(no_legend.series_names),
            "img.png has no text legend, only per-line value badges",
        )

        with_legend = extract_chart(os.path.join(REAL_MULTILINE_DIR, "img_3.png"))
        self.assertTrue(with_legend.y_is_log, "img_3.png's log y-axis was not detected")
        self.assertEqual(
            set(with_legend.series_names),
            {
                "NOW: Price Target High",
                "NOW: Price Target",
                "NOW: Price Target Low",
            },
        )

    def test_legend_area_override_is_recorded_and_restricts_the_search(self):
        """
        Regression for extract_chart(..., legend_area=...): the override must
        reach find_legend_entries (not be silently ignored, e.g. dropped
        somewhere between the API and extract_chart), while the omitted case
        keeps behaving exactly as before the search-area split.
        """
        image_path = os.path.join(REAL_MULTILINE_DIR, "img_3.png")
        default = extract_chart(image_path)
        expected_default_area = tuple(
            int(round(v)) for v in default_legend_search_area(default.chart_area)
        )
        self.assertEqual(default.legend_area, expected_default_area)

        # Passing that same region explicitly must reproduce identical results.
        explicit_default = extract_chart(image_path, legend_area=expected_default_area)
        self.assertEqual(explicit_default.legend_area, expected_default_area)
        self.assertEqual(explicit_default.series_names, default.series_names)

        # A small region deep in the plot's interior, away from the legend
        # text, holds no legend rows -- the override must actually restrict
        # the search rather than falling back to the default corner.
        x1, y1, x2, y2 = default.chart_area
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        interior = (mid_x - 20, mid_y - 20, mid_x + 20, mid_y + 20)
        restricted = extract_chart(image_path, legend_area=interior)
        self.assertEqual(restricted.legend_area, interior)
        self.assertEqual(restricted.series_names, [None] * len(restricted.series_names))

    def test_chart_area_override_is_recorded_and_used(self):
        """
        Regression for extract_chart(..., chart_area=...): the override must
        replace auto-detection (cut_chart_area) rather than being ignored, while
        the omitted case keeps auto-detecting exactly as before.
        """
        image_path = os.path.join(REAL_MULTILINE_DIR, "img_3.png")
        default = extract_chart(image_path)

        # Passing the auto-detected box back explicitly must reproduce
        # identical results -- the override path is behaviorally equivalent
        # to auto-detection when given the same box.
        explicit_default = extract_chart(image_path, chart_area=default.chart_area)
        self.assertEqual(explicit_default.chart_area, default.chart_area)
        self.assertEqual(explicit_default.time_series, default.time_series)
        self.assertEqual(explicit_default.series_names, default.series_names)

        # A slightly shrunk box must be honored exactly (not silently widened
        # back to the auto-detected one), and the legend's own default search
        # area -- derived from chart_area -- must shift with it.
        x1, y1, x2, y2 = default.chart_area
        shrunk = (x1 + 5, y1 + 5, x2 - 5, y2 - 5)
        custom = extract_chart(image_path, chart_area=shrunk)
        self.assertEqual(custom.chart_area, shrunk)
        self.assertNotEqual(custom.legend_area, default.legend_area)


class TestScrabStyleExtraction(TestCase):
    """
    End-to-end extraction of the dashboard style in data/scrab/: pale
    off-white background, right-hand y-axis, no title, a colored bullet+name+
    value legend, and step-shaped (quarterly-revised estimate) lines alongside
    a smooth "actual" line -- see tests/data_generation.generate_scrab_style_chart.

    Real data/scrab/*.png screenshots have no ground-truth CSV and their tiny
    axis text pushes OCR to its limits (see docs/fixes.txt), so this exercises
    the same visual design against synthetic charts with known values instead:
    it is deliberately lenient (series count and coverage, not tight value
    accuracy) since the point is catching pipeline regressions on this chart
    shape, not measuring OCR legibility.
    """

    MAX_MEAN_REL_ERROR = 0.2
    MIN_RESOLVED_FRACTION = 0.6

    def test_step_and_actual_series_are_recovered(self):
        expected_results = _load_expected_results(SCRAB_STYLE_DIR)
        self.assertTrue(expected_results, f"no fixtures in {SCRAB_STYLE_DIR}")

        for idx, expected in expected_results.items():
            with self.subTest(chart=idx):
                image_path = os.path.join(SCRAB_STYLE_DIR, f"{idx}.png")
                extraction = extract_chart(image_path)
                n_expected = len(next(iter(expected.values())))
                n_extracted = (
                    len(extraction.time_series[0][1]) if extraction.time_series else 0
                )
                self.assertEqual(
                    n_extracted, n_expected, f"series count mismatch for {idx}"
                )

                extracted_by_date = {
                    dt.date(): values for dt, values in extraction.time_series
                }
                for series_idx in range(n_expected):
                    resolved = [
                        values[series_idx]
                        for values in extracted_by_date.values()
                        if values[series_idx] is not None
                    ]
                    resolved_fraction = len(resolved) / len(extracted_by_date)
                    self.assertGreater(
                        resolved_fraction,
                        self.MIN_RESOLVED_FRACTION,
                        f"{idx} series {series_idx} resolved too few columns",
                    )

                expected_by_col = [
                    {d: v[i] for d, v in expected.items()} for i in range(n_expected)
                ]
                used_series = set()
                for col_idx, expected_col in enumerate(expected_by_col):
                    best_idx, best_error = None, None
                    for series_idx in range(n_expected):
                        got_col = {
                            d: values[series_idx]
                            for d, values in extracted_by_date.items()
                        }
                        common = [d for d in expected_col if got_col.get(d) is not None]
                        if not common:
                            continue
                        error = float(
                            np.mean(
                                [
                                    abs(got_col[d] - expected_col[d]) / expected_col[d]
                                    for d in common
                                ]
                            )
                        )
                        if best_error is None or error < best_error:
                            best_idx, best_error = series_idx, error
                    self.assertIsNotNone(
                        best_idx, f"{idx}: no series matched csv column {col_idx}"
                    )
                    self.assertNotIn(
                        best_idx,
                        used_series,
                        f"{idx}: csv column {col_idx} matched an already-claimed series",
                    )
                    used_series.add(best_idx)
                    self.assertLess(
                        best_error,
                        self.MAX_MEAN_REL_ERROR,
                        f"{idx}: closest matching series for column {col_idx} is still far off",
                    )


class TestDenseCrossingExtraction(TestCase):
    """
    End-to-end extraction of the scrab_style shape with `dense_crossing=True` --
    a noisy "actual" line pinned to the SAME level as its step lines, so it
    crosses every one of them over and over across the whole date range, unlike
    the default scrab_style fixtures where the steps sit comfortably above the
    actual line and barely touch it. This is the shape of the real screenshots
    that motivated it (data/scrab/nvo-pt.png, avgo-pt.png, panw-pt.png) and is
    what exposed two real bugs:

    - chart_extraction.extract_chart's x-axis pixel-to-value formula silently
      dropped the chart area's own left-edge offset whenever geometry.cut_chart_area
      happened to trim no further than it (see the now-removed grid_l), mislabeling
      every point with the wrong date -- invisible on a slow-moving line (nearby
      dates have similar values) but glaring on a fast-moving one.
    - chart_extraction.select_series_clusters gated a color cluster's eligibility
      on its OWN resolved-column coverage before ever trying to merge it with
      others, so a line whose anti-aliasing got split into shades by all those
      crossings (each shade individually under the bar) lost whichever columns
      the other shade would have covered, instead of the two reuniting into one
      fully-covered series (see _is_same_trajectory).

    Unlike TestScrabStyleExtraction this does not require the extracted series
    count to equal the CSV's: an occasional un-merged duplicate shade of a line
    is a known, milder residual (it still resolves correctly, just twice) and
    asserting exact counts here would make this test flaky for the wrong reason.
    What matters is that every true column is recovered accurately and with high
    coverage SOMEWHERE among the extracted series.
    """

    MAX_MEAN_REL_ERROR = 0.05
    MIN_RESOLVED_FRACTION = 0.6

    def test_every_line_is_recovered_despite_dense_crossings(self):
        expected_results = _load_expected_results(DENSE_CROSSING_DIR)
        self.assertTrue(expected_results, f"no fixtures in {DENSE_CROSSING_DIR}")

        for idx, expected in expected_results.items():
            with self.subTest(chart=idx):
                image_path = os.path.join(DENSE_CROSSING_DIR, f"{idx}.png")
                extraction = extract_chart(image_path)
                n_expected = len(next(iter(expected.values())))
                n_extracted = (
                    len(extraction.time_series[0][1]) if extraction.time_series else 0
                )
                self.assertGreaterEqual(
                    n_extracted,
                    n_expected,
                    f"{idx}: fewer series extracted than are in the chart",
                )

                extracted_by_date = {
                    dt.date(): values for dt, values in extraction.time_series
                }
                expected_by_col = [
                    {d: v[i] for d, v in expected.items()} for i in range(n_expected)
                ]
                used_series = set()
                for col_idx, expected_col in enumerate(expected_by_col):
                    best_idx, best_error, best_resolved_fraction = None, None, None
                    for series_idx in range(n_extracted):
                        if series_idx in used_series:
                            continue
                        got_col = {
                            d: values[series_idx]
                            for d, values in extracted_by_date.items()
                        }
                        common = [d for d in expected_col if got_col.get(d) is not None]
                        if not common:
                            continue
                        error = float(
                            np.mean(
                                [
                                    abs(got_col[d] - expected_col[d]) / expected_col[d]
                                    for d in common
                                ]
                            )
                        )
                        if best_error is None or error < best_error:
                            resolved = sum(v is not None for v in got_col.values())
                            best_idx, best_error, best_resolved_fraction = (
                                series_idx,
                                error,
                                resolved / len(got_col),
                            )
                    self.assertIsNotNone(
                        best_idx, f"{idx}: no series matched csv column {col_idx}"
                    )
                    used_series.add(best_idx)
                    self.assertLess(
                        best_error,
                        self.MAX_MEAN_REL_ERROR,
                        f"{idx}: closest matching series for column {col_idx} is still far off",
                    )
                    self.assertGreater(
                        best_resolved_fraction,
                        self.MIN_RESOLVED_FRACTION,
                        f"{idx}: closest matching series for column {col_idx} resolved too few columns",
                    )


class TestCrossingMultilineExtraction(TestCase):
    """
    End-to-end extraction of a legend-bearing chart where a continuous "Actual"
    line and one quarterly step line cross a controlled, comparable number of
    times (requested as 2 / 5 / 10 / 20 -- see
    tests/data_generation._crossing_actual_series),
    unlike dense_crossing's random-walk actual line whose crossing count is an
    unmeasured property of noise. crossing_multiline_0 (only 2 crossings) is a
    regression test in its own right: with few, widely-spaced crossings the two
    lines linger close together for a long stretch each time, which used to
    split the step line's anti-aliasing into a shade covering 40% of the width
    that _is_same_trajectory's overlap gate could not reunite with the main
    color (docs/series-gaps-diagnosis.md) -- a worse gap than any dense_crossing
    fixture produces, even though it has far fewer crossings.

    As in TestDenseCrossingExtraction, exact series-count equality isn't
    asserted: an occasional un-merged duplicate shade of "Actual" itself (more
    likely as crossing density rises) is a known, milder residual.
    """

    MAX_MEAN_REL_ERROR = 0.05
    MIN_RESOLVED_FRACTION = 0.6

    def test_every_line_is_recovered_at_every_crossing_density(self):
        expected_results = _load_expected_results(CROSSING_MULTILINE_DIR)
        self.assertTrue(expected_results, f"no fixtures in {CROSSING_MULTILINE_DIR}")

        for idx, expected in expected_results.items():
            with self.subTest(chart=idx):
                image_path = os.path.join(CROSSING_MULTILINE_DIR, f"{idx}.png")
                extraction = extract_chart(image_path)
                n_expected = len(next(iter(expected.values())))
                n_extracted = (
                    len(extraction.time_series[0][1]) if extraction.time_series else 0
                )
                self.assertGreaterEqual(
                    n_extracted,
                    n_expected,
                    f"{idx}: fewer series extracted than are in the chart",
                )

                extracted_by_date = {
                    dt.date(): values for dt, values in extraction.time_series
                }
                expected_by_col = [
                    {d: v[i] for d, v in expected.items()} for i in range(n_expected)
                ]
                used_series = set()
                for col_idx, expected_col in enumerate(expected_by_col):
                    best_idx, best_error, best_resolved_fraction = None, None, None
                    for series_idx in range(n_extracted):
                        if series_idx in used_series:
                            continue
                        got_col = {
                            d: values[series_idx]
                            for d, values in extracted_by_date.items()
                        }
                        common = [d for d in expected_col if got_col.get(d) is not None]
                        if not common:
                            continue
                        error = float(
                            np.mean(
                                [
                                    abs(got_col[d] - expected_col[d]) / expected_col[d]
                                    for d in common
                                ]
                            )
                        )
                        if best_error is None or error < best_error:
                            resolved = sum(v is not None for v in got_col.values())
                            best_idx, best_error, best_resolved_fraction = (
                                series_idx,
                                error,
                                resolved / len(got_col),
                            )
                    self.assertIsNotNone(
                        best_idx, f"{idx}: no series matched csv column {col_idx}"
                    )
                    used_series.add(best_idx)
                    self.assertLess(
                        best_error,
                        self.MAX_MEAN_REL_ERROR,
                        f"{idx}: closest matching series for column {col_idx} is still far off",
                    )
                    self.assertGreater(
                        best_resolved_fraction,
                        self.MIN_RESOLVED_FRACTION,
                        f"{idx}: closest matching series for column {col_idx} resolved too few columns",
                    )
