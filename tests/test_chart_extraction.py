import csv
import os
from datetime import date, datetime
from typing import Optional
from unittest import TestCase

import numpy as np

from chart_extraction import (
    adjust_knots_to_grid,
    extract_chart,
    extract_time_series,
    extract_time_series_from_chart_area,
    select_axis_tick_group,
    select_series_clusters,
)
from function import Linear
from ocr_utils import texts_to_numbers
from tests.test_data import adjust_knots_to_grid_data, extract_series_interference_data

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LINEAR_SCALE_DIR = os.path.join(TEST_DATA_DIR, "linear_scaled")
LOG_SCALE_DIR = os.path.join(TEST_DATA_DIR, "log_scaled")
IN_AREA_TEXT_DIR = os.path.join(TEST_DATA_DIR, "in_area_text")
MULTILINE_DIR = os.path.join(TEST_DATA_DIR, "multiline")
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
        grid_l=0,
        x_offset=0,
        y_offset=0,
    )
    return [None if v[0] is None else round(v[0], 1) for _, v in time_series]


class TestChartAreaInterference(TestCase):
    """
    Non-series ink inside the plotting area (legend text, value badges, stray
    markers) must be rejected rather than averaged into the series.
    """

    def test_interfering_ink_is_rejected(self):
        for description, ascii_rows, expected in extract_series_interference_data:
            with self.subTest(case=description):
                self.assertEqual(_read_ascii_chart(ascii_rows), expected, description)


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
