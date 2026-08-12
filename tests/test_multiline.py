from unittest import TestCase

import numpy as np

from src.multiline import (
    LEGEND_LEFT_FRACTION,
    LEGEND_TOP_FRACTION,
    cluster_ink_colors,
    default_legend_search_area,
    find_legend_entries,
    match_series_to_legend,
)


def _paint_box(img, box, color):
    left, top, right, bottom = box
    img[top:bottom, left:right] = color


class TestClusterInkColors(TestCase):
    def test_separates_distinct_colors(self):
        color_area = np.zeros((4, 6, 3), dtype=np.uint8)
        color_area[0:2, :] = (0, 0, 200)  # red band (BGR)
        color_area[2:4, :] = (200, 0, 0)  # blue band
        ink_mask = np.ones((4, 6), dtype=bool)

        clusters = cluster_ink_colors(color_area, ink_mask)

        self.assertEqual(len(clusters), 2)
        colors = sorted(c["color"] for c in clusters)
        self.assertEqual(colors, [(0, 0, 200), (200, 0, 0)])

    def test_faint_antialiased_edge_joins_the_core_color(self):
        # A thin line's anti-aliased edge (gray ~243, past CORE_INK_MAX_GRAY) must
        # not form its own cluster: it should be assigned to the nearest solid
        # color discovered from confidently-inked pixels.
        color_area = np.zeros((2, 5, 3), dtype=np.uint8)
        color_area[0, :] = (0, 0, 180)  # solid red core
        color_area[1, :] = (245, 235, 250)  # faint pink edge, close to white
        ink_mask = np.ones((2, 5), dtype=bool)

        clusters = cluster_ink_colors(color_area, ink_mask)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["count"], 10)

    def test_empty_ink_mask_returns_no_clusters(self):
        color_area = np.zeros((3, 3, 3), dtype=np.uint8)
        ink_mask = np.zeros((3, 3), dtype=bool)
        self.assertEqual(cluster_ink_colors(color_area, ink_mask), [])


class TestDefaultLegendSearchArea(TestCase):
    def test_top_left_corner_sized_by_fractions(self):
        chart_area = (100, 50, 500, 350)  # width 400, height 300
        search_area = default_legend_search_area(chart_area)
        self.assertEqual(
            search_area,
            (
                100,
                50,
                100 + LEGEND_LEFT_FRACTION * 400,
                50 + LEGEND_TOP_FRACTION * 300,
            ),
        )


class TestFindLegendEntries(TestCase):
    def test_two_rows_multiword_name_and_trailing_value(self):
        # Scenario: a legend with a multi-word name (OCR emits one token per word,
        # as it does for the real dashboards in data/multiline/) and a second row
        # whose trailing token is a value badge that must be stripped from the name.
        width, height = 200, 150
        color_img = np.full((height, width, 3), 255, dtype=np.uint8)
        texts = ["Revenue", "Growth", "Costs", "42"]
        bboxes = [
            [5, 5, 45, 16],
            [50, 5, 85, 16],
            [5, 22, 40, 33],
            [45, 22, 60, 33],
        ]
        red, blue = (30, 30, 200), (200, 30, 30)
        _paint_box(color_img, bboxes[0], red)
        _paint_box(color_img, bboxes[1], red)
        _paint_box(color_img, bboxes[2], blue)
        _paint_box(color_img, bboxes[3], blue)

        entries = find_legend_entries(texts, bboxes, (0, 0, width, height), color_img)

        self.assertEqual([e["name"] for e in entries], ["Revenue Growth", "Costs"])
        self.assertLess(
            np.linalg.norm(np.array(entries[0]["color"]) - np.array(red)), 20
        )
        self.assertLess(
            np.linalg.norm(np.array(entries[1]["color"]) - np.array(blue)), 20
        )

    def test_no_candidates_in_legend_region_returns_empty(self):
        width, height = 200, 150
        color_img = np.full((height, width, 3), 255, dtype=np.uint8)
        # A token far outside the legend's corner region (e.g. an axis label).
        texts = ["100"]
        bboxes = [[width - 30, height - 20, width - 5, height - 8]]

        entries = find_legend_entries(texts, bboxes, (0, 0, width, height), color_img)

        self.assertEqual(entries, [])

    def test_leading_bullet_icon_and_trailing_scale_toggle_are_stripped(self):
        # Scenario observed on data/scrab/anet-peg.png: a dashboard legend row is
        # "<bullet icon> ANET: PEG Ratio (1-Year Forward) 1.84 Lin v", where the
        # bullet is OCR garbage with no alphanumeric content, and "Lin"/"v" is a
        # scale-toggle control sharing the legend's row, past the value badge.
        width, height = 400, 150
        color_img = np.full((height, width, 3), 255, dtype=np.uint8)
        texts = ["●", "ANET:", "PEG", "1.84", "Lin"]
        bboxes = [
            [5, 5, 11, 11],
            [20, 5, 47, 16],
            [51, 5, 69, 16],
            [201, 5, 219, 16],
            [252, 5, 275, 16],
        ]
        red = (30, 30, 200)
        for box in bboxes[1:3]:
            _paint_box(color_img, box, red)

        entries = find_legend_entries(texts, bboxes, (0, 0, width, height), color_img)

        self.assertEqual([e["name"] for e in entries], ["ANET: PEG"])

    def test_tick_label_row_is_not_mistaken_for_a_legend(self):
        width, height = 200, 150
        color_img = np.full((height, width, 3), 255, dtype=np.uint8)
        texts = ["10", "20", "30"]
        bboxes = [[5, 5, 20, 16], [25, 5, 40, 16], [45, 5, 60, 16]]
        for box in bboxes:
            _paint_box(color_img, box, (0, 0, 0))

        entries = find_legend_entries(texts, bboxes, (0, 0, width, height), color_img)

        self.assertEqual(entries, [])

    def test_explicit_search_area_finds_a_legend_outside_the_default_corner(self):
        # A legend placed bottom-right sits entirely outside the default
        # top-left search region, so it's invisible without a `search_area`
        # override -- e.g. one the user highlighted in the UI.
        width, height = 300, 200
        color_img = np.full((height, width, 3), 255, dtype=np.uint8)
        texts = ["Total", "Revenue"]
        bboxes = [[210, 170, 240, 185], [245, 170, 285, 185]]
        for box in bboxes:
            _paint_box(color_img, box, (30, 30, 200))
        chart_area = (0, 0, width, height)

        self.assertEqual(find_legend_entries(texts, bboxes, chart_area, color_img), [])

        entries = find_legend_entries(
            texts, bboxes, chart_area, color_img, search_area=(200, 150, 300, 200)
        )
        self.assertEqual([e["name"] for e in entries], ["Total Revenue"])

    def test_explicit_search_area_replaces_the_default_rather_than_widening_it(self):
        # These tokens sit inside the default top-left region, so they'd be
        # found with no override at all. Passing a `search_area` that no
        # longer covers them must still exclude them -- an override replaces
        # the default search region, it doesn't just add to it.
        width, height = 300, 200
        color_img = np.full((height, width, 3), 255, dtype=np.uint8)
        texts = ["Total", "Revenue"]
        bboxes = [[10, 10, 40, 22], [45, 10, 85, 22]]
        for box in bboxes:
            _paint_box(color_img, box, (30, 30, 200))
        chart_area = (0, 0, width, height)

        entries = find_legend_entries(
            texts, bboxes, chart_area, color_img, search_area=(100, 100, 300, 200)
        )
        self.assertEqual(entries, [])


class TestMatchSeriesToLegend(TestCase):
    def test_matches_nearest_color(self):
        series_colors = [(0, 0, 200), (200, 0, 0), (0, 200, 0)]
        legend_entries = [
            {"name": "Red", "color": (0, 0, 190)},
            {"name": "Blue", "color": (190, 0, 0)},
        ]
        names = match_series_to_legend(series_colors, legend_entries)
        self.assertEqual(names, ["Red", "Blue", None])

    def test_too_far_is_unmatched(self):
        # Fewer legend entries than series: unmatched series stay too far and are
        # left as None rather than forced onto a distant color.
        names = match_series_to_legend(
            [(0, 0, 200), (0, 255, 0)],
            [{"name": "Far", "color": (255, 255, 255)}],
            max_distance=10,
        )
        self.assertEqual(names, [None, None])

    def test_each_legend_entry_used_at_most_once(self):
        # Two series both within range of the same single legend entry: only the
        # nearer one (series_colors[1], distance 5 vs 10) may claim it.
        series_colors = [(0, 0, 190), (0, 0, 205)]
        legend_entries = [{"name": "Red", "color": (0, 0, 200)}]
        names = match_series_to_legend(series_colors, legend_entries)
        self.assertEqual(names, [None, "Red"])

    def test_full_coverage_when_legend_has_enough_entries(self):
        # len(legend_entries) >= len(series_colors): every series must be
        # assigned, even past max_distance, choosing the globally minimal-
        # distance pairing rather than leaving any series unmatched.
        series_colors = [(0, 0, 200), (200, 0, 0)]
        legend_entries = [
            {"name": "Red", "color": (0, 0, 190)},
            {"name": "Blue", "color": (190, 0, 0)},
            {"name": "Unused", "color": (0, 200, 0)},
        ]
        names = match_series_to_legend(series_colors, legend_entries, max_distance=1)
        self.assertEqual(names, ["Red", "Blue"])

    def test_full_coverage_minimizes_total_distance(self):
        # Both series are nearest to "A"; a per-series-nearest-first pick would
        # give one of them "A" and leave the other stuck with the far entry "C"
        # (distance 90). The minimal-total-distance assignment instead routes
        # series 1 to "B" (distance 6), which is far cheaper overall.
        series_colors = [(0, 0, 100), (0, 0, 104)]
        legend_entries = [
            {"name": "A", "color": (0, 0, 101)},
            {"name": "B", "color": (0, 0, 110)},
            {"name": "C", "color": (0, 0, 190)},
        ]
        names = match_series_to_legend(series_colors, legend_entries)
        self.assertEqual(names, ["A", "B"])
