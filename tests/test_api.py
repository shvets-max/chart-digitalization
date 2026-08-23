import os
import tempfile
from unittest import TestCase, mock

from fastapi import HTTPException

from api.api import (
    PromoteToTestsetBody,
    SeriesEditBody,
    StoredChart,
    _apply_series_edits,
    _series_payload,
    approve_staged_testset_entry,
    get_staged_testset_image,
    get_testset_categories,
    list_staged_testset,
    promote_to_testset,
    reject_staged_testset_entry,
    store,
)
from src.chart_extraction import ChartExtraction
from src.function import Linear


def _extraction(series_names, time_series, chart_area=(0, 0, 10, 10)):
    y_scale = Linear(knots=[0, 100], values=[0, 100])
    return ChartExtraction(
        image_size=(10, 10),
        chart_area=chart_area,
        time_series=time_series,
        x_pixels=list(range(len(time_series))),
        y_scale=y_scale,
        series_names=series_names,
    )


class TestSeriesPayload(TestCase):
    """
    api._series_payload feeds the frontend's legend (static/app.js reads
    series[].name). It must surface ChartExtraction.series_names -- detected
    from the chart's legend -- rather than hardcoding a generic label, or a
    correctly-detected legend never reaches the UI.
    """

    def test_uses_detected_legend_name(self):
        extraction = _extraction(
            series_names=["Revenue"],
            time_series=[(0.0, [10.0]), (1.0, [20.0])],
        )
        payload = _series_payload(extraction)
        self.assertEqual(payload[0]["name"], "Revenue")

    def test_falls_back_to_generic_name_when_no_legend_match(self):
        extraction = _extraction(
            series_names=[None, None],
            time_series=[(0.0, [10.0, 5.0]), (1.0, [20.0, 6.0])],
        )
        payload = _series_payload(extraction)
        self.assertEqual([s["name"] for s in payload], ["series 1", "series 2"])

    def test_mixed_legend_matches_fall_back_per_series(self):
        extraction = _extraction(
            series_names=["Revenue", None],
            time_series=[(0.0, [10.0, 5.0]), (1.0, [20.0, 6.0])],
        )
        payload = _series_payload(extraction)
        self.assertEqual([s["name"] for s in payload], ["Revenue", "series 2"])

    def test_names_override_replaces_the_detected_name(self):
        extraction = _extraction(
            series_names=["Revenue", "Costs"],
            time_series=[(0.0, [10.0, 5.0]), (1.0, [20.0, 6.0])],
        )
        payload = _series_payload(extraction, names_override=["Sales", None])
        self.assertEqual([s["name"] for s in payload], ["Sales", "Costs"])

    def test_names_override_swaps_two_series(self):
        extraction = _extraction(
            series_names=["Revenue", "Costs"],
            time_series=[(0.0, [10.0, 5.0]), (1.0, [20.0, 6.0])],
        )
        payload = _series_payload(extraction, names_override=["Costs", "Revenue"])
        self.assertEqual([s["name"] for s in payload], ["Costs", "Revenue"])

    def test_empty_override_entry_falls_back_to_the_detected_name(self):
        extraction = _extraction(
            series_names=["Revenue"],
            time_series=[(0.0, [10.0]), (1.0, [20.0])],
        )
        payload = _series_payload(extraction, names_override=[""])
        self.assertEqual(payload[0]["name"], "Revenue")

    def test_removed_flags_the_series_but_keeps_its_points(self):
        extraction = _extraction(
            series_names=["Revenue", "Costs"],
            time_series=[(0.0, [10.0, 5.0]), (1.0, [20.0, 6.0])],
        )
        payload = _series_payload(extraction, removed={0})
        self.assertEqual([s["removed"] for s in payload], [True, False])
        # Points stay in place so index-aligned callers (e.g. the shared x-axis
        # row read off series[0]) keep working once a series is removed.
        self.assertEqual(len(payload[0]["points"]), 2)

    def test_no_removed_set_flags_nothing(self):
        extraction = _extraction(
            series_names=["Revenue"],
            time_series=[(0.0, [10.0]), (1.0, [20.0])],
        )
        payload = _series_payload(extraction)
        self.assertEqual(payload[0]["removed"], False)


class TestApplySeriesEdits(TestCase):
    """
    api._apply_series_edits backs the hand-drawn line corrections. The frontend
    drives undo/redo by re-sending an earlier edit list, so replaying a list must
    depend only on that list -- never on what was applied before it.
    """

    def setUp(self):
        # The y scale is the identity, so a y pixel of 2 reads as the value 2.0.
        # Ten columns, at x pixels 0..9.
        self.extraction = _extraction(
            series_names=["a", "b"],
            time_series=[(float(i), [1.0, 9.0]) for i in range(10)],
        )
        self.base = [(x, list(values)) for x, values in self.extraction.time_series]

    def _values(self, edits, series_index=0):
        result = _apply_series_edits(self.extraction, self.base, edits)
        return [values[series_index] for _, values in result]

    def test_overwrites_only_the_columns_the_line_spans(self):
        edit = SeriesEditBody(series_index=0, x1=2, y1=2, x2=6, y2=6)
        self.assertEqual(
            self._values([edit]),
            [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0],
        )

    def test_leaves_other_series_and_the_pristine_base_untouched(self):
        edit = SeriesEditBody(series_index=0, x1=0, y1=2, x2=9, y2=2)
        self.assertEqual(self._values([edit], series_index=1), [9.0] * 10)
        self.assertEqual([values[0] for _, values in self.base], [1.0] * 10)

    def test_endpoints_may_sit_outside_the_plot_area(self):
        # y = -50 px is far above the chart; the value clamps to the axis top.
        edit = SeriesEditBody(series_index=0, x1=0, y1=-50, x2=9, y2=-50)
        self.assertEqual(self._values([edit]), [0.0] * 10)

    def test_later_edits_overwrite_overlapping_earlier_ones(self):
        edits = [
            SeriesEditBody(series_index=0, x1=0, y1=2, x2=9, y2=2),
            SeriesEditBody(series_index=0, x1=4, y1=5, x2=6, y2=5),
        ]
        self.assertEqual(
            self._values(edits),
            [2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0],
        )

    def test_replaying_an_earlier_list_restores_it_exactly(self):
        edit = SeriesEditBody(series_index=0, x1=2, y1=2, x2=6, y2=6)
        self._values([edit])
        self.assertEqual(self._values([]), [1.0] * 10)

    def test_anchored_endpoints_meet_their_datapoints_and_flatten_the_span(self):
        wiggly = [0.0, 5.0, 2.0, 9.0, 1.0, 8.0, 3.0, 7.0, 4.0, 6.0]
        extraction = _extraction(
            series_names=["a"],
            time_series=[(float(i), [v]) for i, v in enumerate(wiggly)],
        )
        base = [(x, list(values)) for x, values in extraction.time_series]
        # The clicked pixels are deliberately wrong: the anchored datapoints win.
        edit = SeriesEditBody(
            series_index=0, x1=0, y1=0, x2=0, y2=0, anchor1=2, anchor2=6
        )
        result = _apply_series_edits(extraction, base, [edit])
        # Both endpoints keep their own value (2.0, 3.0) and the 9.0/1.0/8.0
        # datapoints between them are gone, replaced by the straight line.
        self.assertEqual(
            [values[0] for _, values in result],
            [0.0, 5.0, 2.0, 2.25, 2.5, 2.75, 3.0, 7.0, 4.0, 6.0],
        )

    def test_anchor_resolves_against_the_result_of_earlier_edits(self):
        edits = [
            SeriesEditBody(series_index=0, x1=0, y1=5, x2=4, y2=5),
            SeriesEditBody(series_index=0, x1=0, y1=0, x2=8, y2=9, anchor1=4),
        ]
        # The second line starts from the 5.0 the first one wrote, not the base 1.0.
        self.assertEqual(
            self._values(edits),
            [5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0],
        )

    def test_stale_anchor_falls_back_to_the_clicked_position(self):
        edit = SeriesEditBody(
            series_index=0, x1=2, y1=2, x2=6, y2=6, anchor1=99, anchor2=None
        )
        self.assertEqual(
            self._values([edit]),
            [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0],
        )

    def test_anchor_to_a_gap_falls_back_to_the_clicked_position(self):
        extraction = _extraction(
            series_names=["a"],
            time_series=[(float(i), [None if i == 4 else 1.0]) for i in range(10)],
        )
        base = [(x, list(values)) for x, values in extraction.time_series]
        edit = SeriesEditBody(series_index=0, x1=4, y1=6, x2=8, y2=6, anchor1=4)
        result = _apply_series_edits(extraction, base, [edit])
        self.assertEqual([values[0] for _, values in result][4:9], [6.0] * 5)

    def test_delete_clears_the_span_to_gaps(self):
        edit = SeriesEditBody(series_index=0, x1=3, y1=0, x2=6, y2=0, kind="delete")
        self.assertEqual(
            self._values([edit]),
            [1.0, 1.0, 1.0, None, None, None, None, 1.0, 1.0, 1.0],
        )

    def test_delete_honours_anchors_and_ignores_the_clicked_y(self):
        # y values are nonsense here: a delete only cares about the x span.
        edit = SeriesEditBody(
            series_index=0,
            x1=0,
            y1=99,
            x2=0,
            y2=99,
            anchor1=2,
            anchor2=4,
            kind="delete",
        )
        self.assertEqual(
            self._values([edit]),
            [1.0, 1.0, None, None, None, 1.0, 1.0, 1.0, 1.0, 1.0],
        )

    def test_delete_leaves_other_series_alone(self):
        edit = SeriesEditBody(series_index=0, x1=0, y1=0, x2=9, y2=0, kind="delete")
        self.assertEqual(self._values([edit], series_index=1), [9.0] * 10)

    def test_delete_is_undone_by_replaying_without_it(self):
        edit = SeriesEditBody(series_index=0, x1=3, y1=0, x2=6, y2=0, kind="delete")
        self._values([edit])
        self.assertEqual(self._values([]), [1.0] * 10)

    def test_a_line_can_be_drawn_back_over_a_deleted_span(self):
        edits = [
            SeriesEditBody(series_index=0, x1=3, y1=0, x2=6, y2=0, kind="delete"),
            SeriesEditBody(series_index=0, x1=3, y1=4, x2=6, y2=4),
        ]
        self.assertEqual(
            self._values(edits),
            [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0],
        )

    def test_a_zero_width_delete_still_clears_its_column(self):
        edit = SeriesEditBody(series_index=0, x1=5, y1=0, x2=5, y2=0, kind="delete")
        self.assertEqual(self._values([edit])[5], None)

    def test_skips_degenerate_and_out_of_range_edits(self):
        edits = [
            SeriesEditBody(series_index=0, x1=3, y1=1, x2=3, y2=9),  # zero width
            SeriesEditBody(series_index=7, x1=0, y1=1, x2=9, y2=9),  # no such series
        ]
        self.assertEqual(
            _apply_series_edits(self.extraction, self.base, edits), self.base
        )


class TestPromoteToTestset(TestCase):
    """POST .../promote-to-testset resolves the chart's current (corrected/
    renamed/removed) series the same way the CSV export does, and hands them to
    eval.authoring.stage_chart."""

    def _add_chart(self, **overrides) -> StoredChart:
        base_time_series = [(0.0, [10.0, 1.0]), (1.0, [20.0, 2.0])]
        extraction = _extraction(
            series_names=["Revenue", "Dropped"], time_series=list(base_time_series)
        )
        chart = StoredChart(
            chart_id="test-promote-chart",
            filename="my-chart.png",
            media_type="image/png",
            image_bytes=b"fake-png-bytes",
            image_data=None,
            grid_data=None,
            extraction=extraction,
            base_time_series=base_time_series,
            **overrides,
        )
        store.add(chart)
        self.addCleanup(store.remove, chart.chart_id)
        return chart

    def test_stages_corrected_series_excluding_removed(self):
        chart = self._add_chart(removed_series={1})
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch("eval.authoring.STAGING_DIR", tmp_dir):
                meta = promote_to_testset(
                    chart.chart_id,
                    PromoteToTestsetBody(category="scrab_style", notes="looks right"),
                )
            with open(os.path.join(tmp_dir, meta["id"], "series.csv")) as f:
                content = f.read()

        self.assertEqual(meta["category"], "scrab_style")
        self.assertEqual(meta["n_series"], 1)  # "Dropped" excluded
        self.assertEqual(meta["correction_fraction"], 0.0)  # no edits applied yet
        self.assertIn("date;Revenue", content)
        self.assertNotIn("Dropped", content)

    def test_correction_fraction_reflects_hand_drawn_edits(self):
        chart = self._add_chart()
        chart.extraction.time_series = [(0.0, [10.0, 1.0]), (1.0, [99.0, 2.0])]
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch("eval.authoring.STAGING_DIR", tmp_dir),
        ):
            meta = promote_to_testset(
                chart.chart_id, PromoteToTestsetBody(category="scrab_style")
            )
        self.assertAlmostEqual(meta["correction_fraction"], 1 / 4)

    def test_raises_when_chart_has_no_series(self):
        chart = self._add_chart()
        chart.extraction.time_series = []
        with self.assertRaises(HTTPException):
            promote_to_testset(chart.chart_id, PromoteToTestsetBody(category="x"))


class TestTestsetReviewEndpoints(TestCase):
    """The reviewer-facing endpoints (list/image/approve/reject staged entries)
    read/write through eval.authoring.STAGING_DIR and eval.manifest.CANONICAL_DIR
    at call time, so patching those module attributes redirects them to a temp
    dir without touching the real ground-truth store."""

    def _stage_one(self, staging_dir: str, **overrides) -> dict:
        with mock.patch("eval.authoring.STAGING_DIR", staging_dir):
            chart = StoredChart(
                chart_id="test-review-chart",
                filename="chart.png",
                media_type="image/png",
                image_bytes=b"fake-bytes",
                image_data=None,
                grid_data=None,
                extraction=_extraction(
                    series_names=["Actual"], time_series=[(0.0, [1.0])]
                ),
                base_time_series=[(0.0, [1.0])],
            )
            store.add(chart)
            self.addCleanup(store.remove, chart.chart_id)
            return promote_to_testset(
                chart.chart_id,
                PromoteToTestsetBody(category="scrab_style", **overrides),
            )

    def test_get_testset_categories_lists_known_categories(self):
        result = get_testset_categories()
        self.assertIn("scrab_style", result["categories"])

    def test_list_staged_is_empty_for_a_fresh_dir(self):
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch("eval.authoring.STAGING_DIR", tmp_dir),
        ):
            self.assertEqual(list_staged_testset(), {"entries": []})

    def test_list_and_fetch_image_for_a_staged_entry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta = self._stage_one(tmp_dir)
            with mock.patch("eval.authoring.STAGING_DIR", tmp_dir):
                listed = list_staged_testset()
                self.assertEqual([e["id"] for e in listed["entries"]], [meta["id"]])

                response = get_staged_testset_image(meta["id"])
                self.assertEqual(response.body, b"fake-bytes")
                self.assertEqual(response.media_type, "image/png")

    def test_fetch_image_for_unknown_entry_404s(self):
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch("eval.authoring.STAGING_DIR", tmp_dir),
            self.assertRaises(HTTPException) as ctx,
        ):
            get_staged_testset_image("nope")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_approve_moves_entry_and_reject_discards_it(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            staging_dir = os.path.join(tmp_dir, "staging")
            canonical_dir = os.path.join(tmp_dir, "canonical")
            meta_a = self._stage_one(staging_dir)
            meta_b = self._stage_one(staging_dir)

            with (
                mock.patch("eval.authoring.STAGING_DIR", staging_dir),
                mock.patch("eval.manifest.CANONICAL_DIR", canonical_dir),
            ):
                result = approve_staged_testset_entry(meta_a["id"])
                self.assertEqual(result["approved"], meta_a["id"])
                self.assertFalse(os.path.isdir(os.path.join(staging_dir, meta_a["id"])))

                reject_staged_testset_entry(meta_b["id"])
                self.assertFalse(os.path.isdir(os.path.join(staging_dir, meta_b["id"])))

                with self.assertRaises(HTTPException) as ctx:
                    approve_staged_testset_entry("nope")
                self.assertEqual(ctx.exception.status_code, 404)
