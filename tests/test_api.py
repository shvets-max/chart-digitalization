from unittest import TestCase

from api import _series_payload
from chart_extraction import ChartExtraction
from function import Linear


def _extraction(series_names, time_series):
    y_scale = Linear(knots=[0, 100], values=[0, 100])
    return ChartExtraction(
        image_size=(10, 10),
        chart_area=(0, 0, 10, 10),
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
