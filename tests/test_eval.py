import os
import tempfile
from datetime import date
from typing import ClassVar
from unittest import TestCase

from eval import history as eval_history
from eval import promote as eval_promote
from eval.align import align_extracted_to_expected
from eval.authoring import correction_fraction, list_staged, stage_chart
from eval.manifest import (
    CATEGORIES,
    REPO_ROOT,
    discover_entries,
    load_ground_truth,
)
from eval.metrics import (
    mae,
    normalized_mae,
    resolved_fraction,
    rmse,
    series_count_match,
)
from eval.report import build_report
from eval.run_eval import aggregate, compare

D1, D2, D3 = date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)


class TestMetrics(TestCase):
    """Unit tests for eval.metrics against small, hand-built series."""

    def test_resolved_fraction_counts_missing_and_none_cells(self):
        expected = {D1: [1.0], D2: [2.0], D3: [3.0]}
        extracted = {D1: [1.0], D2: [None]}  # D3 missing entirely, D2 unresolved
        self.assertAlmostEqual(resolved_fraction(expected, extracted, 1), 1 / 3)

    def test_resolved_fraction_empty_expected_is_zero(self):
        self.assertEqual(resolved_fraction({}, {}, 1), 0.0)

    def test_mae_only_over_jointly_resolved_cells(self):
        expected = {D1: [10.0], D2: [20.0], D3: [30.0]}
        extracted = {D1: [12.0], D2: [None], D3: [30.0]}
        self.assertAlmostEqual(mae(expected, extracted, 1), 1.0)  # (2 + 0) / 2

    def test_mae_none_when_nothing_resolved(self):
        expected = {D1: [10.0]}
        extracted = {D1: [None]}
        self.assertIsNone(mae(expected, extracted, 1))

    def test_rmse_penalizes_larger_errors_more_than_mae(self):
        expected = {D1: [0.0], D2: [0.0]}
        extracted = {D1: [1.0], D2: [3.0]}
        self.assertAlmostEqual(mae(expected, extracted, 1), 2.0)
        self.assertGreater(rmse(expected, extracted, 1), mae(expected, extracted, 1))

    def test_normalized_mae_divides_by_value_range(self):
        expected = {D1: [0.0], D2: [100.0]}
        extracted = {D1: [10.0], D2: [100.0]}
        self.assertAlmostEqual(
            normalized_mae(expected, extracted, 1), 0.05
        )  # mae=5, range=100

    def test_normalized_mae_none_for_constant_ground_truth(self):
        expected = {D1: [5.0], D2: [5.0]}
        extracted = {D1: [5.0], D2: [6.0]}
        self.assertIsNone(normalized_mae(expected, extracted, 1))

    def test_series_count_match(self):
        self.assertTrue(series_count_match(2, 2))
        self.assertFalse(series_count_match(2, 1))


class TestAlignExtractedToExpected(TestCase):
    """extract_chart's series order need not match the CSV's column order (see
    tests/test_chart_extraction.py's TestMultilineExtraction) -- alignment must
    recover the right pairing by value, not position."""

    def test_reorders_swapped_series(self):
        expected = {D1: [1.0, 100.0], D2: [2.0, 200.0]}
        # Extraction found the series in the opposite order from the CSV columns.
        raw = [(D1, [100.0, 1.0]), (D2, [200.0, 2.0])]
        aligned = align_extracted_to_expected(
            expected, raw, n_series_expected=2, n_series_extracted=2
        )
        self.assertEqual(aligned[D1], [1.0, 100.0])
        self.assertEqual(aligned[D2], [2.0, 200.0])

    def test_missing_extracted_series_yields_none_column(self):
        expected = {D1: [1.0, 100.0]}
        raw = [(D1, [100.0])]  # only one series extracted out of two expected
        aligned = align_extracted_to_expected(
            expected, raw, n_series_expected=2, n_series_extracted=1
        )
        self.assertEqual(aligned[D1], [None, 100.0])

    def test_no_extracted_series_yields_all_none(self):
        expected = {D1: [1.0]}
        aligned = align_extracted_to_expected(
            expected, [], n_series_expected=1, n_series_extracted=0
        )
        self.assertEqual(aligned[D1], [None])


class TestManifest(TestCase):
    """eval.manifest against the real tests/data/ fixtures."""

    def test_discover_entries_pairs_every_category(self):
        entries = discover_entries()
        found_categories = {e.category for e in entries}
        self.assertEqual(found_categories, set(CATEGORIES))
        for entry in entries:
            self.assertTrue(os.path.isfile(os.path.join(REPO_ROOT, entry.image_path)))
            self.assertTrue(os.path.isfile(os.path.join(REPO_ROOT, entry.csv_path)))

    def test_load_ground_truth_parses_wide_csv(self):
        entries = discover_entries(categories=("multiline",))
        entry = next(e for e in entries if e.id == "multiline_legend")
        expected, n_series = load_ground_truth(entry.csv_path)
        self.assertEqual(n_series, 2)
        self.assertTrue(all(len(v) == 2 for v in expected.values()))


class TestAggregateAndCompare(TestCase):
    """eval.run_eval's aggregation and regression-gate logic, without running extraction."""

    def _row(self, category, resolved_fraction_, mae_norm, count_match=True):
        return {
            "id": f"{category}-x",
            "category": category,
            "n_series_expected": 1,
            "n_series_extracted": 1 if count_match else 2,
            "series_count_match": count_match,
            "resolved_fraction": resolved_fraction_,
            "mae": mae_norm,
            "rmse": mae_norm,
            "mae_norm": mae_norm,
            "runtime_s": 0.1,
        }

    def test_aggregate_groups_by_category(self):
        rows = [
            self._row("a", 1.0, 0.01),
            self._row("a", 0.5, 0.03),
            self._row("b", 1.0, 0.02),
        ]
        report = aggregate(rows)
        self.assertEqual(report["overall"]["n_charts"], 3)
        self.assertAlmostEqual(report["by_category"]["a"]["resolved_fraction"], 0.75)
        self.assertEqual(report["by_category"]["b"]["n_charts"], 1)

    def test_compare_flags_resolved_fraction_drop_past_tolerance(self):
        baseline = aggregate([self._row("a", 0.90, 0.01)])
        current = aggregate([self._row("a", 0.80, 0.01)])  # 10pp drop
        problems = compare(current, baseline)
        self.assertTrue(any("resolved_fraction" in p for p in problems))

    def test_compare_ignores_small_fluctuation(self):
        baseline = aggregate([self._row("a", 0.90, 0.01)])
        current = aggregate([self._row("a", 0.895, 0.0105)])  # within tolerance
        self.assertEqual(compare(current, baseline), [])

    def test_compare_flags_mae_norm_relative_increase(self):
        baseline = aggregate([self._row("a", 1.0, 0.01)])
        current = aggregate([self._row("a", 1.0, 0.02)])  # +100% relative
        problems = compare(current, baseline)
        self.assertTrue(any("mae_norm" in p for p in problems))

    def test_compare_ignores_category_missing_from_current(self):
        baseline = aggregate(
            [self._row("a", 0.90, 0.01), self._row("removed", 0.90, 0.01)]
        )
        current = aggregate([self._row("a", 0.90, 0.01)])
        self.assertEqual(compare(current, baseline), [])


def _fake_report(git_sha: str, resolved_fraction_: float) -> dict:
    """A full run_eval.run()-shaped report, for exercising history.py/report.py
    without running actual extraction."""
    category_stats = {
        "n_charts": 1,
        "resolved_fraction": resolved_fraction_,
        "mae": 1.0,
        "rmse": 1.0,
        "mae_norm": 0.1,
        "series_count_accuracy": 1.0,
        "runtime_s": 0.1,
    }
    return {
        "run_at": "2026-01-01T00:00:00+00:00",
        "git_sha": git_sha,
        "manifest": "eval/ground_truth/v1.jsonl",
        "overall": dict(category_stats),
        "by_category": {"linear_scaled": dict(category_stats)},
        "per_chart": [
            {
                "id": "linear_scaled_0",
                "category": "linear_scaled",
                "resolved_fraction": resolved_fraction_,
                "mae": 1.0,
                "rmse": 1.0,
                "mae_norm": 0.1,
                "series_count_match": True,
                "runtime_s": 0.1,
            }
        ],
    }


class TestHistory(TestCase):
    """eval.history's SQLite-backed record/query, against a temp db per test."""

    def test_record_and_list_runs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "eval.db")
            eval_history.record_run(_fake_report("aaa111", 1.0), db_path)
            eval_history.record_run(_fake_report("bbb222", 0.8), db_path)

            runs = eval_history.list_runs(db_path)
            self.assertEqual([r["git_sha"] for r in runs], ["aaa111", "bbb222"])

    def test_category_history_is_ordered_oldest_first(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "eval.db")
            eval_history.record_run(_fake_report("aaa111", 1.0), db_path)
            eval_history.record_run(_fake_report("bbb222", 0.8), db_path)

            history = eval_history.category_history("linear_scaled", db_path)
            self.assertEqual([row["resolved_fraction"] for row in history], [1.0, 0.8])

            overall = eval_history.category_history("overall", db_path)
            self.assertEqual(len(overall), 2)

    def test_categories_lists_overall_and_chart_categories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "eval.db")
            eval_history.record_run(_fake_report("aaa111", 1.0), db_path)
            self.assertEqual(
                set(eval_history.categories(db_path)), {"overall", "linear_scaled"}
            )

    def test_chart_history_tracks_one_chart_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "eval.db")
            eval_history.record_run(_fake_report("aaa111", 1.0), db_path)
            eval_history.record_run(_fake_report("bbb222", 0.8), db_path)

            history = eval_history.chart_history("linear_scaled_0", db_path)
            self.assertEqual([row["resolved_fraction"] for row in history], [1.0, 0.8])


class TestReport(TestCase):
    """eval.report.build_report against a temp db."""

    def test_empty_db_reports_no_runs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "eval.db")
            html = build_report(db_path)
            self.assertIn("No eval runs recorded", html)

    def test_renders_category_section_with_chart_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "eval.db")
            eval_history.record_run(_fake_report("aaa111", 1.0), db_path)

            html = build_report(db_path)
            self.assertIn("linear_scaled", html)
            self.assertIn("data:image/png;base64,", html)


def _series_point(x_value: str, y_value) -> dict:
    return {"x_pixel": 0, "y_pixel": 0.0, "x_value": x_value, "y_value": y_value}


class TestCorrectionFraction(TestCase):
    """eval.authoring.correction_fraction against small, hand-built time series."""

    def test_counts_changed_cells_across_series(self):
        base = [(D1, [1.0, 2.0]), (D2, [3.0, 4.0])]
        corrected = [(D1, [1.0, 2.0]), (D2, [3.0, 5.0])]  # 1 of 4 cells changed
        self.assertAlmostEqual(correction_fraction(base, corrected), 0.25)

    def test_none_vs_value_counts_as_changed(self):
        base = [(D1, [1.0])]
        corrected = [(D1, [None])]
        self.assertEqual(correction_fraction(base, corrected), 1.0)

    def test_none_when_shapes_differ(self):
        self.assertIsNone(correction_fraction([(D1, [1.0])], []))

    def test_none_when_no_series_at_all(self):
        self.assertIsNone(correction_fraction([], []))


class TestStageChart(TestCase):
    """eval.authoring.stage_chart writes image/series.csv/meta.json into staging."""

    SERIES: ClassVar = [
        {
            "name": "Revenue",
            "removed": False,
            "points": [
                _series_point("2024-01-01T00:00:00", 10.0),
                _series_point("2024-01-02T00:00:00", 20.0),
            ],
        },
        {
            "name": "Dropped",
            "removed": True,
            "points": [
                _series_point("2024-01-01T00:00:00", 1.0),
                _series_point("2024-01-02T00:00:00", 2.0),
            ],
        },
    ]

    def test_writes_image_csv_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta = stage_chart(
                image_bytes=b"fake-png-bytes",
                image_suffix=".png",
                series=self.SERIES,
                category="scrab_style",
                correction_fraction=0.1,
                source="production",
                annotator="a@b.com",
                notes="fixed a gap",
                staging_dir=tmp_dir,
            )
            entry_dir = os.path.join(tmp_dir, meta["id"])

            self.assertTrue(meta["id"].startswith("scrab_style_"))
            self.assertEqual(meta["category"], "scrab_style")
            self.assertEqual(meta["n_series"], 1)  # "Dropped" is removed
            self.assertEqual(meta["correction_fraction"], 0.1)
            self.assertEqual(meta["annotator"], "a@b.com")

            with open(os.path.join(entry_dir, "image.png"), "rb") as f:
                self.assertEqual(f.read(), b"fake-png-bytes")

            with open(os.path.join(entry_dir, "series.csv")) as f:
                content = f.read()
            self.assertIn("date;Revenue", content)
            self.assertNotIn("Dropped", content)
            self.assertIn("2024-01-01;10.0", content)

    def test_staged_entry_is_listed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta = stage_chart(
                image_bytes=b"x",
                image_suffix=".png",
                series=self.SERIES,
                category="scrab_style",
                staging_dir=tmp_dir,
            )
            staged = list_staged(tmp_dir)
            self.assertEqual([s["id"] for s in staged], [meta["id"]])

    def test_list_staged_missing_dir_is_empty(self):
        self.assertEqual(list_staged("/no/such/dir"), [])


class TestPromote(TestCase):
    """eval.promote.approve/reject move or discard a staged entry."""

    SERIES: ClassVar = [
        {
            "name": "Actual",
            "removed": False,
            "points": [_series_point("2024-01-01T00:00:00", 1.0)],
        }
    ]

    def _stage(self, staging_dir: str) -> dict:
        return stage_chart(
            image_bytes=b"x",
            image_suffix=".png",
            series=self.SERIES,
            category="scrab_style",
            staging_dir=staging_dir,
        )

    def test_approve_moves_staging_into_canonical(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            staging_dir = os.path.join(tmp_dir, "staging")
            canonical_dir = os.path.join(tmp_dir, "canonical")
            meta = self._stage(staging_dir)

            dest = eval_promote.approve(
                meta["id"],
                staging_dir=staging_dir,
                canonical_dir=canonical_dir,
                version="v1",
            )

            self.assertFalse(os.path.isdir(os.path.join(staging_dir, meta["id"])))
            self.assertTrue(os.path.isfile(os.path.join(dest, "meta.json")))
            self.assertTrue(os.path.isfile(os.path.join(dest, "series.csv")))

    def test_approve_missing_entry_raises(self):
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            self.assertRaises(FileNotFoundError),
        ):
            eval_promote.approve("nope", staging_dir=tmp_dir, canonical_dir=tmp_dir)

    def test_approve_already_promoted_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            staging_dir = os.path.join(tmp_dir, "staging")
            canonical_dir = os.path.join(tmp_dir, "canonical")
            meta = self._stage(staging_dir)
            eval_promote.approve(
                meta["id"],
                staging_dir=staging_dir,
                canonical_dir=canonical_dir,
                version="v1",
            )
            os.makedirs(os.path.join(staging_dir, meta["id"]))  # re-stage the same id

            with self.assertRaises(FileExistsError):
                eval_promote.approve(
                    meta["id"],
                    staging_dir=staging_dir,
                    canonical_dir=canonical_dir,
                    version="v1",
                )

    def test_reject_discards_entry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta = self._stage(tmp_dir)
            eval_promote.reject(meta["id"], staging_dir=tmp_dir)
            self.assertFalse(os.path.isdir(os.path.join(tmp_dir, meta["id"])))

    def test_reject_missing_entry_raises(self):
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            self.assertRaises(FileNotFoundError),
        ):
            eval_promote.reject("nope", staging_dir=tmp_dir)


class TestManifestCanonicalEntries(TestCase):
    """eval.manifest.discover_entries also picks up promoted canonical charts."""

    def test_promoted_chart_is_discovered(self):
        # Inside REPO_ROOT so relative paths written into the manifest resolve
        # back through it the same way they would for a real promotion.
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            staging_dir = os.path.join(tmp_dir, "staging")
            canonical_dir = os.path.join(tmp_dir, "canonical")
            series = [
                {
                    "name": "Actual",
                    "removed": False,
                    "points": [
                        _series_point("2024-01-01T00:00:00", 1.0),
                        _series_point("2024-01-02T00:00:00", 2.0),
                    ],
                }
            ]
            meta = stage_chart(
                image_bytes=b"fake",
                image_suffix=".png",
                series=series,
                category="scrab_style",
                source="production",
                staging_dir=staging_dir,
            )
            eval_promote.approve(
                meta["id"],
                staging_dir=staging_dir,
                canonical_dir=canonical_dir,
                version="v1",
            )

            entries = discover_entries(
                dataset_dir=os.path.join(tmp_dir, "no-such-dataset"),
                categories=(),
                canonical_dir=canonical_dir,
                canonical_version="v1",
            )

            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry.id, meta["id"])
            self.assertEqual(entry.category, "scrab_style")
            self.assertEqual(entry.source, "production")

            expected, n_series = load_ground_truth(entry.csv_path)
            self.assertEqual(n_series, 1)
            self.assertEqual(len(expected), 2)
