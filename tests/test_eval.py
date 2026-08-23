import os
import tempfile
from datetime import date
from unittest import TestCase

from eval import history as eval_history
from eval.align import align_extracted_to_expected
from eval.manifest import CATEGORIES, REPO_ROOT, discover_entries, load_ground_truth
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
