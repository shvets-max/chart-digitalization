from unittest import TestCase

import numpy as np

from src.function import Linear, Logarithmic
from src.scale import create_y_scale, drop_monotonicity_outliers, is_log_scale


class TestIsLogScale(TestCase):
    def test_perfectly_linear_ticks_are_not_log(self):
        values = np.array([100, 200, 300, 400, 500], dtype=float)
        knots = np.array([0, 10, 20, 30, 40], dtype=float)
        self.assertFalse(is_log_scale(values, knots))

    def test_constant_ratio_ticks_are_log(self):
        values = np.array([1, 2, 4, 8, 16, 32], dtype=float)
        knots = np.array([0, 10, 20, 30, 40, 50], dtype=float)
        self.assertTrue(is_log_scale(values, knots))

    def test_nice_round_number_log_ticks_with_small_base(self):
        """
        Real log axes label round numbers, not a constant-ratio sequence: the
        ratio between consecutive ticks varies even though the axis genuinely is
        log. Comparing consecutive value ratios alone (the previous algorithm)
        misclassified this as linear -- see data/multiline/img.png and img_3.png,
        whose "small base" (ratio close to 1) log axes triggered exactly this.
        """
        # y-axis ticks read from data/multiline/img_3.png (value, pixel row).
        values = np.array([2400, 1800, 580, 360, 280, 230, 190, 155, 127], dtype=float)
        knots = np.array([28.5, 87.5, 318.5, 416.5, 467.5, 507.5, 546.5, 588.5, 628.5])
        self.assertTrue(is_log_scale(values, knots))

        # y-axis ticks read from data/multiline/img.png (value, pixel row).
        values = np.array([1400, 850, 610, 535, 455, 395, 345, 295], dtype=float)
        knots = np.array([29.5, 176.5, 274.5, 313.5, 360.5, 402.5, 442.5, 488.5])
        self.assertTrue(is_log_scale(values, knots))

    def test_few_points_default_to_linear(self):
        self.assertFalse(is_log_scale(np.array([1.0, 2.0]), np.array([0.0, 10.0])))

    def test_non_positive_values_default_to_linear(self):
        values = np.array([-1, 0, 1, 2, 4], dtype=float)
        knots = np.array([0, 10, 20, 30, 40], dtype=float)
        self.assertFalse(is_log_scale(values, knots))


class TestCreateYScale(TestCase):
    def test_picks_logarithmic_for_nice_log_ticks(self):
        values = [2400, 1800, 580, 360, 280, 230, 190, 155, 127]
        knots = np.array([28.5, 87.5, 318.5, 416.5, 467.5, 507.5, 546.5, 588.5, 628.5])
        scale = create_y_scale(values, knots)
        self.assertIsInstance(scale, Logarithmic)

    def test_picks_linear_for_evenly_spaced_ticks(self):
        values = [100, 200, 300, 400, 500]
        knots = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        scale = create_y_scale(values, knots)
        self.assertIsInstance(scale, Linear)

    def test_fewer_than_two_values_returns_none(self):
        self.assertIsNone(create_y_scale([1.0], np.array([0.0])))

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            create_y_scale([1.0, 2.0], np.array([0.0]))


class TestDropMonotonicityOutliers(TestCase):
    def test_single_ocr_slip_is_dropped(self):
        # "1.4" misread as "14" (missing decimal point), a real failure seen on
        # data/scrab/anet-peg.png that flipped its log/linear scale decision.
        values = np.array([3.8, 3.6, 3.4, 3.2, 2.8, 2.6, 2.2, 1.6, 14.0, 0.8, 0.6])
        knots = np.array(
            [40, 81, 121, 162, 243, 284, 365, 486, 526, 648, 689], dtype=float
        )
        kept_values, kept_knots = drop_monotonicity_outliers(values, knots)
        self.assertNotIn(14.0, kept_values)
        self.assertEqual(len(kept_values), len(values) - 1)
        self.assertEqual(len(kept_values), len(kept_knots))

    def test_clean_monotonic_ticks_are_untouched(self):
        values = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        knots = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        kept_values, kept_knots = drop_monotonicity_outliers(values, knots)
        np.testing.assert_array_equal(kept_values, values)
        np.testing.assert_array_equal(kept_knots, knots)

    def test_majority_corrupted_column_is_left_untouched(self):
        """
        When most values disagree with the trend (a systemically misread column,
        not a single OCR slip), dropping down to a small monotonic remainder
        would do more harm than good, so nothing is dropped.
        """
        values = np.array([29.0, 26.0, 23.0, 21.0, 19.0, 15.5e9, 12.8e9, 10.4e9])
        knots = np.arange(len(values), dtype=float)
        kept_values, kept_knots = drop_monotonicity_outliers(values, knots)
        np.testing.assert_array_equal(kept_values, values)
        np.testing.assert_array_equal(kept_knots, knots)

    def test_fewer_than_three_values_are_untouched(self):
        values = np.array([1.0, 2.0])
        knots = np.array([0.0, 10.0])
        kept_values, kept_knots = drop_monotonicity_outliers(values, knots)
        np.testing.assert_array_equal(kept_values, values)
        np.testing.assert_array_equal(kept_knots, knots)
