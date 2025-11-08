from unittest import TestCase

import numpy as np

from data_integrity import minimal_power_of_10, ensure_numeric_consistency
from tests.test_data import minimal_power_of_10_data, ensure_numeric_consistency_data


class TestUtils(TestCase):
    def setUp(self):
        pass

    def test_minimal_power_of_10(self):
        for i, (arr, exp) in enumerate(minimal_power_of_10_data):
            result = minimal_power_of_10(arr)
            self.assertEqual(result, exp, f"Failed for input {i}: {arr}")

    def test_ensure_numeric_consistency(self):
        for i, (input_data, exp_out) in enumerate(ensure_numeric_consistency_data):
            input_numbers = np.array(input_data["column_numbers"])
            input_bboxes_y_centers = np.array(input_data["bboxes_y_centers"])
            exp_numbers = np.array(exp_out[0])
            exp_bboxes_y_centers = np.array(exp_out[1])

            _numbers, _bboxes_y_centers = ensure_numeric_consistency(
                input_numbers, input_bboxes_y_centers
            )

            self.assertTrue(
                exp_numbers.shape == _numbers.shape and
                exp_bboxes_y_centers.shape == _bboxes_y_centers.shape,
                f"Shape mismatch for setup {i}: {exp_numbers}"
            )

            self.assertEqual(
                sum(exp_numbers - _numbers), 0,
                f"Failed for setup {i}: {exp_numbers}",
            )
            self.assertEqual(
                sum(exp_bboxes_y_centers - _bboxes_y_centers), 0,
                f"Bboxes centers calculations failed for setup {i}: {exp_numbers}",
            )
