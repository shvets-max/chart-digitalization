from unittest import TestCase

import numpy as np

from data_integrity import ensure_linear_continuity, find_minimal_powers_of_10
from tests.test_data import (
    ensure_linear_continuity_data,
    find_minimal_powers_of_10_data,
)


class TestUtils(TestCase):
    def setUp(self):
        pass

    def test_minimal_power_of_10(self):
        for i, (arr, exp) in enumerate(find_minimal_powers_of_10_data):
            result = find_minimal_powers_of_10(arr)
            self.assertEqual(result, exp, f"Failed for input {i}: {arr}")

    def test_ensure_linear_continuity(self):
        for i, (input_data, exp_out) in enumerate(ensure_linear_continuity_data):
            print(i)
            x1 = np.array(input_data["x1"])
            x2 = np.array(input_data["x2"])

            exp_x1 = np.array(exp_out[0])
            exp_x2 = np.array(exp_out[1])

            x1_out, x2_out = ensure_linear_continuity(x1, x2)

            self.assertTrue(
                exp_x1.shape == x1_out.shape and exp_x2.shape == x2_out.shape,
                f"Shape mismatch for setup {i}: {exp_x1}",
            )

            self.assertEqual(
                sum(exp_x1 - x1_out),
                0,
                f"Failed for setup {i}: {exp_x1}",
            )
            self.assertTrue(
                abs(np.mean(exp_x2 - x2_out).item()) < 1,
                f"Bboxes centers calculations failed for setup {i}: {exp_x1}",
            )
