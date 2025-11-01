from unittest import TestCase

from data_integrity import minimal_power_of_10
from tests.test_data import minimal_power_of_10_data


class TestUtils(TestCase):
    def setUp(self):
        pass

    def test_minimal_power_of_10(self):
        for i, (arr, exp) in enumerate(minimal_power_of_10_data):
            result = minimal_power_of_10(arr)
            self.assertEqual(result, exp, f"Failed for input {i}: {arr}")
