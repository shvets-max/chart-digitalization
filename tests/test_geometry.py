from unittest import TestCase

from geometry import cluster_data
from tests.test_data import cluster_data_data


class TestGeometry(TestCase):
    def test_cluster_data(self):
        for i, (points, margin, expected) in enumerate(cluster_data_data):
            result = cluster_data(points, margin)
            self.assertEqual(result, expected, f"Failed for input {i}: {points}")
