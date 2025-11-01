from unittest import TestCase

from ocr_utils import texts_to_datetimes
from tests.test_data import texts_to_datetimes_data


class TestUtils(TestCase):
    def setUp(self):
        pass

    def test_texts_to_datetimes(self):
        for i, (texts, expected) in enumerate(texts_to_datetimes_data):
            result = texts_to_datetimes(texts)
            self.assertEqual(result, expected, f"Failed for input {i}: {texts}")
