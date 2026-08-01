from unittest import TestCase

from date_utils import DateComponentClassifier
from tests.test_data import date_component_classify_data


class TestDateComponentClassifier(TestCase):
    def setUp(self):
        self.classifier = DateComponentClassifier()

    def test_classify(self):
        for i, (text, expected) in enumerate(date_component_classify_data):
            result = self.classifier.classify(text)
            self.assertEqual(result, expected, f"Failed for input {i}: {text!r}")
