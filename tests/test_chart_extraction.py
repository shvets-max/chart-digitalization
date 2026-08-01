import csv
import os
from datetime import date, datetime
from unittest import TestCase

import numpy as np

from chart_extraction import extract_time_series

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LINEAR_SCALE_DIR = os.path.join(TEST_DATA_DIR, "linear_scaled")
LOG_SCALE_DIR = os.path.join(TEST_DATA_DIR, "log_scaled")

SEP = ";"

ExpectedSeries = dict[date, list[float]]


def _load_expected_results(data_dir: str) -> dict[str, ExpectedSeries]:
    """Load ground-truth time series from every CSV in `data_dir`, keyed by file stem."""
    expected_results = {}
    for file in sorted(os.listdir(data_dir)):
        if not file.endswith(".csv"):
            continue
        idx = os.path.splitext(file)[0]
        with open(os.path.join(data_dir, file), newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=SEP)
            next(reader)  # skip header
            expected_results[idx] = {
                datetime.strptime(row[0], "%Y-%m-%d").date(): [
                    float(val.replace(",", ".")) for val in row[1:]
                ]
                for row in reader
            }
    return expected_results


def _compute_errors(
    expected: ExpectedSeries, extracted: ExpectedSeries, n_series: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (abs_errors, rel_errors) arrays of shape (n_dates, n_series) for expected vs extracted values."""
    abs_errors = []
    rel_errors = []
    for key, expected_values in expected.items():
        extracted_values = extracted.get(key, [None] * n_series)
        extracted_values = [
            round(np.float32(v or 0), 4).item() for v in extracted_values
        ]
        abs_errors.append(
            [abs(ev - dv) for ev, dv in zip(expected_values, extracted_values)]
        )
        rel_errors.append(
            [
                abs(ev - dv) / ev if ev != 0 else 0.0
                for ev, dv in zip(expected_values, extracted_values)
            ]
        )
    return np.array(abs_errors), np.array(rel_errors)


def _log_error_stats(
    label: str, abs_errors: np.ndarray, rel_errors: np.ndarray
) -> None:
    """Print mean/std of absolute and relative error for `label`."""
    print(
        f"{label}: "
        f"mean_abs={abs_errors.mean():.4f} std_abs={abs_errors.std():.4f} "
        f"mean_rel={rel_errors.mean():.4f} std_rel={rel_errors.std():.4f}"
    )


class TestChartExtraction(TestCase):
    """Runs chart extraction against generated charts and checks it recovers the right shape of data."""

    def setUp(self):
        self.linear_expected = _load_expected_results(LINEAR_SCALE_DIR)
        self.log_expected = _load_expected_results(LOG_SCALE_DIR)

    def _check_scale(
        self, data_dir: str, expected_results: dict[str, ExpectedSeries], label: str
    ) -> None:
        """Extract every chart in `data_dir`, asserting series count matches and reporting error stats."""
        overall_abs_errors = []
        overall_rel_errors = []
        for idx, expected in expected_results.items():
            with self.subTest(chart=idx):
                image_path = os.path.join(data_dir, f"{idx}.png")
                extracted_data = extract_time_series(image_path)
                extracted_data = {dt.date(): values for dt, values in extracted_data}

                n_series_expected = len(next(iter(expected.values())))
                n_series_extracted = len(next(iter(extracted_data.values())))
                self.assertEqual(
                    n_series_extracted,
                    n_series_expected,
                    f"Number of series mismatch for chart {idx}",
                )

                abs_errors, rel_errors = _compute_errors(
                    expected, extracted_data, n_series_expected
                )
                overall_abs_errors.append(abs_errors)
                overall_rel_errors.append(rel_errors)
                _log_error_stats(f"{label}/{idx}", abs_errors, rel_errors)

        _log_error_stats(
            f"{label}/overall",
            np.concatenate(overall_abs_errors),
            np.concatenate(overall_rel_errors),
        )

    def test_linear_scale_extraction(self):
        """Extraction should recover the correct series count from linear-scaled charts."""
        self._check_scale(LINEAR_SCALE_DIR, self.linear_expected, "linear")

    def test_log_scale_extraction(self):
        """Extraction should recover the correct series count from log-scaled charts."""
        self._check_scale(LOG_SCALE_DIR, self.log_expected, "log")
