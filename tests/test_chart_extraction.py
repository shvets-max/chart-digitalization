import csv
import os
from datetime import datetime
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from chart_extraction import extract_time_series

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LINE_SCALE_DIR = os.path.join(TEST_DATA_DIR, "linear_scaled")
LOG_SCALE_DIR = os.path.join(TEST_DATA_DIR, "log_scaled")

SEP = ";"


def _plot_time_series(extracted, expected, n_series_expected, name: str = "NoName"):
    # plot for visual inspection
    dates = sorted(expected.keys())
    for series_idx in range(n_series_expected):
        expected_values = [expected[dt][series_idx] for dt in dates]
        extracted_values = [
            extracted.get(dt, [None] * n_series_expected)[series_idx] for dt in dates
        ]
        extracted_dates = [
            dt
            for dt in dates
            if extracted.get(dt, [None] * n_series_expected)[series_idx] is not None
        ]
        all_dates = sorted({d for d in dates + extracted_dates})

        plt.plot(all_dates, expected_values, label=f"Expected Series {series_idx + 1}")
        plt.plot(
            all_dates,
            extracted_values,
            label=f"Extracted Series {series_idx + 1}",
            linestyle="-",
        )
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(f"Chart Extraction Comparison for {name}")
    plt.legend()
    plt.grid()
    plt.show()


class TestChartExtraction(TestCase):
    def setUp(self):
        # load expected results from CSV
        self.expected_results = {}
        for file in os.listdir(LINE_SCALE_DIR):
            if file.endswith(".csv"):
                idx = file.rstrip(".csv")
                with open(os.path.join(LINE_SCALE_DIR, file), newline="") as csvfile:
                    reader = csv.reader(csvfile, delimiter=SEP)
                    next(reader)  # skip header
                    # load timeseries data
                    self.expected_results[idx] = {
                        datetime.strptime(row[0], "%Y-%m-%d").date(): [
                            float(val.replace(",", ".")) for val in row[1:]
                        ]
                        for row in reader
                    }

        # for file in os.listdir(LOG_SCALE_DIR):
        #     if file.endswith(".csv"):
        #         idx = file.rstrip(".csv")
        #         with open(os.path.join(LOG_SCALE_DIR, file), newline="") as csvfile:
        #             reader = csv.reader(csvfile, delimiter=SEP)
        #             next(reader)  # skip header
        #             # load timeseries data
        #             self.expected_results[idx] = {
        #                 datetime.strptime(row[0], "%Y-%m-%d").date(): [
        #                     float(val.replace(",", ".")) for val in row[1:]
        #                 ]
        #                 for row in reader
        #             }

    def test_chart_extraction(self):
        for idx, expected in self.expected_results.items():
            print(idx)
            if idx.startswith("linear_"):
                image_path = os.path.join(LINE_SCALE_DIR, f"{idx}.png")
            else:
                image_path = os.path.join(LOG_SCALE_DIR, f"{idx}.png")

            extracted_data = extract_time_series(image_path)
            extracted_data = {dt.date(): val for dt, val in extracted_data}

            n_series_extracted = len(next(iter(expected.values())))
            n_series_expected = len(next(iter(extracted_data.values())))
            self.assertEqual(
                n_series_expected,
                n_series_extracted,
                f"Number of series mismatch for image {idx}",
            )

            abs_errors = []
            rel_errors = []
            for key, expected_values in expected.items():
                extracted_datapoints = extracted_data.get(
                    key, [None] * n_series_expected
                )
                extracted_datapoints = [
                    round(np.float32(v or 0), 4).item() for v in extracted_datapoints
                ]

                abs_errors.append(
                    [
                        abs(ev - dv)
                        for ev, dv in zip(expected_values, extracted_datapoints)
                    ]
                )
                rel_errors.append(
                    [
                        abs(ev - dv) / ev if ev != 0 else 0.0
                        for ev, dv in zip(expected_values, extracted_datapoints)
                    ]
                )

            print(40 * "-")
            print(f"Results for image img_{idx}:")

            abs_errors = np.array(abs_errors)
            rel_errors = np.array(rel_errors)
            mean_abs_errors = np.mean(abs_errors).round(4)
            std_abs_errors = np.std(abs_errors).round(4)
            mean_rel_errors = np.mean(rel_errors).round(4)
            std_rel_errors = np.std(rel_errors).round(4)
            print(f"Mean absolute error for img_{idx}: {mean_abs_errors}")
            print(f"Std of absolute error: {std_abs_errors}")
            print(f"Mean relative error for img_{idx}: {mean_rel_errors}")
            print(f"Std of relative error: {std_rel_errors}")

            _plot_time_series(
                extracted_data, expected, n_series_expected, name=f"img_{idx}"
            )
