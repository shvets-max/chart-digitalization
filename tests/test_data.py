from datetime import datetime

texts_to_datetimes_data = [
    (
        [
            "22.06.2019",
            "22.12.2019",
            "22.06.2020",
            "22.12.2020",
            "22.06.2021",
            "22.12.2021",
            "22.06.2022",
        ],
        [
            datetime(2019, 6, 22),
            datetime(2019, 12, 22),
            datetime(2020, 6, 22),
            datetime(2020, 12, 22),
            datetime(2021, 6, 22),
            datetime(2021, 12, 22),
            datetime(2022, 6, 22),
        ],
    ),
    (
        [
            "2020",
            "Jul",
            "2021",
            "Jul",
            "2022",
            "Jul",
            "2023",
            "Jul",
            "2024",
            "Jul",
            "2025",
            "Jul",
        ],
        [
            datetime(2020, 1, 1),
            datetime(2020, 7, 1),
            datetime(2021, 1, 1),
            datetime(2021, 7, 1),
            datetime(2022, 1, 1),
            datetime(2022, 7, 1),
            datetime(2023, 1, 1),
            datetime(2023, 7, 1),
            datetime(2024, 1, 1),
            datetime(2024, 7, 1),
            datetime(2025, 1, 1),
            datetime(2025, 7, 1),
        ],
    ),
    (
        ["2023", "May", "Aug", "2024", "May", "Aug", "2025", "May", "Aug"],
        [
            datetime(2023, 1, 1),
            datetime(2023, 5, 1),
            datetime(2023, 8, 1),
            datetime(2024, 1, 1),
            datetime(2024, 5, 1),
            datetime(2024, 8, 1),
            datetime(2025, 1, 1),
            datetime(2025, 5, 1),
            datetime(2025, 8, 1),
        ],
    ),
    (
        ["Apr", "Jul", "Oct", "2024", "Apr", "Jul", "Oct", "2025", "Apr", "Jul"],
        [
            datetime(2023, 4, 1),
            datetime(2023, 7, 1),
            datetime(2023, 10, 1),
            datetime(2024, 1, 1),
            datetime(2024, 4, 1),
            datetime(2024, 7, 1),
            datetime(2024, 10, 1),
            datetime(2025, 1, 1),
            datetime(2025, 4, 1),
            datetime(2025, 7, 1),
        ],
    ),
    (
        ["Dec", "15", "2025", "15", "Feb", "15", "Mar", "15", "Apr", "15"],
        [
            datetime(2024, 12, 1),
            datetime(2024, 12, 15),
            datetime(2025, 1, 1),
            datetime(2025, 1, 15),
            datetime(2025, 2, 1),
            datetime(2025, 2, 15),
            datetime(2025, 3, 1),
            datetime(2025, 3, 15),
            datetime(2025, 4, 1),
            datetime(2025, 4, 15),
        ],
    ),
    (
        [
            "12.19",
            "03.20",
            "06.20",
            "09.20",
            "12.20",
            "03.21",
            "06.21",
            "09.21",
            "12.21",
            "03.22",
            "06.22",
            "09.22",
        ],
        [
            datetime(2019, 12, 1),
            datetime(2020, 3, 1),
            datetime(2020, 6, 1),
            datetime(2020, 9, 1),
            datetime(2020, 12, 1),
            datetime(2021, 3, 1),
            datetime(2021, 6, 1),
            datetime(2021, 9, 1),
            datetime(2021, 12, 1),
            datetime(2022, 3, 1),
            datetime(2022, 6, 1),
            datetime(2022, 9, 1),
        ],
    ),
]

minimal_power_of_10_data = [
    ([0.1, 0.2, 0.3], 1),
    ([0.01, 0.02, 0.03], 2),
    ([1.5, 2.5, 3.5], 1),
    ([10.0, 20.0, 30.0], 0),
    ([0.123, 0.456, 0.789], 3),
    ([5.6789, 6.7890, 7.8901], 4),
    ([100.0, 200.0, 300.0], 0),
    ([0.0001, 0.0002, 0.0003], 4),
]

ensure_numeric_consistency_data = [
    # Test case 1: Simple linear descending numbers
    # (no missing points, 5.0 instead of 0.05, minuses missing)
    (
        {"column_numbers": [0.25, 0.2, 0.15, 0.1, 5.0, 0.0, 0.05, 0.1],
         "bboxes_y_centers": [423.5, 456.5, 490.5, 523.5, 557.5, 591.5, 624.5, 658.5]},
        ([0.25, 0.2, 0.15, 0.1, 0.05, 0.0, -0.05, -0.1],
         [423.5, 456.5, 490.5, 523.5, 557.5, 591.5, 624.5, 658.5])
    ),

    # Test case 2: Simple linear descending numbers
    # (no missing points, 2 instead of 0.2, 5.0 instead of 0.05, minuses missing)
    (
        {"column_numbers": [1.0, 0.8, 0.6, 2.0, 0.0, 5.0, 0.2],
         "bboxes_y_centers": [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]},
        ([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2],
         [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
    ),

    # Test case 3: Linear descending numbers with missing minuses
    (
        {"column_numbers": [15.0, 10.0, 0.0, 5.0],
         "bboxes_y_centers": [100.0, 150.0, 250.0, 300.0]},
        ([15.0, 10.0, 5.0, 0.0, -5.0],
         [100.0, 150.0, 200.0, 250.0, 300.0])
    ),

    # Test case 4: Linear descending numbers with one missing point
    (
        {"column_numbers": [100.0, 80.0, 60.0, 20.0],
         "bboxes_y_centers": [100.0, 150.0, 200.0, 300.0]},
        ([100.0, 80.0, 60.0, 40.0, 20.0],
         [100.0, 150.0, 200.0, 250.0, 300.0])
    ),

    # Test case 5: Linear descending numbers with two missing points
    (
        {"column_numbers": [1000.0, 800.0, 200.0],
         "bboxes_y_centers": [50.0, 100.0, 250.0]},
        ([1000.0, 800.0, 600.0, 400.0, 200.0],
         [50.0, 100.0, 150.0, 200.0, 250.0])
    ),
]
