"""
Match extracted series to ground-truth CSV columns.

Extraction order isn't guaranteed to match the CSV's column order -- for
multi-series charts the legend or color-separation order can differ from how
`tests/data_generation.py` wrote the columns (see
`tests/test_chart_extraction.py`'s `TestMultilineExtraction`, which resolves
this the same way: greedy best-match by error, not position).
"""

from datetime import date
from typing import Optional

from eval.manifest import ExpectedSeries

RawTimeSeries = list[tuple[object, list[Optional[float]]]]
ExtractedSeries = dict[date, list[Optional[float]]]


def _series_by_index(raw_time_series: RawTimeSeries, n_series: int) -> list[dict]:
    """`{date: value}` per extracted series index, from `extract_chart`'s raw time series."""
    by_series = [dict() for _ in range(n_series)]
    for dt, values in raw_time_series:
        key = dt.date() if hasattr(dt, "date") else dt
        for i, value in enumerate(values[:n_series]):
            by_series[i][key] = value
    return by_series


def _mean_abs_error(expected_col: dict, got_col: dict) -> Optional[float]:
    common = [d for d in expected_col if got_col.get(d) is not None]
    if not common:
        return None
    return sum(abs(got_col[d] - expected_col[d]) for d in common) / len(common)


def align_extracted_to_expected(
    expected: ExpectedSeries,
    raw_time_series: RawTimeSeries,
    n_series_expected: int,
    n_series_extracted: int,
) -> ExtractedSeries:
    """
    Greedily match each extracted series to the ground-truth column it best
    fits (lowest mean absolute error first, each side used at most once), and
    reindex it to `expected`'s column order.

    Returns `{date: [value_or_None, ...]}` aligned to `expected`'s columns, with
    `None` for any expected column no extracted series was matched to.
    """
    expected_by_col = [
        {d: v[i] for d, v in expected.items()} for i in range(n_series_expected)
    ]
    by_series = _series_by_index(raw_time_series, n_series_extracted)

    candidates = []
    for col_idx, expected_col in enumerate(expected_by_col):
        for series_idx, got_col in enumerate(by_series):
            error = _mean_abs_error(expected_col, got_col)
            if error is not None:
                candidates.append((error, col_idx, series_idx))
    candidates.sort(key=lambda c: c[0])

    col_to_series = {}
    used_series = set()
    for _, col_idx, series_idx in candidates:
        if col_idx in col_to_series or series_idx in used_series:
            continue
        col_to_series[col_idx] = series_idx
        used_series.add(series_idx)

    return {
        d: [
            by_series[col_to_series[col_idx]].get(d)
            if col_idx in col_to_series
            else None
            for col_idx in range(n_series_expected)
        ]
        for d in expected
    }
