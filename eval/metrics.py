"""
Accuracy metrics comparing an extracted time series against its ground truth.

Both `expected` and `extracted` are `{date: [value_or_None, ...]}`, one entry
per series in a chart's declared order (see `eval.manifest.load_ground_truth`
for `expected`; `extract_time_series`'s per-point value list for `extracted`).
All metrics are computed over (date, series) cells that exist in `expected`,
falling back to `None` for any date `extracted` is missing entirely.
"""

from typing import Optional

from eval.manifest import ExpectedSeries

ExtractedSeries = dict[object, list[Optional[float]]]


def _paired_values(
    expected: ExpectedSeries, extracted: ExtractedSeries, n_series: int
) -> list[tuple[float, float]]:
    """(expected, extracted) value pairs for every ground-truth cell the extraction resolved."""
    pairs = []
    for key, expected_values in expected.items():
        extracted_values = extracted.get(key, [None] * n_series)
        for ev, xv in zip(expected_values[:n_series], extracted_values[:n_series]):
            if xv is not None:
                pairs.append((ev, xv))
    return pairs


def resolved_fraction(
    expected: ExpectedSeries, extracted: ExtractedSeries, n_series: int
) -> float:
    """Fraction of ground-truth (date, series) cells the extraction produced a value for.

    0.0 if `expected` is empty.
    """
    total = sum(min(len(values), n_series) for values in expected.values())
    if total == 0:
        return 0.0
    resolved = len(_paired_values(expected, extracted, n_series))
    return resolved / total


def mae(
    expected: ExpectedSeries, extracted: ExtractedSeries, n_series: int
) -> Optional[float]:
    """Mean absolute error over jointly-resolved cells; `None` if nothing resolved."""
    pairs = _paired_values(expected, extracted, n_series)
    if not pairs:
        return None
    return sum(abs(ev - xv) for ev, xv in pairs) / len(pairs)


def rmse(
    expected: ExpectedSeries, extracted: ExtractedSeries, n_series: int
) -> Optional[float]:
    """Root-mean-square error over jointly-resolved cells; `None` if nothing resolved."""
    pairs = _paired_values(expected, extracted, n_series)
    if not pairs:
        return None
    return (sum((ev - xv) ** 2 for ev, xv in pairs) / len(pairs)) ** 0.5


def normalized_mae(
    expected: ExpectedSeries, extracted: ExtractedSeries, n_series: int
) -> Optional[float]:
    """MAE divided by the ground truth's own value range (max - min), so error is
    comparable across charts with different units/scale. `None` if nothing resolved
    or the ground truth is constant (zero range)."""
    mae_value = mae(expected, extracted, n_series)
    if mae_value is None:
        return None
    all_values = [v for values in expected.values() for v in values[:n_series]]
    value_range = max(all_values) - min(all_values)
    return mae_value / value_range if value_range else None


def series_count_match(n_expected: int, n_extracted: int) -> bool:
    """Whether the extraction found the right number of series."""
    return n_expected == n_extracted
