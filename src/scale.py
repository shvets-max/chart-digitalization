import logging
from collections.abc import Callable, Sequence
from typing import Optional

import numpy as np

from src.data_integrity import ensure_linear_continuity
from src.function import Linear, LinearDatetime, Logarithmic

logger = logging.getLogger(__name__)


def _r_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Goodness of fit of the best straight line through (x, y)."""
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    centered = y - y.mean()
    ss_res, ss_tot = (
        float(np.dot(residuals, residuals)),
        float(np.dot(centered, centered)),
    )
    return 1.0 - ss_res / ss_tot if ss_tot else 1.0


MAX_DROPPED_FRACTION = 0.2  # see drop_monotonicity_outliers


def drop_monotonicity_outliers(
    values: np.ndarray, knots: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only the tick values that continue the running monotonic trend in knot
    (pixel) order, dropping the rest.

    Axis ticks are always monotonic in pixel position; a single OCR slip (e.g.
    "1.4" misread as "14") produces a value wildly inconsistent with its
    neighbors and would otherwise corrupt both the log/linear scale decision
    (see is_log_scale) and the fitted scale itself.

    Only a small minority of ticks are dropped this way (MAX_DROPPED_FRACTION);
    when more than that would be discarded, the column isn't a "mostly clean,
    one OCR slip" case this heuristic is meant for, so it is left untouched
    rather than risk gutting a badly-misread column down to a handful of
    coincidentally-monotonic values.
    """
    if len(values) < 3:
        return values, knots
    order = np.argsort(knots)
    sorted_values = values[order]
    direction = 1 if sorted_values[-1] >= sorted_values[0] else -1

    keep = np.ones(len(sorted_values), dtype=bool)
    last = sorted_values[0]
    for i in range(1, len(sorted_values)):
        if direction * (sorted_values[i] - last) < 0:
            keep[i] = False
        else:
            last = sorted_values[i]

    if np.count_nonzero(~keep) > MAX_DROPPED_FRACTION * len(sorted_values):
        return values, knots

    kept = order[keep]
    kept.sort()
    return values[kept], knots[kept]


def is_log_scale(values: np.ndarray, knots: np.ndarray) -> bool:
    """
    Whether `values` are better explained by an exponential curve over their
    pixel positions `knots` than by a straight line.

    Comparing consecutive value ratios (the previous approach) assumes ticks are
    evenly spaced in pixels and breaks whenever they aren't -- e.g. a real log
    axis labelling round numbers (1, 2, 5 x 10^n) has ticks with varying ratios
    even though it is genuinely a log scale. Fitting against pixel position
    instead sidesteps that: ticks are evenly spaced in pixels by construction,
    so whichever of value/log(value) is actually linear in pixel position wins.
    """
    values = np.asarray(values, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if len(values) < 3 or np.any(values <= 0):
        return False
    return _r_squared(knots, np.log(values)) > _r_squared(knots, values)


def nice_ticks(vmin: float, vmax: float, count: int = 6) -> list[float]:
    """
    Round tick values covering [vmin, vmax], spaced by a 1/2/5 x 10^n step.
    All returned ticks lie inside the range.
    """
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    span = vmax - vmin
    if span <= 0 or count < 2:
        return [vmin]

    raw_step = span / (count - 1)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    candidates = [multiple * magnitude for multiple in (1, 2, 2.5, 5, 10)]
    step = min(candidates, key=lambda s: abs(np.log(s / raw_step)))

    start = np.ceil(vmin / step) * step
    ticks = np.arange(start, vmax + step * 1e-6, step)
    decimals = max(0, int(-np.floor(np.log10(step))) + 1)
    return [round(float(t), decimals) for t in ticks if vmin <= t <= vmax]


def log_ticks(vmin: float, vmax: float, count: int = 6) -> list[float]:
    """Ticks at round 1/2/5 x 10^n values covering [vmin, vmax] on a log axis."""
    vmin, vmax = max(min(vmin, vmax), 1e-12), max(vmin, vmax)
    if vmax <= vmin:
        return [vmin]

    candidates = []
    exponents = range(int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax))) + 1)
    for exponent in exponents:
        for mantissa in (1, 2, 5):
            value = mantissa * 10.0**exponent
            if vmin <= value <= vmax:
                candidates.append(value)

    if len(candidates) > count:  # thin out evenly, keeping both endpoints
        keep = np.linspace(0, len(candidates) - 1, count).round().astype(int)
        candidates = [candidates[i] for i in dict.fromkeys(keep.tolist())]
    return candidates or nice_ticks(vmin, vmax, count)


def create_y_scale(values: Sequence[float], knots: np.ndarray) -> Optional[Callable]:
    """
    Fit a y-axis scale (log or linear, whichever fits better) mapping pixel
    coordinate to value.

    :param values: axis tick values
    :param knots: pixel coordinate of each tick
    :return: reversible function mapping y-coordinate to value, or None if there
        are fewer than 2 ticks.
    :raises ValueError: if `values` and `knots` differ in length.
    """
    if len(knots) != len(values):
        raise ValueError("Number of bounding boxes and numbers must match")

    if len(values) < 2:
        return None

    if is_log_scale(values, knots):
        logger.info("Using logarithmic scale for y-axis")
        arg_sorted = np.argsort(knots)
        y_sorted = knots[arg_sorted]
        n_sorted = np.array(values)[arg_sorted]
        return Logarithmic(knots=y_sorted, values=n_sorted)

    logger.info("Using linear scale for y-axis")
    values, knots = ensure_linear_continuity(x1=np.array(values), x2=np.array(knots))
    arg_sorted = np.argsort(knots)
    y_sorted = knots[arg_sorted]
    n_sorted = np.array(values)[arg_sorted]
    return Linear(knots=y_sorted, values=n_sorted)


def create_x_scale(row_index: Sequence, knots: np.ndarray) -> Optional[Callable]:
    """
    Fit an x-axis scale (datetime or linear) mapping pixel coordinate to value.

    :param row_index: axis tick values (datetimes or numbers)
    :param knots: pixel coordinate of each tick
    :return: reversible function mapping x-coordinate to value, or None if there
        are fewer than 2 ticks.
    :raises ValueError: if `row_index` and `knots` differ in length.
    """
    if len(knots) != len(row_index):
        raise ValueError("Number of bounding boxes and index values must match")

    if len(row_index) < 2:
        return None

    argsort = np.argsort(knots)
    x_sorted = knots[argsort]
    idx_sorted = np.array(row_index)[argsort]

    if hasattr(idx_sorted[0], "year") and hasattr(idx_sorted[-1], "year"):
        return LinearDatetime(knots=x_sorted, datetimes=idx_sorted)
    return Linear(knots=x_sorted, values=idx_sorted)
