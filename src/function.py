import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class FunctionBase(ABC):
    @abstractmethod
    def __call__(self, x: float):
        """Forward mapping: pixel to value."""

    @abstractmethod
    def invert(self, v):
        """Inverse mapping: value to pixel."""


class Linear(FunctionBase):
    def __init__(self, knots: Sequence[float], values: Sequence[float]):
        # knots: pixel coordinates, values: corresponding values
        self.knots = np.array(knots)
        self.values = np.array(values)
        self.interpolator = interp1d(
            self.knots, self.values, kind="linear", fill_value="extrapolate"
        )
        self.inverse_interpolator = interp1d(
            self.values, self.knots, kind="linear", fill_value="extrapolate"
        )

    def __call__(self, px: float) -> float:
        return float(self.interpolator(px))

    def invert(self, v: float) -> float:
        return float(self.inverse_interpolator(v))


class LinearDatetime(FunctionBase):
    def __init__(self, knots: Sequence[float], datetimes: Sequence[datetime]):
        # knots: pixel coordinates, datetimes: corresponding datetime objects
        self.knots = np.array(knots)
        self.timestamps = np.array([dt.timestamp() for dt in datetimes])
        self.interpolator = interp1d(
            self.knots, self.timestamps, kind="linear", fill_value="extrapolate"
        )
        self.inverse_interpolator = interp1d(
            self.timestamps, self.knots, kind="linear", fill_value="extrapolate"
        )

    def __call__(self, px: float) -> datetime:
        """Datetime for the given pixel coordinate."""
        ts = float(self.interpolator(px))
        return datetime.fromtimestamp(ts)

    def invert(self, dt: datetime) -> float:
        """Pixel coordinate for the given datetime."""
        return float(self.inverse_interpolator(dt.timestamp()))


class Logarithmic(FunctionBase):
    def __init__(self, knots: Sequence[float], values: Sequence[float]):
        # knots: pixel coordinates, values: corresponding values
        self.knots = np.array(knots)
        self.log_values = np.log(np.array(values))
        self.interpolator = interp1d(
            self.knots, self.log_values, kind="linear", fill_value="extrapolate"
        )
        self.inverse_interpolator = interp1d(
            self.log_values, self.knots, kind="linear", fill_value="extrapolate"
        )

    def __call__(self, px: float) -> float:
        """Value for the given pixel coordinate."""
        val = float(np.exp(self.interpolator(px)))
        if not val:
            logger.warning("Logarithmic scale evaluated to zero at pixel %s", px)
        return val

    def invert(self, v: float) -> float:
        """Pixel coordinate for the given value."""
        return float(self.inverse_interpolator(np.log(v)))
