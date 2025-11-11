from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from scipy.interpolate import interp1d


class FunctionBase(ABC):
    @abstractmethod
    def __call__(self, x):
        """Forward mapping: pixel to value."""
        pass

    @abstractmethod
    def invert(self, v):
        """Inverse mapping: value to pixel."""
        pass


class Linear(FunctionBase):
    def __init__(self, knots, values):
        # knots: pixel coordinates, values: corresponding values
        self.knots = np.array(knots)
        self.values = np.array(values)
        self.interpolator = interp1d(
            self.knots, self.values, kind="linear", fill_value="extrapolate"
        )
        self.inverse_interpolator = interp1d(
            self.values, self.knots, kind="linear", fill_value="extrapolate"
        )

    def __call__(self, px: int):
        return float(self.interpolator(px))

    def invert(self, v):
        return float(self.inverse_interpolator(v))


class LinearDatetime(FunctionBase):
    def __init__(self, knots, datetimes):
        # knots: pixel coordinates, datetimes: corresponding datetime objects
        self.knots = np.array(knots)
        self.timestamps = np.array([dt.timestamp() for dt in datetimes])
        self.interpolator = interp1d(
            self.knots, self.timestamps, kind="linear", fill_value="extrapolate"
        )
        self.inverse_interpolator = interp1d(
            self.timestamps, self.knots, kind="linear", fill_value="extrapolate"
        )

    def __call__(self, px: int):
        # Return datetime for given pixel coordinate
        ts = float(self.interpolator(px))
        return datetime.fromtimestamp(ts)

    def invert(self, dt: datetime):
        # Return pixel coordinate for given datetime
        ts = dt.timestamp()
        return float(self.inverse_interpolator(ts))


class Logarithmic(FunctionBase):
    def __init__(self, knots, values):
        # knots: pixel coordinates, values: corresponding values
        self.knots = np.array(knots)
        self.log_values = np.log(np.array(values))
        self.interpolator = interp1d(
            self.knots, self.log_values, kind="linear", fill_value="extrapolate"
        )
        self.inverse_interpolator = interp1d(
            self.log_values, self.knots, kind="linear", fill_value="extrapolate"
        )

    def __call__(self, px: int):
        # Returns value for given pixel coordinate
        val = float(np.exp(self.interpolator(px)))
        if not val:
            print(val)
        return val

    def invert(self, v):
        # Returns pixel coordinate for given value
        log_v = np.log(v)
        return float(self.inverse_interpolator(log_v))
