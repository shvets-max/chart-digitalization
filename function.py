from datetime import datetime

import numpy as np


class Linear:
    def __init__(self, px0, px1, v0, v1):
        # y = pixel coordinate, v = value
        self.px0 = px0
        self.px1 = px1
        self.v0 = v0
        self.v1 = v1
        self.slope = (v1 - v0) / (px1 - px0)  # slope = delta v / delta pixel
        self.intercept = v0 - self.slope * px0

    def __call__(self, px: int):
        return self.slope * px + self.intercept

    def invert(self, v):
        return (v - self.intercept) / self.slope


class LinearDatetime:
    def __init__(self, px0, px1, dt0: datetime, dt1: datetime):
        # Convert datetimes to timestamps (seconds since epoch)
        self.px0 = px0
        self.px1 = px1
        self.ts0 = dt0.timestamp()
        self.ts1 = dt1.timestamp()
        self.slope = (self.ts1 - self.ts0) / (
            px1 - px0
        )  # slope = delta time / delta pixel
        self.intercept = self.ts0 - self.slope * px0

    def __call__(self, px: int):
        """
        return timestamp value for given y-coordinate
        :param px:
        :return:
        """
        return datetime.fromtimestamp(self.slope * px + self.intercept)

    def invert(self, dt: datetime):
        """
        return y-coordinate for given timestamp value
        :param dt:
        :return:
        """
        return (dt.timestamp() - self.intercept) / self.slope


class Logarithmic:
    def __init__(self, y0, y1, v0, v1):
        # v = a * exp(b * y)
        self.y0 = y0
        self.y1 = y1
        self.v0 = v0
        self.v1 = v1
        self.b = np.log(v1 / v0) / (y1 - y0)
        self.a = v0 / np.exp(self.b * y0)

    def __call__(self, y):
        return self.a * np.exp(self.b * y)

    def invert(self, v):
        return np.log(v / self.a) / self.b
