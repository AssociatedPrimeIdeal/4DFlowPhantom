"""Native-frame selection and spline interpolation of CFD time series."""

import warnings

import numpy as np
from scipy.interpolate import make_interp_spline

from .config import TemporalInterpolationWarning, interpolation_degree


def native_frame_indices(native_times, requested_times):
    """Return matching native indices, or None when resampling is required."""
    native = np.asarray(native_times, dtype=float)
    requested = np.asarray(requested_times, dtype=float)
    distance = np.abs(requested[:, None] - native[None, :])
    indices = distance.argmin(axis=1)
    tolerance = max(float(np.ptp(native)), 1.0) * 1e-10
    return indices if np.all(distance[np.arange(len(requested)), indices] <= tolerance) else None


def make_time_axis(info: dict, frames: int | None = None,
                   interpolation_order: int = 1) -> tuple[np.ndarray, list[str]]:
    """Keep native timestamps, decimate exactly, or warn and resample uniformly.

    Periodic grids exclude the duplicated endpoint; nonperiodic grids include
    both endpoints. A one-frame request selects the first native frame.
    """
    order = interpolation_degree(interpolation_order)
    native = np.asarray(info["times_s"], dtype=float)
    if frames is None:
        return native.copy(), []
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
        raise ValueError("frames must be a positive integer or None")
    times = np.linspace(native[0], info["period_s"] if info["periodic"] else native[-1],
                        frames, endpoint=not info["periodic"])
    indices = native_frame_indices(native, times)
    if indices is not None:
        return native[indices].copy(), []
    if order >= len(native):
        raise ValueError(f"Order {order} interpolation requires at least {order + 1} native frames")
    name = ("nearest-neighbour", "linear", "quadratic", "cubic")[order]
    message = (f"Cannot select {frames} uniformly spaced k-space frames directly from the "
               f"{len(native)} native CFD frames. Resampling with order {order} ({name}) interpolation. "
               "Interpolation does not add measured temporal information.")
    warnings.warn(message, TemporalInterpolationWarning, stacklevel=2)
    return times, [message]


def frame_windows_s(times, info):
    """Local frame widths for optional complex-signal temporal averaging."""
    times = np.asarray(times)
    if len(times) == 1:
        duration = info["period_s"] if info["periodic"] else info["times_s"][-1]
        return np.array([duration])
    if info["periodic"]:
        previous = np.r_[times[-1] - info["period_s"], times[:-1]]
        following = np.r_[times[1:], times[0] + info["period_s"]]
        return (following - previous) / 2
    intervals = np.diff(times)
    return np.r_[intervals[0], (intervals[:-1] + intervals[1:]) / 2, intervals[-1]]


class TemporalInterpolator:
    """Cache small spline weight matrices, without copying the mesh time series."""

    def __init__(self, times_s, period_s=None):
        self.times = np.asarray(times_s, dtype=float)
        self.period = period_s
        self._splines = {}

    def weights(self, time_s: float, order: int = 1):
        order = interpolation_degree(order)
        if order >= len(self.times):
            raise ValueError(f"Order {order} interpolation requires at least {order + 1} native frames")
        t = float(time_s % self.period if self.period is not None
                  else np.clip(time_s, self.times[0], self.times[-1]))
        if order == 0:
            distance = np.abs(self.times - t)
            if self.period is not None:
                distance = np.minimum(distance, self.period - distance)
            weights = np.zeros(len(self.times))
            weights[distance.argmin()] = 1
            return weights, np.zeros_like(weights)
        if order not in self._splines:
            times, identity = self.times, np.eye(len(self.times))
            if self.period is not None:
                times = np.r_[times, self.period]
                identity = np.vstack([identity, identity[0]])
            self._splines[order] = make_interp_spline(
                times, identity, k=order,
                bc_type="periodic" if self.period is not None and order >= 2 else None)
        spline = self._splines[order]
        weights = spline(t)
        # Preserve native mesh fields exactly, including when splines are selected.
        matches = native_frame_indices(self.times, [t])
        if matches is not None:
            weights = np.zeros(len(self.times))
            weights[matches[0]] = 1
        derivative = spline(t, nu=1)
        if self.period is None and (time_s < self.times[0] or time_s > self.times[-1]):
            derivative = np.zeros_like(derivative)
        return weights, derivative


def weighted_fields(weights, fields):
    nonzero = np.flatnonzero(weights)
    if len(nonzero) == 1 and weights[nonzero[0]] == 1:
        return fields[nonzero[0]]
    result = np.zeros(fields.shape[1:], dtype=np.float32)
    for index in nonzero:
        result += np.float32(weights[index]) * fields[index]
    return result
