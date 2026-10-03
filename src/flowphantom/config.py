"""Public parameters. Lengths are mm, times are ms, and VENC is cm/s."""

from dataclasses import dataclass
import math

import numpy as np


def positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def xyz(value, name: str) -> tuple[float, float, float]:
    values = (float(value),) * 3 if isinstance(value, (float, int)) else tuple(value)
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar or three values")
    for v in values:
        positive(float(v), name)
    return tuple(float(v) for v in values)


def finite_xyz(value, name: str) -> tuple[float, float, float]:
    values = (float(value),) * 3 if isinstance(value, (float, int)) else tuple(value)
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar or three values in x, y, z order")
    if not all(math.isfinite(float(component)) for component in values):
        raise ValueError(f"{name} must contain finite values")
    return tuple(float(component) for component in values)


def venc_values(value) -> tuple[float, ...]:
    """Positive VENC entries applied cyclically to z, y, x directions."""
    values = (float(value),) * 3 if np.isscalar(value) else tuple(value)
    if not values:
        raise ValueError("venc_cm_s must contain at least one value")
    for component in values:
        positive(float(component), "venc_cm_s")
    return tuple(float(component) for component in values)


@dataclass(frozen=True)
class Tissue:
    T1_ms: float
    T2_ms: float
    PD: float = 1.0
    T2star_ms: float | None = None
    off_resonance_hz: float = 0.0

    def __post_init__(self):
        positive(self.T1_ms, "T1_ms")
        positive(self.T2_ms, "T2_ms")
        positive(self.PD, "PD", allow_zero=True)
        if self.T2star_ms is not None:
            positive(self.T2star_ms, "T2star_ms")
        if not math.isfinite(self.off_resonance_hz):
            raise ValueError("off_resonance_hz must be finite")


@dataclass(frozen=True)
class Sequence:
    model: str = "spgr"
    TR_ms: float = 5.0
    TE_ms: float = 2.5
    flip_deg: float = 15.0
    rf_phase_increment_deg: float = 180.0
    off_resonance_hz: float = 0.0

    def __post_init__(self):
        if self.model not in ("spgr", "bssfp"):
            raise ValueError("sequence.model must be 'spgr' or 'bssfp'")
        positive(self.TR_ms, "TR_ms")
        positive(self.TE_ms, "TE_ms", allow_zero=True)
        if self.TE_ms > self.TR_ms:
            raise ValueError("TE_ms must not exceed TR_ms")
        if not 0 < self.flip_deg < 180:
            raise ValueError("flip_deg must lie between 0 and 180")
        if not all(math.isfinite(v) for v in (self.rf_phase_increment_deg, self.off_resonance_hz)):
            raise ValueError("RF phase increment and off resonance must be finite")


@dataclass(frozen=True)
class B0Field:
    """Static off-resonance in Hz: polynomial plus optional smooth residual.

    The residual is a continuous random Fourier field with Gaussian spatial
    correlation. Its nominal standard deviation is smooth_std_hz, and its
    correlation lengths use x/y/z millimetres. Fixed coordinates and seed
    give the same field independently of sampling resolution or FOV.
    This is a phenomenological shim-residual model, not an anatomical
    susceptibility calculation.
    """

    offset_hz: float = 0.0
    gradient_hz_per_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quadratic_hz_per_mm2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    smooth_std_hz: float = 0.0
    correlation_length_mm: tuple[float, float, float] = (20.0, 20.0, 20.0)
    seed: int = 0

    def __post_init__(self):
        if not math.isfinite(self.offset_hz):
            raise ValueError("b0.offset_hz must be finite")
        object.__setattr__(self, "gradient_hz_per_mm", finite_xyz(self.gradient_hz_per_mm,
                                                                    "b0.gradient_hz_per_mm"))
        object.__setattr__(self, "quadratic_hz_per_mm2", finite_xyz(self.quadratic_hz_per_mm2,
                                                                      "b0.quadratic_hz_per_mm2"))
        positive(self.smooth_std_hz, "b0.smooth_std_hz", allow_zero=True)
        object.__setattr__(self, "correlation_length_mm", xyz(self.correlation_length_mm,
                                                             "b0.correlation_length_mm"))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("b0.seed must be a nonnegative integer")

    def evaluate(self, axes_xyz) -> object:
        """Evaluate the field on 1-D x, y, z axes and return a z, y, x array."""
        x_axis, y_axis, z_axis = (np.asarray(axis, dtype=np.float32) for axis in axes_xyz)
        gradient = self.gradient_hz_per_mm
        quadratic = self.quadratic_hz_per_mm2
        field = np.float32(self.offset_hz)
        field = field + gradient[0] * x_axis[None, None, :]
        field = field + gradient[1] * y_axis[None, :, None]
        field = field + gradient[2] * z_axis[:, None, None]
        field = field + quadratic[0] * x_axis[None, None, :] ** 2
        field = field + quadratic[1] * y_axis[None, :, None] ** 2
        field = field + quadratic[2] * z_axis[:, None, None] ** 2
        if self.smooth_std_hz:
            rng = np.random.default_rng(self.seed)
            terms = 64
            wavevectors = (rng.standard_normal((terms, 3)) / self.correlation_length_mm).astype(np.float32)
            phases = rng.uniform(0, 2 * np.pi, terms).astype(np.float32)
            amplitude = np.float32(self.smooth_std_hz * np.sqrt(2 / terms))
            for wave, phase in zip(wavevectors, phases):
                angle = (wave[0] * x_axis[None, None, :] + wave[1] * y_axis[None, :, None]
                         + wave[2] * z_axis[:, None, None] + phase)
                field += amplitude * np.cos(angle)
        return np.asarray(field, dtype=np.float32)


@dataclass(frozen=True)
class CoilArray:
    count: int = 8
    layout: str = "cylinder"
    rings: int = 1
    radius_mm: float | None = None
    axial_span_mm: float | None = None
    sensitivity_width_mm: float | None = None
    centers_mm: tuple[tuple[float, float, float], ...] | None = None
    phase_scale: float = 0.5

    def __post_init__(self):
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 1:
            raise ValueError("coils.count must be a positive integer")
        if self.layout not in ("ring", "cylinder", "custom"):
            raise ValueError("coils.layout must be 'ring', 'cylinder', or 'custom'")
        if not isinstance(self.rings, int) or isinstance(self.rings, bool) or not 1 <= self.rings <= self.count:
            raise ValueError("coils.rings must be an integer between 1 and count")
        for name in ("radius_mm", "sensitivity_width_mm"):
            if getattr(self, name) is not None:
                positive(getattr(self, name), name)
        if self.axial_span_mm is not None:
            positive(self.axial_span_mm, "axial_span_mm", allow_zero=True)
        if not math.isfinite(self.phase_scale):
            raise ValueError("coils.phase_scale must be finite")
        if self.layout == "custom":
            if self.centers_mm is None or len(self.centers_mm) != self.count:
                raise ValueError("custom coils require count centers_mm, in phantom-centered coordinates")
            if any(len(c) != 3 or not all(math.isfinite(v) for v in c) for c in self.centers_mm):
                raise ValueError("each coil center must contain three finite coordinates")


@dataclass(frozen=True)
class CFDInput:
    """VTU with time-indexed nodal fields and explicit physical units.

    project is optional: pass time_step_s (the solver step in seconds) or
    times_s for timing, and wall_surface for the matching wall VTP. A project
    directory only supplies automatic discovery of solver.inp and wall data.
    """

    cfd: str
    project: str | None = None
    length_unit: str = "cm"
    velocity_unit: str = "cm/s"
    time_step_s: float | None = None
    times_s: tuple[float, ...] | None = None
    period_s: float | None = None
    periodic: bool = True
    wall_surface: str | None = None
    velocity_prefix: str = "velocity_"
    displacement_prefix: str = "displacement_"

    def __post_init__(self):
        if self.length_unit not in ("m", "cm", "mm"):
            raise ValueError("length_unit must be 'm', 'cm', or 'mm'")
        if self.velocity_unit not in ("m/s", "cm/s", "mm/s"):
            raise ValueError("velocity_unit must be 'm/s', 'cm/s', or 'mm/s'")
        for name in ("time_step_s", "period_s"):
            if getattr(self, name) is not None:
                positive(getattr(self, name), name)


class TemporalInterpolationWarning(UserWarning):
    """Requested frames cannot all be selected from native CFD timestamps."""


class VencExceededWarning(UserWarning):
    """Velocity components exceed VENC and their MR phase will wrap."""


def interpolation_degree(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1, 2, 3):
        raise ValueError("interpolation_order must be 0 (nearest), 1 (linear), 2 (quadratic), or 3 (cubic)")
    return value
