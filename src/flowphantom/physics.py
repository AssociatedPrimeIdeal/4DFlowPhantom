"""Steady-state tissue signals, ideal phase contrast, and centered FFTs."""

import numpy as np
from scipy import fft

from .config import CoilArray, Sequence, Tissue


def steady_state_signal(tissue: Tissue, sequence: Sequence, b0_hz=0.0) -> complex | np.ndarray:
    """SPGR analytic signal or bSSFP Bloch fixed point, in the RF frame.

    Phase-contrast encoding is applied separately by the simulator. This is a
    steady-state signal model, not a flowing-spin pulse-sequence simulation.
    """
    alpha = np.deg2rad(sequence.flip_deg)
    tr, te = sequence.TR_ms, sequence.TE_ms
    frequency = np.asarray(sequence.off_resonance_hz + tissue.off_resonance_hz + b0_hz, dtype=float)
    e1 = np.exp(-tr / tissue.T1_ms)
    if sequence.model == "spgr":
        transverse_time = tissue.T2star_ms if tissue.T2star_ms is not None else tissue.T2_ms
        amplitude = tissue.PD * (1 - e1) * np.sin(alpha) / (1 - e1 * np.cos(alpha))
        signal = amplitude * np.exp(-te / transverse_time + 2j * np.pi * frequency * te / 1000)
        return complex(signal) if signal.ndim == 0 else signal
    e2 = np.exp(-tr / tissue.T2_ms)
    angle = 2 * np.pi * frequency * tr / 1000 - np.deg2rad(sequence.rf_phase_increment_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    rz = np.zeros(frequency.shape + (3, 3), dtype=float)
    rz[..., 0, 0] = np.cos(angle)
    rz[..., 0, 1] = -np.sin(angle)
    rz[..., 1, 0] = np.sin(angle)
    rz[..., 1, 1] = np.cos(angle)
    rz[..., 2, 2] = 1
    relaxation = np.zeros(frequency.shape + (3, 3), dtype=float)
    relaxation[..., 0, 0] = e2
    relaxation[..., 1, 1] = e2
    relaxation[..., 2, 2] = e1
    evolution = rz @ relaxation @ rx
    recovery = np.zeros(frequency.shape + (3,), dtype=float)
    recovery[..., 2] = tissue.PD * (1 - e1)
    before_rf = np.linalg.solve(np.eye(3) - evolution, recovery[..., None])[..., 0]
    after_rf = np.einsum("ij,...j->...i", rx, before_rf)
    echo = (after_rf[..., 0] + 1j * after_rf[..., 1]) * np.exp(
        -te / tissue.T2_ms + 2j * np.pi * frequency * te / 1000)
    signal = 1j * echo  # Put the on-resonance passband on the positive real axis.
    return complex(signal) if signal.ndim == 0 else signal


def encoding_matrix(venc_cm_s) -> np.ndarray:
    """Reference plus z/y/x velocity encodings; matrix columns remain x/y/z."""
    from .config import venc_values
    venc_m_s = np.array(venc_values(venc_cm_s)) / 100
    matrix = np.zeros((1 + len(venc_m_s), 3))
    for index, venc in enumerate(venc_m_s):
        matrix[index + 1, 2 - index % 3] = np.pi / venc
    return matrix


def fft3c(image: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    if device != "cpu":
        from .compute import cuda_fft, select_device
        selected = select_device(device)
        if selected != "cpu":
            return cuda_fft(image, device=selected)
    axes = (-3, -2, -1)
    return fft.fftshift(fft.fftn(fft.ifftshift(image, axes=axes), axes=axes, norm="ortho"), axes=axes)


def ifft3c(kspace: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    if device != "cpu":
        from .compute import cuda_fft, select_device
        selected = select_device(device)
        if selected != "cpu":
            return cuda_fft(kspace, device=selected, inverse=True)
    axes = (-3, -2, -1)
    return fft.fftshift(fft.ifftn(fft.ifftshift(kspace, axes=axes), axes=axes, norm="ortho"), axes=axes)


def _crop_parameters(fine_shape, shape_zyx, resolution_xyz):
    """Shared CPU/CUDA crop indices and cell-centre phase/amplitude correction."""
    fine = np.array(fine_shape)
    coarse = np.array(shape_zyx)
    if np.any(coarse > fine):
        raise ValueError("Cannot crop k-space to a larger matrix")
    starts = fine // 2 - coarse // 2
    slices = tuple(slice(int(i), int(i + n)) for i, n in zip(starts, coarse))
    spacing = np.array(resolution_xyz)[::-1]
    fine_spacing = spacing * coarse / fine
    coarse_anchor = (coarse // 2 - coarse / 2 + .5) * spacing
    fine_anchor = (fine // 2 - fine / 2 + .5) * fine_spacing
    offset = coarse_anchor - fine_anchor
    frequencies = [fft.fftshift(fft.fftfreq(int(n), d=float(d))) for n, d in zip(coarse, spacing)]
    phase = np.exp(2j * np.pi * (frequencies[0][:, None, None] * offset[0]
                              + frequencies[1][None, :, None] * offset[1]
                              + frequencies[2][None, None, :] * offset[2]))
    return slices, phase * np.sqrt(np.prod(coarse) / np.prod(fine))


def crop_kspace(kspace: np.ndarray, shape_zyx: tuple[int, int, int], resolution_xyz) -> np.ndarray:
    """Band-limit a fine cell-center grid, preserving amplitude and grid origin."""
    slices, correction = _crop_parameters(kspace.shape[-3:], shape_zyx, resolution_xyz)
    return (kspace[slices] * correction).astype(np.complex64)


def coil_centers(config: CoilArray, fov_xyz: np.ndarray, grid_center_xyz: np.ndarray) -> np.ndarray:
    if config.layout == "custom":
        return np.asarray(config.centers_mm, dtype=float)
    radius = config.radius_mm or float(.65 * max(fov_xyz[:2]) + 5)
    rings = 1 if config.layout == "ring" else config.rings
    span = config.axial_span_mm if config.axial_span_mm is not None else .7 * fov_xyz[2]
    locations = [0.] if rings == 1 else np.linspace(-span / 2, span / 2, rings)
    counts = np.full(rings, config.count // rings)
    counts[:config.count % rings] += 1
    centers = []
    for ring, (z, count) in enumerate(zip(locations, counts)):
        angles = np.arange(count) * 2 * np.pi / count + ring * np.pi / max(count, 1)
        centers.extend((radius * np.cos(a), radius * np.sin(a), z) for a in angles)
    return np.asarray(centers) + grid_center_xyz


def coil_sensitivities(config: CoilArray, centers: np.ndarray, axes_xyz, fov_xyz) -> np.ndarray:
    """Smooth complex Gaussian receive profiles, pointwise RSS normalized.

    This configurable geometric model is not an electromagnetic field solver.
    """
    x = np.asarray(axes_xyz[0], dtype=np.float32)[None, None, :]
    y = np.asarray(axes_xyz[1], dtype=np.float32)[None, :, None]
    z = np.asarray(axes_xyz[2], dtype=np.float32)[:, None, None]
    width = config.sensitivity_width_mm or float(.6 * max(fov_xyz))
    log_amplitude = np.stack([-((x - cx)**2 + (y - cy)**2 + (z - cz)**2) / (2 * width**2)
                              for cx, cy, cz in centers])
    log_amplitude -= log_amplitude.max(axis=0)
    amplitudes = np.exp(log_amplitude)
    amplitudes /= np.sqrt(np.sum(amplitudes**2, axis=0))
    result = np.empty(amplitudes.shape, dtype=np.complex64)
    midpoint = (centers.min(axis=0) + centers.max(axis=0)) / 2
    for i, (cx, cy, cz) in enumerate(centers):
        angle = np.arctan2(cy - midpoint[1], cx - midpoint[0])
        phase = config.phase_scale * (-(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
                                      + .2 * (z - cz)) / width
        result[i] = amplitudes[i] * np.exp(1j * phase)
    return result
