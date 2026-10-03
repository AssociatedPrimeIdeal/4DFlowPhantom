"""Simple CFD-to-k-space interface, with optional streaming HDF5 output."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import warnings

import h5py
import numpy as np
from numpy.polynomial.legendre import leggauss

from .config import B0Field, CFDInput, CoilArray, Sequence, Tissue, VencExceededWarning, positive, venc_values, xyz
from .compute import CudaComputer, select_device
from .geometry import Grid, block_mean, sample_fields
from .io import CFDCase, load_case
from .physics import coil_centers, coil_sensitivities, crop_kspace, encoding_matrix, fft3c, steady_state_signal
from .temporal import frame_windows_s, make_time_axis, native_frame_indices


@dataclass
class SimulationResult:
    kspace: object
    coil_maps: np.ndarray
    times_s: np.ndarray
    encoding_matrix: np.ndarray
    blood_fraction: object
    wall_fraction: object
    velocity_gt_m_s: object
    metadata: dict
    b0_hz: object = None
    _file: h5py.File | None = field(default=None, repr=False)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def simulate(
    cfd: str | CFDInput | CFDCase,
    *,
    project: str | None = None,
    blood: Tissue = Tissue(T1_ms=1600, T2_ms=200, PD=1, T2star_ms=80),
    wall: Tissue = Tissue(T1_ms=1000, T2_ms=50, PD=.7, T2star_ms=30),
    background: Tissue | None = None,
    b0: B0Field | None = None,
    sequence: Sequence = Sequence(),
    resolution_mm=(2.0, 2.0, 2.0),
    frames: int | None = None,
    interpolation_order: int = 1,
    venc_cm_s=(150.0, 150.0, 150.0),
    coils: CoilArray = CoilArray(),
    wall_thickness_mm: float | None = None,
    padding_mm: float = 5.0,
    fov_mm=None,
    center_mm=None,
    subvoxels: int = 2,
    temporal_samples: int = 1,
    snr: float | None = None,
    seed: int = 0,
    output: str | Path | None = None,
    device: str = "auto",
    progress=None,
) -> SimulationResult:
    """Return fully sampled Cartesian complex64 k-space [V, T, C, Z, Y, X].

    With output=..., arrays are written frame by frame to HDF5; result.kspace
    is a sliceable h5py.Dataset. Without output, result.kspace is an ndarray.
    Coils are smooth geometric sensitivity profiles. Default tissue parameters
    are illustrative. frames=None retains all native CFD timestamps;
    an explicit count selects native frames when possible, otherwise warns
    and resamples with interpolation_order=0/1/2/3. Velocities above VENC
    retain their physical complex phase wrapping and emit a warning.
    resolution_mm is (dz, dy, dx); venc_cm_s is a flat list cycling through
    z, y, x. V = 1 + len(venc_cm_s), including a shared reference encoding.
    snr is a linear amplitude ratio: ideal pure-blood steady-state magnitude
    at the phantom origin divided by the noise standard deviation of each
    real/imaginary component. None disables noise. Partial volume and phase
    cancellation can lower local SNR. temporal_samples=1 samples each frame
    center; larger values integrate the complex signal over the frame window.
    device='auto' uses optional CuPy/Warp CUDA for wall distances, phase
    encoding and FFTs. VTK volume interpolation stays on CPU. Outputs remain
    NumPy/HDF5.
    """
    resolution = np.array(xyz(resolution_mm, "resolution_mm"))[::-1]
    selected_device = select_device(device)
    positive(padding_mm, "padding_mm", allow_zero=True)
    if snr is not None:
        if isinstance(snr, (bool, np.bool_)):
            raise ValueError("snr must be finite and positive, or None for no noise")
        positive(snr, "snr")
        snr = float(snr)
    for name, value in (("subvoxels", subvoxels), ("temporal_samples", temporal_samples)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if wall_thickness_mm is not None:
        positive(wall_thickness_mm, "wall_thickness_mm")
    if b0 is not None and not isinstance(b0, B0Field):
        raise TypeError("b0 must be a B0Field or None")
    reference_b0_hz = float(b0.evaluate((np.array([0.]),) * 3).item()) if b0 else 0.0
    reference_signal = float(abs(steady_state_signal(blood, sequence, reference_b0_hz)))
    if snr is not None and reference_signal == 0:
        raise ValueError("snr requires a nonzero pure-blood reference signal")
    noise_std = reference_signal / snr if snr is not None else 0.0
    if isinstance(cfd, CFDCase):
        case = cfd
    else:
        source = cfd if isinstance(cfd, CFDInput) else CFDInput(str(cfd), project=project)
        case = load_case(source)
    times, messages = make_time_axis(case.info, frames, interpolation_order)
    selected_indices = native_frame_indices(case.times_s, times)
    windows = frame_windows_s(times, case.info)
    if temporal_samples == 1:
        quadrature_nodes, weights = np.array([0.]), np.array([1.])
    else:
        quadrature_nodes, weights = leggauss(temporal_samples)
        weights = weights / 2
    sample_times = (times[:, None] + windows[:, None] * quadrature_nodes[None, :] / 2).ravel()
    bounds = case.bounds_mm(sample_times, interpolation_order)
    if center_mm is not None:
        center = np.asarray(center_mm, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center_mm must contain three finite phantom-centered coordinates")
    else:
        center = bounds.mean(axis=0)
    if fov_mm is None:
        extent = 2 * np.maximum(center - bounds[0], bounds[1] - center) + 2 * padding_mm
    else:
        extent = np.array(xyz(fov_mm, "fov_mm"))
    if wall.PD > 0:
        if case.wall is None:
            raise ValueError("Wall PD > 0 requires a matching wall_surface or SimVascular project directory")
        if wall_thickness_mm is None and case.wall_thickness_mm is None:
            raise ValueError("No thickness field in this wall surface: specify wall_thickness_mm")
    matrix = tuple(np.maximum(2, np.ceil(extent / resolution).astype(int)).tolist())
    grid = Grid(matrix, resolution, center, subvoxels)
    encoding = encoding_matrix(venc_cm_s)
    shape = (len(encoding), len(times), coils.count, *grid.shape_zyx)
    centers = coil_centers(coils, grid.fov_mm, grid.center_mm)
    maps_high = coil_sensitivities(coils, centers, grid.axes(fine=True), grid.fov_mm)
    maps_low = coil_sensitivities(coils, centers, grid.axes(), grid.fov_mm)
    b0_high = b0.evaluate(grid.axes(fine=True)) if b0 else 0.0
    b0_low = b0.evaluate(grid.axes()) if b0 else np.zeros(grid.shape_zyx, dtype=np.float32)
    blood_base = steady_state_signal(blood, sequence, b0_high)
    wall_base = steady_state_signal(wall, sequence, b0_high)
    background_base = steady_state_signal(background, sequence, b0_high) if background else 0j
    venc = np.array(venc_values(venc_cm_s)) / 100
    encoding_axes_xyz = np.array([2 - i % 3 for i in range(len(venc))])
    warned = {name: np.zeros(len(venc), dtype=bool) for name in ("blood", "wall")}
    sampled_peaks = {name: np.zeros(3) for name in ("blood", "wall")}

    def report_venc(peaks, tissue, label):
        encoded_peaks = peaks[encoding_axes_xyz]
        exceeded = encoded_peaks > venc
        if np.any(exceeded & ~warned[tissue]):
            details = "; ".join(f"v{'xyz'[axis]}: {peak * 100:.2f} > {limit * 100:g} cm/s"
                                for axis, peak, limit, over in zip(encoding_axes_xyz, encoded_peaks, venc, exceeded) if over)
            message = (f"{label} exceeds VENC ({details}). Complex phase encoding is retained without "
                       "velocity clipping; decoded principal phase wraps into [-VENC, VENC].")
            warnings.warn(message, VencExceededWarning, stacklevel=2)
            messages.append(message)
        warned[tissue] |= exceeded

    native_peaks = np.max(np.abs(case.velocity_m_s), axis=(0, 1))
    report_venc(native_peaks, "blood", "Native CFD velocity")
    interpolation_name = ("nearest", "linear", "quadratic", "cubic")[interpolation_order]
    metadata = {
        "format_version": "2.0", "tool": "SimVascular-based 4D Flow digital phantom",
        "device": selected_device, "compute_backend": "numpy-scipy",
        "geometry_backend": "vtk-cpu",
        "source": case.info, "blood": asdict(blood), "wall": asdict(wall),
        "background": asdict(background) if background else None,
        "b0": asdict(b0) if b0 else None,
        "sequence": asdict(sequence), "coils": asdict(coils),
        "coil_centers_mm": centers.tolist(), "coil_model": "RSS-normalized complex Gaussian sensitivity profiles",
        "resolution_mm_zyx": resolution[::-1].tolist(),
        "resolution_mm_xyz": resolution.tolist(), "matrix_xyz": list(matrix),
        "fov_mm_xyz": grid.fov_mm.tolist(), "grid_center_mm": grid.center_mm.tolist(),
        "world_center_mm": case.world_center_mm.tolist(),
        "frames": frames, "output_frame_count": len(times),
        "output_frame_intervals_ms": (np.diff(times) * 1000).tolist(),
        "temporal_sampling": ("native" if frames is None else "selection")
                             if selected_indices is not None else "interpolation",
        "native_frame_indices": selected_indices.tolist() if selected_indices is not None else None,
        "interpolation_order": interpolation_order,
        "temporal_interpolation": f"{interpolation_name} nodal velocity and displacement; periodic boundary if periodic=True",
        "temporal_samples": temporal_samples,
        "temporal_window": "instantaneous frame centers" if temporal_samples == 1 else "rectangular complex-signal average with Gauss-Legendre quadrature",
        "temporal_window_widths_s": windows.tolist(),
        "venc_cm_s": (venc * 100).tolist(), "venc_direction_order": "zyx, repeated",
        "encoding_order": ["reference"] + [f"v{'xyz'[axis]}" for axis in encoding_axes_xyz],
        "encoding_venc_cm_s": [None] + (venc * 100).tolist(),
        "encoding_matrix_units": "rad/(m/s)", "kspace_axes": ["V", "T", "C", "Z", "Y", "X"],
        "encoding_matrix_columns": ["vx", "vy", "vz"],
        "velocity_gt_component_order": ["vx", "vy", "vz"],
        "fft": "centered, orthonormal; fine-grid complex coil signals Fourier-cropped to target resolution",
        "subvoxels_per_axis": subvoxels, "snr": snr,
        "snr_definition": "linear ideal pure-blood magnitude / per-real-or-imaginary noise standard deviation; not dB",
        "snr_reference": "blood steady-state signal at phantom origin, before partial volume and phase cancellation",
        "snr_reference_b0_hz": reference_b0_hz,
        "snr_reference_signal_magnitude": reference_signal,
        "noise_std_per_real_imag_component": noise_std,
        "noise_model": "independent Gaussian real and imaginary k-space noise for each encoding, frame and coil; added after signal averaging and Fourier crop",
        "seed": seed,
        "warnings": messages,
        "native_peak_abs_velocity_cm_s_xyz": (native_peaks * 100).tolist(),
        "velocity_aliasing": "unclipped complex phase exp(i*pi*v/VENC); principal phase wraps on reconstruction",
        "wall_thickness_mm": wall_thickness_mm,
        "wall_thickness_source": "constant override" if wall_thickness_mm is not None else "wall surface thickness at nearest wall vertex",
        "wall_velocity_model": f"time derivative of {interpolation_name} FSI displacement; nearest surface vertex; zero between jumps for order 0",
        "velocity_gt_definition": "blood volume-weighted arithmetic CFD mean, also time-averaged if temporal_samples > 1; not phase-derived MRI velocity",
        "signal_model_scope": "steady-state tissue signal times ideal PC phase; no spin transport, inflow saturation, diffusion, or pulse-sequence Bloch tracking",
        "spgr_T2star_fallback": "T2_ms is used only when T2star_ms is None; this approximation is explicit",
        "data_source_url": "https://www.vascularmodel.com/dataset.html",
        "data_license_url": "https://www.vascularmodel.com/FAQs.html",
        "data_acknowledgement": "Vascular Model Repository; Wilson et al. 2013, doi:10.1115/1.4025983",
    }
    if case.info.get("project"):
        notice_path = Path(case.info["project"]) / "README-COPYRIGHT"
        if notice_path.is_file():
            metadata["input_data_copyright_notice"] = notice_path.read_text(encoding="utf-8-sig")
    origin = np.array([a[0] for a in grid.axes()]) + case.world_center_mm
    affine = np.eye(4)
    affine[:3, :3] = np.diag(resolution)
    affine[:3, 3] = origin
    metadata["affine_xyz_to_world_mm"] = affine.tolist()
    output_path, partial, h5 = None, None, None
    if output is not None:
        output_path = Path(output).resolve()
        if output_path.suffix.lower() not in (".h5", ".hdf5"):
            raise ValueError("output must end in .h5 or .hdf5")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = output_path.with_suffix(output_path.suffix + ".partial")
        h5 = h5py.File(partial, "w")
        h5.attrs["complete"] = False
        kspace = h5.create_dataset("kspace", shape=shape, dtype="complex64",
                                   chunks=(1, 1, 1, *grid.shape_zyx))
        kspace.attrs["axes"] = "V,T,C,Z,Y,X"
        blood_fraction = h5.create_dataset("blood_fraction", shape=(len(times), *grid.shape_zyx), dtype="float32")
        wall_fraction = h5.create_dataset("wall_fraction", shape=(len(times), *grid.shape_zyx), dtype="float32")
        truth = h5.create_dataset("velocity_gt_m_s", shape=(len(times), 3, *grid.shape_zyx), dtype="float32")
        b0_dataset = h5.create_dataset("b0_hz", data=b0_low)
        b0_dataset.attrs["units"] = "Hz"
        h5.create_dataset("coil_maps", data=maps_low)
        h5.create_dataset("times_s", data=times)
        h5.create_dataset("encoding_matrix", data=encoding)
        h5.create_dataset("affine_xyz_to_world_mm", data=affine)
    else:
        kspace = np.empty(shape, dtype=np.complex64)
        blood_fraction = np.empty((len(times), *grid.shape_zyx), dtype=np.float32)
        wall_fraction = np.empty_like(blood_fraction)
        truth = np.empty((len(times), 3, *grid.shape_zyx), dtype=np.float32)
    rng = np.random.default_rng(seed)
    high_shape = tuple(np.array(grid.shape_zyx) * subvoxels)
    blood_signal = np.broadcast_to(np.asarray(blood_base), high_shape)
    wall_signal = np.broadcast_to(np.asarray(wall_base), high_shape)
    background_signal = np.broadcast_to(np.asarray(background_base), high_shape)
    try:
        gpu = (CudaComputer(selected_device, maps_high, encoding, blood_signal, wall_signal,
                            background_signal, grid.shape_zyx, resolution)
               if selected_device != "cpu" else None)
        if gpu is not None:
            metadata.update(gpu.metadata)
        wall_sampler = None
        if gpu is not None and wall.PD > 0:
            from .cuda_geometry import CudaWallSampler
            wall_sampler = CudaWallSampler(selected_device)
            metadata.update(wall_sampler.metadata)
        elif gpu is not None:
            metadata["cuda_scope"] = "complex phase encoding and FFT; VTK volume probe on CPU"
        for index, time_s in enumerate(times):
            signals = gpu.zeros() if gpu else np.zeros((len(encoding), *high_shape), dtype=np.complex64)
            occupancy_blood = np.zeros(high_shape, dtype=np.float32)
            occupancy_wall = np.zeros(high_shape, dtype=np.float32)
            velocity_sum = np.zeros((*high_shape, 3), dtype=np.float32)
            offsets = quadrature_nodes * windows[index] / 2
            for offset, weight in zip(offsets, weights):
                blood_mask, velocity, wall_mask, wall_velocity = sample_fields(
                    case, grid, float(time_s + offset), wall_thickness_mm, wall.PD > 0,
                    interpolation_order, wall_sampler=wall_sampler)
                for tissue, mask, field in (("blood", blood_mask, velocity), ("wall", wall_mask, wall_velocity)):
                    if np.any(mask):
                        peaks = np.max(np.abs(field[mask]), axis=0)
                        sampled_peaks[tissue] = np.maximum(sampled_peaks[tissue], peaks)
                        report_venc(peaks, tissue, f"Sampled {tissue} velocity")
                occupancy_blood += weight * blood_mask
                occupancy_wall += weight * wall_mask
                velocity_sum += weight * velocity
                if gpu:
                    gpu.accumulate(signals, blood_mask, velocity, wall_mask, wall_velocity, weight)
                else:
                    for e, row in enumerate(encoding):
                        signal = background_signal.astype(np.complex64, copy=True)
                        signal[blood_mask] = blood_signal[blood_mask] * np.exp(1j * (velocity[blood_mask] @ row))
                        signal[wall_mask] = wall_signal[wall_mask] * np.exp(1j * (wall_velocity[wall_mask] @ row))
                        signals[e] += weight * signal
            bf = block_mean(occupancy_blood, grid)
            wf = block_mean(occupancy_wall, grid)
            gt = block_mean(velocity_sum, grid)
            np.divide(gt, bf[..., None], out=gt, where=bf[..., None] > 0)
            gt[bf == 0] = 0
            blood_fraction[index], wall_fraction[index] = bf, wf
            truth[index] = np.moveaxis(gt, -1, 0)
            gpu_frame = gpu.kspace(signals) if gpu else None
            for e in range(len(encoding)):
                for c in range(coils.count):
                    frame = (gpu_frame[e, c] if gpu else
                             crop_kspace(fft3c(maps_high[c] * signals[e]), grid.shape_zyx, resolution))
                    if noise_std:
                        frame += (noise_std * (rng.standard_normal(frame.shape) + 1j * rng.standard_normal(frame.shape))).astype(np.complex64)
                    kspace[e, index, c] = frame
            if progress:
                progress(index + 1, len(times))
        if not np.any(np.asarray(blood_fraction[:]) > 0):
            raise ValueError("No blood samples fall inside the requested FOV/grid; check units, FOV, and subvoxels")
        metadata["sampled_peak_abs_velocity_cm_s_xyz"] = {name: (peak * 100).tolist()
                                                         for name, peak in sampled_peaks.items()}
        if h5 is not None:
            h5.attrs["metadata_json"] = json.dumps(metadata)
            h5.attrs["complete"] = True
            h5.close()
            h5 = None
            partial.replace(output_path)
            h5 = h5py.File(output_path, "r")
            kspace, blood_fraction, wall_fraction, truth = (h5[n] for n in ("kspace", "blood_fraction", "wall_fraction", "velocity_gt_m_s"))
    except BaseException:
        if h5 is not None:
            h5.close()
        if partial is not None and partial.exists():
            partial.unlink()
        raise
    return SimulationResult(kspace, maps_low, times, encoding, blood_fraction, wall_fraction, truth, metadata, b0_low, h5)


def load_result(path: str | Path) -> SimulationResult:
    file = h5py.File(path, "r")
    if not file.attrs.get("complete", False):
        file.close()
        raise ValueError("Simulation output is incomplete")
    metadata = json.loads(file.attrs["metadata_json"])
    if metadata.get("kspace_axes") != ["V", "T", "C", "Z", "Y", "X"]:
        file.close()
        raise ValueError("This output uses legacy k-space axes; regenerate it for [V, T, C, Z, Y, X]")
    b0_hz = file["b0_hz"][:] if "b0_hz" in file else None
    return SimulationResult(file["kspace"], file["coil_maps"][:], file["times_s"][:], file["encoding_matrix"][:],
                            file["blood_fraction"], file["wall_fraction"], file["velocity_gt_m_s"],
                            metadata, b0_hz, file)
