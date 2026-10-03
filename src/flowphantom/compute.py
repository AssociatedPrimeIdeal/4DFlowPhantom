"""Optional CuPy CUDA signal processing; public outputs stay NumPy."""

import numpy as np

from .physics import _crop_parameters


def select_device(device: str = "auto") -> str:
    """Choose usable CUDA when available, otherwise CPU without CUDA packages."""
    if device == "cpu":
        return "cpu"
    if not isinstance(device, str) or (device not in ("auto", "cuda") and not device.startswith("cuda:")):
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'")
    try:
        import cupy as cp
        import warp as wp
    except (ImportError, OSError) as error:
        if device == "auto":
            return "cpu"
        raise RuntimeError('CUDA requires CuPy and Warp; install with pip install ".[cuda]"') from error
    try:
        index = cp.cuda.runtime.getDevice() if device in ("auto", "cuda") else int(device.split(":")[1])
        if index < 0 or index >= cp.cuda.runtime.getDeviceCount():
            raise ValueError(f"CUDA device {index} does not exist")
        with cp.cuda.Device(index):
            # Check a real kernel, including the compiler/runtime and GPU support.
            cp.zeros(1).fill(0)
            cp.cuda.get_current_stream().synchronize()
        wp.config.log_level = wp.LOG_WARNING
        wp.init()
        if not wp.get_device(f"cuda:{index}").is_cuda:
            raise RuntimeError("Warp CUDA is unavailable")
    except Exception as error:
        if device == "auto":
            return "cpu"
        raise RuntimeError(f"Cannot use {device}: {error}") from error
    return f"cuda:{index}"


def cuda_fft(array, *, device: str, inverse: bool = False):
    import cupy as cp
    with cp.cuda.Device(int(device.split(":")[1])):
        values = cp.asarray(np.ascontiguousarray(array))
        axes = (-3, -2, -1)
        transform = cp.fft.ifftn if inverse else cp.fft.fftn
        output = cp.fft.fftshift(transform(cp.fft.ifftshift(values, axes=axes),
                                          axes=axes, norm="ortho"), axes=axes)
        return cp.asnumpy(output)


class CudaComputer:
    """Keep static maps on CUDA and transfer sampled fields one frame at a time."""

    def __init__(self, device, maps, encoding, blood, wall, background, shape_zyx, resolution_xyz):
        import cupy as cp
        self.cp = cp
        self.device = device
        self.cuda_device = cp.cuda.Device(int(device.split(":")[1]))
        self.shape_zyx = shape_zyx
        self.slices, correction = _crop_parameters(maps.shape[-3:], shape_zyx, resolution_xyz)
        with self.cuda_device:
            self.maps = cp.asarray(maps)
            # Match CPU double precision phase before complex64 signal/FFTs.
            self.encoding = cp.asarray(encoding, dtype=cp.float64)
            self.blood = cp.asarray(blood, dtype=cp.complex128)
            self.wall = cp.asarray(wall, dtype=cp.complex128)
            self.background = cp.asarray(background, dtype=cp.complex64)
            self.correction = cp.asarray(correction, dtype=cp.complex64)

    @property
    def metadata(self):
        with self.cuda_device:
            name = self.cp.cuda.runtime.getDeviceProperties(self.cuda_device.id)["name"]
        return {"device": self.device, "compute_backend": "cupy-cuda",
                "gpu_name": name.decode() if isinstance(name, bytes) else name,
                "cupy_version": self.cp.__version__}

    def zeros(self):
        with self.cuda_device:
            return self.cp.zeros((len(self.encoding), *self.maps.shape[-3:]), dtype=self.cp.complex64)

    def accumulate(self, signals, blood_mask, velocity, wall_mask, wall_velocity, weight):
        cp = self.cp
        with self.cuda_device:
            signal = cp.broadcast_to(self.background, signals.shape).copy()
            for mask, field, base in ((blood_mask, velocity, self.blood),
                                     (wall_mask, wall_velocity, self.wall)):
                if np.any(mask):
                    selected = cp.asarray(mask)
                    values = cp.asarray(field[mask], dtype=cp.float64)
                    phase = (values @ self.encoding.T).T
                    signal[:, selected] = (base[selected][None] * cp.exp(1j * phase)).astype(cp.complex64)
            signals += signal * np.float32(weight)

    def kspace(self, signals):
        cp = self.cp
        axes = (-3, -2, -1)
        result = np.empty((len(self.encoding), len(self.maps), *self.shape_zyx), dtype=np.complex64)
        with self.cuda_device:
            for coil, sensitivity in enumerate(self.maps):
                image = signals * sensitivity[None]
                transformed = cp.fft.fftshift(
                    cp.fft.fftn(cp.fft.ifftshift(image, axes=axes), axes=axes, norm="ortho"), axes=axes)
                cropped = transformed[(slice(None), *self.slices)] * self.correction
                result[:, coil] = cp.asnumpy(cropped)
        return result
