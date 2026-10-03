# 4DFlowPhantom

Generate multi-coil 4D Flow MRI digital phantoms from SimVascular CFD data. The notebook uses **FSI + SPGR** from VMR case `0141_H_CORO_KD`.

## Installation

```bash
pip install .
```

For optional GPU acceleration (CUDA 13 driver): `pip install ".[cuda]"`. The notebook automatically uses CuPy for phase encoding/FFTs and NVIDIA Warp for wall distances; VTK volume interpolation runs on CPU. `device="cpu"` forces CPU.

Notebook dependencies are included. Run [`case0141_demo.ipynb`](case0141_demo.ipynb).

## FSI files and timing

Download the [FSI VTU](https://www.vascularmodel.com/svresults/0141_H_CORO_KD/0141_H_CORO_KD_3D_FSI_VTU.zip) and [SimVascular project](https://www.vascularmodel.com/svprojects/0141_H_CORO_KD.zip), then extract:

```text
data/0141_H_CORO_KD/
  results/fsi/KDR32_FSI.vtu
  Simulations/FSI_ptsE_cor115/varwallprop.vtp
```

The VTU provides volume velocity/displacement; the VTP provides matching wall geometry and thickness. `project` is optional. Set `time_step_s` from the matching FSI `solver.inp`: **0.00061747 s** here. The 20 output frames are 50 solver steps apart, giving **30.8735 ms** spacing and a **0.61747 s** cycle.

The optional [FSI results VTP](https://www.vascularmodel.com/svresults/0141_H_CORO_KD/0141_H_CORO_KD_3D_FSI_VTP.zip), extracted as `results/fsi/KDR32_FSI.vtp`, supplies surface pressure, velocity, displacement, and WSS for visualization. Volume streamlines use the VTU.

## Python API

```python
from flowphantom import CFDInput, simulate

source = CFDInput(
    cfd="data/0141_H_CORO_KD/results/fsi/KDR32_FSI.vtu",
    wall_surface="data/0141_H_CORO_KD/Simulations/FSI_ptsE_cor115/varwallprop.vtp",
    time_step_s=0.00061747,
    length_unit="cm",
    velocity_unit="cm/s",
)
result = simulate(
    source,
    resolution_mm=[2.5, 2.5, 2.5],  # z, y, x
    venc_cm_s=[150, 150, 150],      # z, y, x; append further encodings for multi-VENC
    frames=None,                  # retain all native timestamps
    interpolation_order=1,        # 0 nearest, 1 linear, 2 quadratic, 3 cubic
    snr=20,                       # pure-blood amplitude / noise std; None disables noise
    output=None,                  # return NumPy arrays; no HDF5 file
)
print(result.kspace.shape)         # [V, T, C, Z, Y, X]
```

`V` includes one shared reference plus each supplied VENC. `[150,150,150,30,30,30]` gives `V=7`, ordered reference/z/y/x/z/y/x.

An explicit `frames` count selects native frames directly when the requested timestamps match; otherwise it warns and interpolates. Velocities exceeding each encoding's VENC trigger a warning and retain physical phase wrapping without clipping.

`snr` is a linear ratio of ideal pure-blood signal at the phantom origin to the noise standard deviation of each real/imaginary component; local SNR can decrease with partial volume or phase cancellation. `temporal_samples=1` samples frame centers; larger values average complex signals across each frame's time window.

Set `output="outputs/phantom.h5"` to stream results to HDF5; use `with simulate(...) as result:` to close the file after use. HDF5 contains k-space, coil maps, B0, timestamps, encoding matrix, tissue fractions, CFD velocity truth, and metadata.

The notebook shows FSI streamlines, the synthetic B0 field, orthogonal k-space/magnitude/phase views, and phase-derived velocity. Tissue and coil models are illustrative steady-state approximations.

`B0Field(smooth_std_hz=40, correlation_length_mm=(15,20,25), seed=0)` adds a smooth spatially correlated residual to the polynomial field. This synthetic shim model retains common B0 phase in the reference encoding; it is not an anatomical susceptibility simulation.

## License

Code: [MIT](LICENSE). VMR data: [terms](https://www.vascularmodel.com/FAQs.html) and [attribution notice](THIRD_PARTY_NOTICES.md); preserve `README-COPYRIGHT` when redistributing data.
