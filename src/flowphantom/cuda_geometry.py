"""CUDA triangle-distance queries, with VTK refinement at shell boundaries."""

import numpy as np
import vtk
import warp as wp
from vtk.util.numpy_support import vtk_to_numpy


@wp.kernel
def _wall_distances(mesh: wp.uint64,
                    x: wp.array(dtype=float), y: wp.array(dtype=float), z: wp.array(dtype=float),
                    nx: int, ny: int, radius: float, distance: wp.array(dtype=float)):
    i = wp.tid()
    p = wp.vec3(x[i % nx], y[(i // nx) % ny], z[i // (nx * ny)])
    query = wp.mesh_query_point_no_sign(mesh, p, radius)
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        distance[i] = wp.length(closest - p)
    else:
        distance[i] = wp.inf


class CudaWallSampler:
    """Reuse grid coordinates and refit the BVH as the FSI wall moves."""

    def __init__(self, device):
        self.device = wp.get_device(device)
        self.mesh = None
        self.grid = None
        self.axes = None
        self.triangles = None

    @property
    def metadata(self):
        return {"geometry_backend": "vtk-cpu-volume+warp-cuda-wall",
                "wall_distance_backend": "warp-cuda",
                "warp_version": wp.__version__,
                "cuda_scope": "wall triangle distances, complex phase encoding and FFT; VTK volume probe and shell boundary refinement on CPU"}

    def distances(self, wall, grid, thickness_max):
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(wall)
        triangulate.PassVertsOff()
        triangulate.PassLinesOff()
        triangulate.Update()
        surface = triangulate.GetOutput()
        points = vtk_to_numpy(surface.GetPoints().GetData())
        triangles = vtk_to_numpy(surface.GetPolys().GetConnectivityArray()).astype(np.int32)
        if triangles.size == 0:
            raise ValueError("Wall surface has no triangles")
        axes = grid.axes(fine=True)
        shape = tuple(np.array(grid.shape_zyx) * grid.subvoxels)
        if self.axes is None or any(not np.array_equal(a, b) for a, b in zip(axes, self.axes)):
            self.axes = axes
            self.grid = tuple(wp.array(a.astype(np.float32), dtype=float, device=self.device) for a in axes)
            self.output = wp.empty(int(np.prod(shape)), dtype=float, device=self.device)
        gpu_points = wp.array(points.astype(np.float32), dtype=wp.vec3, device=self.device)
        if self.mesh is None or not np.array_equal(triangles, self.triangles) or len(points) != len(self.mesh.points):
            self.triangles = triangles
            self.mesh = wp.Mesh(gpu_points, wp.array(triangles, dtype=int, device=self.device))
        else:
            wp.copy(self.mesh.points, gpu_points)
            self.mesh.refit()
        # Account for float32 coordinates/BVH arithmetic. Only points within
        # this numerical margin of their local thickness need double VTK queries.
        scale = max(1.0, float(np.max(np.abs(points))), *(float(np.max(np.abs(a))) for a in axes))
        margin = 32 * np.finfo(np.float32).eps * max(scale, float(thickness_max))
        wp.launch(_wall_distances, dim=self.output.size,
                  inputs=[self.mesh.id, *self.grid, shape[2], shape[1], float(thickness_max + 2 * margin), self.output],
                  device=self.device)
        return self.output.numpy().reshape(shape), margin
