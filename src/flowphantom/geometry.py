"""Cell-based CFD interpolation and wall-shell sampling on an Eulerian grid."""

from dataclasses import dataclass
import numpy as np
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy

from .io import CFDCase


@dataclass
class Grid:
    shape_xyz: tuple[int, int, int]
    resolution_mm: np.ndarray
    center_mm: np.ndarray
    subvoxels: int

    @property
    def shape_zyx(self):
        return self.shape_xyz[::-1]

    @property
    def fov_mm(self):
        return np.array(self.shape_xyz) * self.resolution_mm

    def axes(self, fine=False):
        factor = self.subvoxels if fine else 1
        return tuple(c - f / 2 + (np.arange(n * factor) + .5) * d / factor
                     for c, f, n, d in zip(self.center_mm, self.fov_mm, self.shape_xyz, self.resolution_mm))

    def vtk_image(self):
        axes = self.axes(fine=True)
        image = vtk.vtkImageData()
        image.SetDimensions(*(np.array(self.shape_xyz) * self.subvoxels).tolist())
        image.SetSpacing(*(self.resolution_mm / self.subvoxels).tolist())
        image.SetOrigin(*(float(a[0]) for a in axes))
        return image


def block_mean(field, grid: Grid):
    """Subvoxel volume average; trailing component axes, if any, are preserved."""
    nz, ny, nx = grid.shape_zyx
    s = grid.subvoxels
    return field.reshape(nz, s, ny, s, nx, s, *field.shape[3:]).mean(axis=(1, 3, 5))


def sample_fields(case: CFDCase, grid: Grid, time_s: float, wall_thickness_mm: float | None,
                  include_wall: bool, interpolation_order: int = 1, wall_sampler=None):
    mesh, wall, wall_node_velocity = case.fields(time_s, interpolation_order)
    image = grid.vtk_image()
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(image)
    probe.SetSourceData(mesh)
    locator = vtk.vtkStaticCellLocator()
    if hasattr(probe, "SetCellLocator"):
        locator.SetDataSet(mesh)
        locator.BuildLocator()
        probe.SetCellLocator(locator)
    else:
        probe.SetCellLocatorPrototype(locator)
    probe.ComputeToleranceOff()
    probe.SetTolerance(1e-7)
    probe.Update()
    sampled = probe.GetOutput().GetPointData()
    shape = tuple(np.array(grid.shape_zyx) * grid.subvoxels)
    valid = sampled.GetArray("vtkValidPointMask")
    values = sampled.GetArray("velocity_m_s")
    if valid is None or values is None:
        raise RuntimeError("VTK could not sample the CFD volume")
    blood_mask = vtk_to_numpy(valid).reshape(shape).astype(bool)
    velocity = vtk_to_numpy(values).reshape(*shape, 3).astype(np.float32)
    velocity[~blood_mask] = 0
    wall_mask = np.zeros(shape, dtype=bool)
    wall_velocity = np.zeros((*shape, 3), dtype=np.float32)
    if include_wall:
        if wall is None:
            raise ValueError("Wall signal requires project or wall_surface, excluding inlet/outlet caps")
        thickness_max = wall_thickness_mm if wall_thickness_mm is not None else float(case.wall_thickness_mm.max())
        implicit = vtk.vtkImplicitPolyDataDistance()
        implicit.SetInput(wall)
        axes = grid.axes(fine=True)
        if wall_sampler is None:
            distances = vtk.vtkSampleFunction()
            distances.SetImplicitFunction(implicit)
            distances.SetSampleDimensions(*(np.array(grid.shape_xyz) * grid.subvoxels).tolist())
            distances.SetModelBounds(*(v for a in axes for v in (float(a[0]), float(a[-1]))))
            distances.ComputeNormalsOff()
            distances.Update()
            distance = np.abs(vtk_to_numpy(distances.GetOutput().GetPointData().GetScalars()).reshape(shape))
            margin = 0.0
        else:
            distance, margin = wall_sampler.distances(wall, grid, thickness_max)
        candidate = np.flatnonzero((~blood_mask & (distance <= thickness_max + margin)).ravel())
        if candidate.size:
            zz, yy, xx = np.unravel_index(candidate, shape)
            query = np.column_stack([axes[0][xx], axes[1][yy], axes[2][zz]])
            wall_points = vtk_to_numpy(wall.GetPoints().GetData())
            _, nearest = cKDTree(wall_points).query(query, workers=-1)
            thickness = wall_thickness_mm if wall_thickness_mm is not None else case.wall_thickness_mm[nearest]
            local_distance = distance.ravel()[candidate].astype(float)
            if margin:
                boundary = np.flatnonzero(np.abs(local_distance - thickness) <= margin)
                for i in boundary:
                    local_distance[i] = abs(implicit.EvaluateFunction(query[i]))
            selected = local_distance <= thickness
            candidate = candidate[selected]
            wall_mask.ravel()[candidate] = True
            wall_velocity.reshape(-1, 3)[candidate] = wall_node_velocity[nearest[selected]]
    return blood_mask, velocity, wall_mask, wall_velocity
