"""Read packed SimVascular VTU fields, their project metadata, and FSI surfaces."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import re

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from .config import CFDInput
from .temporal import TemporalInterpolator, weighted_fields


def _header(source: Path) -> tuple[bytes, str]:
    if source.suffix.lower() != ".vtu":
        raise ValueError("CFD input must be an extracted .vtu file")
    with source.open("rb") as stream:
        return _read_xml_prefix(stream), source.name


def _read_xml_prefix(stream) -> bytes:
    chunks = bytearray()
    while len(chunks) < 16 * 1024 * 1024:
        chunk = stream.read(65536)
        if not chunk:
            break
        chunks.extend(chunk)
        marker = chunks.find(b"<AppendedData")
        if marker >= 0:
            return bytes(chunks[:marker])
    return bytes(chunks)


def _project_metadata(project: Path | None, deformable: bool) -> dict:
    result = {"time_step_s": None, "solver_file": None, "wall_member": None}
    if project is None:
        return result
    if not project.is_dir():
        raise ValueError("project must be an extracted SimVascular project directory")
    names = [p.relative_to(project).as_posix() for p in project.rglob("*") if p.is_file()]
    solvers = [n for n in names if n.lower().endswith("solver.inp")
               and ("fsi" in n.lower()) == deformable]
    time_steps = []
    for name in solvers:
        match = re.search(r"^\s*Time Step Size\s*:\s*([\deE.+-]+)",
                          (project / name).read_text(errors="replace"), re.M | re.I)
        if match:
            time_steps.append((name, float(match.group(1))))
    if time_steps:
        if not np.allclose([dt for _, dt in time_steps], time_steps[0][1], rtol=1e-8, atol=0):
            raise ValueError("Project has conflicting CFD time steps; specify time_step_s explicitly")
        result["solver_file"], result["time_step_s"] = time_steps[0]
    preferred = "varwallprop.vtp" if deformable else "walls_combined.vtp"
    walls = [n for n in names if n.lower().endswith(preferred)]
    if not walls:
        walls = [n for n in names if n.lower().endswith("walls_combined.vtp")]
    if len(walls) > 1:
        raise ValueError("Project has multiple wall surfaces; specify wall_surface explicitly")
    if walls:
        result["wall_member"] = walls[0]
    return result


def inspect_case(cfd: str | CFDInput, project: str | None = None) -> dict:
    source = cfd if isinstance(cfd, CFDInput) else CFDInput(str(cfd), project=project)
    path = Path(source.cfd).resolve()
    header, member = _header(path)
    tags = re.findall(rb"<DataArray\b[^>]*>", header)
    fields = [dict((k.decode(), v.decode()) for k, v in
                   re.findall(rb'(\w+)="([^"]*)"', tag)) for tag in tags]
    pattern = re.compile(re.escape(source.velocity_prefix) + r"(\d+)$")
    velocities = sorted(((int(pattern.fullmatch(d.get("Name", "")).group(1)), d)
                         for d in fields if pattern.fullmatch(d.get("Name", ""))))
    if not velocities:
        raise ValueError(f"No time-indexed {source.velocity_prefix}<step> velocity arrays in {member}")
    if len(velocities) < 2:
        raise ValueError("A time-resolved phantom requires at least two CFD velocity frames")
    if any(d.get("NumberOfComponents", "1") != "3" for _, d in velocities):
        raise ValueError("Each CFD velocity array must have three components")
    displacement_names = [source.displacement_prefix + d["Name"][len(source.velocity_prefix):]
                          for _, d in velocities]
    all_names = {d.get("Name") for d in fields}
    present = [n in all_names for n in displacement_names]
    if any(present) and not all(present):
        raise ValueError("Displacement is missing for some velocity frames")
    deformable = all(present)
    if "FSI" in (member.upper() + path.name.upper()) and not deformable:
        raise ValueError("FSI input lacks time-resolved displacements; refusing to silently freeze geometry")
    project_path = Path(source.project).resolve() if source.project else None
    meta = _project_metadata(project_path, deformable)
    steps = np.array([n for n, _ in velocities], dtype=np.int64)
    step_s = source.time_step_s if source.time_step_s is not None else meta["time_step_s"]
    if source.times_s is not None:
        times = np.asarray(source.times_s, dtype=float)
        if times.shape != steps.shape:
            raise ValueError("times_s must have one entry per velocity array, sorted by step")
    elif step_s is not None:
        times = steps * step_s
    else:
        raise ValueError("Physical timestamps are required: supply project, time_step_s, or times_s")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("CFD times_s must be finite and strictly increasing")
    time_origin_s = float(times[0])
    times = times - times[0]
    intervals = np.diff(times)
    period = source.period_s
    inferred_period = False
    if source.periodic:
        if period is None:
            if not np.allclose(intervals, intervals[0], rtol=1e-6, atol=1e-12):
                raise ValueError("Nonuniform periodic CFD frames require an explicit period_s")
            period = float(times[-1] + intervals[0])
            inferred_period = True
        if period <= times[-1]:
            raise ValueError("period_s must be greater than the last relative CFD timestamp")
    piece = re.search(rb'<Piece[^>]*NumberOfPoints="(\d+)"[^>]*NumberOfCells="(\d+)"', header)
    return {
        "source": str(path), "project": str(project_path) if project_path else None,
        "vtu_member": member, "n_points": int(piece.group(1)) if piece else None,
        "n_cells": int(piece.group(2)) if piece else None,
        "n_frames": len(velocities), "steps": steps.tolist(),
        "velocity_arrays": [d["Name"] for _, d in velocities],
        "displacement_arrays": displacement_names if deformable else [],
        "deformable": deformable, "time_step_s": step_s,
        "solver_file": meta["solver_file"], "times_s": times.tolist(),
        "input_time_origin_s": time_origin_s, "native_dt_ms": float(np.median(intervals) * 1000),
        "native_min_dt_ms": float(intervals.min() * 1000),
        "native_max_dt_ms": float(intervals.max() * 1000),
        "period_s": period, "period_inferred_from_complete_cycle": inferred_period,
        "periodic": source.periodic, "wall_member": meta["wall_member"],
        "length_unit": source.length_unit, "velocity_unit": source.velocity_unit,
    }


def vtk_points(points_mm: np.ndarray) -> vtk.vtkPoints:
    result = vtk.vtkPoints()
    result.SetData(numpy_to_vtk(np.ascontiguousarray(points_mm), deep=True))
    return result


@dataclass
class CFDCase:
    mesh: vtk.vtkUnstructuredGrid
    reference_points_mm: np.ndarray
    velocity_m_s: np.ndarray
    displacement_mm: np.ndarray | None
    times_s: np.ndarray
    period_s: float | None
    periodic: bool
    wall: vtk.vtkPolyData | None
    wall_point_indices: np.ndarray | None
    wall_thickness_mm: np.ndarray | None
    world_center_mm: np.ndarray
    info: dict

    def bracket(self, time_s: float) -> tuple[int, int, float, float]:
        if self.periodic:
            time_s %= self.period_s
            timestamps = np.append(self.times_s, self.period_s)
        else:
            time_s = float(np.clip(time_s, self.times_s[0], self.times_s[-1]))
            timestamps = self.times_s
        left = int(np.clip(np.searchsorted(timestamps, time_s, side="right") - 1, 0, len(timestamps) - 2))
        right = (left + 1) % len(self.times_s)
        interval = float(timestamps[left + 1] - timestamps[left])
        return left, right, float((time_s - timestamps[left]) / interval), interval

    @cached_property
    def temporal_interpolator(self):
        return TemporalInterpolator(self.times_s, self.period_s if self.periodic else None)

    def fields(self, time_s: float, interpolation_order: int = 1):
        weights, derivative = self.temporal_interpolator.weights(time_s, interpolation_order)
        velocity = weighted_fields(weights, self.velocity_m_s)
        if self.displacement_mm is None:
            points = self.reference_points_mm
            wall_velocity = np.zeros((self.wall.GetNumberOfPoints(), 3), dtype=np.float32) if self.wall else None
        else:
            disp = weighted_fields(weights, self.displacement_mm)
            points = self.reference_points_mm + disp
            wall_velocity = (weighted_fields(derivative, self.displacement_mm[:, self.wall_point_indices]) / 1000
                             if self.wall is not None else None)
        mesh = vtk.vtkUnstructuredGrid()
        mesh.ShallowCopy(self.mesh)
        mesh.SetPoints(vtk_points(points))
        mesh.GetPointData().Initialize()
        array = numpy_to_vtk(np.ascontiguousarray(velocity), deep=True)
        array.SetName("velocity_m_s")
        mesh.GetPointData().AddArray(array)
        surface = None
        if self.wall is not None:
            surface = vtk.vtkPolyData()
            surface.ShallowCopy(self.wall)
            surface.SetPoints(vtk_points(points[self.wall_point_indices]))
        return mesh, surface, wall_velocity

    def bounds_mm(self, sample_times=None, interpolation_order: int = 1) -> np.ndarray:
        minimum = self.reference_points_mm.min(axis=0)
        maximum = self.reference_points_mm.max(axis=0)
        if self.displacement_mm is not None:
            for disp in self.displacement_mm:
                minimum = np.minimum(minimum, (self.reference_points_mm + disp).min(axis=0))
                maximum = np.maximum(maximum, (self.reference_points_mm + disp).max(axis=0))
            if sample_times is not None and interpolation_order > 1:
                for t in sample_times:
                    weights, _ = self.temporal_interpolator.weights(float(t), interpolation_order)
                    moved = self.reference_points_mm + weighted_fields(weights, self.displacement_mm)
                    minimum = np.minimum(minimum, moved.min(axis=0))
                    maximum = np.maximum(maximum, moved.max(axis=0))
        return np.stack([minimum, maximum])


def load_case(source: CFDInput) -> CFDCase:
    info = inspect_case(source)
    cfd = Path(source.cfd).resolve()
    filename = cfd
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(filename))
    reader.UpdateInformation()
    reader.GetPointDataArraySelection().DisableAllArrays()
    reader.GetCellDataArraySelection().DisableAllArrays()
    for name in info["velocity_arrays"] + info["displacement_arrays"] + ["GlobalNodeID"]:
        reader.GetPointDataArraySelection().EnableArray(name)
    reader.Update()
    raw = reader.GetOutput()
    if raw.GetNumberOfPoints() == 0 or reader.GetErrorCode():
        raise ValueError(f"Cannot read CFD mesh: {filename}")
    length_scale = {"m": 1000., "cm": 10., "mm": 1.}[source.length_unit]
    velocity_scale = {"m/s": 1., "cm/s": .01, "mm/s": .001}[source.velocity_unit]
    points = vtk_to_numpy(raw.GetPoints().GetData()).astype(np.float64) * length_scale
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    points -= center
    arrays = raw.GetPointData()
    velocity = np.stack([vtk_to_numpy(arrays.GetArray(n)).astype(np.float32) * velocity_scale
                         for n in info["velocity_arrays"]])
    displacement = (np.stack([vtk_to_numpy(arrays.GetArray(n)).astype(np.float32) * length_scale
                              for n in info["displacement_arrays"]]) if info["deformable"] else None)
    if not np.all(np.isfinite(velocity)) or (displacement is not None and not np.all(np.isfinite(displacement))):
        raise ValueError("CFD velocities/displacements contain nonfinite values")
    node_ids = arrays.GetArray("GlobalNodeID")
    ids = vtk_to_numpy(node_ids).copy() if node_ids is not None else None
    mesh = vtk.vtkUnstructuredGrid()
    mesh.ShallowCopy(raw)
    mesh.SetPoints(vtk_points(points))
    mesh.GetPointData().Initialize()
    mesh.GetCellData().Initialize()
    wall_path = None
    if source.wall_surface:
        wall_path = Path(source.wall_surface).resolve()
    elif info["wall_member"]:
        project = Path(source.project).resolve()
        wall_path = project / info["wall_member"]
    surface, indices, thickness = None, None, None
    if wall_path:
        wall_reader = vtk.vtkXMLPolyDataReader()
        wall_reader.SetFileName(str(wall_path))
        wall_reader.Update()
        surface = vtk.vtkPolyData()
        surface.ShallowCopy(wall_reader.GetOutput())
        if surface.GetNumberOfPoints() == 0:
            raise ValueError(f"Cannot read wall surface: {wall_path}")
        wall_ids = surface.GetPointData().GetArray("GlobalNodeID")
        if ids is not None and wall_ids is not None:
            wall_ids = vtk_to_numpy(wall_ids)
            order = np.argsort(ids)
            locations = np.searchsorted(ids[order], wall_ids)
            if np.any(locations >= len(ids)) or not np.array_equal(ids[order[locations]], wall_ids):
                raise ValueError("Wall surface node IDs do not match this CFD mesh")
            indices = order[locations]
        else:
            from scipy.spatial import cKDTree
            surface_points = vtk_to_numpy(surface.GetPoints().GetData()) * length_scale - center
            distances, indices = cKDTree(points).query(surface_points)
            if np.max(distances) > 0.01:
                raise ValueError("Wall surface does not match the CFD reference mesh")
        original_wall_points = vtk_to_numpy(surface.GetPoints().GetData()) * length_scale - center
        if not np.allclose(original_wall_points, points[indices], atol=0.01, rtol=0):
            raise ValueError("Wall IDs match but reference coordinates differ; check project/CFD pairing")
        wall_prop = surface.GetPointData().GetArray("thickness")
        if wall_prop is not None:
            thickness = vtk_to_numpy(wall_prop).astype(np.float32) * length_scale
            if np.any(~np.isfinite(thickness)) or np.any(thickness <= 0):
                raise ValueError("Project wall thickness must be finite and positive")
        surface.SetPoints(vtk_points(points[indices]))
        surface.GetPointData().Initialize()
        surface.GetCellData().Initialize()
    info["reference_bounds_mm"] = np.stack([points.min(axis=0), points.max(axis=0)]).tolist()
    info["world_center_mm"] = center.tolist()
    info["wall_thickness_range_mm"] = [float(thickness.min()), float(thickness.max())] if thickness is not None else None
    return CFDCase(mesh, points, velocity, displacement, np.array(info["times_s"]),
                   info["period_s"], source.periodic, surface, indices, thickness, center, info)
