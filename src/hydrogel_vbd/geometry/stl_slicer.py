"""Slice an STL mesh into 2D cross-section contours at each print layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def load_stl(path: str | Path) -> trimesh.Trimesh:
    """Read an STL file and return a single Trimesh mesh."""
    mesh = trimesh.load(str(path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"STL file did not produce a triangle mesh: {type(mesh).__name__}")
    return mesh


def slice_stl(
    path: str | Path,
    layer_height: float,
    z_min: float | None = None,
    z_max: float | None = None,
) -> list[dict[str, Any]]:
    """Slice an STL file from *z_min* to *z_max* at *layer_height* intervals.

    Returns a list of dicts, one per slice::

        {"layer": int, "z": float, "polygons": [np.ndarray, ...]}

    Each polygon is an *(N, 2)* array of XY vertices in the projection plane.
    The projection plane is always XY (i.e. Z is the slicing axis).
    """
    mesh = load_stl(path)

    if z_min is None:
        z_min = float(mesh.bounds[0][2])
    if z_max is None:
        z_max = float(mesh.bounds[1][2])

    eps = layer_height * 0.01
    start = z_min + eps
    stop = z_max - eps

    if start >= stop:
        return []

    num = max(1, int(np.ceil((stop - start) / layer_height)) + 1)
    heights = np.linspace(start, stop, num)

    slices: list[dict[str, Any]] = []
    for i, z in enumerate(heights):
        polys: list[np.ndarray] = []
        sliced = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sliced is not None:
            for entity in sliced.entities:
                pts_3d = entity.discrete(sliced.vertices)
                if len(pts_3d) >= 3:
                    polys.append(pts_3d[:, :2])  # XY projection
        slices.append({"layer": i, "z": float(z), "polygons": polys})

    return slices


def slice_polygons_to_2d_triangulation(
    polygons: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a set of 2D polygons and return ``(vertices, triangles)``.

    This is a convenience wrapper around ``trimesh.creation.triangulate_polygon``.
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    all_verts: list[np.ndarray] = []
    all_tris: list[np.ndarray] = []
    offset = 0

    for poly in polygons:
        if len(poly) < 3:
            continue
        # shapely expects (N, 2) closed (first == last)
        shell = np.vstack([poly, poly[0:1]])
        sp = ShapelyPolygon(shell)
        verts, tris = trimesh.creation.triangulate_polygon(sp, engine="earcut")
        all_verts.append(verts)
        all_tris.append(np.asarray(tris) + offset)
        offset += len(verts)

    if not all_verts:
        return np.empty((0, 2), dtype=float), np.empty((0, 3), dtype=int)

    return np.vstack(all_verts), np.vstack(all_tris)
