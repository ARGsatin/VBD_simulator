"""Tetrahedral mesh generation from STL files using TetGen."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tetgen
import trimesh


def from_stl(path: str | Path, quality: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a linear tetrahedral mesh from an STL file.

    Parameters
    ----------
    path:
        Path to the STL file (binary or ASCII).
    quality:
        Mesh refinement factor.  Larger values produce finer meshes
        (0.1 … 5.0, default 1.0).

    Returns
    -------
    vertices:
        ``(N, 3)`` float array of vertex coordinates.
    tets:
        ``(T, 4)`` int array of tetrahedron vertex indices.
    """
    mesh = trimesh.load(str(path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    max_vol = diag * 0.02 * quality

    tet = tetgen.TetGen(np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=int))
    result = tet.tetrahedralize(
        order=1,
        mindihedral=10.0,
        minratio=2.0,
        maxvolume=float(max_vol),
        verbose=0,
    )
    return np.asarray(result[0], dtype=float), np.asarray(result[1], dtype=int)
