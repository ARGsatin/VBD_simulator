from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import MeshState


def peel_force(
    mesh: MeshState,
    pressure: float,
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    vertex_area: float = 1.0,
) -> np.ndarray:
    forces = np.zeros_like(mesh.vertices)
    direction = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("normal must be nonzero")
    direction = direction / norm
    forces[mesh.active_vertex_mask] = float(pressure) * float(vertex_area) * direction
    return forces
