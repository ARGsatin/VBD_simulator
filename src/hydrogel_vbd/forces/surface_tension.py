from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import MeshState


def surface_tension_force(mesh: MeshState, coefficient: float, direction: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    forces = np.zeros_like(mesh.vertices)
    direction_array = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction_array)
    if norm == 0.0:
        raise ValueError("direction must be nonzero")
    forces[mesh.active_vertex_mask] = float(coefficient) * direction_array / norm
    return forces
