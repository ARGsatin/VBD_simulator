from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import MeshState


def gravity_force(mesh: MeshState, density: float, g: tuple[float, float, float] = (0.0, 0.0, -9.81)) -> np.ndarray:
    forces = np.zeros_like(mesh.vertices)
    direction = np.asarray(g, dtype=float)
    forces[mesh.active_vertex_mask] = density * direction
    return forces
