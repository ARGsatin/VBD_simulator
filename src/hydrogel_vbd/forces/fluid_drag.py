from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import MeshState


def fluid_drag_force(mesh: MeshState, coefficient: float) -> np.ndarray:
    forces = np.zeros_like(mesh.vertices)
    forces[mesh.active_vertex_mask] = -float(coefficient) * mesh.velocities[mesh.active_vertex_mask]
    return forces
