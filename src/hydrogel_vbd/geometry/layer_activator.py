from __future__ import annotations

import numpy as np

from hydrogel_vbd.forces.czm import CZMState
from hydrogel_vbd.state import MeshState


class LayerActivator:
    """Updates mesh active masks for the current print layer."""

    def activate(self, mesh: MeshState, current_layer: int) -> MeshState:
        mesh.activate_layer(current_layer)
        return mesh

    def activate_with_inheritance(self, mesh: MeshState, current_layer: int, z_fep: float = 0.0) -> MeshState:
        if current_layer > 0:
            previous_bottom = mesh.bottom_nodes(current_layer - 1)
            collided = previous_bottom[mesh.vertices[previous_bottom, 2] < z_fep]
            mesh.vertices[collided, 2] = z_fep
            mesh.velocities[collided, 2] = 0.0

        mesh.activate_layer(current_layer)
        new_nodes = np.flatnonzero(mesh.first_active_layer == current_layer)
        top = mesh.top_nodes(current_layer)
        bottom = mesh.bottom_nodes(current_layer)

        if len(top):
            mesh.vertices[top] = np.minimum(mesh.vertices[top], mesh.ideal_vertices[top])
        if len(bottom):
            mesh.vertices[bottom, 2] = z_fep
        interior = np.setdiff1d(new_nodes, np.union1d(top, bottom), assume_unique=False)
        for idx in interior:
            ideal_z = mesh.ideal_vertices[idx, 2]
            mesh.vertices[idx, 2] = min(max(ideal_z, z_fep), ideal_z)
        mesh.velocities[new_nodes] = 0.0
        mesh.is_top_fixed[:] = False
        mesh.is_top_fixed[top] = True
        mesh.czm_state[new_nodes] = CZMState.FREE
        mesh.czm_state[bottom] = CZMState.FIXED
        mesh.damage[new_nodes] = 0.0
        mesh.time_free[new_nodes] = 0.0
        return mesh
