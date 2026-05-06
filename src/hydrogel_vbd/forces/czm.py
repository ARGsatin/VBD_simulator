from __future__ import annotations

from enum import IntEnum

import numpy as np

from hydrogel_vbd.state import MeshState


class CZMState(IntEnum):
    FIXED = 0
    DAMAGING = 1
    FREE = 2


def update_czm_states(
    mesh: MeshState,
    bottom_nodes: np.ndarray,
    internal_pull_z: np.ndarray,
    area: float,
    t_max: float,
    k_czm: float,
    delta_f: float,
    z_fep: float,
    dt: float,
) -> None:
    bottom_nodes = np.asarray(bottom_nodes, dtype=int)
    pulls = np.asarray(internal_pull_z, dtype=float)
    for local_idx, node_id in enumerate(bottom_nodes):
        state = CZMState(int(mesh.czm_state[node_id]))
        if state == CZMState.FIXED:
            if pulls[local_idx] / max(float(area), 1e-12) >= float(t_max):
                mesh.czm_state[node_id] = CZMState.DAMAGING
        elif state == CZMState.DAMAGING:
            gap = max(float(mesh.vertices[node_id, 2] - z_fep), 0.0)
            damage = max(float(mesh.damage[node_id]), (gap / max(delta_f, 1e-12)) * (t_max / max(k_czm, 1e-12)))
            if gap >= delta_f:
                damage = 1.0
            mesh.damage[node_id] = float(np.clip(damage, 0.0, 1.0))
            if gap >= delta_f or mesh.damage[node_id] >= 1.0:
                mesh.czm_state[node_id] = CZMState.FREE
        elif state == CZMState.FREE:
            mesh.time_free[node_id] += float(dt)
