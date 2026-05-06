from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydrogel_vbd.config import SimulationConfig
from hydrogel_vbd.forces.czm import CZMState
from hydrogel_vbd.state import MeshState


@dataclass
class LocalPhysicsTerms:
    force: np.ndarray
    hessian: np.ndarray


def build_local_physics_terms(
    mesh: MeshState,
    config: SimulationConfig,
    e_z: float,
    x_prev: np.ndarray,
) -> LocalPhysicsTerms:
    force = np.zeros_like(mesh.vertices)
    hessian = np.zeros((mesh.vertices.shape[0], 3, 3), dtype=float)
    active = mesh.active_vertex_mask
    masses = mesh.masses
    g = np.asarray(config.g, dtype=float)
    force[active] += masses[active, None] * g
    force[active, 2] += config.q_ion * float(e_z)

    stiffness = max(float(config.mu), 1.0) * 1e-4
    displacement = mesh.vertices - mesh.ideal_vertices
    force[active] += -stiffness * displacement[active]
    hessian[active] += stiffness * np.eye(3)

    for node_id in np.flatnonzero(active):
        state = CZMState(int(mesh.czm_state[node_id]))
        if state == CZMState.DAMAGING:
            gap = max(float(mesh.vertices[node_id, 2] - config.z_fep), 0.0)
            softening = max(0.0, 1.0 - gap / max(config.delta_f, 1e-12))
            traction = (1.0 - float(mesh.damage[node_id])) * config.T_max * softening
            force[node_id, 2] -= traction * config.node_area
            hessian[node_id, 2, 2] += (1.0 - float(mesh.damage[node_id])) * config.T_max * config.node_area / max(config.delta_f, 1e-12)
        elif state == CZMState.FREE:
            gap = max(float(mesh.vertices[node_id, 2] - config.z_fep), config.d_min)
            if gap < config.d_fluid_max and mesh.time_free[node_id] < config.t_fluid_max:
                v_z_imp = float(mesh.vertices[node_id, 2] - x_prev[node_id, 2]) / max(config.dt, 1e-12)
                coeff = config.C_0 * config.eta * (config.fluid_radius**4) / (gap**3)
                force[node_id, 2] -= coeff * v_z_imp
                hessian[node_id, 2, 2] += coeff / max(config.dt, 1e-12)

    return LocalPhysicsTerms(force=force, hessian=hessian)
