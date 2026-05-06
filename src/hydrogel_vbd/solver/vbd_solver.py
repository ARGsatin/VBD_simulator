from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydrogel_vbd.config import SimulationConfig
from hydrogel_vbd.forces.czm import CZMState
from hydrogel_vbd.forces.local_terms import build_local_physics_terms
from hydrogel_vbd.state import MeshState


@dataclass
class VBDSolveResult:
    x: np.ndarray
    v: np.ndarray
    iterations: int
    max_dx: float
    kinetic_energy: float
    stable_steps: int
    all_free: bool
    chebyshev_skipped_damaging: int


class PythonReferenceVBDSolver:
    """Reference VBD architecture with local 3x3 Newton updates."""

    def __init__(self, damping: float | SimulationConfig = 0.05) -> None:
        if isinstance(damping, SimulationConfig):
            self.config = damping
            self.damping = float(damping.k_d)
        else:
            self.config = SimulationConfig(k_d=float(damping))
            self.damping = float(damping)

    def step(
        self,
        mesh: MeshState,
        forces: np.ndarray,
        constraints: np.ndarray | None,
        dt: float,
        substeps: int,
        iterations: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del iterations
        forces = np.asarray(forces, dtype=float)
        if forces.shape != mesh.vertices.shape:
            raise ValueError("forces must match mesh vertices shape")
        fixed = np.zeros(mesh.vertices.shape[0], dtype=bool) if constraints is None else np.asarray(constraints, dtype=bool)
        if fixed.shape != (mesh.vertices.shape[0],):
            raise ValueError("constraints must have shape (N,)")

        x = mesh.vertices.copy()
        v = mesh.velocities.copy()
        sub_dt = float(dt) / max(int(substeps), 1)
        movable = mesh.active_vertex_mask & ~fixed
        masses = mesh.masses[:, None]

        for _ in range(max(int(substeps), 1)):
            acceleration = forces / masses
            v[movable] = (1.0 - self.damping) * v[movable] + sub_dt * acceleration[movable]
            x[movable] = x[movable] + sub_dt * v[movable]
            x[fixed] = mesh.vertices[fixed]
            v[fixed] = 0.0

        mesh.prev_vertices = mesh.vertices.copy()
        mesh.vertices = x
        mesh.velocities = v
        return x, v

    def solve_until_stable(self, mesh: MeshState, layer_id: int, e_z: float) -> VBDSolveResult:
        config = self.config
        x_prev = mesh.vertices.copy()
        terms = build_local_physics_terms(mesh, config, e_z=e_z, x_prev=x_prev)
        masses = mesh.masses
        adaptive_accel = np.zeros_like(mesh.vertices)
        adaptive_accel[mesh.active_vertex_mask] = config.c_init * terms.force[mesh.active_vertex_mask] / masses[mesh.active_vertex_mask, None]
        y = x_prev + config.dt * mesh.velocities + (config.dt**2) * adaptive_accel
        fixed = mesh.is_top_fixed | (mesh.czm_state == CZMState.FIXED) | ~mesh.active_vertex_mask
        colors = mesh.colors if mesh.colors is not None else np.zeros(mesh.vertices.shape[0], dtype=int)
        max_dx = 0.0
        iterations_done = 0
        damaging_count = int(np.sum(mesh.active_vertex_mask & (mesh.czm_state == CZMState.DAMAGING)))

        for iteration in range(1, config.max_iters + 1):
            iterations_done = iteration
            x_old_iter = mesh.vertices.copy()
            terms = build_local_physics_terms(mesh, config, e_z=e_z, x_prev=x_prev)
            max_dx = 0.0
            for color in sorted(set(int(c) for c in colors)):
                for node_id in np.flatnonzero(colors == color):
                    if fixed[node_id]:
                        continue
                    h_elastic = terms.hessian[node_id]
                    h_total = (
                        (masses[node_id] / (config.dt**2)) * np.eye(3)
                        + h_elastic
                        + (config.k_d / max(config.dt, 1e-12)) * h_elastic
                        + 1e-9 * np.eye(3)
                    )
                    f_inertia = -(masses[node_id] / (config.dt**2)) * (mesh.vertices[node_id] - y[node_id])
                    f_damp = -(config.k_d / max(config.dt, 1e-12)) * h_elastic @ (mesh.vertices[node_id] - x_prev[node_id])
                    f_total = terms.force[node_id] + f_inertia + f_damp
                    dx = np.linalg.solve(h_total, f_total)
                    length = float(np.linalg.norm(dx))
                    if length > 0.01:
                        dx *= 0.01 / length
                        length = 0.01
                    mesh.vertices[node_id] += dx
                    max_dx = max(max_dx, length)

            if iteration > 5:
                omega = self._chebyshev_omega(iteration, config.rho_cheb)
                free_mask = mesh.active_vertex_mask & ~fixed & (mesh.czm_state != CZMState.DAMAGING)
                mesh.vertices[free_mask] += omega * (mesh.vertices[free_mask] - x_old_iter[free_mask])

            if max_dx < config.epsilon:
                break

        free = mesh.active_vertex_mask & ~fixed
        mesh.velocities[free] = (mesh.vertices[free] - x_prev[free]) / max(config.dt, 1e-12)
        mesh.velocities[fixed] = 0.0
        mesh.prev_vertices = x_prev
        free_bottom = mesh.bottom_nodes(layer_id)
        all_free = bool(len(free_bottom) == 0 or np.all(mesh.czm_state[free_bottom] == CZMState.FREE))
        kinetic = float(0.5 * np.sum(masses[free] * np.sum(mesh.velocities[free] ** 2, axis=1)))
        stable_steps = config.N_stable if kinetic < 1e-6 and max_dx < config.epsilon else 0
        return VBDSolveResult(
            x=mesh.vertices.copy(),
            v=mesh.velocities.copy(),
            iterations=iterations_done,
            max_dx=float(max_dx),
            kinetic_energy=kinetic,
            stable_steps=int(stable_steps),
            all_free=all_free,
            chebyshev_skipped_damaging=damaging_count,
        )

    @staticmethod
    def _chebyshev_omega(iteration: int, rho_cheb: float) -> float:
        return float(min(0.5, (rho_cheb**iteration) / (1.0 + rho_cheb**iteration)))
