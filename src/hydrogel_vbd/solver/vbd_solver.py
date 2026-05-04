from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import MeshState


class PythonReferenceVBDSolver:
    """Small reference stepper behind the future VBD core interface.

    This is not a full elastic VBD implementation. It keeps the API stable while
    the rest of the closed loop is developed and tested.
    """

    def __init__(self, damping: float = 0.05) -> None:
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
