from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import FieldCommand, MeshState


class ElectricForceModel:
    def __init__(self, alpha: float | np.ndarray, direction: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> None:
        self.alpha = alpha
        direction_array = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction_array)
        if norm == 0.0:
            raise ValueError("direction must be nonzero")
        self.direction = direction_array / norm

    def compute(self, mesh: MeshState, command: FieldCommand) -> np.ndarray:
        forces = np.zeros_like(mesh.vertices)
        alpha = np.asarray(self.alpha, dtype=float)
        if alpha.ndim == 0:
            alpha_per_vertex = np.full(mesh.vertices.shape[0], float(alpha))
        elif alpha.shape == (mesh.vertices.shape[0],):
            alpha_per_vertex = alpha
        else:
            raise ValueError("alpha must be scalar or one value per vertex")
        voltage_sum = float(np.sum(command.voltage))
        forces[mesh.active_vertex_mask] = alpha_per_vertex[mesh.active_vertex_mask, None] * voltage_sum * self.direction
        return forces

    @staticmethod
    def from_mapping(mapping: np.ndarray, voltage: np.ndarray, vertex_count: int) -> np.ndarray:
        flat_force = np.asarray(mapping, dtype=float) @ np.asarray(voltage, dtype=float)
        return flat_force.reshape(vertex_count, 3)
