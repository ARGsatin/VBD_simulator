from __future__ import annotations

import numpy as np


def linear_tetrahedral_placeholder_force(vertices: np.ndarray, rest_vertices: np.ndarray, stiffness: float) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    rest_vertices = np.asarray(rest_vertices, dtype=float)
    if vertices.shape != rest_vertices.shape:
        raise ValueError("vertices and rest_vertices must have the same shape")
    return -float(stiffness) * (vertices - rest_vertices)
