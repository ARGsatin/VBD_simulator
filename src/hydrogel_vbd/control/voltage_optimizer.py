from __future__ import annotations

import numpy as np


def solve_regularized_voltage(mapping: np.ndarray, desired_force: np.ndarray, regularization: float) -> np.ndarray:
    b = np.asarray(mapping, dtype=float)
    f = np.asarray(desired_force, dtype=float)
    if b.ndim != 2:
        raise ValueError("mapping must be a 2D matrix")
    if f.shape != (b.shape[0],):
        raise ValueError("desired_force length must match mapping rows")
    lhs = b.T @ b + float(regularization) * np.eye(b.shape[1])
    rhs = b.T @ f
    return np.linalg.solve(lhs, rhs)
