from __future__ import annotations

import numpy as np

from hydrogel_vbd.evaluation.metrics import max_norm, rms_norm


def compare_shapes(x_sim: np.ndarray, x_target: np.ndarray) -> dict[str, float]:
    simulated = np.asarray(x_sim, dtype=float)
    target = np.asarray(x_target, dtype=float)
    if simulated.shape != target.shape or simulated.ndim != 2 or simulated.shape[1] != 3:
        raise ValueError("x_sim and x_target must both have shape (N, 3)")
    error = target - simulated
    return {
        "rms_error": rms_norm(error),
        "max_error": max_norm(error),
        "max_z_sag": float(np.max(error[:, 2])),
        "mean_z_error": float(np.mean(error[:, 2])),
    }
