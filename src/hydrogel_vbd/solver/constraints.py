from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import MeshState


def fixed_z_constraints(mesh: MeshState, z_value: float, tolerance: float = 1e-9) -> np.ndarray:
    return np.isclose(mesh.vertices[:, 2], float(z_value), atol=tolerance)
