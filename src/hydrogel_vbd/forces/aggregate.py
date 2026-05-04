from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import ForceState


def aggregate_forces(gravity: np.ndarray, peel: np.ndarray, fluid: np.ndarray, surface: np.ndarray, electric: np.ndarray) -> ForceState:
    return ForceState(gravity=gravity, peel=peel, fluid=fluid, surface=surface, electric=electric)
