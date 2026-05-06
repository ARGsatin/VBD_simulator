from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        return tuple(float(part.strip()) for part in text[1:-1].split(",") if part.strip())
    try:
        if any(marker in text.lower() for marker in (".", "e")):
            return float(text)
        return int(text)
    except ValueError:
        return text


@dataclass
class SimulationConfig:
    g: tuple[float, float, float] = (0.0, 0.0, -9.81)
    rho: float = 1050.0
    mu: float = 50000.0
    kappa: float = 1.0e7
    k_d: float = 0.5
    c_shrink: float = 0.98
    T_max: float = 5000.0
    K_czm: float = 1.0e8
    delta_f: float = 1.0e-4
    eta: float = 0.8
    d_min: float = 1.0e-6
    d_fluid_max: float = 2.0e-3
    t_fluid_max: float = 0.5
    dt: float = 0.01
    epsilon: float = 1.0e-6
    max_iters: int = 20
    N_stable: int = 10
    rho_cheb: float = 0.95
    c_init: float = 0.1
    err_target: float = 5.0e-4
    K_p: float = 150.0
    K_i: float = 20.0
    K_d: float = 5.0
    q_ion: float = 1.2e-3
    E_max: float = 500.0
    layer_thickness: float = 0.05
    z_fep: float = 0.0
    v_lift: float = 0.001
    C_0: float = 1.0
    fluid_radius: float = 0.001
    node_area: float = 1.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationConfig":
        values: dict[str, Any] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            if not hasattr(cls, key):
                continue
            values[key] = _parse_scalar(raw_value.split("#", 1)[0])
        return cls(**values)
