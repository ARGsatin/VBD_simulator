from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import MaterialState


class HydrogelMaterialModel:
    def __init__(
        self,
        density: float,
        young_modulus_min: float,
        young_modulus_max: float,
        poisson_ratio: float,
        damping: float,
        curing_exponent: float,
        peel_stress_crit: float,
        electric_response_alpha: float,
    ) -> None:
        self.density = float(density)
        self.young_modulus_min = float(young_modulus_min)
        self.young_modulus_max = float(young_modulus_max)
        self.poisson_ratio = float(poisson_ratio)
        self.damping = float(damping)
        self.curing_exponent = float(curing_exponent)
        self.peel_stress_crit = float(peel_stress_crit)
        self.electric_response_alpha = float(electric_response_alpha)

    def create_state(self, curing_degree: np.ndarray) -> MaterialState:
        phi = np.clip(np.asarray(curing_degree, dtype=float), 0.0, 1.0)
        young = self.young_modulus_min + (self.young_modulus_max - self.young_modulus_min) * (
            phi**self.curing_exponent
        )
        nu = self.poisson_ratio
        mu = young / (2.0 * (1.0 + nu))
        lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return MaterialState(
            density=self.density,
            young_modulus=young,
            poisson_ratio=nu,
            mu=mu,
            lam=lam,
            damping=np.full_like(phi, self.damping, dtype=float),
            curing_degree=phi,
            peel_stress_crit=self.peel_stress_crit,
            electric_response_alpha=np.full_like(phi, self.electric_response_alpha, dtype=float),
        )
