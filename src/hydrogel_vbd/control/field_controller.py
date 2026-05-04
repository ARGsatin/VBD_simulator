from __future__ import annotations

import numpy as np

from hydrogel_vbd.control.voltage_optimizer import solve_regularized_voltage
from hydrogel_vbd.state import FieldCommand


class FieldController:
    def __init__(
        self,
        force_mapping: np.ndarray,
        kp: float,
        kd: float = 0.0,
        regularization: float = 1e-3,
        voltage_limits: tuple[float, float] | None = None,
        electrode_ids: list[str] | None = None,
    ) -> None:
        self.force_mapping = np.asarray(force_mapping, dtype=float)
        self.kp = float(kp)
        self.kd = float(kd)
        self.regularization = float(regularization)
        self.voltage_limits = voltage_limits
        self.electrode_ids = electrode_ids or [f"e{i}" for i in range(self.force_mapping.shape[1])]
        self._previous_error: np.ndarray | None = None

    def compute(self, nodal_error: np.ndarray, previous_command: FieldCommand | None = None) -> FieldCommand:
        del previous_command
        error = np.asarray(nodal_error, dtype=float).reshape(-1)
        derivative = np.zeros_like(error) if self._previous_error is None else error - self._previous_error
        desired_force = self.kp * error + self.kd * derivative
        voltage = solve_regularized_voltage(self.force_mapping, desired_force, self.regularization)
        if self.voltage_limits is not None:
            voltage = np.clip(voltage, self.voltage_limits[0], self.voltage_limits[1])
        self._previous_error = error
        return FieldCommand(voltage=voltage, electrode_ids=self.electrode_ids)
