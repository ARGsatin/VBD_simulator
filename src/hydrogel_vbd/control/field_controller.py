from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydrogel_vbd.config import SimulationConfig
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


@dataclass
class PIDFieldState:
    E_z: float
    err_avg: float
    PID_integral: float
    prev_error: float
    delta_E: float


class PIDFieldController:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.E_z = 0.0
        self.PID_integral = 0.0
        self.prev_error = 0.0

    def update(self, err_avg: float) -> PIDFieldState:
        delta_e = 0.0
        error_input = 0.0
        if err_avg > self.config.err_target:
            error_input = float(err_avg - self.config.err_target)
            self.PID_integral += error_input * self.config.dt
            derivative = (error_input - self.prev_error) / max(self.config.dt, 1e-12)
            delta_e = self.config.K_p * error_input + self.config.K_i * self.PID_integral + self.config.K_d * derivative
            self.E_z = float(np.clip(self.E_z + delta_e, 0.0, self.config.E_max))
            self.prev_error = error_input
        return PIDFieldState(
            E_z=self.E_z,
            err_avg=float(err_avg),
            PID_integral=self.PID_integral,
            prev_error=self.prev_error,
            delta_E=float(delta_e),
        )
