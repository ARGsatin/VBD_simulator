# -*- coding: utf-8 -*-
"""电场控制器实现 —— 基于形状误差反馈的 PD/PID 电压补偿。

本模块提供两种控制器，用于在 DLP 打印仿真中通过调节电场
来补偿形状变形：

1. **FieldController**：通用 PD 映射型控制器
   将节点形状误差通过 PD 反馈转换为期望力，再用力映射矩阵
   求解最优电极电压。适用于有空间分布的多电极系统。

2. **PIDFieldController**：标量 PID 控制器
   基于宏观平均形状误差的标量 PID 反馈，输出均匀电场强度 E_z。
   适用于简化模型或单一电场的场景。

控制架构
--------
两种控制器都将形状误差转换为电场修正，但粒度不同：

* **FieldController** → 逐节点误差 → 逐电极电压向量
* **PIDFieldController** → 宏观平均误差 → 均匀标量场强

它们共享相同的错误信号来源（``shape_error.py`` 模块），
并在主循环中每步调用一次。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.control.voltage_optimizer import solve_regularized_voltage
from hydrogel_vbd.core.state import FieldCommand


class FieldController:
    """通用 PD 映射型电场控制器。

    工作流程：
    1. 接收**平坦化的节点形状误差** ``nodal_error`` (M,)
    2. 通过 PD 控制律计算期望力：``f_desired = Kp·e + Kd·Δe/Δt``
    3. 调用 ``solve_regularized_voltage()`` 求解最优电极电压
    4. 电压限幅后返回 ``FieldCommand``

    Attributes
    ----------
    force_mapping : np.ndarray, shape (M, N)
        力映射矩阵，M 为自由度总数，N 为电极数。
    kp : float
        比例增益，将形状误差 (m) 转换为期望力 (N)。
    kd : float
        微分增益，用于抑制振荡。
    regularization : float
        Tikhonov 正则化系数，抑制电压幅值。
    voltage_limits : tuple[float, float] or None
        电压限幅 (V_min, V_max)，None 表示不限幅。
    electrode_ids : list[str]
        各电极的标识符，用于生成 FieldCommand。
    """

    def __init__(
        self,
        force_mapping: np.ndarray,
        kp: float,
        kd: float = 0.0,
        regularization: float = 1e-3,
        voltage_limits: tuple[float, float] | None = None,
        electrode_ids: list[str] | None = None,
    ) -> None:
        """初始化 PD 映射型控制器。

        Parameters
        ----------
        force_mapping : np.ndarray, shape (M, N)
            力映射矩阵。
        kp : float
            比例增益（N/m）。
        kd : float, optional
            微分增益（N/(m/s)）。默认 0（纯比例控制）。
        regularization : float, optional
            正则化系数 λ。默认 1e-3。
        voltage_limits : tuple[float, float] | None, optional
            电压限幅 (V_min, V_max)。默认 None。
        electrode_ids : list[str] | None, optional
            电极标识符列表。默认自动生成 e0...e(N-1)。
        """
        self.force_mapping = np.asarray(force_mapping, dtype=float)
        self.kp = float(kp)
        self.kd = float(kd)
        self.regularization = float(regularization)
        self.voltage_limits = voltage_limits
        self.electrode_ids = electrode_ids or [
            f"e{i}" for i in range(self.force_mapping.shape[1])
        ]
        self._previous_error: np.ndarray | None = None

    def compute(
        self,
        nodal_error: np.ndarray,
        previous_command: FieldCommand | None = None,
    ) -> FieldCommand:
        """计算当前步的电极控制命令。

        根据节点形状误差，通过 PD 反馈和正则化最小二乘
        求解最优电极电压。

        Parameters
        ----------
        nodal_error : np.ndarray, shape (M,)
            平坦化的节点形状误差向量。
        previous_command : FieldCommand | None, optional
            上一步的电场命令（当前未使用，保留用于未来扩展）。

        Returns
        -------
        FieldCommand
            当前步的电场控制命令，包含电压向量和电极标识符。
        """
        del previous_command  # 保留用于未来扩展
        # ── 平坦化误差 ──
        error = np.asarray(nodal_error, dtype=float).reshape(-1)
        # ── 误差微分 ──
        derivative = (
            np.zeros_like(error)
            if self._previous_error is None
            else error - self._previous_error
        )
        # ── PD 期望力 ──
        desired_force = self.kp * error + self.kd * derivative
        # ── 正则化最小二乘求解最优电压 ──
        voltage = solve_regularized_voltage(
            self.force_mapping, desired_force, self.regularization
        )
        # ── 电压限幅 ──
        if self.voltage_limits is not None:
            voltage = np.clip(
                voltage, self.voltage_limits[0], self.voltage_limits[1]
            )
        # ── 存储误差用于下一步的微分计算 ──
        self._previous_error = error
        return FieldCommand(voltage=voltage, electrode_ids=self.electrode_ids)


@dataclass
class BottomZFieldState:
    """底部 Z 向局部控制器状态快照。"""

    E_z: float
    bottom_z_mean_error: float
    bottom_z_max_error: float
    unclipped_E_z: float
    err_avg: float
    PID_integral: float = 0.0
    prev_error: float = 0.0
    delta_E: float = 0.0


class BottomZFieldController:
    """基于当前层底部节点 Z 向误差的一维电场控制器。

    v1 使用现有均匀电场力模型，将每个底部节点的单位场强响应写成
    ``q_ion``，即 ``B_z = q_ion * 1``。求解出的单个标量仍作为
    求解器已有的 ``e_z`` 参数传入，不改变底层力模型。
    """

    def __init__(
        self,
        config: SimulationConfig,
        regularization: float | None = None,
    ) -> None:
        self.config = config
        self.regularization = float(
            getattr(config, "field_regularization", 1e-3)
            if regularization is None
            else regularization
        )
        self.E_z = 0.0
        self._previous_error_by_node: dict[int, float] = {}
        self._integral_by_node: dict[int, float] = {}

    def update(
        self,
        bottom_nodes: np.ndarray,
        target_vertices: np.ndarray,
        simulated_vertices: np.ndarray,
    ) -> BottomZFieldState:
        """根据底部节点 Z 误差计算下一步均匀电场强度。"""
        bottom = np.asarray(bottom_nodes, dtype=int).reshape(-1)
        target = np.asarray(target_vertices, dtype=float)
        simulated = np.asarray(simulated_vertices, dtype=float)

        if bottom.size == 0:
            self._previous_error_by_node = {}
            self._integral_by_node = {}
            return self._set_state(
                e_z=0.0,
                mean_error=0.0,
                max_error=0.0,
                unclipped=0.0,
                prev_error=0.0,
                pid_integral=0.0,
            )

        z_error = target[bottom, 2] - simulated[bottom, 2]
        mean_error = float(np.mean(z_error))
        max_error = float(np.max(z_error))
        active_error = np.maximum(
            z_error - float(self.config.err_target), 0.0
        )

        derivative = np.zeros_like(active_error)
        integral = np.zeros_like(active_error)
        dt = max(float(self.config.dt), 1e-12)
        k_i = float(getattr(self.config, "K_i", 0.0))
        q_ion = float(self.config.q_ion)
        integral_limit = math.inf
        if abs(k_i) > 1e-12 and abs(q_ion) > 1e-12:
            integral_limit = abs(q_ion) * float(self.config.E_max) / abs(k_i)
        for local_idx, node_id in enumerate(bottom):
            node_key = int(node_id)
            previous = self._previous_error_by_node.get(node_key)
            if previous is not None:
                derivative[local_idx] = (active_error[local_idx] - previous) / dt
            integral[local_idx] = self._integral_by_node.get(node_key, 0.0)
            integral[local_idx] += active_error[local_idx] * dt
        if math.isfinite(integral_limit):
            integral = np.clip(integral, -integral_limit, integral_limit)

        desired_force = (
            float(self.config.K_p) * active_error
            + k_i * integral
            + float(self.config.K_d) * derivative
        )

        if abs(q_ion) < 1e-12 or not np.any(desired_force):
            unclipped = 0.0
        else:
            mapping = q_ion * np.ones((bottom.size, 1), dtype=float)
            unclipped = float(
                solve_regularized_voltage(
                    mapping, desired_force, self.regularization
                )[0]
            )

        clipped = float(np.clip(unclipped, 0.0, float(self.config.E_max)))
        self._previous_error_by_node = {
            int(node_id): float(error)
            for node_id, error in zip(bottom, active_error)
        }
        self._integral_by_node = {
            int(node_id): float(value)
            for node_id, value in zip(bottom, integral)
        }
        return self._set_state(
            e_z=clipped,
            mean_error=mean_error,
            max_error=max_error,
            unclipped=unclipped,
            prev_error=float(np.mean(active_error)),
            pid_integral=float(np.mean(integral)),
        )

    def _set_state(
        self,
        *,
        e_z: float,
        mean_error: float,
        max_error: float,
        unclipped: float,
        prev_error: float,
        pid_integral: float,
    ) -> BottomZFieldState:
        previous_e_z = self.E_z
        self.E_z = float(e_z)
        return BottomZFieldState(
            E_z=self.E_z,
            bottom_z_mean_error=float(mean_error),
            bottom_z_max_error=float(max_error),
            unclipped_E_z=float(unclipped),
            err_avg=float(mean_error),
            PID_integral=float(pid_integral),
            prev_error=float(prev_error),
            delta_E=float(self.E_z - previous_e_z),
        )


@dataclass
class PIDFieldState:
    """标量 PID 控制器的状态快照。

    记录 PID 控制器在单步内的完整状态，用于日志记录和调试。

    Attributes
    ----------
    E_z : float
        当前电场强度（V/m）。
    err_avg : float
        宏观平均形状误差（m）。
    PID_integral : float
        PID 积分项的累积值（m·s）。
    prev_error : float
        上一步的误差（m），用于微分计算。
    delta_E : float
        本步电场强度的增量（V/m）。
    """
    E_z: float
    err_avg: float
    PID_integral: float
    prev_error: float
    delta_E: float


class PIDFieldController:
    """标量 PID 电场控制器。

    基于宏观平均形状误差（一个标量）的 PID 反馈，
    输出均匀电场强度 E_z。误差死区设置为 ``err_target``：
    仅当 ``err_avg > err_target`` 时才激活反馈。

    控制律（标准位置式 PID，杜绝双重积分爆炸）：
    .. math::
        E_z = clip( K_p·e + K_i·∫e dt + K_d·de/dt, 0, E_max )

    抗积分饱和策略
    --------------
    * **限幅前裁剪（Clamping Pre-clip）**：积分项 ``PID_integral`` 累加后
      立即裁剪到 ``[-E_max/K_i, +E_max/K_i]``，防止积分项指数级发散。
    * **反计算饱和（Clamping Anti-windup Back-calculation）**：若输出
      ``target_E`` 被 ``[0, E_max]`` 截断，则根据有效输出反算出校正后的
      积分项，使积分器始终保持在合理范围内。
    * **死区抑制**：误差低于 ``err_target`` 时不激活 PID，保持当前场强不变。

    特性
    ----
    * **纯正值**：E_z 始终在 [0, E_max] 范围内
    * **位置式公式**：直接设定目标场强，不再增量叠加，消除双重积分
    * **严格抗积分饱和**：上下限裁剪 + 反算回退，双重保护
    """

    def __init__(self, config: SimulationConfig) -> None:
        """初始化标量 PID 控制器。

        Parameters
        ----------
        config : SimulationConfig
            仿真配置对象，从中读取 PID 参数（K_p, K_i, K_d,
            err_target, E_max, dt）。
        """
        self.config = config
        self.E_z = 0.0          # 当前电场强度
        self.PID_integral = 0.0 # 积分累积项
        self.prev_error = 0.0   # 上一步误差

    def update(self, err_avg: float) -> PIDFieldState:
        """根据宏观平均误差更新电场强度。

        采用**位置式 PID**（Position Form）控制律：
        ``E_z = Kp·e + Ki·∫e dt + Kd·de/dt``，限幅后直接赋给
        ``self.E_z``，杜绝"增量叠加导致的双重积分爆炸"。

        仅在误差超过死区阈值时激活反馈；低于阈值时保持当前场强。
        积分项 ``PID_integral`` 具有严格的抗积分饱和（Anti-windup）
        上下限裁剪逻辑。

        Parameters
        ----------
        err_avg : float
            宏观平均形状误差（m）。

        Returns
        -------
        PIDFieldState
            包含当前 PID 状态所有字段的快照。
        """
        # ── 位置式 PID 输出计算 ──
        delta_e = 0.0
        error_input = 0.0
        P_term = 0.0
        I_term = 0.0
        D_term = 0.0

        if err_avg > self.config.err_target:
            # ── 误差超出死区，激活 PID ──
            error_input = float(err_avg - self.config.err_target)

            # 比例项
            P_term = self.config.K_p * error_input

            # 积分项：先累加，再抗积分饱和裁剪
            self.PID_integral += error_input * self.config.dt
            # 抗积分饱和：将积分项限制在 [-E_max/Ki, +E_max/Ki]
            # 防止因长期累加导致积分项指数发散
            integral_limit = (
                self.config.E_max / max(abs(self.config.K_i), 1e-12)
                if abs(self.config.K_i) > 1e-12
                else 0.0
            )
            self.PID_integral = float(
                np.clip(self.PID_integral, -integral_limit, +integral_limit)
            )
            I_term = self.config.K_i * self.PID_integral

            # 微分项
            derivative = (
                (error_input - self.prev_error)
                / max(self.config.dt, 1e-12)
            )
            D_term = self.config.K_d * derivative

            # ── 位置式 PID 输出：E = P + I + D ──
            target_E = P_term + I_term + D_term
            delta_e = target_E - self.E_z  # 记录本步增量（仅用于日志/状态快照）

            # ── 直接赋值并限幅（非增量累加！） ──
            self.E_z = float(
                np.clip(target_E, 0.0, self.config.E_max)
            )

            # 若 target_E 被 clip 截断且处于饱和边界，实施
            # 积分反算饱和（clamping anti-windup）：将积分项回退到
            # 恰好使 target_E 落在边界内的值
            if self.E_z != target_E:
                # 反算：I_corrected = E_z - P - D，仅当 clip 触及时回退
                corrected_I = self.E_z - P_term - D_term
                self.PID_integral = corrected_I / max(abs(self.config.K_i), 1e-12)

            self.prev_error = error_input

        # ── 误差低于死区：保持 E_z 不变，但清除积分器以防下次跳跃 ──
        # (可选保守策略——若想保留历史状态可注释掉下面这行)
        # self.PID_integral *= 0.9  # 让积分项逐渐衰减

        return PIDFieldState(
            E_z=self.E_z,
            err_avg=float(err_avg),
            PID_integral=self.PID_integral,
            prev_error=self.prev_error,
            delta_E=float(delta_e),
        )
