# -*- coding: utf-8 -*-
"""VBD（Vertex-Block Descent）求解器 —— 仿真核心引擎。

本模块实现了整个仿真系统的核心求解算法，基于顶点块坐标下降（VBD）框架，
结合 Neo-Hookean 弹性、Chebyshev 半隐式加速、内聚力模型（CZM）以及平台运动学。

核心类
------
- **VBDSolveResult**：求解器返回的结果数据结构
- **PythonReferenceVBDSolver**：纯 Python 实现的 VBD 参考求解器

求解流程
--------
1. **显式欧拉步（step）**：简单前向积分，用于快速预览（测试用）
2. **隐式 VBD 稳定求解（solve_until_stable）**：
   - 构建局部物理项（弹性力 + Hessian）
   - 按图着色分组并行进行逐顶点 3×3 Newton 迭代
   - Chebyshev 半隐式加速收敛
   - CZM 损伤节点 Hessian 正定化
   - 收敛判定：连续 N_stable 步 max_dx < epsilon
3. **带平台提升的求解（solve_with_lift）**：
   - 阶段 1：平台提升阶段（刚性抬升顶层节点 + CZM 更新）
   - 阶段 2：静平衡阶段（调用 solve_until_stable 内联逻辑）
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.physics.czm import CZMState
from hydrogel_vbd.physics.local_terms import build_local_physics_terms
from hydrogel_vbd.core.state import MeshState


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      VBDSolveResult —— 求解结果                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class VBDSolveResult:
    """VBD 求解器返回的单次求解结果。

    Parameters
    ----------
    x : np.ndarray, shape (N, 3)
        求解结束后的顶点坐标。
    v : np.ndarray, shape (N, 3)
        求解结束后的顶点速度。
    iterations : int
        实际执行的主迭代次数。
    max_dx : float
        最后一次迭代的最大顶点位移 (m)。
    kinetic_energy : float
        系统的动能 (J)。
    stable_steps : int
        收敛判据达标的连续步数。
    all_free : bool
        底部节点是否已全部脱膜（CZM 状态均为 FREE）。
    chebyshev_skipped_damaging : int
        Chebyshev 加速跳过的损伤节点数量。
    """
    x: np.ndarray                         # (N, 3) 终态坐标
    v: np.ndarray                         # (N, 3) 终态速度
    iterations: int                       # 迭代次数
    max_dx: float                         # 最大位移
    kinetic_energy: float                 # 动能
    stable_steps: int                     # 连续收敛步数
    all_free: bool                        # 底部是否全脱膜
    chebyshev_skipped_damaging: int       # 跳过的损伤节点数


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              PythonReferenceVBDSolver —— VBD 参考求解器                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PythonReferenceVBDSolver:
    """基于顶点块坐标下降（VBD）的参考求解器。

    采用**逐顶点**的局部 Newton 法：对每个顶点独立求解一个 3×3 线性系统，
    考虑惯性项、阻尼项和弹性 Hessian，按图着色分组以保证并行安全性。

    Parameters
    ----------
    damping : float or SimulationConfig
        阻尼参数。若传入 ``SimulationConfig`` 则从中提取 ``k_d``，
        否则作为独立的阻尼系数。
    """

    def __init__(self, damping: float | SimulationConfig = 0.05) -> None:
        """初始化求解器。

        - 若传入 SimulationConfig，同时保存配置引用（用于后续稳定求解）
        - 若传入 float，构造一个最小配置对象
        """
        if isinstance(damping, SimulationConfig):
            self.config = damping              # 保存完整配置引用
            self.damping = float(damping.k_d)   # 提取阻尼系数
        else:
            self.config = SimulationConfig(k_d=float(damping))
            self.damping = float(damping)

    # ───────────────────────────────────────────────────────────────────────
    # 显式欧拉步（轻量级、用于快速预览/测试）
    # ───────────────────────────────────────────────────────────────────────

    def step(
        self,
        mesh: MeshState,
        forces: np.ndarray,
        constraints: np.ndarray | None,
        dt: float,
        substeps: int,
        iterations: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """执行一次显式欧拉积分步。

        使用半隐式欧拉法（Symplectic Euler）：
        1. v_new = (1-damping)*v_old + dt * a
        2. x_new = x_old + dt * v_new
        3. 固定节点恢复原坐标、速度归零

        .. warning::
           此方法**不使用**隐式求解或收敛检查，
           主要用于快速验证力模型和测试框架。

        Parameters
        ----------
        mesh : MeshState
            网格状态（将被原地修改）。
        forces : np.ndarray, shape (N, 3)
            节点力向量。
        constraints : np.ndarray or None, shape (N,)
            布尔数组，标记固定（不可动）节点。
        dt : float
            时间步长 (s)。
        substeps : int
            子步数（dt 被均分）。
        iterations : int
            迭代次数（此方法中未使用，保留兼容性）。

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (新顶点坐标, 新速度)。
        """
        del iterations  # 显式方法不需要迭代参数
        forces = np.asarray(forces, dtype=float)
        if forces.shape != mesh.vertices.shape:
            raise ValueError("forces 的形状必须与 mesh.vertices 相同")

        # ── 处理约束 ──
        fixed = (
            np.zeros(mesh.vertices.shape[0], dtype=bool)
            if constraints is None
            else np.asarray(constraints, dtype=bool)
        )
        if fixed.shape != (mesh.vertices.shape[0],):
            raise ValueError("constraints 的形状必须为 (N,)")

        # ── 初始值 ──
        x = mesh.vertices.copy()
        v = mesh.velocities.copy()
        sub_dt = float(dt) / max(int(substeps), 1)
        movable = mesh.active_vertex_mask & ~fixed      # 可动节点掩码
        masses = mesh.masses[:, None]                    # (N, 1) 用于广播

        # ── 子步循环（半隐式欧拉） ──
        for _ in range(max(int(substeps), 1)):
            acceleration = forces / masses                # a = F / m
            # 速度更新（带阻尼衰减）
            v[movable] = (1.0 - self.damping) * v[movable] + sub_dt * acceleration[movable]
            # 位置更新
            x[movable] = x[movable] + sub_dt * v[movable]
            # 固定节点保持原位
            x[fixed] = mesh.vertices[fixed]
            v[fixed] = 0.0

        # ── 写回网格状态 ──
        mesh.prev_vertices = mesh.vertices.copy()
        mesh.vertices = x
        mesh.velocities = v
        return x, v

    # ───────────────────────────────────────────────────────────────────────
    # 隐式 VBD 稳定求解（核心算法）
    # ───────────────────────────────────────────────────────────────────────

    def solve_until_stable(
        self,
        mesh: MeshState,
        layer_id: int,
        e_z: float,
        on_iteration: Callable[[int, float], None] | None = None,
    ) -> VBDSolveResult:
        """执行 VBD 隐式迭代直到网格达到静力平衡。

        这是求解器的**核心算法**，采用逐顶点的 3×3 Newton 迭代：

        1. **预测步（Y 向量）**：基于显式预测 + 自适应加速度构造惯性参考点
        2. **主循环**：
           a. 按图着色分组遍历所有顶点
           b. 对每个顶点求解 ``H_total · dx = f_total``
           c. 其中 ``H_total = M/dt² + (1+ζ)·H_elastic + ε·I``
           d. 对损伤节点，将 H_elastic 投影到正定锥
           e. 步长限制（max 0.01）防止过度外推
           f. Chebyshev 加速（迭代 > 5 后启用）
        3. **收敛判定**：连续 N_stable 步 max_dx < epsilon 则退出

        Parameters
        ----------
        mesh : MeshState
            网格状态（将被原地修改）。
        layer_id : int
            当前求解的层 ID（用于 CZM 底面追踪）。
        e_z : float
            z 方向电场强度（来自 PID 控制器）。

        Returns
        -------
        VBDSolveResult
            包含终态坐标、速度、迭代统计等。
        """
        config = self.config
        x_prev = mesh.vertices.copy()           # 本时间步初态
        masses = mesh.masses

        # ── 构建局部物理项（弹性力 + Hessian） ──
        terms = build_local_physics_terms(mesh, config, e_z=e_z, x_prev=x_prev)

        # ── 自适应加速度（基于初始固化度） ──
        adaptive_accel = np.zeros_like(mesh.vertices)
        adaptive_accel[mesh.active_vertex_mask] = (
            config.c_init
            * terms.force[mesh.active_vertex_mask]
            / masses[mesh.active_vertex_mask, None]
        )

        # ── Y 向量：（惯性参考点，用于 VBD 的动量项） ──
        #   y = x_prev + dt·v + dt²·a_adaptive
        y = (
            x_prev
            + config.dt * mesh.velocities
            + (config.dt ** 2) * adaptive_accel
        )

        # ── 固定节点：平台夹持、CZM 固定、未激活 ──
        fixed = (
            mesh.is_top_fixed
            | (mesh.czm_state == CZMState.FIXED)
            | ~mesh.active_vertex_mask
        )

        # ── 图着色分组（用于并行化/分组计算） ──
        colors = (
            mesh.colors
            if mesh.colors is not None
            else np.zeros(mesh.vertices.shape[0], dtype=int)
        )

        max_dx = 0.0
        iterations_done = 0
        stable_counter = 0                                              # 连续收敛步数
        damaging_count = int(
            np.sum(
                mesh.active_vertex_mask
                & (mesh.czm_state == CZMState.DAMAGING)
            )
        )

        N_stable = int(config.N_stable)
        target_epsilon = float(config.epsilon)

        # ════════════════ 主迭代循环 ════════════════
        for iteration in range(1, config.max_iters + 1):
            iterations_done = iteration
            x_old_iter = mesh.vertices.copy()   # 记录本迭代初始位置（用于 Chebyshev）

            # TODO: 未来重构需将弹力与Hessian计算下沉移入着色循环的最内层，
            #       以恢复真正的Gauss-Seidel性能。当前在着色循环外部统一计算全局
            #       物理力（基于上一轮旧坐标），使Gauss-Seidel跌落为Jacobi迭代，
            #       降低了收敛速度，也使"图着色"失去了其核心加速意义。
            # ── 重新计算物理项（位置变化后力场变化） ──
            terms = build_local_physics_terms(mesh, config, e_z=e_z, x_prev=x_prev)
            max_dx = 0.0

            # ── 按颜色分组遍历（同色顶点互不影响，可安全并行） ──
            for color in sorted(set(int(c) for c in colors)):
                for node_id in np.flatnonzero(colors == color):
                    # 跳过固定节点
                    if fixed[node_id]:
                        continue

                    # ── 离型膜（FEP）穿透约束：底面节点不能低于 FEP 平面 ──
                    if (
                        not mesh.is_top_fixed[node_id]
                        and mesh.is_bottom_surface[node_id]
                    ):
                        if mesh.vertices[node_id, 2] < config.z_fep:
                            mesh.vertices[node_id, 2] = config.z_fep

                    # ── 弹性 Hessian ──
                    h_elastic = terms.hessian[node_id]

                    # ── 损伤节点 Hessian 正定化 ──
                    #   当 CZM 处于 DAMAGING 状态时，局部刚度可能为负，
                    #   导致求解发散。此步骤将 Hessian 特征值裁剪到 ≥0。
                    if mesh.czm_state[node_id] == CZMState.DAMAGING:
                        eigvals = np.linalg.eigvalsh(h_elastic)
                        if np.min(eigvals) < 0:
                            # 裁剪负特征值并重构矩阵
                            eigvals_psd = np.maximum(eigvals, 0.0)
                            eigvecs = np.linalg.eigh(h_elastic)[1]
                            h_elastic = eigvecs @ np.diag(eigvals_psd) @ eigvecs.T

                    # ── 总 Hessian：惯性项 + 弹性项 + 阻尼项 + 正则化 ──
                    #   H_total = M/dt² · I + (1 + k_d/dt) · H_elastic + ε·I
                    h_total = (
                        (masses[node_id] / (config.dt ** 2)) * np.eye(3)  # 惯性项
                        + h_elastic                                        # 弹性项
                        + (config.k_d / max(config.dt, 1e-12)) * h_elastic # 阻尼项
                        + 1e-9 * np.eye(3)                                # 正则化（防奇异）
                    )

                    # ── 惯性力：F_inertia = -M/dt² · (x - y) ──
                    f_inertia = (
                        -(masses[node_id] / (config.dt ** 2))
                        * (mesh.vertices[node_id] - y[node_id])
                    )

                    # ── 阻尼力：F_damp = -k_d/dt · H_elastic · (x - x_prev) ──
                    f_damp = (
                        -(config.k_d / max(config.dt, 1e-12))
                        * h_elastic @ (mesh.vertices[node_id] - x_prev[node_id])
                    )

                    # ── 合力：弹性力 + 惯性力 + 阻尼力 ──
                    f_total = terms.force[node_id] + f_inertia + f_damp

                    # ── 求解 3×3 线性系统：H·dx = f ──
                    dx = np.linalg.solve(h_total, f_total)
                    length = float(np.linalg.norm(dx))

                    # ── 步长限制：单步位移不超过 0.01 m ──
                    #   防止损伤节点或大变形区域过度外推导致发散
                    if length > 0.01:
                        dx *= 0.01 / length
                        length = 0.01

                    # ── 更新顶点位置 ──
                    mesh.vertices[node_id] += dx

                    # ── 再次检查 FEP 穿透 ──
                    if (
                        not mesh.is_top_fixed[node_id]
                        and mesh.is_bottom_surface[node_id]
                    ):
                        if mesh.vertices[node_id, 2] < config.z_fep:
                            mesh.vertices[node_id, 2] = config.z_fep

                    # 跟踪本次迭代的最大位移
                    max_dx = max(max_dx, length)

            # ── Chebyshev 半隐式加速（迭代 > 5 后启用） ──
            #   基于 Chebyshev 多项式的外推，加速收敛。
            #   跳过损伤节点（DAMAGING），避免其不稳定行为被放大。
            if iteration > 5:
                omega = self._chebyshev_omega(iteration, config.rho_cheb)
                free_mask = (
                    mesh.active_vertex_mask
                    & ~fixed
                    & (mesh.czm_state != CZMState.DAMAGING)
                )
                mesh.vertices[free_mask] += omega * (
                    mesh.vertices[free_mask] - x_old_iter[free_mask]
                )

            # ── 收敛判定 ──
            if max_dx < target_epsilon:
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter >= N_stable:
                break

            # ── 迭代回调：用于 GUI 事件泵 / 进度更新 ──
            if on_iteration is not None:
                on_iteration(iteration, max_dx)

        # ── 收敛后处理：更新速度和上一帧坐标 ──
        free = mesh.active_vertex_mask & ~fixed
        mesh.velocities[free] = (
            mesh.vertices[free] - x_prev[free]
        ) / max(config.dt, 1e-12)
        mesh.velocities[fixed] = 0.0
        mesh.prev_vertices = x_prev

        # ── 检查底部脱膜状态 ──
        free_bottom = mesh.bottom_nodes(layer_id)
        all_free = bool(
            len(free_bottom) == 0
            or np.all(mesh.czm_state[free_bottom] == CZMState.FREE)
        )

        # ── 计算动能 ──
        kinetic = float(
            0.5 * np.sum(
                masses[free] * np.sum(mesh.velocities[free] ** 2, axis=1)
            )
        )

        return VBDSolveResult(
            x=mesh.vertices.copy(),
            v=mesh.velocities.copy(),
            iterations=iterations_done,
            max_dx=float(max_dx),
            kinetic_energy=kinetic,
            stable_steps=int(stable_counter),
            all_free=all_free,
            chebyshev_skipped_damaging=damaging_count,
        )

    # ───────────────────────────────────────────────────────────────────────
    # 带平台提升的求解（打印工艺专用）
    # ───────────────────────────────────────────────────────────────────────

    def solve_with_lift(
        self,
        mesh: MeshState,
        layer_id: int,
        e_z: float,
        lifting_top: np.ndarray,
        on_iteration: Callable[[int, float], None] | None = None,
    ) -> VBDSolveResult:
        """单步"提升 + 静平衡"求解（供 Python 控制循环调用）。

        控制反转（Inversion of Control）架构：
        本函数已降级为"单步求解器"，仅执行**一次**微小提升
        和**一次** VBD 静平衡迭代循环。不含任何 while 循环
        控制提升距离，时间流逝由 Python 端循环接管。

        每一调用步：
        1. 刚性抬升顶部节点 Z 坐标 ``v_lift * dt`` 米
        2. 更新 CZM 内聚力状态
        3. 调用核心 VBD 隐式迭代直到静平衡

        Parameters
        ----------
        mesh : MeshState
            网格状态（将被原地修改）。
        layer_id : int
            当前打印层 ID。
        e_z : float
            z 方向电场强度。
        lifting_top : np.ndarray of int
            平台夹持的顶层节点索引数组。

        Returns
        -------
        VBDSolveResult
        """
        config = self.config
        x_prev = mesh.vertices.copy()
        masses = mesh.masses
        fixed = (
            mesh.is_top_fixed
            | (mesh.czm_state == CZMState.FIXED)
            | ~mesh.active_vertex_mask
        )

        # ════════════════════ 单步提升 ════════════════════
        v_lift = float(config.v_lift)
        bottom = mesh.bottom_nodes(layer_id)

        # 刚性抬升顶部节点（仅 Z 方向）— 单步 v_lift * dt
        mesh.vertices[lifting_top, 2] += v_lift * config.dt

        # ── 更新 CZM 状态（提离型膜脱粘）──
        from hydrogel_vbd.physics.czm import update_czm_states

        if len(bottom):
            update_czm_states(
                mesh,
                bottom,
                internal_pull_z=np.full(len(bottom), config.T_max * 1.05),
                area=config.node_area,
                t_max=config.T_max,
                k_czm=config.K_czm,
                delta_f=config.delta_f,
                z_fep=config.z_fep,
                dt=config.dt,
            )

        # ════════════════════ 静平衡迭代 ════════════════════
        fixed[:] = (
            mesh.is_top_fixed
            | (mesh.czm_state == CZMState.FIXED)
            | ~mesh.active_vertex_mask
        )

        terms = build_local_physics_terms(mesh, config, e_z=e_z, x_prev=x_prev)
        adaptive_accel = np.zeros_like(mesh.vertices)
        adaptive_accel[mesh.active_vertex_mask] = (
            config.c_init
            * terms.force[mesh.active_vertex_mask]
            / masses[mesh.active_vertex_mask, None]
        )
        y = (
            x_prev
            + config.dt * mesh.velocities
            + (config.dt ** 2) * adaptive_accel
        )

        colors = (
            mesh.colors
            if mesh.colors is not None
            else np.zeros(mesh.vertices.shape[0], dtype=int)
        )

        max_dx = 0.0
        iterations_done = 0
        stable_counter = 0
        damaging_count = int(
            np.sum(
                mesh.active_vertex_mask
                & (mesh.czm_state == CZMState.DAMAGING)
            )
        )

        N_stable = int(config.N_stable)
        target_epsilon = float(config.epsilon)

        for iteration in range(1, config.max_iters + 1):
            iterations_done = iteration
            x_old_iter = mesh.vertices.copy()
            terms = build_local_physics_terms(mesh, config, e_z=e_z, x_prev=x_prev)
            max_dx = 0.0

            for color in sorted(set(int(c) for c in colors)):
                for node_id in np.flatnonzero(colors == color):
                    if fixed[node_id]:
                        continue

                    if (
                        not mesh.is_top_fixed[node_id]
                        and mesh.is_bottom_surface[node_id]
                    ):
                        if mesh.vertices[node_id, 2] < config.z_fep:
                            mesh.vertices[node_id, 2] = config.z_fep

                    h_elastic = terms.hessian[node_id]

                    if mesh.czm_state[node_id] == CZMState.DAMAGING:
                        eigvals = np.linalg.eigvalsh(h_elastic)
                        if np.min(eigvals) < 0:
                            eigvals_psd = np.maximum(eigvals, 0.0)
                            eigvecs = np.linalg.eigh(h_elastic)[1]
                            h_elastic = eigvecs @ np.diag(eigvals_psd) @ eigvecs.T

                    h_total = (
                        (masses[node_id] / (config.dt ** 2)) * np.eye(3)
                        + h_elastic
                        + (config.k_d / max(config.dt, 1e-12)) * h_elastic
                        + 1e-9 * np.eye(3)
                    )
                    f_inertia = (
                        -(masses[node_id] / (config.dt ** 2))
                        * (mesh.vertices[node_id] - y[node_id])
                    )
                    f_damp = (
                        -(config.k_d / max(config.dt, 1e-12))
                        * h_elastic @ (mesh.vertices[node_id] - x_prev[node_id])
                    )
                    f_total = terms.force[node_id] + f_inertia + f_damp
                    dx = np.linalg.solve(h_total, f_total)
                    length = float(np.linalg.norm(dx))
                    if length > 0.01:
                        dx *= 0.01 / length
                        length = 0.01
                    mesh.vertices[node_id] += dx

                    if (
                        not mesh.is_top_fixed[node_id]
                        and mesh.is_bottom_surface[node_id]
                    ):
                        if mesh.vertices[node_id, 2] < config.z_fep:
                            mesh.vertices[node_id, 2] = config.z_fep

                    max_dx = max(max_dx, length)

            if iteration > 5:
                omega = self._chebyshev_omega(iteration, config.rho_cheb)
                free_mask = (
                    mesh.active_vertex_mask
                    & ~fixed
                    & (mesh.czm_state != CZMState.DAMAGING)
                )
                mesh.vertices[free_mask] += omega * (
                    mesh.vertices[free_mask] - x_old_iter[free_mask]
                )

            if max_dx < target_epsilon:
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter >= N_stable:
                break

            if on_iteration is not None:
                on_iteration(iteration, max_dx)

        # ── 后处理 ──
        free = mesh.active_vertex_mask & ~fixed
        mesh.velocities[free] = (
            mesh.vertices[free] - x_prev[free]
        ) / max(config.dt, 1e-12)
        mesh.velocities[fixed] = 0.0
        mesh.prev_vertices = x_prev

        free_bottom = mesh.bottom_nodes(layer_id)
        all_free = bool(
            len(free_bottom) == 0
            or np.all(mesh.czm_state[free_bottom] == CZMState.FREE)
        )
        kinetic = float(
            0.5 * np.sum(
                masses[free] * np.sum(mesh.velocities[free] ** 2, axis=1)
            )
        )
        return VBDSolveResult(
            x=mesh.vertices.copy(),
            v=mesh.velocities.copy(),
            iterations=iterations_done,
            max_dx=float(max_dx),
            kinetic_energy=kinetic,
            stable_steps=int(stable_counter),
            all_free=all_free,
            chebyshev_skipped_damaging=damaging_count,
        )

    # ───────────────────────────────────────────────────────────────────────
    # Chebyshev 加速因子计算
    # ───────────────────────────────────────────────────────────────────────

    @staticmethod
    def _chebyshev_omega(iteration: int, rho_cheb: float) -> float:
        """计算 Chebyshev 半隐式加速因子。

        公式：ω = min(0.5, ρ^k / (1 + ρ^k))

        其中 ρ 为谱半径（``rho_cheb``），k 为当前迭代次数。
        ω 随迭代递减，实现由激进的加速逐渐趋于保守。

        Parameters
        ----------
        iteration : int
            当前迭代次数。
        rho_cheb : float
            Chebyshev 谱半径（0 < ρ < 1，越大越激进）。

        Returns
        -------
        float
            加速因子 ω ∈ (0, 0.5]。
        """
        return float(
            min(0.5, (rho_cheb ** iteration) / (1.0 + rho_cheb ** iteration))
        )
