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
from hydrogel_vbd.physics.elastic_energy import (
    compute_tet_deformation_gradient,
    neo_hookean_energy_density,
)
from hydrogel_vbd.physics.local_terms import build_local_physics_terms
from hydrogel_vbd.core.state import MeshState


def _poisson_to_lame(mu: float, kappa: float) -> tuple[float, float]:
    """Lamé 参数转换."""
    lam = kappa - (2.0 / 3.0) * mu
    return mu, lam


def _make_psd(hessian: np.ndarray) -> np.ndarray:
    """Project a 3x3 Hessian to the positive semidefinite cone."""
    eigvals, eigvecs = np.linalg.eigh(hessian)
    if float(np.min(eigvals)) >= 0.0:
        return hessian
    return eigvecs @ np.diag(np.maximum(eigvals, 0.0)) @ eigvecs.T


def _normal_pull_from_terms(
    terms_force: np.ndarray,
    bottom_nodes: np.ndarray,
) -> np.ndarray:
    """Extract upward normal pull for CZM updates from assembled nodal forces."""
    if len(bottom_nodes) == 0:
        return np.zeros(0, dtype=float)
    force_z = np.asarray(terms_force[bottom_nodes, 2], dtype=float)
    return np.maximum(force_z, 0.0)


def _current_czm_masks(mesh: MeshState, layer_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return FIXED/DAMAGING CZM masks limited to the current layer bottom."""
    current_bottom = _current_bottom_mask(mesh, layer_id)
    fixed = current_bottom & (mesh.czm_state == CZMState.FIXED)
    damaging = current_bottom & (mesh.czm_state == CZMState.DAMAGING)
    return fixed, damaging


def _current_bottom_mask(mesh: MeshState, layer_id: int) -> np.ndarray:
    """Return the nodes that can contact the FEP for the current layer only."""
    current_bottom = np.zeros(mesh.vertices.shape[0], dtype=bool)
    bottom_nodes = mesh.bottom_nodes(layer_id)
    if bottom_nodes.size:
        current_bottom[bottom_nodes] = True
    elif not np.any(mesh.is_top_surface_of_layer >= 0):
        current_bottom = np.asarray(mesh.is_bottom_surface, dtype=bool).copy()
    current_bottom &= mesh.active_vertex_mask
    return current_bottom


def _fep_floor_for_nodes(
    mesh: MeshState,
    config: SimulationConfig,
    layer_id: int,
    nodes: np.ndarray,
    current_bottom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return which nodes have a FEP collision floor and the floor z value."""
    node_ids = np.asarray(nodes, dtype=int)
    applies = np.asarray(mesh.active_vertex_mask[node_ids], dtype=bool).copy()
    base_z_fep = float(config.z_fep) - int(layer_id) * float(config.layer_thickness)
    node_layers = np.asarray(mesh.first_active_layer[node_ids], dtype=int)
    surface_layers = np.asarray(mesh.is_top_surface_of_layer[node_ids], dtype=int)
    node_layers = np.where(
        surface_layers >= 0,
        np.maximum(node_layers, surface_layers),
        node_layers,
    ).astype(float)
    floor_z = base_z_fep + node_layers * float(config.layer_thickness)

    current = np.asarray(current_bottom[node_ids], dtype=bool)
    floor_z[current] = float(config.z_fep)

    global_bottom = np.asarray(mesh.is_bottom_surface[node_ids], dtype=bool)
    previous_global_bottom = global_bottom & ~current
    if np.any(previous_global_bottom):
        floor_z[previous_global_bottom] = base_z_fep
        applies[previous_global_bottom] = True

    return applies, floor_z


def _apply_fep_floor_to_mask(
    mesh: MeshState,
    config: SimulationConfig,
    layer_id: int,
    mask: np.ndarray,
    current_bottom: np.ndarray,
) -> None:
    node_ids = np.flatnonzero(np.asarray(mask, dtype=bool) & mesh.active_vertex_mask)
    if len(node_ids) == 0:
        return
    applies, floor_z = _fep_floor_for_nodes(
        mesh, config, layer_id, node_ids, current_bottom
    )
    fix = applies & (mesh.vertices[node_ids, 2] < floor_z)
    mesh.vertices[node_ids[fix], 2] = floor_z[fix]


def _apply_fep_floor(
    mesh: MeshState,
    config: SimulationConfig,
    layer_id: int,
    node_id: int,
    current_bottom: np.ndarray,
) -> None:
    applies, floor_z = _fep_floor_for_nodes(
        mesh, config, layer_id, np.asarray([node_id], dtype=int), current_bottom
    )
    if bool(applies[0]) and mesh.vertices[node_id, 2] < floor_z[0]:
        mesh.vertices[node_id, 2] = floor_z[0]


def _vertex_local_elastic_energy(
    mesh: MeshState,
    node_id: int,
    mu: float,
    lam: float,
) -> float:
    """计算单个顶点关联的所有 tet 的弹性能之和。

    用于 backtracking line search 的能量下降验证。
    """
    energy = 0.0
    for tet_id in mesh.vertex2tets[node_id]:
        if not mesh.active_tet_mask[tet_id]:
            continue
        v = mesh.vertices[mesh.tets[tet_id]]
        dmi = mesh.dm_inv[tet_id]
        F = compute_tet_deformation_gradient(v, dmi)
        energy += mesh.tet_volumes[tet_id] * neo_hookean_energy_density(F, mu, lam)
    return energy


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
        terms = build_local_physics_terms(
            mesh, config, e_z=e_z, x_prev=x_prev, layer_id=layer_id
        )

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
        czm_fixed, czm_damaging = _current_czm_masks(mesh, layer_id)
        fixed = (
            mesh.is_top_fixed
            | czm_fixed
            | ~mesh.active_vertex_mask
        )
        current_bottom = _current_bottom_mask(mesh, layer_id)

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
                czm_damaging
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
            terms = build_local_physics_terms(
                mesh, config, e_z=e_z, x_prev=x_prev, layer_id=layer_id
            )
            max_dx = 0.0

            # ── 按颜色分组遍历 ──
            # 同色顶点互不相邻，可安全批量求解。
            # 批量装配 H_total、合力、3×3 solve（numpy 自动并行），
            # 只保留 line search 逐节点（需要评估局部能量）。
            dt2 = config.dt ** 2
            inv_dt2 = 1.0 / max(dt2, 1e-24)
            damp_factor = config.k_d / max(config.dt, 1e-12)
            mu_lame, lam_lame = _poisson_to_lame(config.mu, config.kappa)
            eye3 = np.eye(3, dtype=np.float64)

            for color in sorted(set(int(c) for c in colors)):
                # ── 提取同色节点并过滤固定节点 ──
                color_nodes = np.flatnonzero(colors == color)
                if len(color_nodes) == 0:
                    continue
                active_mask_c = ~fixed[color_nodes]
                if not np.any(active_mask_c):
                    continue
                active_nodes = color_nodes[active_mask_c]            # (K,)
                K = len(active_nodes)

                # ── FEP 穿透批量修正 ──
                not_top = ~mesh.is_top_fixed[active_nodes]
                fep_applies, fep_floor = _fep_floor_for_nodes(
                    mesh, config, layer_id, active_nodes, current_bottom
                )
                fep_fix = (
                    not_top
                    & fep_applies
                    & (mesh.vertices[active_nodes, 2] < fep_floor)
                )
                mesh.vertices[active_nodes[fep_fix], 2] = fep_floor[fep_fix]

                # ── 批量提取 Hessian ──
                h_elastic_batch = terms.hessian[active_nodes].copy()  # (K, 3, 3)

                # ── DAMAGING 节点 PSD 批量修正 ──
                for idx in range(K):
                    h_elastic_batch[idx] = _make_psd(h_elastic_batch[idx])

                # ── 批量装配 H_total: M/dt²·I + (1 + k_d/dt)·H_elastic + ε·I ──
                mass_batch = masses[active_nodes]                       # (K,)
                h_total_batch = (
                    inv_dt2 * mass_batch[:, None, None] * eye3[None, :, :]
                    + (1.0 + damp_factor) * h_elastic_batch
                    + 1e-9 * eye3[None, :, :]
                )                                                        # (K, 3, 3)

                # ── 批量装配惯性力：-M/dt² · (x - y) ──
                f_inertia_batch = (
                    -inv_dt2 * mass_batch[:, None]
                    * (mesh.vertices[active_nodes] - y[active_nodes])
                )                                                        # (K, 3)

                # ── 批量装配阻尼力：-k_d/dt · H_elastic · (x - x_prev) ──
                dx_prev_batch = (
                    mesh.vertices[active_nodes] - x_prev[active_nodes]
                )                                                        # (K, 3)
                f_damp_batch = -damp_factor * np.einsum(
                    'kij,kj->ki', h_elastic_batch, dx_prev_batch
                )                                                        # (K, 3)

                # ── 合力 + 批量 3×3 solve ──
                f_total_batch = (
                    terms.force[active_nodes]
                    + f_inertia_batch
                    + f_damp_batch
                )                                                        # (K, 3)
                # NumPy 2.0: f_total_batch (K,3) 需 reshape 为 (K,3,1)
                dx_batch = np.linalg.solve(
                    h_total_batch, f_total_batch[..., None]
                ).squeeze(-1)  # (K, 3)

                # ── Per-node line search（能量评估无法批量） ──
                x_saved_batch = mesh.vertices[active_nodes].copy()       # (K, 3)

                for i in range(K):
                    node_id = int(active_nodes[i])
                    dx = dx_batch[i]
                    x_saved = x_saved_batch[i]

                    e_before = _vertex_local_elastic_energy(
                        mesh, node_id, mu_lame, lam_lame
                    )
                    alpha = 1.0
                    accepted = False
                    for _ in range(12):
                        mesh.vertices[node_id] = x_saved + alpha * dx
                        if not mesh.is_top_fixed[node_id]:
                            _apply_fep_floor(
                                mesh, config, layer_id, node_id, current_bottom
                            )

                        e_after = _vertex_local_elastic_energy(
                            mesh, node_id, mu_lame, lam_lame
                        )
                        if e_after <= e_before + 1e-10 or alpha < 1e-6:
                            accepted = True
                            break
                        alpha *= 0.5

                    if not accepted:
                        mesh.vertices[node_id] = x_saved

                    actual_dx = mesh.vertices[node_id] - x_saved
                    length = float(np.linalg.norm(actual_dx))
                    if length > max_dx:
                        max_dx = length

            # ── Chebyshev 半隐式加速（迭代 > 5 后启用） ──
            #   基于 Chebyshev 多项式的外推，加速收敛。
            #   跳过损伤节点（DAMAGING），避免其不稳定行为被放大。
            if iteration > 5:
                omega = self._chebyshev_omega(iteration, config.rho_cheb)
                free_mask = (
                    mesh.active_vertex_mask
                    & ~fixed
                    & ~czm_damaging
                )
                mesh.vertices[free_mask] += omega * (
                    mesh.vertices[free_mask] - x_old_iter[free_mask]
                )
                _apply_fep_floor_to_mask(
                    mesh, config, layer_id, free_mask, current_bottom
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
        czm_fixed, czm_damaging = _current_czm_masks(mesh, layer_id)
        fixed = (
            mesh.is_top_fixed
            | czm_fixed
            | ~mesh.active_vertex_mask
        )
        current_bottom = _current_bottom_mask(mesh, layer_id)

        # ════════════════════ 单步提升 ════════════════════
        v_lift = float(config.v_lift)
        bottom = mesh.bottom_nodes(layer_id)

        # 刚性抬升顶部节点（仅 Z 方向）— 单步 v_lift * dt
        mesh.vertices[lifting_top, 2] += v_lift * config.dt
        terms = build_local_physics_terms(
            mesh, config, e_z=e_z, x_prev=x_prev, layer_id=layer_id
        )

        # ════════════════════ 静平衡迭代 ════════════════════
        czm_fixed, czm_damaging = _current_czm_masks(mesh, layer_id)
        fixed[:] = (
            mesh.is_top_fixed
            | czm_fixed
            | ~mesh.active_vertex_mask
        )

        terms = build_local_physics_terms(
            mesh, config, e_z=e_z, x_prev=x_prev, layer_id=layer_id
        )
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
                czm_damaging
            )
        )

        N_stable = int(config.N_stable)
        target_epsilon = float(config.epsilon)

        for iteration in range(1, config.max_iters + 1):
            iterations_done = iteration
            x_old_iter = mesh.vertices.copy()
            terms = build_local_physics_terms(
                mesh, config, e_z=e_z, x_prev=x_prev, layer_id=layer_id
            )
            max_dx = 0.0

            for color in sorted(set(int(c) for c in colors)):
                for node_id in np.flatnonzero(colors == color):
                    if fixed[node_id]:
                        continue

                    if not mesh.is_top_fixed[node_id]:
                        _apply_fep_floor(
                            mesh, config, layer_id, node_id, current_bottom
                        )

                    h_elastic = _make_psd(terms.hessian[node_id])

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

                    # ── Backtracking line search ──
                    mu_lame, lam_lame = _poisson_to_lame(
                        config.mu, config.kappa
                    )
                    x_saved = mesh.vertices[node_id].copy()
                    e_before = _vertex_local_elastic_energy(
                        mesh, node_id, mu_lame, lam_lame
                    )

                    alpha = 1.0
                    accepted = False
                    for _ in range(12):
                        mesh.vertices[node_id] = x_saved + alpha * dx
                        if not mesh.is_top_fixed[node_id]:
                            _apply_fep_floor(
                                mesh, config, layer_id, node_id, current_bottom
                            )
                        e_after = _vertex_local_elastic_energy(
                            mesh, node_id, mu_lame, lam_lame
                        )
                        if e_after <= e_before + 1e-10 or alpha < 1e-6:
                            accepted = True
                            break
                        alpha *= 0.5

                    if not accepted:
                        mesh.vertices[node_id] = x_saved

                    actual_dx = mesh.vertices[node_id] - x_saved
                    length = float(np.linalg.norm(actual_dx))

                    if not mesh.is_top_fixed[node_id]:
                        _apply_fep_floor(
                            mesh, config, layer_id, node_id, current_bottom
                        )

                    max_dx = max(max_dx, length)

            if iteration > 5:
                omega = self._chebyshev_omega(iteration, config.rho_cheb)
                free_mask = (
                    mesh.active_vertex_mask
                    & ~fixed
                    & ~czm_damaging
                )
                mesh.vertices[free_mask] += omega * (
                    mesh.vertices[free_mask] - x_old_iter[free_mask]
                )
                _apply_fep_floor_to_mask(
                    mesh, config, layer_id, free_mask, current_bottom
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
        if len(free_bottom):
            from hydrogel_vbd.physics.czm import update_czm_states

            terms_after = build_local_physics_terms(
                mesh, config, e_z=e_z, x_prev=x_prev, layer_id=layer_id
            )
            update_czm_states(
                mesh,
                free_bottom,
                internal_pull_z=_normal_pull_from_terms(
                    terms_after.force, free_bottom
                ),
                area=config.node_area,
                t_max=config.T_max,
                k_czm=config.K_czm,
                delta_f=config.delta_f,
                z_fep=config.z_fep,
                dt=config.dt,
            )

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
