# -*- coding: utf-8 -*-
"""局部物理项组装器 —— 综合弹性体、CZM 粘聚力和流体拖曳力的逐节点组装。

本模块是力的**核心计算引擎**，在每个时间步被调用一次。
它将以下物理贡献按节点聚合为统一的力与 Hessian：

1. **重力**：均匀体积力 f = m · g
2. **电场力**：f_z = q_ion · E_z（仅 Z 方向）
3. **Neo-Hookean 超弹性**：通过四面体有限元组装的弹性力与 Hessian
4. **CZM 粘聚区**：基于牵引-分离法则的损伤软化力（仅底面节点）
5. **流体挤压流拖曳**：FREE 状态下节点接近离型膜时的挤压膜阻尼
   （Reynolds 润滑近似，f ∝ v_z / gap³）

组装策略
--------
* 先遍历所有活动四面体组装超弹性贡献（主要计算热点）
* 再遍历所有活动节点，根据 CZM 状态施加界面力或流体拖曳力
* 所有贡献直接累加到 ``force`` 和 ``hessian`` 数组中

.. note::
   ``hessian`` 存储的是**对角块** Hessian（每个节点一个 3×3 矩阵），
   这是 VBD（Vertex-Block Diagonal）求解器的要求。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.physics.czm import CZMState
from hydrogel_vbd.physics.elastic_energy import compute_tet_force_and_hessian_contributions
from hydrogel_vbd.core.state import MeshState


def _poisson_to_lame(mu: float, kappa: float) -> tuple[float, float]:
    """从剪切模量 μ 和体积模量 κ 计算 Lamé 第一参数 λ。

    各向同性线弹性材料需要 Lamé 参数 (λ, μ) 来描述本构关系，
    但输入配置文件通常提供工程常数 (μ, κ)。
    本函数完成从 (μ, κ) 到 (μ, λ) 的转换。

    转换公式（基于 κ = λ + 2μ/3）：
    .. math::
        λ = κ - \\frac{2}{3}μ

    Parameters
    ----------
    mu : float
        剪切模量（Pa），即 Lamé 第二参数。
    kappa : float
        体积模量（Pa）。

    Returns
    -------
    tuple[float, float]
        ``(mu, lam)`` —— 注意第一个元素仍是 μ（不变），
        第二个元素是计算得到的 λ。
    """
    lam = kappa - (2.0 / 3.0) * mu
    return mu, lam


@dataclass
class LocalPhysicsTerms:
    """局部物理项组装结果 —— 单步的力向量与对角块 Hessian。

    本数据结构用于存储 ``build_local_physics_terms()`` 的输出，
    包含当前时间步所有局部物理贡献的累加结果。

    Attributes
    ----------
    force : np.ndarray, shape (N, 3)
        每节点的净力向量（N）。非活动节点保持零。
    hessian : np.ndarray, shape (N, 3, 3)
        每节点的对角块 Hessian 矩阵（N/m）。用于 VBD 求解器
        的局部线性化。非活动节点为单位矩阵（已在外层初始化）。
    """
    force: np.ndarray
    hessian: np.ndarray


def build_local_physics_terms(
    mesh: MeshState,
    config: SimulationConfig,
    e_z: float,
    x_prev: np.ndarray,
) -> LocalPhysicsTerms:
    """组装当前时间步的所有局部物理贡献。

    这是每个时间步的**核心力计算函数**，按如下顺序累加各物理项的
    贡献到节点的力向量和对角块 Hessian：

    1. **体积力**（重力 + 电场力）→ 直接按节点质量 / 电荷累加
    2. **Neo-Hookean 超弹性** → 遍历所有活动四面体，
       调用 ``compute_tet_force_and_hessian_contributions``
       将力与 Hessian 分散到四面体的 4 个节点
    3. **CZM 粘聚区损伤力** → 对底面 DAMAGING 节点施加软化牵引力
    4. **流体挤压流拖曳** → 对底面 FREE 节点施加间隙相关阻尼力

    Parameters
    ----------
    mesh : MeshState
        当前网格状态（顶点、四面体、CZM 状态、掩码等）。
    config : SimulationConfig
        全局仿真参数（材料常数、CZM 参数、流体参数等）。
    e_z : float
        当前 z 方向电场强度（V/m），来自 PID 控制器。
    x_prev : np.ndarray, shape (N, 3)
        上一时间步的顶点坐标（用于计算挤压流速度）。

    Returns
    -------
    LocalPhysicsTerms
        组装完成的力向量和 Hessian 对角块。
    """
    force = np.zeros_like(mesh.vertices)
    hessian = np.zeros((mesh.vertices.shape[0], 3, 3), dtype=float)
    active = mesh.active_vertex_mask
    masses = mesh.masses
    g = np.asarray(config.g, dtype=float)

    # ---- 重力 + 电场力 ----
    force[active] += masses[active, None] * g
    force[active, 2] += config.q_ion * float(e_z)

    # ---- Neo-Hookean 超弹性（经四面体组装）----
    mu_val = float(config.mu)
    kappa_val = float(config.kappa)
    mu_lame, lam_lame = _poisson_to_lame(mu_val, kappa_val)

    tet_active = mesh.active_tet_mask
    dm_inv = mesh.dm_inv
    tet_volumes = mesh.tet_volumes
    # 收缩系数补偿（参考形状可能被 c_shrink 缩放）
    c_shrink = float(config.c_shrink)

    for tet_id in np.flatnonzero(tet_active):
        tet = mesh.tets[tet_id]
        tet_verts = mesh.vertices[tet]  # (4, 3)
        dm_inv_tet = dm_inv[tet_id]     # (3, 3)
        rest_vol = float(tet_volumes[tet_id]) * (c_shrink ** 3)

        try:
            forces_per_vertex, hessian_per_vertex = compute_tet_force_and_hessian_contributions(
                tet_verts,
                dm_inv_tet,
                rest_vol,
                mu_lame,
                lam_lame,
                inverted_penalty=1e8,
            )
        except (np.linalg.LinAlgError, ValueError):
            # 退化单元：回退到线性弹簧
            stiffness = max(mu_lame, 1.0) * 1e-4
            for local_idx, node_id in enumerate(tet):
                if active[node_id]:
                    force[node_id] += -stiffness * (mesh.vertices[node_id] - mesh.ideal_vertices[node_id])
                    hessian[node_id] += stiffness * np.eye(3)
            continue

        for local_idx, node_id in enumerate(tet):
            if active[node_id]:
                force[node_id] += forces_per_vertex[local_idx]
                hessian[node_id] += hessian_per_vertex[local_idx]

    # ---- CZM 粘聚区 + 流体拖曳（逐顶点）----
    for node_id in np.flatnonzero(active):
        state = CZMState(int(mesh.czm_state[node_id]))
        if state == CZMState.DAMAGING:
            gap = max(float(mesh.vertices[node_id, 2] - config.z_fep), 0.0)
            softening = max(0.0, 1.0 - gap / max(config.delta_f, 1e-12))
            traction = (1.0 - float(mesh.damage[node_id])) * config.T_max * softening
            force[node_id, 2] -= traction * config.node_area
            hessian[node_id, 2, 2] += (1.0 - float(mesh.damage[node_id])) * config.T_max * config.node_area / max(config.delta_f, 1e-12)
        elif state == CZMState.FREE:
            gap = max(float(mesh.vertices[node_id, 2] - config.z_fep), config.d_min)
            if gap < config.d_fluid_max and mesh.time_free[node_id] < config.t_fluid_max:
                v_z_imp = float(mesh.vertices[node_id, 2] - x_prev[node_id, 2]) / max(config.dt, 1e-12)
                coeff = config.C_0 * config.eta * (config.fluid_radius**4) / (gap**3)
                force[node_id, 2] -= coeff * v_z_imp
                hessian[node_id, 2, 2] += coeff / max(config.dt, 1e-12)

    return LocalPhysicsTerms(force=force, hessian=hessian)
