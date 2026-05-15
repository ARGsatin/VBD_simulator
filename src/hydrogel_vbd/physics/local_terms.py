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
    layer_id: int | None = None,
) -> LocalPhysicsTerms:
    """组装当前时间步的所有局部物理贡献（每次分配新数组）。"""
    force = np.zeros_like(mesh.vertices)
    hessian = np.zeros((mesh.vertices.shape[0], 3, 3), dtype=float)
    _assemble_physics_terms(mesh, config, e_z, x_prev, force, hessian, layer_id)
    return LocalPhysicsTerms(force=force, hessian=hessian)


def update_local_physics_terms(
    mesh: MeshState,
    config: SimulationConfig,
    e_z: float,
    x_prev: np.ndarray,
    out_force: np.ndarray,
    out_hessian: np.ndarray,
    layer_id: int | None = None,
) -> None:
    """组装物理贡献到预分配的输出数组中（零新分配）。

    与 ``build_local_physics_terms`` 功能相同，但复用已分配的数组，
    避免每迭代一次就分配一次大数组。

    Parameters
    ----------
    mesh : MeshState
    config : SimulationConfig
    e_z : float
    x_prev : np.ndarray
    out_force : np.ndarray, shape (N, 3)
        预分配的力数组，原地修改。
    out_hessian : np.ndarray, shape (N, 3, 3)
        预分配的 Hessian 数组，原地修改。
    """
    out_force.fill(0.0)
    out_hessian.fill(0.0)
    _assemble_physics_terms(
        mesh, config, e_z, x_prev, out_force, out_hessian, layer_id
    )


def _assemble_physics_terms(
    mesh: MeshState,
    config: SimulationConfig,
    e_z: float,
    x_prev: np.ndarray,
    force: np.ndarray,
    hessian: np.ndarray,
    layer_id: int | None = None,
) -> None:
    """组装当前时间步的所有局部物理贡献到预分配的数组中。

    这是每个时间步的**核心力计算函数**。
    """
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
    if not bool(getattr(config, "enable_czm", True)):
        return LocalPhysicsTerms(force=force, hessian=hessian)

    czm_nodes = active
    if layer_id is not None:
        bottom_nodes = mesh.bottom_nodes(int(layer_id))
        has_layer_interfaces = np.any(mesh.is_top_surface_of_layer >= 0)
        if bottom_nodes.size or has_layer_interfaces:
            czm_nodes = np.zeros_like(active, dtype=bool)
            czm_nodes[bottom_nodes] = True
            czm_nodes &= active

    for node_id in np.flatnonzero(czm_nodes):
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
                coeff = min(
                    config.C_0 * config.eta * (config.fluid_radius**4) / (gap**3),
                    1e12  # 防止 squeeze-film 1/gap³ 奇点溢出
                )
                force[node_id, 2] -= coeff * v_z_imp
                hessian[node_id, 2, 2] += coeff / max(config.dt, 1e-12)
