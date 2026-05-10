# -*- coding: utf-8 -*-
"""重力模型 —— 作用于所有节点的均匀体积力。

在 DLP 仿真中，重力是最基本的体积力之一。本模块提供
简单的均匀重力场模型，对每个活动节点施加相同的加速度。

物理模型
--------
采用均匀重力场假设：

.. math::
    f_{gravity} = ρ · g

其中 **ρ** 为密度（kg/m³），**g** 为重力加速度向量（m/s²）。

.. note::
   此模块输出的力是一个**体力密度**（N/m³），
   在主循环或局部物理项组装中需乘以节点等效体积才能得到实际力。
   但在当前框架中，密度参数已被直接用作节点力系数。
"""

from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import MeshState


def gravity_force(
    mesh: MeshState,
    density: float,
    g: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> np.ndarray:
    """计算均匀重力场的节点力。

    对所有活动节点施加相同的重力加速度，
    非活动节点受力为零。

    Parameters
    ----------
    mesh : MeshState
        网格状态（用于确定顶点数和活动掩码）。
    density : float
        密度系数（在简化模型中直接作为力系数使用）。
    g : tuple[float, float, float], optional
        重力加速度向量（m/s²），默认为 (0, 0, -9.81)。

    Returns
    -------
    np.ndarray, shape (N, 3)
        每个顶点的重力向量（N）。
    """
    forces = np.zeros_like(mesh.vertices)
    direction = np.asarray(g, dtype=float)
    forces[mesh.active_vertex_mask] = density * direction
    return forces
