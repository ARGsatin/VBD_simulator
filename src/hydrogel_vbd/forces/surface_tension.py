# -*- coding: utf-8 -*-
"""表面张力模型 —— 作用于活动节点的均匀表面张力。

在 DLP 打印中，液态树脂的表面张力对固化界面施加附加应力，
特别是在微尺度结构中可能影响成型精度。本模块提供一个简化的
均匀表面张力模型。

物理模型
--------
假设表面张力沿固定方向的均匀分布力：

.. math::
    f_{st} = γ · d

其中 **γ** 为表面张力系数（N），**d** 为单位方向向量
（默认 (0, 0, 1) 即 +Z 方向）。

.. note::
   此模型是简化近似，未考虑曲率相关的 Laplace 压力项。
   对于需要精确描述气液界面的场景，建议将表面张力集成到
   CZM 或附加的压力边界条件中。
"""

from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import MeshState


def surface_tension_force(
    mesh: MeshState,
    coefficient: float,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """计算均匀表面张力节点力。

    对每个活动节点施加大小相同、方向固定的表面张力。

    Parameters
    ----------
    mesh : MeshState
        网格状态（用于确定顶点数和活动掩码）。
    coefficient : float
        表面张力系数（N），作用于每个节点的大小。
    direction : tuple[float, float, float], optional
        表面张力方向向量（将被归一化），默认为 (0, 0, 1)。

    Returns
    -------
    np.ndarray, shape (N, 3)
        每个顶点的表面张力向量（N）。

    Raises
    ------
    ValueError
        若 ``direction`` 是零向量。
    """
    forces = np.zeros_like(mesh.vertices)
    direction_array = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction_array)
    if norm == 0.0:
        raise ValueError("direction must be nonzero")
    forces[mesh.active_vertex_mask] = (
        float(coefficient) * direction_array / norm
    )
    return forces
