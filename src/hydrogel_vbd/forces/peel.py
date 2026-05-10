# -*- coding: utf-8 -*-
"""剥离力模型 —— 平台抬升过程中打印件承受的均匀剥离压力。

在 DLP 打印中，每层曝光固化后，打印平台向上抬升使已固化部分
与离型膜（FEP）分离。这一分离过程在打印件底部施加剥离力。
本模块提供一个简化的均匀剥离压力模型。

物理模型
--------
假设剥离力为均匀分布的面压力：

.. math::
    f_{peel} = p · A_{node} · n

其中：
- **p** 为剥离压力（Pa，正值指向法向）
- **A_node** 为每个节点的等效承载面积（m²）
- **n** 为单位法向量（默认 (0, 0, 1) 即 +Z 抬升方向）
"""

from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import MeshState


def peel_force(
    mesh: MeshState,
    pressure: float,
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    vertex_area: float = 1.0,
) -> np.ndarray:
    """计算均匀剥离压力产生的节点力。

    对每个活动节点施加相同大小的剥离力，
    方向沿指定法向量，大小 = pressure × vertex_area。

    Parameters
    ----------
    mesh : MeshState
        网格状态（用于确定顶点数和活动掩码）。
    pressure : float
        剥离压力（Pa），正值表示沿法向方向的推力。
    normal : tuple[float, float, float], optional
        剥离力方向向量（将被归一化），默认为 (0, 0, 1)。
    vertex_area : float, optional
        每个节点的等效承载面积（m²），默认 1.0。

    Returns
    -------
    np.ndarray, shape (N, 3)
        每个顶点的剥离力向量（N）。

    Raises
    ------
    ValueError
        若 ``normal`` 是零向量。
    """
    forces = np.zeros_like(mesh.vertices)
    direction = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("normal must be nonzero")
    direction = direction / norm
    forces[mesh.active_vertex_mask] = (
        float(pressure) * float(vertex_area) * direction
    )
    return forces
