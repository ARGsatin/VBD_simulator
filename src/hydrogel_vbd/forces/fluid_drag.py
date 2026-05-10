# -*- coding: utf-8 -*-
"""流体拖曳力 —— 树脂/空气对节点运动的粘性阻尼。

在水凝胶 DLP 打印中，节点（尤其是与液态树脂接触的节点）
在抬升运动时会受到周围流体的粘性阻力。本模块提供一个简单的
线性阻尼模型，阻力与节点速度成正比且方向相反。

物理模型
--------
采用 Stokes 流假设下的线性拖曳近似：

.. math::
    f_{drag} = -c · v

其中 **c** 为阻尼系数（N·s/m），**v** 为节点速度（m/s）。

````
"""

from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import MeshState


def fluid_drag_force(
    mesh: MeshState,
    coefficient: float,
) -> np.ndarray:
    """计算线性流体拖曳力。

    对每个活动节点施加与其速度成正比的阻尼力。
    非活动节点受力为零。

    Parameters
    ----------
    mesh : MeshState
        网格状态，包含 ``velocities`` (N, 3) 速度数组。
    coefficient : float
        线性阻尼系数（N·s/m）。正值产生阻力，
        值越大阻尼越强。

    Returns
    -------
    np.ndarray, shape (N, 3)
        每个顶点的流体拖曳力向量（N）。
    """
    forces = np.zeros_like(mesh.vertices)
    forces[mesh.active_vertex_mask] = (
        -float(coefficient) * mesh.velocities[mesh.active_vertex_mask]
    )
    return forces
