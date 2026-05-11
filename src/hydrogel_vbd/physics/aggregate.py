# -*- coding: utf-8 -*-
"""力聚合器 —— 将各物理分力合并为统一的 ForceState。

在 DLP 仿真中，每个时间步会计算多种物理力分量：
重力（gravity）、剥离力（peel）、流体拖曳（fluid）、
表面张力（surface）和电场力（electric）。
本模块将所有分量打包为 ``ForceState``，供主循环
统一施加并记录到报告中。

.. note::
   本模块本身不执行任何力的计算，仅做**数据结构聚合**。
   各分量的具体计算在 ``forces/`` 子模块中分别实现。
"""

from __future__ import annotations

import numpy as np

from hydrogel_vbd.core.state import ForceState


def aggregate_forces(
    gravity: np.ndarray,
    peel: np.ndarray,
    fluid: np.ndarray,
    surface: np.ndarray,
    electric: np.ndarray,
) -> ForceState:
    """聚合五种物理力分量。

    将所有分力数组（均为 (N, 3) ndarray）封装进 ``ForceState``
    数据容器，便于主循环统一处理和记录。

    Parameters
    ----------
    gravity : np.ndarray, shape (N, 3)
        重力分量（N/vertex）。
    peel : np.ndarray, shape (N, 3)
        剥离力分量（N/vertex）。
    fluid : np.ndarray, shape (N, 3)
        流体拖曳力分量（N/vertex）。
    surface : np.ndarray, shape (N, 3)
        表面张力分量（N/vertex）。
    electric : np.ndarray, shape (N, 3)
        电场力分量（N/vertex）。

    Returns
    -------
    ForceState
        包含所有分力的结构体，各字段与输入对应。
    """
    return ForceState(
        gravity=gravity,
        peel=peel,
        fluid=fluid,
        surface=surface,
        electric=electric,
    )
