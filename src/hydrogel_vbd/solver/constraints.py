# -*- coding: utf-8 -*-
"""求解器约束条件 —— 固定节点判定。

本模块提供轻量级的约束判定函数，用于在求解过程中
将特定节点标记为固定（位移为零），确保边界条件正确施加。

当前支持的约束类型
-------------------
* **Z 平面约束** (`fixed_z_constraints`)：锁定处于某一 Z 坐标的节点
  （如平台夹持平面、离型膜底面等）。

约束数组在 VBD 求解器内部以布尔掩码形式使用，
固定节点在 Newton 迭代中被完全跳过。
"""

from __future__ import annotations

import numpy as np

from hydrogel_vbd.core.state import MeshState


def fixed_z_constraints(
    mesh: MeshState,
    z_value: float,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """返回处于指定 Z 坐标的节点布尔掩码。

    通过容差比较识别所有当前 Z 坐标等于 ``z_value`` 的顶点。
    这些顶点在 VBD 求解器中被视为**固定边界**，
    不参与 Newton 迭代且位移保持为零。

    典型用途
    --------
    * 锁定平台夹持平面（顶层节点 z = platform_z）
    * 固定底面边界（离型膜 FEP 平面 z = z_fep）
    * 约束特定工艺步骤中的边界条件

    Parameters
    ----------
    mesh : MeshState
        当前网格状态，包含 ``vertices`` (N, 3) 坐标。
    z_value : float
        目标 Z 坐标值（被识别为"固定"的平面高度）。
    tolerance : float, optional
        浮点比较的绝对容差。满足
        ``|vertices[i, 2] - z_value| <= tolerance``
        的顶点被标记为固定。默认为 1e-9。

    Returns
    -------
    np.ndarray, shape (N,), dtype bool
        布尔掩码数组，``True`` 表示该顶点应被固定。

    Notes
    -----
    * 使用 ``np.isclose(..., atol=tolerance)`` 进行安全比较。
    * 返回的掩码与 VBD 求解器中的 ``fixed`` 掩码含义一致。
    """
    return np.isclose(mesh.vertices[:, 2], float(z_value), atol=tolerance)
