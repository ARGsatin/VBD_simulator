# -*- coding: utf-8 -*-
"""电压优化求解器 —— Tikhonov 正则化最小二乘求解最优电压。

本模块解决了电场控制中的核心数学问题：
已知力映射矩阵 **M** (num_dofs × num_electrodes) 和目标力向量
**f_desired**，求满足以下正则化最小二乘的电压向量 **V**：

.. math::
    \\min_V \\| M V - f_{desired} \\|_2^2 + \\lambda \\| V \\|_2^2

闭式解为：

.. math::
    V = (M^T M + \\lambda I)^{-1} M^T f_{desired}

其中 **λ** 为正则化系数，用于抑制电压幅值、提高数值稳定性。

.. note::
   当电极数较少时（通常 4-16 个），直接求解小规模线性方程组
   即可满足实时性要求。对于大规模电极阵列，可考虑迭代法。
"""

from __future__ import annotations

import numpy as np


def solve_regularized_voltage(
    mapping: np.ndarray,
    desired_force: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """求解正则化最小二乘电压。

    根据映射矩阵和目标力向量，计算最优电极电压。

    Parameters
    ----------
    mapping : np.ndarray, shape (M, N)
        力映射矩阵，M 为自由度总数（通常是 3 × num_vertices），
        N 为电极数量。
    desired_force : np.ndarray, shape (M,)
        期望的节点力向量（平坦化后的形状误差反馈力）。
    regularization : float
        Tikhonov 正则化系数 λ。值越大，电压幅值越受抑制，
        但可能导致力跟踪精度下降。

    Returns
    -------
    np.ndarray, shape (N,)
        最优电极电压向量（V）。

    Raises
    ------
    ValueError
        若 ``mapping`` 不是 2D 矩阵，或 ``desired_force``
        长度与 ``mapping`` 行数不匹配。
    np.linalg.LinAlgError
        若正则化后的法方程矩阵奇异（通常不会发生，因为 λ > 0）。
    """
    b = np.asarray(mapping, dtype=float)
    f = np.asarray(desired_force, dtype=float)
    if b.ndim != 2:
        raise ValueError("mapping must be a 2D matrix")
    if f.shape != (b.shape[0],):
        raise ValueError("desired_force length must match mapping rows")
    # 构造法方程： (M^T M + λI) V = M^T f
    lhs = b.T @ b + float(regularization) * np.eye(b.shape[1])
    rhs = b.T @ f
    return np.linalg.solve(lhs, rhs)
