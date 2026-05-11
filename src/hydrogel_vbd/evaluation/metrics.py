# -*- coding: utf-8 -*-
"""形状误差范数 —— RMS 误差与最大误差的计算。

本模块提供两种向量场范数，用于测量仿真节点位置
与目标位置之间的偏差：

1. **RMS 误差**：均方根误差，反映整体偏差水平
   .. math::
       \\text{RMS} = \\sqrt{\\frac{1}{N} \\sum_i \\|v_i\\|_2^2}

2. **最大误差**：逐节点 L2 范数的最大值，反映局部最差偏差
   .. math::
       \\text{max} = \\max_i \\|v_i\\|_2

这些范数构成形状误差评估的底层基础。
"""

from __future__ import annotations

import numpy as np


def rms_norm(vectors: np.ndarray) -> float:
    """计算均方根 (RMS) 范数。

    对每个向量的 L2 范数求平方的均值再开方。
    RMS 值越小，整体形状偏差越小。

    Parameters
    ----------
    vectors : np.ndarray, shape (N, d)
        向量集合，每行是一个 d 维向量（通常是误差向量）。

    Returns
    -------
    float
        均方根范数。
    """
    vectors = np.asarray(vectors, dtype=float)
    return float(np.sqrt(np.mean(np.sum(vectors * vectors, axis=1))))


def max_norm(vectors: np.ndarray) -> float:
    """计算最大范数。

    取所有向量 L2 范数中的最大值。
    最大范数反映最严重的局部偏差。

    Parameters
    ----------
    vectors : np.ndarray, shape (N, d)
        向量集合，每行是一个 d 维向量（通常是误差向量）。

    Returns
    -------
    float
        所有向量 L2 范数中的最大值。
    """
    vectors = np.asarray(vectors, dtype=float)
    return float(np.max(np.linalg.norm(vectors, axis=1)))
