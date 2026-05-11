# -*- coding: utf-8 -*-
"""形状偏差对比 —— 仿真结果与目标形状的逐节点误差分析与指标综合。

本模块是评估流程的入口，调用 ``metrics`` 模块的范数计算函数，
将仿真节点位置与目标位置进行对比，输出一组综合的形状误差指标。

输出指标
--------
* **rms_error**：均方根误差（m），反映整体偏差水平
* **max_error**：最大逐节点偏差（m），反映最差局部精度
* **max_z_sag**：Z 方向最大下沉量（m），对 DLP 打印尤为关键
* **mean_z_error**：Z 方向平均偏差（m），正值表示下沉，负值表示上翘

这些指标被 ``ReportWriter`` 用于生成仿真报告，也被 PID 控制器
用作反馈信号来源（通常使用 ``mean_z_error`` 或 ``max_z_sag``）。
"""

from __future__ import annotations

import numpy as np

from hydrogel_vbd.evaluation.metrics import max_norm, rms_norm


def compare_shapes(
    x_sim: np.ndarray, x_target: np.ndarray
) -> dict[str, float]:
    """对比仿真形状与目标形状，计算综合误差指标。

    对每对节点位置计算误差向量 ``e = target - simulated``，
    然后统计多个范数和方向性指标。

    Parameters
    ----------
    x_sim : np.ndarray, shape (N, 3)
        仿真得到的节点位置（m）。
    x_target : np.ndarray, shape (N, 3)
        目标节点位置（m）。

    Returns
    -------
    dict[str, float]
        包含以下键的指标字典：
        - ``"rms_error"``: 均方根误差（m）
        - ``"max_error"``: 最大逐节点偏差（m）
        - ``"max_z_sag"``: Z 方向最大正偏差（下沉量，m）
        - ``"mean_z_error"``: Z 方向平均偏差（m）

    Raises
    ------
    ValueError
        若 ``x_sim`` 和 ``x_target`` 形状不是 (N, 3) 或不一致。
    """
    simulated = np.asarray(x_sim, dtype=float)
    target = np.asarray(x_target, dtype=float)
    if (
        simulated.shape != target.shape
        or simulated.ndim != 2
        or simulated.shape[1] != 3
    ):
        raise ValueError(
            "x_sim and x_target must both have shape (N, 3)"
        )
    # 误差向量：目标位置减去仿真位置（正值=仿真偏低/下沉）
    error = target - simulated
    return {
        "rms_error": rms_norm(error),
        "max_error": max_norm(error),
        "max_z_sag": float(np.max(error[:, 2])),
        "mean_z_error": float(np.mean(error[:, 2])),
    }
