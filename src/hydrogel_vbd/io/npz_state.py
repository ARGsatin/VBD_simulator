# -*- coding: utf-8 -*-
"""NPZ 层状态持久化 —— 将每层仿真结果保存为 NumPy 压缩格式或从中恢复。

本模块提供层仿真结果（``LayerResult``）的序列化/反序列化功能。
使用 ``.npz`` 格式存储，这是一种基于 NumPy 的压缩存档格式，
便于后续分析和与 MATLAB 等工具的互操作。

存储内容
--------
每个 NPZ 文件包含该层的一次完整子系统仿真结果：
* ``layer_id``：层索引
* ``x_sim``：仿真后的节点位置 (N, 3)
* ``v_sim``：仿真后的节点速度 (N, 3)
* ``voltage``：该层施加的电极电压向量
* ``max_deformation``：该层最大变形量 (m)
* ``rms_error``：该层 RMS 形状误差 (m)
* ``success``：求解是否收敛
* ``error_metrics``：完整的误差指标字典（通过键值对分别存储）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hydrogel_vbd.core.state import LayerResult


def save_layer_state(path: str | Path, result: LayerResult) -> Path:
    """将层仿真结果保存为 NPZ 文件。

    创建必要的输出目录，将 ``LayerResult`` 的所有字段
    序列化到单个压缩 NumPy 存档中。

    Parameters
    ----------
    path : str or Path
        输出文件路径（含 ``.npz`` 扩展名）。
    result : LayerResult
        待保存的层仿真结果。

    Returns
    -------
    Path
        写入的 NPZ 文件路径。
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = list(result.error_metrics.keys())
    metric_values: list[str] = []
    metric_value_types: list[str] = []
    for value in result.error_metrics.values():
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            metric_values.append(repr(float(value)))
            metric_value_types.append("float")
        else:
            metric_values.append(str(value))
            metric_value_types.append("str")
    np.savez(
        output,
        layer_id=np.array(result.layer_id, dtype=int),
        x_sim=result.x_sim,
        v_sim=result.v_sim,
        voltage=result.field_command_next.voltage,
        max_deformation=np.array(result.max_deformation, dtype=float),
        rms_error=np.array(result.rms_error, dtype=float),
        success=np.array(result.success, dtype=bool),
        metric_keys=np.array(metric_keys),
        metric_values=np.array(metric_values),
        metric_value_types=np.array(metric_value_types),
    )
    return output


def load_layer_state(path: str | Path) -> dict:
    """从 NPZ 文件加载层仿真结果。

    读取之前保存的层状态文件，返回包含所有字段的字典。
    注意：返回的字典不是 ``LayerResult`` 对象，而是其字段的
    扁平化表示，便于灵活使用。

    Parameters
    ----------
    path : str or Path
        输入 NPZ 文件路径。

    Returns
    -------
    dict
        包含以下键的字典：
        - ``"layer_id"`` (int)：层索引
        - ``"x_sim"`` (np.ndarray, shape (N, 3))：节点位置
        - ``"v_sim"`` (np.ndarray, shape (N, 3))：节点速度
        - ``"voltage"`` (np.ndarray)：电极电压向量
        - ``"max_deformation"`` (float)：最大变形量
        - ``"rms_error"`` (float)：RMS 误差
        - ``"success"`` (bool)：求解是否收敛
        - ``"error_metrics"`` (dict[str, float | str])：完整误差指标
    """
    with np.load(Path(path), allow_pickle=False) as data:
        metric_keys = [str(item) for item in data["metric_keys"]]
        raw_values = [str(item) for item in data["metric_values"]]
        if "metric_value_types" in data:
            value_types = [str(item) for item in data["metric_value_types"]]
        else:
            value_types = ["float"] * len(raw_values)
        metric_values = [
            float(value) if value_type == "float" else value
            for value, value_type in zip(raw_values, value_types)
        ]
        return {
            "layer_id": int(data["layer_id"]),
            "x_sim": data["x_sim"].copy(),
            "v_sim": data["v_sim"].copy(),
            "voltage": data["voltage"].copy(),
            "max_deformation": float(data["max_deformation"]),
            "rms_error": float(data["rms_error"]),
            "success": bool(data["success"]),
            "error_metrics": dict(zip(metric_keys, metric_values)),
        }
