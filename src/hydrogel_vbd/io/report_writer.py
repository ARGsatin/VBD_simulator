# -*- coding: utf-8 -*-
"""仿真报告生成器 —— 将多层仿真结果汇总为 CSV 指标文件。

本模块将每层仿真完成后产生的 ``LayerResult`` 批量导出为
CSV 格式的表格文件，便于用 Excel、Python pandas 或 MATLAB
进行后续数据分析和统计。

输出格式
--------
CSV 文件包含以下列：
* ``layer_id``：层序号
* ``success``：求解是否收敛（True/False）
* ``max_deformation``：该层最大变形量（m）
* ``rms_error``：该层 RMS 形状误差（m）
* 各层 ``error_metrics`` 中出现的所有自定义指标列

缺失值自动留空（CSV DictWriter 默认行为）。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from hydrogel_vbd.state import LayerResult


def write_metrics_csv(
    path: str | Path, results: Iterable[LayerResult]
) -> Path:
    """将多层仿真结果的误差指标写入 CSV 文件。

    遍历所有层结果，自动发现所有自定义指标名称，
    并输出统一列格式的 CSV 表格。

    Parameters
    ----------
    path : str or Path
        输出 CSV 文件路径。
    results : Iterable[LayerResult]
        各层的仿真结果（可按任意顺序提供，输出按迭代顺序排列）。

    Returns
    -------
    Path
        写入的 CSV 文件路径。
    """
    rows = list(results)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 固定列 + 自动发现的动态指标列
    fixed_names = {"layer_id", "success", "max_deformation", "rms_error"}
    metric_names = sorted(
        {
            name
            for result in rows
            for name in result.error_metrics
            if name not in fixed_names
        }
    )
    fieldnames = [
        "layer_id",
        "success",
        "max_deformation",
        "rms_error",
        *metric_names,
    ]

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in rows:
            row = {
                "layer_id": result.layer_id,
                "success": result.success,
                "max_deformation": result.max_deformation,
                "rms_error": result.rms_error,
            }
            row.update(result.error_metrics)
            writer.writerow(row)
    return output
