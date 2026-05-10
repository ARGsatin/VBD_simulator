# -*- coding: utf-8 -*-
"""G-code 命令注入器 —— 将仿真求出的电场控制命令嵌入到打印机 G-code 中。

本模块提供两个函数，分别处理空间分布电场命令（``FieldCommand``）
和标量 PID 命令（``PIDFieldState``），通过解析包含 ``;LAYER:`` 标记的
G-code 文件，在每层的打印指令后插入对应的电场控制命令。

注入格式
--------
* **FieldCommand 模式**：每层插入多行注释格式的命令：
  ``;E_FIELD: ELECTRODE=e0, VOLTAGE=1.234567, DURATION=0.123456``
  后跟 ``;E_FIELD: OFF`` 关闭电场。
* **PIDFieldState 模式**：每层插入 Marlin 兼容的 M150 命令：
  ``M150 E12.345678`` （设置电场强度，V/m）。

这些 G-code 输出可直接发送到支持电场辅助的 DLP 打印机控制板。
"""

from __future__ import annotations

from hydrogel_vbd.control.field_controller import PIDFieldState
from hydrogel_vbd.state import FieldCommand


def insert_field_commands(
    source_gcode: str, commands_by_layer: dict[int, FieldCommand]
) -> str:
    """将空间分布电场命令注入 G-code。

    遍历源 G-code 的每一行，当遇到 ``;LAYER:<id>`` 标记时，
    在该层打印指令后插入所有电极的电压命令。

    Parameters
    ----------
    source_gcode : str
        原始 G-code 字符串（含 ``;LAYER:`` 层标记）。
    commands_by_layer : dict[int, FieldCommand]
        层 ID 到电场控制命令的映射。

    Returns
    -------
    str
        注入电场命令后的完整 G-code 字符串。
    """
    output_lines: list[str] = []
    for line in source_gcode.splitlines():
        output_lines.append(line)
        if not line.startswith(";LAYER:"):
            continue
        # 解析层号
        layer_id = int(line.split(":", 1)[1].strip())
        command = commands_by_layer.get(layer_id)
        if command is None:
            continue
        # 插入各电极电压命令
        electrode_ids = command.electrode_ids or [
            f"e{i}" for i in range(len(command.voltage))
        ]
        for electrode_id, voltage in zip(
            electrode_ids, command.voltage
        ):
            output_lines.append(
                f";E_FIELD: ELECTRODE={electrode_id},"
                f" VOLTAGE={float(voltage):.6f},"
                f" DURATION={command.duration:.6f}"
            )
        output_lines.append(";E_FIELD: OFF")
    return "\n".join(output_lines) + "\n"


def insert_pid_field_commands(
    source_gcode: str,
    commands_by_layer: dict[int, PIDFieldState],
) -> str:
    """将标量 PID 电场命令注入 G-code。

    遍历源 G-code，当遇到 ``;LAYER:<id>`` 标记时，
    插入 Marlin 兼容的 M150 命令设置均匀场强。

    Parameters
    ----------
    source_gcode : str
        原始 G-code 字符串（含 ``;LAYER:`` 层标记）。
    commands_by_layer : dict[int, PIDFieldState]
        层 ID 到 PID 场状态的映射。

    Returns
    -------
    str
        注入 M150 电场命令后的完整 G-code 字符串。
    """
    output_lines: list[str] = []
    for line in source_gcode.splitlines():
        output_lines.append(line)
        if not line.startswith(";LAYER:"):
            continue
        layer_id = int(line.split(":", 1)[1].strip())
        command = commands_by_layer.get(layer_id)
        if command is not None:
            output_lines.append(f"M150 E{command.E_z:.6f}")
    return "\n".join(output_lines) + "\n"
