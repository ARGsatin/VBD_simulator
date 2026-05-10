# -*- coding: utf-8 -*-
"""输入输出 (IO) 模块 —— 仿真状态持久化、报告生成、VTK 可视化和 G-code 导出。

本模块负责仿真数据的外部交互：

1. **npz_state.py**：层仿真结果的 NPZ 格式保存与加载
2. **vtk_writer.py**：四面体网格和位移场的 VTK 文件输出（用于 ParaView 可视化）
3. **report_writer.py**：仿真统计报告的文本文件生成
4. **gcode_exporter.py**：将仿真结果转换为打印机 G-code 指令
"""
