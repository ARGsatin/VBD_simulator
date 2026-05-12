# -*- coding: utf-8 -*-
"""图形用户界面 (GUI) —— 基于 PySide6 的水凝胶 DLP VBD 模拟器操作界面。

本模块提供友好的桌面应用程序，允许用户通过可视化面板配置参数、
加载 STL 模型、运行仿真并实时监控进度。

1. **main_window.py**：主窗口及配套控件（参数面板、进度条、日志窗口）

运行方式
--------
.. code-block:: bash

    python run_gui.py

或在 Python 中直接调用：

.. code-block:: python

    from hydrogel_vbd.gui.main_window import launch_gui
    launch_gui()
"""
