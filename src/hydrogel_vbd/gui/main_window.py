# -*- coding: utf-8 -*-
"""主 GUI 窗口 —— 水凝胶 DLP VBD 模拟器。

本模块是 GUI 应用的核心，包含以下组件：

1. **ParameterPanel**：可折叠的参数配置面板，提供 18 个仿真参数的
   可视化调节控件（双精度/整数微调框），实时触发参数变更信号。
2. **ProgressWidget**：仿真实时进度条与当前层数指示。
3. **LogWidget**：只读文本区域的仿真日志输出。
4. **MainWindow**：主窗口，整合上述组件，配套 4 步工作流：

   - **加载 STL 模型**：导入 3D 模型，自动计算层数与分辨率
   - **划分网格**：生成共形四面体网格，在 3D 视图中预览
   - **运行仿真**：执行逐层 VBD 物理求解，实时显示变形
   - **结果汇总**：输出 CSV 报告与 VTK 可视化文件

运行方式
--------
* 直接运行本文件或使用项目根目录的 ``run_gui.py``
* 需要 PySide6：``pip install pyside6``
* 加载 STL 需要 trimesh：``pip install trimesh``
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PySide6 import QtWidgets, QtCore, QtGui
except ImportError:
    raise ImportError(
        "PySide6 未安装。请运行: pip install pyside6"
    )

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.control.field_controller import PIDFieldController
from hydrogel_vbd.physics.czm import update_czm_states
from hydrogel_vbd.geometry.layer_activator import LayerActivator
from hydrogel_vbd.geometry.stl_mesher import create_demo_or_stl, STLMesher
from hydrogel_vbd.io.report_writer import write_metrics_csv
from hydrogel_vbd.io.vtk_writer import write_vtu
from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver
from hydrogel_vbd.core.state import LayerResult, MeshState
from hydrogel_vbd.gui.mesh_viewer import MeshViewer


# 参数元数据列表：定义每个参数在 GUI 中的显示标签、默认值、取值范围
_PARAM_META: list[dict[str, Any]] = [
    {"key": "mu", "label": "剪切模量 μ (Pa)", "default": 5000.0, "min": 500.0, "max": 1e6},
    {"key": "kappa", "label": "体积模量 κ (Pa)", "default": 20000.0, "min": 2000.0, "max": 1e7},
    {"key": "k_d", "label": "阻尼系数 k_d", "default": 0.5, "min": 0.0, "max": 1.0},
    {"key": "c_shrink", "label": "收缩因子 c_shrink", "default": 0.98, "min": 0.8, "max": 1.0},
    {"key": "T_max", "label": "最大附着力 T_max (Pa)", "default": 5000.0, "min": 100.0, "max": 50000.0},
    {"key": "K_czm", "label": "CZM 刚度 K_czm (Pa/m)", "default": 1.0e8, "min": 1e6, "max": 1e10},
    {"key": "delta_f", "label": "CZM 失效位移 δ_f (m)", "default": 1.0e-4, "min": 1e-6, "max": 1e-2},
    {"key": "dt", "label": "时间步长 dt (s)", "default": 0.001, "min": 0.0001, "max": 0.05},
    {"key": "max_iters", "label": "最大迭代次数", "default": 20, "min": 5, "max": 200},
    {"key": "N_stable", "label": "稳定步数判决", "default": 10, "min": 2, "max": 50},
    {"key": "layer_thickness", "label": "层厚 (mm)", "default": 0.05, "min": 0.01, "max": 10000.0},
    {"key": "v_lift", "label": "平台提升速度 (m/s)", "default": 0.001, "min": 0.0, "max": 0.01},
    {"key": "K_p", "label": "PID K_p", "default": 150.0, "min": 0.0, "max": 1000.0},
    {"key": "K_i", "label": "PID K_i", "default": 20.0, "min": 0.0, "max": 200.0},
    {"key": "K_d", "label": "PID K_d", "default": 5.0, "min": 0.0, "max": 100.0},
    {"key": "E_max", "label": "最大电场 E_max (V/m)", "default": 500.0, "min": 10.0, "max": 5000.0},
]


class ParameterPanel(QtWidgets.QGroupBox):
    """可折叠的参数面板。

    根据 ``_PARAM_META`` 自动生成调节控件，值变化时
    通过 ``params_changed`` 信号通知外部。

    Signals
    -------
    params_changed(dict)
        当任意参数值改变时发射，携带所有当前参数键值对。
    """

    params_changed = QtCore.Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("仿真参数", parent)
        self._spin_map: dict[str, QtWidgets.QDoubleSpinBox | QtWidgets.QSpinBox] = {}
        self._build_ui()
        self._params: dict[str, Any] = {}
        self._collect_params()

    def _build_ui(self) -> None:
        """根据参数元数据列表构建表单布局。"""
        layout = QtWidgets.QFormLayout(self)
        for meta in _PARAM_META:
            default = meta["default"]
            if isinstance(default, float):
                sb = QtWidgets.QDoubleSpinBox()
                sb.setRange(meta["min"], meta["max"])
                sb.setDecimals(6)
                sb.setValue(default)
                sb.setSingleStep((meta["max"] - meta["min"]) / 100.0)
            else:
                sb = QtWidgets.QSpinBox()
                sb.setRange(meta["min"], meta["max"])
                sb.setValue(int(default))
            sb.valueChanged.connect(self._collect_params)  # type: ignore[arg-type]
            self._spin_map[meta["key"]] = sb
            layout.addRow(meta["label"], sb)
        self.setLayout(layout)

    def _collect_params(self) -> None:
        """从所有控件中收集当前参数值并发射变更信号。"""
        self._params = {
            key: widget.value() for key, widget in self._spin_map.items()
        }
        self.params_changed.emit(self._params)

    def get_config(self) -> SimulationConfig:
        """获取当前参数对应的 ``SimulationConfig`` 对象。

        Returns
        -------
        SimulationConfig
            包含当前所有参数值的配置对象。
        """
        self._collect_params()
        return SimulationConfig(**self._params)


class ProgressWidget(QtWidgets.QWidget):
    """仿真进度显示条。

    包含一个进度条和一个文本标签，实时显示当前层计算进度。
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 100)
        self._label = QtWidgets.QLabel("就绪")
        layout.addWidget(self._bar)
        layout.addWidget(self._label)
        self.setLayout(layout)

    def set_layer(self, current: int, total: int) -> None:
        """更新进度为 ``current/total`` 的百分比。

        Parameters
        ----------
        current : int
            当前已完成层序号（从 1 开始）。
        total : int
            总层数。
        """
        pct = int(100 * current / max(total, 1))
        self._bar.setValue(pct)
        self._label.setText(f"计算第 {current}/{total} 层 …")
        QtWidgets.QApplication.processEvents()

    def set_done(self) -> None:
        """标记仿真完成，进度条置为 100%。"""
        self._bar.setValue(100)
        self._label.setText("仿真完成 ✓")
        QtWidgets.QApplication.processEvents()


class LogWidget(QtWidgets.QTextEdit):
    """只读日志输出控件。

    用于实时显示仿真过程中的状态信息、错误提示等。
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("仿真日志 …")

    def append_log(self, text: str) -> None:
        """追加一行日志到显示区域。

        Parameters
        ----------
        text : str
            待显示的日志文本。
        """
        self.append(text)
        QtWidgets.QApplication.processEvents()


class MainWindow(QtWidgets.QMainWindow):
    """VBD 模拟器主窗口。

    整合参数面板、进度条、日志输出和仿真引擎，
    提供 4 步交互式仿真流程：

    1. **加载 STL** → 自动计算层数、分辨率
    2. **划分网格** → 生成共形四面体网格并预览
    3. **运行仿真** → 逐层 VBD 求解
    4. **结果汇总** → CSV 报告 + VTK 文件

    Examples
    --------
    >>> from hydrogel_vbd.gui.main_window import launch_gui
    >>> launch_gui()
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hydrogel VBD Simulator")
        self.resize(1100, 750)

        self._config = SimulationConfig()
        self._last_results: list[LayerResult] = []

        # ── 工作流状态 ──
        self._stl_path: Path | None = None
        self._generated_mesh: MeshState | None = None
        self.mesh: Any = None  # 持久化网格引用，供 _run_simulation 安全访问
        self._actual_layers: int = 0
        self._auto_resolution: float = 0.02  # 单位: m (内部计算)
        self._demo_default_resolution_mm: float = 20.0  # Demo 模式默认分辨率 (mm)

        # ── 自定义分辨率控件引用（在 _init_central 中创建）──
        self._chk_custom_res: QtWidgets.QCheckBox | None = None
        self._lbl_auto_res: QtWidgets.QLabel | None = None
        # ── 网格算法选择器 ──
        self._combo_mesh_algo: QtWidgets.QComboBox | None = None

        # ── 后处理动画回放状态 ──
        self.animation_frames: list[dict[str, Any]] = []
        self._anim_tets: np.ndarray | None = None
        self._anim_timer: QtCore.QTimer | None = None
        self.current_frame_idx: int = 0

        self._init_central()
        self._update_button_states()

        # 确保初始配置与参数面板默认值一致
        self._config = self._param_panel.get_config()

    # ========================================================================
    # UI 搭建
    # ========================================================================
    def _init_central(self) -> None:
        """构建完整 UI 布局。"""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # ── 顶部工具栏 ──
        top_bar = QtWidgets.QHBoxLayout()

        self._btn_load = QtWidgets.QPushButton("📂 加载 CAD 模型")
        self._btn_load.clicked.connect(self._on_load_stl)

        self._btn_clear = QtWidgets.QPushButton("✕ 清除模型")
        self._btn_clear.clicked.connect(self._on_clear_stl)

        self._lbl_model = QtWidgets.QLabel("无模型 (使用 Demo 网格)")

        self._btn_mesh = QtWidgets.QPushButton("🔄 划分网格")
        self._btn_mesh.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #2196F3; "
            "color: white; padding: 6px 20px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:disabled { background-color: #BDBDBD; color: #757575; }"
        )
        self._btn_mesh.clicked.connect(self._on_generate_mesh)

        self._btn_run = QtWidgets.QPushButton("▶ 运行仿真")
        self._btn_run.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #4CAF50; "
            "color: white; padding: 6px 20px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #BDBDBD; color: #757575; }"
        )
        self._btn_run.clicked.connect(self._on_run)

        self._btn_stop_anim = QtWidgets.QPushButton("⏹ 停止回放")
        self._btn_stop_anim.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #f44336; "
            "color: white; padding: 6px 20px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #d32f2f; }"
        )
        self._btn_stop_anim.clicked.connect(self._on_stop_animation)
        self._btn_stop_anim.setVisible(False)

        top_bar.addWidget(self._btn_load)
        top_bar.addWidget(self._btn_clear)
        top_bar.addWidget(self._lbl_model, 1)
        top_bar.addWidget(self._btn_mesh)
        top_bar.addWidget(self._btn_run)
        top_bar.addWidget(self._btn_stop_anim)
        root.addLayout(top_bar)

        # ── 中部：参数 + 进度/日志 + 3D 视图 ──
        mid = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # 左侧面板
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        self._param_panel = ParameterPanel()
        self._param_panel.params_changed.connect(self._on_params)

        # ── 层数 / 分辨率信息行 ──
        info_layout = QtWidgets.QHBoxLayout()
        info_layout.addWidget(QtWidgets.QLabel("层厚 (mm):"))
        self._lbl_thickness = QtWidgets.QLabel("0.05")
        self._lbl_thickness.setStyleSheet("font-weight: bold; color: #1976D2;")
        info_layout.addWidget(self._lbl_thickness)
        info_layout.addSpacing(16)

        info_layout.addWidget(QtWidgets.QLabel("层数:"))
        self._lbl_layers = QtWidgets.QLabel("—")
        self._lbl_layers.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self._lbl_layers)
        self._spin_layers = QtWidgets.QSpinBox()
        self._spin_layers.setRange(1, 100)
        self._spin_layers.setValue(3)
        self._spin_layers.setToolTip("Demo 模式层数")
        self._spin_layers.valueChanged.connect(self._on_demo_layers_changed)
        info_layout.addWidget(self._spin_layers)

        info_layout.addSpacing(16)
        info_layout.addWidget(QtWidgets.QLabel("分辨率 (mm):"))
        self._lbl_resolution = QtWidgets.QLabel("—")
        self._lbl_resolution.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self._lbl_resolution)
        self._spin_resolution = QtWidgets.QDoubleSpinBox()
        self._spin_resolution.setRange(0.5, 500.0)
        self._spin_resolution.setDecimals(1)
        self._spin_resolution.setValue(20.0)
        self._spin_resolution.setSingleStep(1.0)
        self._spin_resolution.setSuffix(" mm")
        self._spin_resolution.setToolTip("Demo 模式 XY 网格间距 (mm)")
        self._spin_resolution.valueChanged.connect(self._on_demo_params_changed)
        info_layout.addWidget(self._spin_resolution)

        # ── 自动推荐分辨率提示标签 ──
        self._lbl_auto_res = QtWidgets.QLabel("")
        self._lbl_auto_res.setStyleSheet("color: #888; font-size: 9pt;")
        info_layout.addWidget(self._lbl_auto_res)

        # ── 自定义分辨率复选框 ──
        self._chk_custom_res = QtWidgets.QCheckBox("自定义分辨率")
        self._chk_custom_res.setToolTip(
            "勾选后可手动调节网格分辨率；\n"
            "不勾选则根据模型尺寸自动计算最佳分辨率"
        )
        self._chk_custom_res.toggled.connect(self._on_custom_resolution_toggled)
        info_layout.addWidget(self._chk_custom_res)

        # ── 网格算法选择器 ──
        info_layout.addSpacing(12)
        info_layout.addWidget(QtWidgets.QLabel("网格算法:"))
        self._combo_mesh_algo = QtWidgets.QComboBox()
        self._combo_mesh_algo.addItem("规整分层算法 (OCC 切片)", "layered")
        self._combo_mesh_algo.addItem("标准非结构化算法 (自由四面体)", "standard")
        self._combo_mesh_algo.setToolTip(
            "规整分层算法: 通过 OCC Boolean Fragment 水平切片，"
            "保证四面体不跨层;\n"
            "标准非结构化算法: 跳过切片，直接生成自由四面体网格"
        )
        info_layout.addWidget(self._combo_mesh_algo)
        info_layout.addStretch()

        left_layout.addLayout(info_layout)
        left_layout.addWidget(self._param_panel)
        left_layout.addStretch()

        # 右侧面板
        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        upper_right = QtWidgets.QWidget()
        upper_layout = QtWidgets.QVBoxLayout(upper_right)
        self._progress = ProgressWidget()
        self._log = LogWidget()
        upper_layout.addWidget(self._progress)
        upper_layout.addWidget(self._log)
        right.addWidget(upper_right)

        self._viewer = MeshViewer()
        right.addWidget(self._viewer)
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 5)  # 3D 视图占绝对主体

        mid.addWidget(left)
        mid.addWidget(right)
        mid.setStretchFactor(0, 1)
        mid.setStretchFactor(1, 4)  # 右侧（含 3D 视图）大幅扩展
        root.addWidget(mid)

        # ── 底部状态栏 ──
        self._status = QtWidgets.QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("就绪 — 请加载 CAD 模型 (*.stl *.step) 或使用 Demo 模式")

        # 初始状态：Demo 模式，分辨率自动（不可编辑）
        self._set_demo_mode()
        self._spin_resolution.setEnabled(False)

    # ========================================================================
    # 模式切换
    # ========================================================================
    def _set_stl_mode(self) -> None:
        """切换到 STL 模式：层数自动显示，分辨率可选择自动/手动。"""
        self._spin_layers.setVisible(False)
        self._spin_resolution.setVisible(True)
        self._lbl_layers.setVisible(True)
        self._lbl_resolution.setVisible(False)
        if self._lbl_auto_res is not None:
            self._lbl_auto_res.setVisible(True)
        if self._chk_custom_res is not None:
            self._chk_custom_res.setVisible(True)
        # 初始状态：自动分辨率（复选框未勾选 → 微调框不可编辑）
        self._chk_custom_res.setChecked(False)
        self._spin_resolution.setEnabled(False)

    def _set_demo_mode(self) -> None:
        """切换到 Demo 模式：层数和分辨率可手动调节。"""
        self._spin_layers.setVisible(True)
        self._spin_resolution.setVisible(True)
        self._lbl_layers.setVisible(False)
        self._lbl_resolution.setVisible(False)
        self._lbl_layers.setText("—")
        self._lbl_resolution.setText("—")
        if self._lbl_auto_res is not None:
            self._lbl_auto_res.setVisible(True)
        if self._chk_custom_res is not None:
            self._chk_custom_res.setVisible(True)
        # 初始状态：自动分辨率（复选框未勾选 → 微调框不可编辑）
        self._chk_custom_res.setChecked(False)
        self._spin_resolution.setEnabled(False)
        self._auto_calculate_demo_resolution()

    # ========================================================================
    # 自定义分辨率复选框
    # ========================================================================
    def _on_custom_resolution_toggled(self, checked: bool) -> None:
        """自定义分辨率复选框切换：启用/禁用微调框编辑。"""
        self._spin_resolution.setEnabled(checked)
        if checked:
            self._lbl_auto_res.setText("(手动模式)")
        else:
            self._lbl_auto_res.setText(
                f"(自动推荐: {self._spin_resolution.value():.1f} mm)"
            )
            self._generated_mesh = None
            self.mesh = None
            if self._stl_path is not None:
                self._auto_calculate_mesh_params()
            else:
                self._auto_calculate_demo_resolution()
            self._update_button_states()

    # ========================================================================
    # 按钮状态管理
    # ========================================================================
    def _update_button_states(self) -> None:
        """根据当前状态启用/禁用按钮。"""
        has_mesh = self._generated_mesh is not None
        has_model = self._stl_path is not None

        # "划分网格" — 总是可用（STL 模式用已加载的 STL，Demo 模式自动生成）
        self._btn_mesh.setEnabled(True)

        # "运行仿真" — 必须有已生成的网格
        self._btn_run.setEnabled(has_mesh)
        if not has_mesh:
            self._btn_run.setToolTip("请先点击「划分网格」生成网格")
        else:
            self._btn_run.setToolTip("开始逐层仿真")

    # ========================================================================
    # 槽函数 — 参数
    # ========================================================================
    @property
    def _layer_thickness_m(self) -> float:
        """当前有效的层厚（m），从参数面板读取 mm 值并转换。"""
        return self._config.layer_thickness * 1e-3 if self._config.layer_thickness > 0 else 5e-5

    def _on_params(self, params: dict[str, Any]) -> None:
        """参数面板变更：更新当前配置。"""
        self._config = SimulationConfig(**params)
        # 动态更新层厚显示
        lt_mm = self._config.layer_thickness
        self._lbl_thickness.setText(f"{lt_mm:.2f}")
        # 自动重新计算层数和分辨率（仅在非自定义模式下）
        is_custom = (
            self._chk_custom_res is not None
            and self._chk_custom_res.isChecked()
        )
        if not is_custom:
            if self._stl_path is not None:
                self._auto_calculate_mesh_params()
            else:
                self._auto_calculate_demo_resolution()
        # 层厚变化会影响层数（STL 模式），需要重新计算
        if self._stl_path is not None:
            self._auto_calculate_mesh_params()
        else:
            if not is_custom:
                self._auto_calculate_demo_resolution()
        self._generated_mesh = None
        self.mesh = None
        self._update_button_states()

    @property
    def _resolution(self) -> float:
        """当前有效的网格分辨率（m）。

        从微调框读取 mm 并转为 m。
        """
        return self._spin_resolution.value() * 1e-3  # mm → m

    def _on_demo_layers_changed(self, _value: int) -> None:
        """Demo 模式下手动调整层数时，清除已生成的网格并重新适配分辨率。"""
        self._generated_mesh = None
        self.mesh = None
        self._actual_layers = 0
        is_custom = (
            self._chk_custom_res is not None
            and self._chk_custom_res.isChecked()
        )
        if not is_custom:
            self._auto_calculate_demo_resolution()
        self._update_button_states()

    def _on_demo_params_changed(self, _value: float) -> None:
        """Demo 模式分辨率微调框变更时，清除已生成的网格。"""
        self._generated_mesh = None
        self.mesh = None
        self._actual_layers = 0
        self._update_button_states()

    def _auto_calculate_demo_resolution(self) -> None:
        """根据 Demo 模型（1x1 m 正方体）和当前层数自动计算推荐分辨率。

        总网格点数预算恒定（约 150000），分辨率随层数增加而自动变粗，
        避免层数过多时网格规模爆炸。同时确保 XY 各方向至少 3 个网格点
        以支持 2×2 quad → 四面体。结果以 mm 为单位写入微调框。
        """
        # 如果是自定义分辨率模式，不自动覆盖
        if self._chk_custom_res is not None and self._chk_custom_res.isChecked():
            return

        xy_area = 1.0  # Demo 正方体底面积 1 m²
        n_layers = max(self._spin_layers.value(), 1)
        total_point_budget = 150000  # 总体网格点数预算
        per_layer_points = total_point_budget / n_layers
        resolution_m = max(0.005, min(0.5, float(np.sqrt(xy_area / per_layer_points))))
        # Demo 正方体 1m × 1m，确保至少 3 个网格点
        min_grid_points = 3
        max_r_for_grid = 1.0 / (min_grid_points - 0.5)
        if resolution_m > max_r_for_grid:
            resolution_m = max_r_for_grid
        resolution_mm = resolution_m * 1000.0
        self._demo_default_resolution_mm = resolution_mm
        self._spin_resolution.blockSignals(True)
        self._spin_resolution.setValue(round(resolution_mm, 1))
        self._spin_resolution.blockSignals(False)
        # 更新自动推荐提示
        if self._lbl_auto_res is not None:
            is_custom = self._chk_custom_res.isChecked() if self._chk_custom_res else False
            if not is_custom:
                self._lbl_auto_res.setText(
                    f"(自动推荐: {resolution_mm:.1f} mm)"
                )

    # ========================================================================
    # 槽函数 — STL 加载 / 清除
    # ========================================================================
    def _on_load_stl(self) -> None:
        """加载 STL 模型文件。

        加载后自动计算：总层数（模型高度 / 0.05mm）、
        XY 网格分辨率（根据包围盒面积与 max_points 自适应）。
        """
        default_dir = str(
            Path(__file__).parent.parent.parent.parent
            / "assets" / "test_models"
        )
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 CAD 模型文件",
            default_dir,
            "CAD 模型 (*.stl *.step *.stp);;STEP 模型 (*.step *.stp);;STL 模型 (*.stl);;所有文件 (*)"
        )
        if not path:
            return

        # ── 停止正在播放的动画回放并隐藏停止按钮 ──
        self._stop_animation_timer()
        self.animation_frames.clear()
        self._anim_tets = None
        self.current_frame_idx = 0
        self._btn_stop_anim.setVisible(False)

        self._stl_path = Path(path)
        self._generated_mesh = None
        self.mesh = None
        self._actual_layers = 0

        # ── 加载 trimesh 并自动计算参数 ──
        try:
            self._auto_calculate_mesh_params()
        except Exception as exc:
            self._log.clear()
            self._log.append_log(f"[错误] 无法解析 CAD 模型文件: {exc}")
            self._status.showMessage("模型加载失败 ✗")
            self._stl_path = None
            self._lbl_model.setText("无模型 (使用 Demo 网格)")
            self._set_demo_mode()
            self._update_button_states()
            return

        self._set_stl_mode()
        self._lbl_model.setText(f"模型: {self._stl_path.name}")
        self._log.clear()
        self._log.append_log(f"已加载 CAD 模型: {self._stl_path}")
        self._log.append_log(
            f"  · 模型包围盒: "
            f"X[{self._bbox_x[0]:.3f}~{self._bbox_x[1]:.3f}] "
            f"Y[{self._bbox_y[0]:.3f}~{self._bbox_y[1]:.3f}] "
            f"Z[{self._bbox_z[0]:.3f}~{self._bbox_z[1]:.3f}] m"
        )
        self._log.append_log(
            f"  · 自动计算: {self._actual_layers} 层, "
            f"分辨率 {self._auto_resolution*1000:.2f} mm"
        )
        self._log.append_log("  · 请点击「划分网格」生成四面体网格")
        self._status.showMessage(
            f"已加载 {self._stl_path.name} | "
            f"{self._actual_layers} 层 | "
            f"{self._auto_resolution*1000:.2f} mm 分辨率"
        )

        # ── 立即在 3D 视图中显示 STL 模型原始表面 ──
        self._show_stl_preview()
        self._update_button_states()

    def _auto_calculate_mesh_params(self) -> None:
        """根据模型包围盒自动计算层数和分辨率。

        优先使用 Gmsh OCC 读取 STEP/B-Rep 格式的精确包围盒；
        STL 格式回退到 trimesh 加载。坐标单位 mm → m 内部转换。
        """
        import os as _os
        ext = _os.path.splitext(str(self._stl_path))[1].lower()
        is_step = ext in ('.step', '.stp', '.igs', '.iges', '.brep')

        if is_step:
            # ── STEP / B-Rep 格式：通过 Gmsh OCC 获取包围盒 ──
            self._stl_trimesh = None  # STEP 无法直接构造 trimesh 对象
            x_min, x_max, y_min, y_max, z_min, z_max = self._get_bounds_via_gmsh()
        else:
            # ── STL 格式：通过 trimesh 加载 ──
            import trimesh

            loaded = trimesh.load(str(self._stl_path))
            if isinstance(loaded, trimesh.Scene):
                geometries = list(loaded.geometry.values())
                if not geometries:
                    raise ValueError("STL 场景中无几何体")
                loaded = trimesh.util.concatenate(geometries)
            if not isinstance(loaded, trimesh.Trimesh):
                raise TypeError(f"不支持的几何类型: {type(loaded).__name__}")

            # STL 坐标从 mm 转换为 m
            loaded.vertices *= 0.001
            self._stl_trimesh = loaded

            bounds = loaded.bounds
            x_min, x_max = float(bounds[0, 0]), float(bounds[1, 0])
            y_min, y_max = float(bounds[0, 1]), float(bounds[1, 1])
            z_min, z_max = float(bounds[0, 2]), float(bounds[1, 2])

        model_height = z_max - z_min
        lt_m = self._layer_thickness_m
        self._actual_layers = max(1, int(np.ceil(model_height / lt_m)))

        # 分辨率自适应：根据 XY 包围盒面积，目标单层 max_points=50000
        xy_area = (x_max - x_min) * (y_max - y_min)
        model_w = x_max - x_min
        model_h = y_max - y_min
        max_points = 50000
        auto_res = max(0.005, float(np.sqrt(xy_area / max_points)))
        # 确保 XY 各方向至少 3 个网格点（以支持 2×2 quad → 四面体）
        min_grid_points = 3
        max_r_for_grid = min(model_w, model_h) / max(min_grid_points - 0.5, 0.1)
        if auto_res > max_r_for_grid:
            auto_res = max_r_for_grid
        self._auto_resolution = auto_res

        self._bbox_x = (x_min, x_max)
        self._bbox_y = (y_min, y_max)
        self._bbox_z = (z_min, z_max)

        # 更新显示和微调框默认值（分辨率仅在自动模式下更新）
        self._lbl_layers.setText(str(self._actual_layers))
        is_custom = (
            self._chk_custom_res is not None
            and self._chk_custom_res.isChecked()
        )
        if not is_custom:
            resolution_mm = auto_res * 1000.0
            self._spin_resolution.blockSignals(True)
            self._spin_resolution.setValue(round(resolution_mm, 1))
            self._spin_resolution.blockSignals(False)
            if self._lbl_auto_res is not None:
                self._lbl_auto_res.setText(
                    f"(自动推荐: {resolution_mm:.1f} mm)"
                )

    def _get_bounds_via_gmsh(self) -> tuple[float, float, float, float, float, float]:
        """通过 Gmsh OCC 导入 STEP/B-Rep 模型并返回包围盒 (m)。

        内部流程：
        1. ``gmsh.initialize()`` 启动 Gmsh
        2. ``gmsh.model.occ.importShapes()`` 导入 STEP 实体
        3. ``gmsh.model.occ.synchronize()`` 同步几何
        4. ``gmsh.model.occ.getBoundingBox()`` 获取包围盒 (mm)
        5. ``gmsh.finalize()`` 清理

        Returns
        -------
        tuple[float, float, float, float, float, float]
            (x_min, x_max, y_min, y_max, z_min, z_max) 单位 m。
        """
        try:
            import gmsh as _gmsh
        except ImportError:
            raise ImportError(
                "STEP 模型需要 Gmsh (OCC 几何内核)。请运行: pip install gmsh"
            )

        _gmsh.initialize()
        try:
            _gmsh.option.setNumber("General.Terminal", 0)
            _gmsh.model.add("_bounds_probe")
            imported = _gmsh.model.occ.importShapes(str(self._stl_path))
            if not imported:
                raise RuntimeError(
                    f"Gmsh OCC 无法导入模型: {self._stl_path}"
                )
            _gmsh.model.occ.synchronize()

            bbox = _gmsh.model.occ.getBoundingBox(dim=-1, tag=-1)
            if not bbox or len(bbox) < 6:
                raise RuntimeError("无法获取 STEP 模型包围盒")

            # Gmsh 返回 mm 单位，转为 m
            x_min_m = float(bbox[0]) * 0.001
            y_min_m = float(bbox[1]) * 0.001
            z_min_m = float(bbox[2]) * 0.001
            x_max_m = float(bbox[3]) * 0.001
            y_max_m = float(bbox[4]) * 0.001
            z_max_m = float(bbox[5]) * 0.001
            return x_min_m, x_max_m, y_min_m, y_max_m, z_min_m, z_max_m
        finally:
            try:
                if _gmsh.isInitialized():
                    _gmsh.finalize()
            except Exception:
                pass

    def _show_stl_preview(self) -> None:
        """在 3D 视图中展示模型的原始表面。

        对 STL 格式使用 trimesh 三角面预览；
        对 STEP 格式（无可直接渲染的三角面）仅显示包围盒信息。
        """
        import os as _os
        try:
            stl = self._stl_trimesh
            if stl is not None:
                # STL 格式：渲染三角面
                self._viewer.show_stl_surface(
                    stl.vertices,
                    stl.faces,
                    title=f"STL 模型 — {self._stl_path.name}",
                )
            else:
                # STEP 格式：无可渲染三角面，显示包围盒线框
                ext = _os.path.splitext(str(self._stl_path))[1].lower() if self._stl_path else ""
                is_step = ext in ('.step', '.stp', '.igs', '.iges', '.brep')
                if is_step and hasattr(self, '_bbox_x'):
                    self._viewer.show_bounding_box(
                        self._bbox_x,
                        self._bbox_y,
                        self._bbox_z,
                        title=f"STEP 模型包围盒 — {self._stl_path.name}",
                    )
        except Exception as exc:
            self._log.append_log(f"  (预览失败: {exc})")

    def _on_clear_stl(self) -> None:
        """清除 STL 模型，恢复到 Demo 模式。"""
        # ── 停止正在播放的动画回放并恢复 3D 视图 ──
        self._stop_animation_timer()
        self.animation_frames.clear()
        self._anim_tets = None
        self.current_frame_idx = 0
        self._btn_stop_anim.setVisible(False)

        self._stl_path = None
        self._generated_mesh = None
        self.mesh = None
        self._actual_layers = 0
        self._lbl_model.setText("无模型 (使用 Demo 网格)")
        self._set_demo_mode()
        self._viewer.clear()  # 清空 3D 视图（恢复到空白状态）
        self._log.clear()
        self._log.append_log("已清除 CAD 模型，将使用 Demo 网格")
        self._status.showMessage("就绪 — Demo 模式")
        self._update_button_states()

    # ========================================================================
    # 槽函数 — 划分网格
    # ========================================================================
    def _on_generate_mesh(self) -> None:
        """生成共形四面体网格（不运行仿真）。"""
        self._log.clear()
        self._log.append_log("===== 划分网格 =====")

        config = self._param_panel.get_config()
        lt_m = self._layer_thickness_m
        lt_mm = self._config.layer_thickness
        config.layer_thickness = lt_m

        try:
            if self._stl_path is not None:
                # ── CAD 模型模式（STEP / STL）──
                import os as _os
                ext = _os.path.splitext(str(self._stl_path))[1].lower()
                is_step = ext in ('.step', '.stp', '.igs', '.iges', '.brep')
                fmt_label = "STEP (B-Rep)" if is_step else "STL"

                self._log.append_log(f"模型: {self._stl_path.name} ({fmt_label})")
                user_res_mm = self._spin_resolution.value()
                self._log.append_log(
                    f"层厚: {lt_mm:.2f} mm | "
                    f"层数: {self._actual_layers} | "
                    f"分辨率: {user_res_mm:.1f} mm"
                )
                self._status.showMessage("正在划分网格（OCC Boolean Fragment）…")

                # 显式使用 OCCFragmentMesher：STEP B-Rep 格式通过
                # gmsh.model.occ.importShapes + fragment 水平切割平面
                # 生成绝对平齐的分层 Delaunay 四面体网格
                from hydrogel_vbd.geometry.stl_mesher import OCCFragmentMesher

                algo_type = self._combo_mesh_algo.currentData() if self._combo_mesh_algo else "layered"
                mesher = OCCFragmentMesher(
                    stl_path=str(self._stl_path),
                    layer_thickness=lt_m,
                    resolution=user_res_mm * 1e-3,
                )
                mesh, actual_layers = mesher.build_layered_mesh(config, algo_type=algo_type)
                self._actual_layers = actual_layers
                self._lbl_layers.setText(str(actual_layers))
                self._log.append_log(
                    f"✓ {fmt_label} 网格生成完成: {actual_layers} 层, "
                    f"{len(mesh.vertices)} 顶点, {len(mesh.tets)} 四面体"
                )
                if not is_step:
                    self._log.append_log(
                        "  ⚠ 提示: STL 为离散三角面片格式，OCC 布尔运算可能不稳定。"
                        " 建议使用 .step/.stp B-Rep 格式获得最佳分层质量。"
                    )
            else:
                # ── Demo 模式 ──
                demo_layers = self._spin_layers.value()
                self._log.append_log(f"模型: Demo 正方体")
                self._log.append_log(
                    f"层厚: {lt_mm:.2f} mm | "
                    f"层数: {demo_layers}"
                )
                self._status.showMessage("正在生成 Demo 网格 …")

                mesh, actual_layers = create_demo_or_stl(
                    stl_path=None,
                    layers=demo_layers,
                    layer_thickness=lt_m,
                    resolution=self._resolution,
                    config=config,
                )
                self._actual_layers = actual_layers
                self._log.append_log(
                    f"✓ Demo 网格生成完成: {actual_layers} 层, "
                    f"{len(mesh.vertices)} 顶点, {len(mesh.tets)} 四面体"
                )

            # 存储网格
            self._generated_mesh = mesh
            # 持久化挂载到 self.mesh —— 供 _run_simulation 安全访问
            # (将关键数据做 .copy() 防止被意外回收)
            self.mesh = mesh
            self.mesh.ideal_vertices = mesh.ideal_vertices.copy()
            # elements 是四面体索引数组 (对应 MeshState.tets)
            self.mesh.elements = mesh.tets.copy()

            # 在 3D 视图中展示初始网格
            self._viewer.show_initial_mesh(mesh.vertices, mesh.tets)
            self._log.append_log("  · 3D 视图已更新为初始网格")
            self._log.append_log("  · 请点击「运行仿真」开始逐层计算")

            self._status.showMessage(
                f"网格就绪 ✓ | {self._actual_layers} 层 | "
                f"{len(mesh.vertices)} 顶点 | {len(mesh.tets)} 四面体"
            )

        except Exception as exc:
            # ── 即使网格生成失败，也隐藏停止回放按钮（无有效网格可播放）──
            self._btn_stop_anim.setVisible(False)

            import traceback
            self._log.append_log(f"\n[错误] 网格生成失败: {exc}")
            self._log.append_log(traceback.format_exc())
            self._status.showMessage("网格划分失败 ✗")
            self._generated_mesh = None
            self.mesh = None

        self._update_button_states()

    # ========================================================================
    # 槽函数 — 运行仿真
    # ========================================================================
    def _on_run(self) -> None:
        """运行仿真按钮槽函数。"""
        # ── 防御性编程：检查 self.mesh 数据完整性 ──
        if self._generated_mesh is None or self.mesh is None:
            self._log.append_log(
                "<span style='color:orange;font-weight:bold;'>"
                "[错误] 网格数据为空，请先点击「划分网格」</span>"
            )
            QtWidgets.QMessageBox.warning(
                self, "网格数据缺失",
                "错误：网格数据为空，请先点击【划分网格】生成网格后再运行仿真。"
            )
            return
        if self.mesh.ideal_vertices is None or self.mesh.elements is None:
            self._log.append_log(
                "<span style='color:orange;font-weight:bold;'>"
                "[错误] 网格关键数据 (ideal_vertices / elements) 缺失，请重新点击「划分网格」</span>"
            )
            QtWidgets.QMessageBox.warning(
                self, "网格数据不完整",
                "错误：网格的关键数据（理想顶点坐标 / 四面体索引）缺失，\n"
                "请重新点击【划分网格】生成完整网格后再运行仿真。"
            )
            return

        # ── 深拷贝关键数据，防止 get_config() 触发的 _on_params 清空 self.mesh ──
        saved_ideal_vertices = self.mesh.ideal_vertices.copy()
        saved_elements = self.mesh.elements.copy()
        saved_generated_mesh = self._generated_mesh
        saved_actual_layers = self._actual_layers

        self._log.clear()
        self._log.append_log("===== 开始仿真 =====")
        if self._stl_path:
            self._log.append_log(f"模型: {self._stl_path.name}")
        else:
            self._log.append_log("模型: Demo (正方体网格)")
        self._log.append_log(
            f"层数: {saved_actual_layers} | "
            f"层厚: {self._config.layer_thickness:.2f} mm | "
            f"顶点: {len(saved_generated_mesh.vertices)} | "
            f"四面体: {len(saved_generated_mesh.tets)}"
        )
        # ── 停止之前可能还在运行的动画回放定时器 ──
        self._stop_animation_timer()
        self.animation_frames.clear()
        self._anim_tets = None
        self.current_frame_idx = 0

        self._status.showMessage("仿真运行中 …")

        # get_config() 会触发 _on_params → 清空 self._generated_mesh / self.mesh
        config = self._param_panel.get_config()

        # ── 恢复网格数据到 self.mesh，供后续 _run_simulation 使用 ──
        self._generated_mesh = saved_generated_mesh
        self.mesh = saved_generated_mesh
        self.mesh.ideal_vertices = saved_ideal_vertices
        self.mesh.elements = saved_elements
        self._actual_layers = saved_actual_layers

        try:
            results = self._run_simulation(config)
            self._last_results = results
            self._on_finish(results)
        except Exception as exc:
            import traceback
            self._log.append_log(f"\n[错误] {exc}")
            self._log.append_log(traceback.format_exc())
            self._status.showMessage("仿真失败 ✗")

    # ========================================================================
    # 仿真核心
    # ========================================================================
    def _run_simulation(self, config: SimulationConfig) -> list[LayerResult]:
        """执行完整的逐层仿真流程。

        使用已生成的 ``_generated_mesh``，逐层激活、施加 CZM 剥离力、
        VBD 求解，PID 控制器调节电场强度，输出 VTK 可视化文件。

        Parameters
        ----------
        config : SimulationConfig
            仿真参数配置。

        Returns
        -------
        list[LayerResult]
            各层的仿真结果列表。
        """
        output_dir = Path("outputs/gui")
        vtk_dir = output_dir / "vtk"
        reports_dir = output_dir / "reports"
        for d in (vtk_dir, reports_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ── 防御性编程：再次校验 mesh 数据完整性（二次保险）──
        if self.mesh is None or self.mesh.ideal_vertices is None or self.mesh.elements is None:
            # 不弹窗（_on_run 已做过 UI 拦截），仅在日志中记录并安全退出
            self._log.append_log(
                "<span style='color:orange;font-weight:bold;'>"
                "[错误] _run_simulation 中检测到网格数据丢失，仿真中断</span>"
            )
            return []

        # 使用 self.mesh 访问持久化数据（self.mesh 即为 self._generated_mesh 的引用）
        mesh = self.mesh
        target_vertices = mesh.ideal_vertices.copy()
        solver = PythonReferenceVBDSolver(config)
        activator = LayerActivator()
        controller = PIDFieldController(config)

        results: list[LayerResult] = []
        n_layers = self._actual_layers

        # ── 帧数据缓存：初始化动画帧列表 ──
        self.animation_frames.clear()
        self._anim_tets = mesh.tets.copy()

        # ── 渲染降频与异步刷新 ─────────────────────────────────
        #   render_interval:  物理引擎跑 N 步，界面才刷新 1 次
        #   displacement_threshold:  形变积累超过此阈值时强制刷新
        #   last_render_displacement:  上次刷新时的形变量 (m)
        step_counter = 0
        render_interval = 50
        displacement_threshold = 0.1e-3  # 0.1 mm
        last_render_displacement = 0.0

        def _on_physics_iteration(
            iteration: int, max_dx: float
        ) -> None:
            """物理求解器每次 VBD 迭代后的回调。

            用于降频泵送 Qt 事件循环并条件性刷新 3D 视图，
            防止长时间密集计算导致 GUI "未响应"。
            """
            nonlocal step_counter, last_render_displacement

            step_counter += 1

            # 取出当前顶点坐标（mesh.vertices 已被求解器原地更新）
            x_current = mesh.vertices.copy()
            current_displacement = float(
                np.max(
                    np.linalg.norm(
                        x_current - mesh.ideal_vertices, axis=1
                    )
                )
            )

            # 条件一：步数计数器取模命中 (降频)
            # 条件二：形变累积超过阈值 (关键帧强制刷新)
            should_render = (
                (step_counter % render_interval == 0)
                or (
                    current_displacement - last_render_displacement
                    >= displacement_threshold
                )
            )

            if should_render:
                self._viewer.show_deformed_mesh(
                    x_current,
                    mesh.tets,
                    mesh.active_vertex_mask,
                    title=(
                        f"第 {layer_id + 1}/{n_layers} 层"
                        f" — 迭代 {iteration}"
                    ),
                )
                QtWidgets.QApplication.processEvents()
                last_render_displacement = current_displacement
            else:
                # 即使不渲染，也定期泵送事件循环以避免"未响应"
                if step_counter % 20 == 0:
                    QtWidgets.QApplication.processEvents()
        # ────────────────────────────────────────────────────────

        for layer_id in range(n_layers):
            self._progress.set_layer(layer_id + 1, n_layers)
            self._log.append_log(
                f"--- 第 {layer_id + 1}/{n_layers} 层"
                f" (E_z={controller.E_z:.3f}) ---"
            )

            # 激活当前层节点
            activator.activate_with_inheritance(
                mesh, layer_id, z_fep=config.z_fep
            )
            bottom = mesh.bottom_nodes(layer_id)

            # 更新 CZM 状态
            update_czm_states(
                mesh,
                bottom,
                internal_pull_z=np.full(len(bottom), config.T_max * 1.05),
                area=config.node_area,
                t_max=config.T_max,
                k_czm=config.K_czm,
                delta_f=config.delta_f,
                z_fep=config.z_fep,
                dt=config.dt,
            )

            # 平台运动学 + VBD 求解（传入回调以支持渲染降频）
            if config.v_lift > 0 and np.any(mesh.is_top_fixed):
                lifting_top = np.flatnonzero(mesh.is_top_fixed)
                solve_result = solver.solve_with_lift(
                    mesh,
                    layer_id=layer_id,
                    e_z=controller.E_z,
                    lifting_top=lifting_top,
                    on_iteration=_on_physics_iteration,
                )
                self._log.append_log(
                    f"  提升完成, 静平衡迭代 {solve_result.iterations} 步, "
                    f"max_dx={solve_result.max_dx:.2e}"
                )
            else:
                solve_result = solver.solve_until_stable(
                    mesh,
                    layer_id=layer_id,
                    e_z=controller.E_z,
                    on_iteration=_on_physics_iteration,
                )

            x_sim, v_sim = solve_result.x, solve_result.v

            # PID 反馈
            err_avg = (
                float(
                    np.mean(
                        target_vertices[bottom, 2] - x_sim[bottom, 2]
                    )
                )
                if len(bottom)
                else 0.0
            )
            pid_state = controller.update(err_avg=err_avg)

            # 形状误差指标
            max_error = float(
                np.max(
                    np.linalg.norm(
                        target_vertices - x_sim, axis=1
                    )
                )
            )
            rms_error = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (target_vertices - x_sim) ** 2,
                            axis=1,
                        )
                    )
                )
            )

            self._log.append_log(
                f"  max_error={max_error:.4f}, rms_error={rms_error:.4f}, "
                f"kinetic={solve_result.kinetic_energy:.2e}, "
                f"E_z_next={pid_state.E_z:.3f}"
            )

            # 封装结果
            result = LayerResult(
                layer_id=layer_id,
                x_sim=x_sim.copy(),
                v_sim=v_sim.copy(),
                error_metrics={
                    "err_avg": err_avg,
                    "E_z": pid_state.E_z,
                    "max_error": max_error,
                    "rms_error": rms_error,
                },
                field_command_next=None,
                max_deformation=max_error,
                rms_error=rms_error,
                success=bool(max_error < 2.0),
            )
            results.append(result)

            # ── 每层完成后的 GUI 更新（保证层间可见）──
            self._viewer.show_deformed_mesh(
                x_sim,
                mesh.tets,
                mesh.active_vertex_mask,
                title=f"第 {layer_id + 1}/{n_layers} 层 — 层完成",
            )
            QtWidgets.QApplication.processEvents()

            # ── 缓存动画帧：每层完成后保存一帧 ──
            self.animation_frames.append({
                "vertices": x_sim.copy(),
                "active_mask": mesh.active_vertex_mask.copy(),
                "title": f"第 {layer_id + 1}/{n_layers} 层 — 层完成",
            })

            # 输出 VTU
            write_vtu(
                vtk_dir / f"layer_{layer_id:04d}.vtu",
                mesh,
                point_data={
                    "active": mesh.active_vertex_mask.astype(float)
                },
            )

        # ── 强制末帧刷新：仿真循环结束后确保显示最终结果 ──
        if results:
            final_result = results[-1]
            self._viewer.show_deformed_mesh(
                final_result.x_sim,
                mesh.tets,
                mesh.active_vertex_mask,
                title=f"仿真完成 — 全部 {n_layers} 层",
            )
            QtWidgets.QApplication.processEvents()
            self._log.append_log(
                "  ✓ 末帧已刷新 — 3D 视图显示最终变形结果"
            )

            # ── 缓存最终帧 ──
            self.animation_frames.append({
                "vertices": final_result.x_sim.copy(),
                "active_mask": mesh.active_vertex_mask.copy(),
                "title": f"仿真完成 — 全部 {n_layers} 层",
            })

        # CSV 汇总
        write_metrics_csv(reports_dir / "error_metrics.csv", results)

        # ── 无缝衔接：仿真结束后自动启动动画回放 ──
        if self.animation_frames:
            self._log.append_log(
                f"  🎬 已缓存 {len(self.animation_frames)} 帧，启动动画回放 …"
            )
            self.start_animation_playback()

        return results

    # ========================================================================
    # 后处理动画回放
    # ========================================================================
    def _stop_animation_timer(self) -> None:
        """安全停止动画回放定时器。"""
        if self._anim_timer is not None and self._anim_timer.isActive():
            self._anim_timer.stop()
            self._anim_timer = None

    def start_animation_playback(self) -> None:
        """启动后处理动画回放。

        使用 ``QtCore.QTimer`` 以约 30 FPS（33 ms 间隔）循环播放
        仿真过程中缓存的所有帧。若动画已在运行则先停止再重启。
        """
        if not self.animation_frames:
            self._log.append_log(
                "  ⚠ 无动画帧可播放（animation_frames 为空）"
            )
            return

        # 停止已有的定时器（如果正在播放）
        self._stop_animation_timer()

        self.current_frame_idx = 0

        # 创建定时器：约 33 ms → ~30 FPS
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._anim_timer.timeout.connect(self._update_anim_frame)
        self._anim_timer.start(33)

        # 显示"停止回放"按钮
        self._btn_stop_anim.setVisible(True)

        self._log.append_log(
            "  ▶ 动画回放开始 (循环播放, ~30 FPS, "
            f"{len(self.animation_frames)} 帧)"
        )
        self._status.showMessage(
            f"动画回放中 … [{len(self.animation_frames)} 帧]"
        )

    def _update_anim_frame(self) -> None:
        """逐帧渲染动画：每次定时器超时时调用。

        按顺序读取 ``self.animation_frames`` 中的顶点数据，
        调用 ``self._viewer.show_deformed_mesh`` 刷新 3D 视图。
        播放到最后一帧后自动回到第 0 帧实现无限循环。
        """
        if not self.animation_frames or self._anim_tets is None:
            self._stop_animation_timer()
            return

        frame = self.animation_frames[self.current_frame_idx]

        self._viewer.show_deformed_mesh(
            frame["vertices"],
            self._anim_tets,
            frame["active_mask"],
            title=f"🔁 回放 — {frame['title']}",
        )
        QtWidgets.QApplication.processEvents()

        # 循环播放：到达末尾后重置到第 0 帧
        self.current_frame_idx += 1
        if self.current_frame_idx >= len(self.animation_frames):
            self.current_frame_idx = 0

    def _on_stop_animation(self) -> None:
        """手动停止动画回放，恢复初始网格视图。

        停止定时器，清空动画帧缓存，隐藏停止按钮，
        并将 3D 视图恢复为划分网格后的初始状态。
        """
        # 停止定时器
        self._stop_animation_timer()

        # 清空动画帧
        self.animation_frames.clear()
        self._anim_tets = None
        self.current_frame_idx = 0

        # 隐藏停止按钮
        self._btn_stop_anim.setVisible(False)

        # 恢复 3D 视图为初始网格（如果存在）
        if self._generated_mesh is not None:
            mesh = self._generated_mesh
            self._viewer.show_initial_mesh(mesh.vertices, mesh.tets)
            self._status.showMessage(
                f"已停止回放 | 网格就绪 ✓ | {self._actual_layers} 层 | "
                f"{len(mesh.vertices)} 顶点 | {len(mesh.tets)} 四面体"
            )
        else:
            self._viewer.clear()
            self._status.showMessage("已停止回放 — 无网格数据")

        self._log.append_log("  ⏹ 动画回放已手动停止")

    def _on_finish(self, results: list[LayerResult]) -> None:
        """仿真完成汇总。

        Parameters
        ----------
        results : list[LayerResult]
            全部层的仿真结果。
        """
        self._progress.set_done()
        max_e = max(r.max_deformation for r in results)
        rms_e = np.sqrt(
            np.mean([r.rms_error**2 for r in results])
        )
        success_count = sum(1 for r in results if r.success)
        self._status.showMessage(
            f"完成 ✓ | 层数: {len(results)} | "
            f"最大形变: {max_e:.4f} | RMS: {rms_e:.4f} | "
            f"成功层: {success_count}/{len(results)}"
        )
        self._log.append_log("\n===== 仿真结束 =====")
        self._log.append_log(
            f"汇总: 共 {len(results)} 层, "
            f"最大形变 {max_e:.4f} m, RMS {rms_e:.4f} m, "
            f"成功 {success_count}/{len(results)} 层"
        )


def launch_gui() -> None:
    """启动 GUI 应用程序（主入口）。

    创建 QApplication 实例，应用 Fusion 主题风格，
    显示主窗口并进入事件循环。
    """
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
