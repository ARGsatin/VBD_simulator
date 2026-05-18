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

from dataclasses import dataclass
import os
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
from hydrogel_vbd.geometry.stl_mesher import (
    create_demo_or_stl,
    STLMesher,
    _effective_top_down_layer_count,
    transform_points_to_print_z,
)
from hydrogel_vbd.io.report_writer import write_metrics_csv
from hydrogel_vbd.io.vtk_writer import write_vtu
from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver
from hydrogel_vbd.solver.cpp_adapter import is_cpp_available, refresh_availability
from hydrogel_vbd.solver.cpp_builder import CppBuilder, CppBuildResult
from hydrogel_vbd.core.state import LayerResult, MeshState
from hydrogel_vbd.gui.mesh_viewer import MeshViewer
from hydrogel_vbd.gui.simulation_worker import SimulationWorker

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


@dataclass(frozen=True)
class _ElectricFieldPlotData:
    """电场分析图使用的已归一化数据。"""

    layers: np.ndarray
    e_z: np.ndarray
    primary: np.ndarray
    secondary: np.ndarray
    aux_pct: np.ndarray
    mode: str
    primary_label: str
    secondary_label: str
    aux_label: str
    primary_title: str
    primary_ylabel: str
    guard_passed: np.ndarray | None = None


def _format_bbox_mm(vertices: np.ndarray) -> str:
    arr = np.asarray(vertices, dtype=float)
    if arr.size == 0:
        return "bbox empty"
    mins = np.min(arr, axis=0) * 1000.0
    maxs = np.max(arr, axis=0) * 1000.0
    size = maxs - mins
    return (
        f"bbox mm: X[{mins[0]:.3f}, {maxs[0]:.3f}] "
        f"Y[{mins[1]:.3f}, {maxs[1]:.3f}] "
        f"Z[{mins[2]:.3f}, {maxs[2]:.3f}] "
        f"size[{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]"
    )


def _frame_layer_id_from_title(title: str) -> int:
    import re

    text = str(title)
    for pattern in (
        "\u7b2c\\s*(\\d+)\\s*/",
        "\u7b2c\\s*(\\d+)\\s*\u5c42",
    ):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)) - 1
    return -1


def _should_retry_standard_meshing(
    *,
    algo_type: str,
    is_step: bool,
    actual_layers: int,
    exc: Exception,
) -> bool:
    """Avoid silently replacing a layer-conformal STEP mesh with a free tet mesh."""
    del algo_type, is_step, actual_layers, exc
    return False


# 参数元数据列表：定义每个参数在 GUI 中的显示标签、默认值、取值范围
_PARAM_META: list[dict[str, Any]] = [
    {"key": "mu", "label": "剪切模量 μ (Pa)", "default": 5000.0, "min": 500.0, "max": 1e6},
    {"key": "kappa", "label": "体积模量 κ (Pa)", "default": 250000.0, "min": 2000.0, "max": 1e7},
    {"key": "k_d", "label": "阻尼系数 k_d", "default": 0.05, "min": 0.0, "max": 1.0},
    {"key": "c_shrink", "label": "收缩因子 c_shrink", "default": 1.0, "min": 0.8, "max": 1.0},
    {"key": "T_max", "label": "最大附着力 T_max (Pa)", "default": 3000.0, "min": 100.0, "max": 50000.0},
    {"key": "K_czm", "label": "CZM 刚度 (Pa/m)", "default": 1.0e7, "min": 1e6, "max": 1e10},
    {"key": "delta_f", "label": "CZM 失效位移 δ_f (m)", "default": 2.0e-3, "min": 1e-6, "max": 1e-2},
    {"key": "node_area", "label": "CZM node area (m^2)", "default": 1.0e-6, "min": 1e-10, "max": 1e-3},
    {"key": "eta", "label": "流体/损伤系数 η", "default": 0.8, "min": 0.0, "max": 10.0, "step": 0.1},
    {"key": "C_0", "label": "流体负压倍率 C_0", "default": 1.0, "min": 0.0, "max": 1000.0, "step": 1.0},
    {"key": "fluid_radius", "label": "流体作用半径 r (m)", "default": 0.001, "min": 1e-5, "max": 1e-2, "step": 1e-4},
    {"key": "d_fluid_max", "label": "流体作用距离 d_fluid_max (m)", "default": 2.0e-3, "min": 0.0, "max": 1e-2, "step": 1e-4},
    {"key": "t_fluid_max", "label": "流体持续时间 t_fluid_max (s)", "default": 0.5, "min": 0.0, "max": 10.0, "step": 0.05},
    {"key": "d_min", "label": "最小液膜间隙 d_min (m)", "default": 1.0e-6, "min": 1e-9, "max": 1e-4, "decimals": 9, "step": 1e-6},
    {"key": "dt", "label": "时间步长 dt (s)", "default": 0.001, "min": 0.0001, "max": 0.05},
    {"key": "max_iters", "label": "最大迭代次数", "default": 50, "min": 5, "max": 200},
    {"key": "N_stable", "label": "稳定步数判决", "default": 3, "min": 2, "max": 50},
    {"key": "layer_thickness", "label": "层厚 (mm)", "default": 0.7993, "min": 0.01, "max": 10.0},
    {"key": "lift_multiplier", "label": "提升距离倍数", "default": 1.5, "min": 0.1, "max": 5.0},
    {"key": "v_lift", "label": "提升速度 (m/s)", "default": 0.01, "min": 0.0, "max": 0.01},
    {"key": "K_p", "label": "PID K_p", "default": 150.0, "min": 0.0, "max": 1000.0},
    {"key": "K_i", "label": "PID K_i", "default": 20.0, "min": 0.0, "max": 200.0},
    {"key": "K_d", "label": "PID K_d", "default": 5.0, "min": 0.0, "max": 100.0},
    {"key": "E_max", "label": "最大电场 (V/m)", "default": 500.0, "min": 10.0, "max": 5000.0},
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
        layout.setContentsMargins(10, 10, 15, 10)
        layout.setVerticalSpacing(4)
        layout.setSpacing(4)
        for meta in _PARAM_META:
            default = meta["default"]
            if isinstance(default, float):
                sb = QtWidgets.QDoubleSpinBox()
                sb.setRange(meta["min"], meta["max"])
                sb.setDecimals(int(meta.get("decimals", 6)))
                sb.setValue(default)
                sb.setSingleStep(float(
                    meta.get("step", (meta["max"] - meta["min"]) / 100.0)
                ))
            else:
                sb = QtWidgets.QSpinBox()
                sb.setRange(meta["min"], meta["max"])
                sb.setValue(int(default))
            sb.setMaximumWidth(140)
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

    @staticmethod
    def _config_values_from_ui(params: dict[str, Any]) -> dict[str, Any]:
        """将 GUI 显示单位转换为 ``SimulationConfig`` 使用的 SI 单位。"""
        values = dict(params)
        if "layer_thickness" in values:
            values["layer_thickness"] = float(values["layer_thickness"]) * 1e-3
        return values

    def get_config(self) -> SimulationConfig:
        """获取当前参数对应的 ``SimulationConfig`` 对象。

        Returns
        -------
        SimulationConfig
            包含当前所有参数值的配置对象。
        """
        self._collect_params()
        return SimulationConfig(**self._config_values_from_ui(self._params))


class ProgressWidget(QtWidgets.QWidget):
    """仿真进度显示条。

    包含外层进度条（层间进度）、一个文本标签，以及
    一条绿色的细粒度子进度条（层内提升进度）。
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setFixedHeight(16)

        self._label = QtWidgets.QLabel("就绪")

        # ── 细粒度子进度条（层内提升进度，绿色、细条）──
        self._sub_bar = QtWidgets.QProgressBar()
        self._sub_bar.setRange(0, 100)
        self._sub_bar.setValue(0)
        self._sub_bar.setFixedHeight(6)
        self._sub_bar.setStyleSheet(
            "QProgressBar { border: none; background: transparent; }"
            "QProgressBar::chunk { background-color: #4CAF50; border-radius: 2px; }"
        )
        self._sub_bar.setToolTip("当前层提升进度 (0–100%)")

        layout.addWidget(self._bar)
        layout.addWidget(self._label)
        layout.addWidget(self._sub_bar)
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

    def set_sub_progress(self, percentage: int) -> None:
        """更新层内细粒度子进度（提升百分比）。

        Parameters
        ----------
        percentage : int
            0–100 的整数值，表示当前层内提升进度。
        """
        self._sub_bar.setValue(max(0, min(100, percentage)))

    def set_done(self) -> None:
        """标记仿真完成，进度条置为 100%，子进度条归零。"""
        self._bar.setValue(100)
        self._sub_bar.setValue(0)
        self._label.setText("仿真完成 ✓")


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
        self.resize(1500, 950)

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
        self._chk_use_cpp: QtWidgets.QCheckBox | None = None
        self._chk_solver_diag: QtWidgets.QCheckBox | None = None
        self._chk_field_debug: QtWidgets.QCheckBox | None = None
        self._combo_print_z_axis: QtWidgets.QComboBox | None = None
        self._chk_disable_czm: QtWidgets.QCheckBox | None = None
        self._chk_disable_chebyshev: QtWidgets.QCheckBox | None = None
        self._solver_diag_env_backup: dict[str, str | None] | None = None

        # ── 后处理动画回放状态 ──
        self.animation_frames: list[dict[str, Any]] = []
        self._anim_tets: np.ndarray | None = None
        self._anim_timer: QtCore.QTimer | None = None
        self.current_frame_idx: int = 0
        self._anim_paused: bool = False
        self._anim_layer_filter: int | None = None  # None = 全部层
        self._anim_slider: QtWidgets.QSlider | None = None
        self._anim_frame_label: QtWidgets.QLabel | None = None
        self._anim_layer_combo: QtWidgets.QComboBox | None = None
        self._btn_anim_play: QtWidgets.QPushButton | None = None
        self._btn_anim_pause: QtWidgets.QPushButton | None = None
        self._frame_indices: list[int] = []  # 筛选后的帧索引映射

        # ── C++ 自动编译状态 ──
        self._build_thread: QtCore.QThread | None = None
        self._build_worker: QtCore.QObject | None = None
        self._pending_build_config: SimulationConfig | None = None

        # ── 仿真线程 / 中断状态 ──
        self._thread: QtCore.QThread | None = None
        self._worker: SimulationWorker | None = None
        self._simulation_stop_requested: bool = False

        # ── DVR 实时时间轴状态 ──
        self._dv_slider: QtWidgets.QSlider | None = None
        self._dv_label: QtWidgets.QLabel | None = None
        self._dv_is_slider_down: bool = False
        self._dv_efield_canvas: Any = None
        self._dv_efield_fig: Any = None
        self._dv_efield_layer_data: list[tuple[int, float, float, float]] = (
            []
        )  # (layer_id, e_z, max_err, rms_err)
        self._is_rendering_worker_frame: bool = False

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
        top_bar.setContentsMargins(2, 2, 2, 2)
        top_bar.setSpacing(4)

        self._btn_load = QtWidgets.QPushButton("加载模型")
        self._btn_load.setFixedHeight(28)
        self._btn_load.setFixedWidth(80)
        self._btn_load.clicked.connect(self._on_load_stl)

        self._btn_clear = QtWidgets.QPushButton("清除")
        self._btn_clear.setFixedHeight(28)
        self._btn_clear.setFixedWidth(50)
        self._btn_clear.clicked.connect(self._on_clear_stl)

        self._lbl_model = QtWidgets.QLabel("无模型 (Demo 网格)")
        self._lbl_model.setMinimumWidth(120)

        self._btn_mesh = QtWidgets.QPushButton("划分网格")
        self._btn_mesh.setFixedHeight(28)
        self._btn_mesh.setFixedWidth(80)
        self._btn_mesh.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #2196F3; "
            "color: white; border-radius: 3px; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:disabled { background-color: #BDBDBD; color: #757575; }"
        )
        self._btn_mesh.clicked.connect(self._on_generate_mesh)

        self._btn_run = QtWidgets.QPushButton("运行仿真")
        self._btn_run.setFixedHeight(28)
        self._btn_run.setFixedWidth(80)
        self._btn_run.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #4CAF50; "
            "color: white; border-radius: 3px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #BDBDBD; color: #757575; }"
        )
        self._btn_run.clicked.connect(self._on_run)

        self._btn_stop_sim = QtWidgets.QPushButton("中断仿真")
        self._btn_stop_sim.setFixedHeight(28)
        self._btn_stop_sim.setFixedWidth(80)
        self._btn_stop_sim.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #E53935; "
            "color: white; border-radius: 3px; }"
            "QPushButton:hover { background-color: #C62828; }"
            "QPushButton:disabled { background-color: #BDBDBD; color: #757575; }"
        )
        self._btn_stop_sim.clicked.connect(self._on_stop_simulation)
        self._btn_stop_sim.setEnabled(False)
        self._btn_stop_sim.setVisible(False)

        self._btn_stop_anim = QtWidgets.QPushButton("停止回放")
        self._btn_stop_anim.setFixedHeight(28)
        self._btn_stop_anim.setFixedWidth(80)
        self._btn_stop_anim.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #f44336; "
            "color: white; border-radius: 3px; }"
            "QPushButton:hover { background-color: #d32f2f; }"
        )
        self._btn_stop_anim.clicked.connect(self._on_stop_animation)
        self._btn_stop_anim.setVisible(False)

        top_bar.addWidget(self._btn_load)
        top_bar.addWidget(self._btn_clear)
        top_bar.addWidget(self._lbl_model, 1)
        _sep = QtWidgets.QFrame()
        _sep.setFrameShape(QtWidgets.QFrame.VLine)
        _sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        top_bar.addWidget(_sep)
        top_bar.addWidget(self._btn_mesh)
        top_bar.addWidget(self._btn_run)
        top_bar.addWidget(self._btn_stop_sim)
        top_bar.addWidget(self._btn_stop_anim)
        root.addLayout(top_bar)
        root.setSpacing(2)
        root.setContentsMargins(4, 4, 4, 4)

        # ── 中部：参数 + 进度/日志 + 3D 视图 ──
        mid = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        mid.setHandleWidth(3)

        # 左侧面板
        left = QtWidgets.QWidget()
        left.setMinimumWidth(480)
        left.setMaximumWidth(620)
        left_layout = QtWidgets.QVBoxLayout(left)
        self._left_layout = left_layout
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)
        self._param_panel = ParameterPanel()
        self._param_panel.params_changed.connect(self._on_params)

        # ── 第 1 行：层厚 / 层数 / 分辨率 ──
        info_1 = QtWidgets.QHBoxLayout()
        info_1.addWidget(QtWidgets.QLabel("层厚 (mm):"))
        self._lbl_thickness = QtWidgets.QLabel("0.05")
        self._lbl_thickness.setStyleSheet("font-weight: bold; color: #1976D2;")
        info_1.addWidget(self._lbl_thickness)
        info_1.addSpacing(12)

        info_1.addWidget(QtWidgets.QLabel("层数:"))
        self._lbl_layers = QtWidgets.QLabel("—")
        self._lbl_layers.setStyleSheet("font-weight: bold;")
        info_1.addWidget(self._lbl_layers)
        self._spin_layers = QtWidgets.QSpinBox()
        self._spin_layers.setRange(1, 100)
        self._spin_layers.setValue(3)
        self._spin_layers.setMaximumWidth(70)
        self._spin_layers.setToolTip("Demo 模式层数")
        self._spin_layers.valueChanged.connect(self._on_demo_layers_changed)
        info_1.addWidget(self._spin_layers)
        info_1.addSpacing(12)

        info_1.addWidget(QtWidgets.QLabel("分辨率 (mm):"))
        self._lbl_resolution = QtWidgets.QLabel("—")
        self._lbl_resolution.setStyleSheet("font-weight: bold;")
        info_1.addWidget(self._lbl_resolution)
        self._spin_resolution = QtWidgets.QDoubleSpinBox()
        self._spin_resolution.setRange(0.01, 500.0)
        self._spin_resolution.setDecimals(3)
        self._spin_resolution.setValue(20.0)
        self._spin_resolution.setSingleStep(0.05)
        self._spin_resolution.setSuffix(" mm")
        self._spin_resolution.setMaximumWidth(110)
        self._spin_resolution.setToolTip("Demo 模式 XY 网格间距 (mm)")
        self._spin_resolution.valueChanged.connect(self._on_demo_params_changed)
        info_1.addWidget(self._spin_resolution)

        self._lbl_auto_res = QtWidgets.QLabel("")
        self._lbl_auto_res.setStyleSheet("color: #888; font-size: 9pt;")
        info_1.addWidget(self._lbl_auto_res)
        info_1.addStretch()
        left_layout.addLayout(info_1)

        # ── 第 2 行：自定义分辨率 + 网格算法 ──
        info_2 = QtWidgets.QHBoxLayout()
        self._chk_custom_res = QtWidgets.QCheckBox("自定义分辨率")
        self._chk_custom_res.setToolTip(
            "勾选后可手动调节网格分辨率；\n"
            "不勾选则根据模型尺寸自动计算最佳分辨率"
        )
        self._chk_custom_res.toggled.connect(self._on_custom_resolution_toggled)
        info_2.addWidget(self._chk_custom_res)
        info_2.addSpacing(12)

        info_2.addWidget(QtWidgets.QLabel("网格算法:"))
        self._combo_mesh_algo = QtWidgets.QComboBox()
        self._combo_mesh_algo.addItem("规整分层 (OCC 切片)", "layered")
        self._combo_mesh_algo.addItem("标准非结构化", "standard")
        self._combo_mesh_algo.setToolTip(
            "规整分层算法: 通过 OCC Boolean Fragment 水平切片，"
            "保证四面体不跨层;\n"
            "标准非结构化算法: 跳过切片，直接生成自由四面体网格"
        )
        info_2.addWidget(self._combo_mesh_algo)
        info_2.addSpacing(12)
        info_2.addWidget(QtWidgets.QLabel("打印 Z:"))
        self._combo_print_z_axis = QtWidgets.QComboBox()
        self._combo_print_z_axis.addItem("model Y", "y")
        self._combo_print_z_axis.addItem("model -Y", "-y")
        self._combo_print_z_axis.addItem("model Z", "z")
        self._combo_print_z_axis.addItem("model -Z", "-z")
        self._combo_print_z_axis.addItem("model X", "x")
        self._combo_print_z_axis.addItem("model -X", "-x")
        self._combo_print_z_axis.setToolTip("选择模型哪个轴作为打印/切片 Z 方向")
        self._combo_print_z_axis.currentIndexChanged.connect(self._on_model_axis_changed)
        info_2.addWidget(self._combo_print_z_axis)
        info_2.addStretch()
        left_layout.addLayout(info_2)

        # ── 第 3 行：C++ 求解器开关 ──
        info_3 = QtWidgets.QHBoxLayout()
        self._chk_use_cpp = QtWidgets.QCheckBox("使用 C++ 加速求解器 (实验性)")
        self._chk_use_cpp.setChecked(False)
        self._chk_use_cpp.setToolTip(
            "勾选后优先使用 C++ 加速求解器；\n"
            "⚠ 实验性功能：在 QThread 中可能不稳定，默认关闭"
        )
        info_3.addWidget(self._chk_use_cpp)
        info_3.addSpacing(12)
        self._chk_solver_diag = QtWidgets.QCheckBox("输出求解器诊断 CSV")
        self._chk_solver_diag.setChecked(False)
        self._chk_solver_diag.setToolTip(
            "勾选后本次仿真写入 outputs/gui/reports/solver_diagnostics.csv"
        )
        info_3.addWidget(self._chk_solver_diag)
        info_3.addSpacing(12)
        self._chk_field_debug = QtWidgets.QCheckBox("电场调试对比")
        self._chk_field_debug.setChecked(False)
        self._chk_field_debug.setToolTip(
            "勾选后每层克隆同一初态，分别计算 E=0 与 Bottom-Z 推导电场后的指标；"
            "会额外增加求解耗时"
        )
        info_3.addWidget(self._chk_field_debug)
        info_3.addSpacing(12)
        self._chk_disable_czm = QtWidgets.QCheckBox("禁用 CZM")
        self._chk_disable_czm.setChecked(False)
        self._chk_disable_czm.setToolTip("调试用：跳过 CZM/流体脱粘力、状态更新和 CZM 固定约束")
        self._chk_disable_czm.toggled.connect(self._on_czm_toggle_changed)
        info_3.addWidget(self._chk_disable_czm)
        info_3.addSpacing(12)
        self._chk_disable_chebyshev = QtWidgets.QCheckBox("禁用 Chebyshev")
        self._chk_disable_chebyshev.setChecked(False)
        self._chk_disable_chebyshev.setToolTip("调试用：将 rho_cheb 置为 0，跳过 Chebyshev 加速")
        self._chk_disable_chebyshev.toggled.connect(self._on_chebyshev_toggle_changed)
        info_3.addWidget(self._chk_disable_chebyshev)
        info_3.addStretch()
        left_layout.addLayout(info_3)

        # ── QScrollArea 包裹参数面板 ──
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_area.setWidget(self._param_panel)
        self._param_scroll_area = scroll_area
        left_layout.addWidget(scroll_area, 1)

        # 右侧面板
        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right.setHandleWidth(3)
        upper_right = QtWidgets.QWidget()
        upper_layout = QtWidgets.QVBoxLayout(upper_right)
        upper_layout.setContentsMargins(0, 0, 0, 2)
        upper_layout.setSpacing(2)
        self._progress = ProgressWidget()
        self._log = LogWidget()
        upper_layout.addWidget(self._progress)
        upper_layout.addWidget(self._log, 1)
        right.addWidget(upper_right)

        self._viewer = MeshViewer()
        right.addWidget(self._viewer)

        # ── 动画播放控制面板（3D 视图下方）──
        anim_panel = QtWidgets.QWidget()
        anim_panel_layout = QtWidgets.QHBoxLayout(anim_panel)
        anim_panel_layout.setContentsMargins(4, 2, 4, 2)
        anim_panel_layout.setSpacing(6)

        self._btn_anim_play = QtWidgets.QPushButton("播放")
        self._btn_anim_play.setFixedHeight(26)
        self._btn_anim_play.setFixedWidth(70)
        self._btn_anim_play.setToolTip("从头开始播放动画")
        self._btn_anim_play.clicked.connect(self._on_anim_play_clicked)
        self._btn_anim_play.setVisible(False)

        self._btn_anim_pause = QtWidgets.QPushButton("暂停")
        self._btn_anim_pause.setFixedHeight(26)
        self._btn_anim_pause.setFixedWidth(70)
        self._btn_anim_pause.setToolTip("暂停/继续动画回放")
        self._btn_anim_pause.clicked.connect(self._on_anim_pause_clicked)
        self._btn_anim_pause.setVisible(False)

        self._btn_stop_anim_from_panel = QtWidgets.QPushButton("停止")
        self._btn_stop_anim_from_panel.setFixedHeight(26)
        self._btn_stop_anim_from_panel.setFixedWidth(70)
        self._btn_stop_anim_from_panel.setToolTip("停止回放并恢复初始网格")
        self._btn_stop_anim_from_panel.clicked.connect(self._on_stop_animation)
        self._btn_stop_anim_from_panel.setVisible(False)

        self._anim_frame_label = QtWidgets.QLabel("帧: 0 / 0")
        self._anim_frame_label.setMinimumWidth(80)
        self._anim_frame_label.setVisible(False)

        # 层筛选下拉框
        self._anim_layer_combo = QtWidgets.QComboBox()
        self._anim_layer_combo.addItem("全部层", -1)
        self._anim_layer_combo.setToolTip("按层筛选动画帧")
        self._anim_layer_combo.currentIndexChanged.connect(self._on_anim_layer_filter_changed)
        self._anim_layer_combo.setVisible(False)
        self._anim_layer_combo.setMinimumWidth(100)

        # 帧滑块（水平拖动定位）
        self._anim_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._anim_slider.setRange(0, 0)
        self._anim_slider.setValue(0)
        self._anim_slider.setToolTip("拖动定位到指定帧")
        self._anim_slider.valueChanged.connect(self._on_anim_slider_changed)
        self._anim_slider.setVisible(False)

        anim_panel_layout.addWidget(self._btn_anim_play)
        anim_panel_layout.addWidget(self._btn_anim_pause)
        anim_panel_layout.addWidget(self._btn_stop_anim_from_panel)
        anim_panel_layout.addWidget(self._anim_frame_label)
        anim_panel_layout.addWidget(QtWidgets.QLabel("层筛选:"))
        anim_panel_layout.addWidget(self._anim_layer_combo)
        anim_panel_layout.addWidget(self._anim_slider, 1)

        right.addWidget(anim_panel)

        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 4)  # 3D 视图占据 4/5 垂直空间
        right.setStretchFactor(2, 0)  # 控制面板保持最小高度

        mid.addWidget(left)
        mid.addWidget(right)
        # 水平拉伸策略：左侧保持紧凑不拉伸，右侧吸纳所有剩余空间
        mid.setStretchFactor(0, 0)
        mid.setStretchFactor(1, 1)
        root.addWidget(mid)

        # ── DVR 实时时间轴（底部水平栏）──
        dvr_bar = QtWidgets.QHBoxLayout()
        dvr_bar.setContentsMargins(6, 2, 6, 4)
        dvr_bar.setSpacing(8)

        dvr_bar.addWidget(QtWidgets.QLabel("📽 时间轴:"))

        self._dv_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._dv_slider.setRange(0, 0)
        self._dv_slider.setValue(0)
        self._dv_slider.setToolTip("拖动定位到已完成的层")
        self._dv_slider.sliderPressed.connect(self._on_dv_slider_pressed)
        self._dv_slider.sliderReleased.connect(self._on_dv_slider_released)
        self._dv_slider.valueChanged.connect(self._on_dv_slider_changed)
        dvr_bar.addWidget(self._dv_slider, 1)

        self._dv_label = QtWidgets.QLabel("层: —")
        self._dv_label.setMinimumWidth(80)
        self._dv_label.setStyleSheet("font-weight: bold; color: #1976D2;")
        dvr_bar.addWidget(self._dv_label)

        # ── 实时电场图表（迷你 matplotlib 嵌板，约 200×120 px）──
        if _HAS_MPL:
            try:
                self._dv_efield_fig = Figure(figsize=(2.8, 1.6), dpi=80)
                self._dv_efield_canvas = FigureCanvas(self._dv_efield_fig)
                self._dv_efield_canvas.setFixedSize(260, 120)
                self._dv_efield_canvas.setToolTip("实时电场强度 E_z (V/m) 逐层曲线")
                dvr_bar.addWidget(self._dv_efield_canvas)
            except Exception:
                self._dv_efield_fig = None
                self._dv_efield_canvas = None
        else:
            self._dv_efield_fig = None
            self._dv_efield_canvas = None

        root.addLayout(dvr_bar)

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
                f"(自动推荐: {self._spin_resolution.value():.3f} mm)"
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
    def _reset_window_size(self) -> None:
        """重置主窗口为默认推荐尺寸并居中显示。"""
        self.resize(1500, 950)
        # 居中显示
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            center = screen.availableGeometry().center()
            frame = self.frameGeometry()
            frame.moveCenter(center)
            self.move(frame.topLeft())

    def _update_button_states(self) -> None:
        """根据当前状态启用/禁用按钮。"""
        has_mesh = self._generated_mesh is not None
        has_model = self._stl_path is not None
        is_running = self._thread is not None

        # "划分网格" — 总是可用（STL 模式用已加载的 STL，Demo 模式自动生成）
        self._btn_mesh.setEnabled(not is_running)

        # "运行仿真" — 必须有已生成的网格
        self._btn_run.setEnabled(has_mesh and not is_running)
        if not has_mesh:
            self._btn_run.setToolTip("请先点击「划分网格」生成网格")
        elif is_running:
            self._btn_run.setToolTip("仿真运行中")
        else:
            self._btn_run.setToolTip("开始逐层仿真")

    def _set_simulation_running(self, running: bool) -> None:
        """Update run/stop controls for the simulation worker lifecycle."""
        if running:
            self._btn_run.setEnabled(False)
            self._btn_run.setText("仿真中...")
            self._btn_stop_sim.setText("中断仿真")
            self._btn_stop_sim.setEnabled(True)
            self._btn_stop_sim.setVisible(True)
            self._btn_mesh.setEnabled(False)
            return

        self._simulation_stop_requested = False
        self._btn_run.setText("运行仿真")
        self._btn_stop_sim.setText("中断仿真")
        self._btn_stop_sim.setEnabled(False)
        self._btn_stop_sim.setVisible(False)
        has_mesh = self._generated_mesh is not None
        self._btn_mesh.setEnabled(True)
        self._btn_run.setEnabled(has_mesh)
        self._btn_run.setToolTip(
            "开始逐层仿真"
            if has_mesh
            else "请先点击「划分网格」生成网格"
        )

    # ========================================================================
    # 槽函数 — 参数
    # ========================================================================
    @property
    def _layer_thickness_m(self) -> float:
        """当前有效的层厚（m），``SimulationConfig`` 内部始终使用 SI 单位。"""
        return self._config.layer_thickness if self._config.layer_thickness > 0 else 5e-5

    @property
    def _layer_thickness_mm(self) -> float:
        """当前有效的层厚（mm），仅用于 GUI 显示。"""
        return self._layer_thickness_m * 1000.0

    def _on_params(self, params: dict[str, Any]) -> None:
        """参数面板变更：更新当前配置。"""
        self._config = SimulationConfig(**ParameterPanel._config_values_from_ui(params))
        self._apply_debug_solver_toggles(self._config)
        # 动态更新层厚显示
        lt_mm = float(params.get("layer_thickness", self._layer_thickness_mm))
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

    def _on_model_axis_changed(self, _index: int) -> None:
        self._generated_mesh = None
        self.mesh = None
        self._actual_layers = 0
        if self._stl_path is not None:
            self._auto_calculate_mesh_params()
            self._show_stl_preview()
        self._update_button_states()

    def _on_czm_toggle_changed(self, _checked: bool) -> None:
        self._apply_debug_solver_toggles(self._config)

    def _on_chebyshev_toggle_changed(self, _checked: bool) -> None:
        self._apply_debug_solver_toggles(self._config)

    def _print_z_axis(self) -> str:
        if self._combo_print_z_axis is None:
            return "z"
        return str(self._combo_print_z_axis.currentData() or "z")

    def _is_czm_disabled(self) -> bool:
        return bool(self._chk_disable_czm and self._chk_disable_czm.isChecked())

    def _is_chebyshev_disabled(self) -> bool:
        return bool(
            self._chk_disable_chebyshev
            and self._chk_disable_chebyshev.isChecked()
        )

    def _apply_debug_solver_toggles(self, config: SimulationConfig) -> None:
        config.enable_czm = not self._is_czm_disabled()
        if self._is_chebyshev_disabled():
            config.rho_cheb = 0.0

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
        self._spin_resolution.setValue(round(resolution_mm, 3))
        self._spin_resolution.blockSignals(False)
        # 更新自动推荐提示
        if self._lbl_auto_res is not None:
            is_custom = self._chk_custom_res.isChecked() if self._chk_custom_res else False
            if not is_custom:
                self._lbl_auto_res.setText(
                    f"(自动推荐: {resolution_mm:.3f} mm)"
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
            f"分辨率 {self._auto_resolution*1000:.3f} mm"
        )
        self._log.append_log("  · 请点击「划分网格」生成四面体网格")
        self._status.showMessage(
            f"已加载 {self._stl_path.name} | "
            f"{self._actual_layers} 层 | "
            f"{self._auto_resolution*1000:.3f} mm 分辨率"
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
            loaded.vertices = transform_points_to_print_z(
                loaded.vertices, self._print_z_axis()
            )
            self._stl_trimesh = loaded

            bounds = loaded.bounds
            x_min, x_max = float(bounds[0, 0]), float(bounds[1, 0])
            y_min, y_max = float(bounds[0, 1]), float(bounds[1, 1])
            z_min, z_max = float(bounds[0, 2]), float(bounds[1, 2])

        model_height = z_max - z_min
        lt_m = self._layer_thickness_m
        self._actual_layers = _effective_top_down_layer_count(
            model_height,
            lt_m,
        )

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
            self._spin_resolution.setValue(round(resolution_mm, 3))
            self._spin_resolution.blockSignals(False)
            if self._lbl_auto_res is not None:
                self._lbl_auto_res.setText(
                    f"(自动推荐: {resolution_mm:.3f} mm)"
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
            if self._print_z_axis() != "z":
                from hydrogel_vbd.geometry.stl_mesher import _apply_gmsh_print_z_transform

                _apply_gmsh_print_z_transform(self._print_z_axis())
                _gmsh.model.occ.synchronize()

            # ── 遍历所有导入的 OCC 实体，分别获取包围盒后合并 ──
            # 避免 dim=-1, tag=-1 全局查询对某些 STEP 拓扑的兼容性问题
            xmin = float("inf")
            ymin = float("inf")
            zmin = float("inf")
            xmax = float("-inf")
            ymax = float("-inf")
            zmax = float("-inf")

            for dim, tag in imported:
                try:
                    bbox = _gmsh.model.occ.getBoundingBox(dim, tag)
                except Exception:
                    # 某些低维实体（如点/线）可能无法获取包围盒，跳过
                    continue
                if not bbox or len(bbox) < 6:
                    continue

                xmin = min(xmin, float(bbox[0]))
                ymin = min(ymin, float(bbox[1]))
                zmin = min(zmin, float(bbox[2]))
                xmax = max(xmax, float(bbox[3]))
                ymax = max(ymax, float(bbox[4]))
                zmax = max(zmax, float(bbox[5]))

            if not np.isfinite(xmin) or not np.isfinite(xmax):
                raise RuntimeError(
                    "无法获取 STEP 模型包围盒 — "
                    "所有 OCC 实体均未返回有效包围盒"
                )

            # Gmsh 返回 mm 单位，转为 m
            return (
                xmin * 0.001, xmax * 0.001,
                ymin * 0.001, ymax * 0.001,
                zmin * 0.001, zmax * 0.001,
            )
        finally:
            try:
                if _gmsh.isInitialized():
                    _gmsh.finalize()
            except Exception:
                pass

    def _show_stl_preview(self) -> None:
        """在 3D 视图中展示模型的原始表面。

        - **STL 格式**：直接使用 trimesh 三角面渲染。
        - **STEP 格式**：启动临时 Gmsh 流程，生成极粗 2D 表面网格
          作为"快速表面代理（Surface Proxy）"。仅生成面网格
          （``gmsh.model.mesh.generate(2)``），跳过昂贵的 3D 体划分，
          使用 Delaunay 算法确保秒开预览。
        """
        import os as _os

        try:
            stl = self._stl_trimesh
            if stl is not None:
                # ── STL 格式：已有三角面，直接渲染 ──
                self._viewer.show_stl_surface(
                    stl.vertices,
                    stl.faces,
                    title=f"STL 模型 — {self._stl_path.name}",
                )
                return

            # ── STEP 格式：快速表面代理 ──
            ext = _os.path.splitext(str(self._stl_path))[1].lower() if self._stl_path else ""
            is_step = ext in ('.step', '.stp', '.igs', '.iges', '.brep')
            if not (is_step and hasattr(self, '_bbox_x')):
                return

            self._status.showMessage("正在生成几何预览 …")

            # 计算包围盒对角线长度（m）
            dx = self._bbox_x[1] - self._bbox_x[0]
            dy = self._bbox_y[1] - self._bbox_y[0]
            dz = self._bbox_z[1] - self._bbox_z[0]
            diag_m = float(np.sqrt(dx * dx + dy * dy + dz * dz))

            # Gmsh 使用 mm 单位 → 转为 mm
            diag_mm = diag_m * 1000.0

            try:
                import gmsh as _gmsh
            except ImportError:
                self._log.append_log(
                    "  (Gmsh 未安装，回退到包围盒预览)"
                )
                self._viewer.show_bounding_box(
                    self._bbox_x, self._bbox_y, self._bbox_z,
                    title=f"STEP 模型包围盒 — {self._stl_path.name}",
                )
                self._status.showMessage(
                    f"已加载 {self._stl_path.name} | "
                    f"{self._actual_layers} 层"
                )
                return

            _gmsh.initialize()
            try:
                _gmsh.option.setNumber("General.Terminal", 0)
                _gmsh.model.add("_surface_preview")

                # 导入 STEP 几何
                imported = _gmsh.model.occ.importShapes(str(self._stl_path))
                if not imported:
                    raise RuntimeError("Gmsh OCC 无法导入 STEP 模型")
                _gmsh.model.occ.synchronize()
                if self._print_z_axis() != "z":
                    from hydrogel_vbd.geometry.stl_mesher import _apply_gmsh_print_z_transform

                    _apply_gmsh_print_z_transform(self._print_z_axis())
                    _gmsh.model.occ.synchronize()

                # ── 极粗网格尺寸：包围盒对角线的 1/20 ──
                coarse_size = max(diag_mm / 20.0, 0.1)
                _gmsh.option.setNumber("Mesh.CharacteristicLengthMin", coarse_size)
                _gmsh.option.setNumber("Mesh.CharacteristicLengthMax", coarse_size * 2.0)

                # ── Delaunay 表面网格（不进行 3D 体划分）──
                _gmsh.option.setNumber("Mesh.Algorithm", 5)          # Delaunay
                _gmsh.option.setNumber("Mesh.Algorithm3D", 1)        # 不影响 2D
                _gmsh.option.setNumber("Mesh.Optimize", 1)           # 基础优化
                _gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)     # 禁用 Netgen

                # 仅生成 2D 表面网格
                _gmsh.model.mesh.generate(2)

                # ── 提取表面三角面片 ──
                node_tags, coords, _ = _gmsh.model.mesh.getNodes()
                coords = np.array(coords).reshape(-1, 3)  # (N, 3) in mm
                vertices_preview = coords * 0.001  # mm → m

                # 获取 2D 单元（三角形）
                elem_types, elem_tags, elem_node_tags = _gmsh.model.mesh.getElements(dim=2)
                faces_preview = None
                for et, ent in zip(elem_types, elem_node_tags):
                    if et == 2:  # 2 = 3-node triangle
                        faces_preview = np.array(ent, dtype=int).reshape(-1, 3)
                        break
                    if et == 3:  # 3 = 4-node quad → 拆分为 2 个三角形
                        quads = np.array(ent, dtype=int).reshape(-1, 4)
                        t1 = quads[:, [0, 1, 2]]
                        t2 = quads[:, [0, 2, 3]]
                        faces_preview = np.vstack([t1, t2])
                        break

                if faces_preview is None or faces_preview.size == 0:
                    raise RuntimeError("未能提取表面三角面片")

                # ── Gmsh 返回 1-based 索引 → 0-based ──
                faces_preview = faces_preview - 1
                self._log.append_log(f"  · STEP preview {_format_bbox_mm(vertices_preview)}")

                # ── 渲染表面代理 ──
                self._viewer.show_stl_surface(
                    vertices_preview,
                    faces_preview,
                    title=f"STEP 几何预览 — {self._stl_path.name}",
                )

            finally:
                try:
                    if _gmsh.isInitialized():
                        _gmsh.finalize()
                except Exception:
                    pass

            self._status.showMessage(
                f"已加载 {self._stl_path.name} | "
                f"{self._actual_layers} 层 | "
                f"表面预览 ✓"
            )

        except Exception as exc:
            self._log.append_log(f"  (预览失败: {exc})")
            # 最终回退：包围盒
            if hasattr(self, '_bbox_x'):
                try:
                    self._viewer.show_bounding_box(
                        self._bbox_x, self._bbox_y, self._bbox_z,
                        title=f"STEP 模型包围盒 — {self._stl_path.name}",
                    )
                except Exception:
                    pass
            self._status.showMessage(
                f"已加载 {self._stl_path.name} | "
                f"{self._actual_layers} 层 | "
                f"预览失败（包围盒回退）"
            )

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
        self._apply_debug_solver_toggles(config)
        lt_m = self._layer_thickness_m
        lt_mm = self._layer_thickness_mm
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
                    f"分辨率: {user_res_mm:.3f} mm"
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
                    print_z_axis=self._print_z_axis(),
                )
                try:
                    mesh, actual_layers = mesher.build_layered_mesh(
                        config, algo_type=algo_type
                    )
                except Exception as mesh_exc:
                    if not _should_retry_standard_meshing(
                        algo_type=algo_type,
                        is_step=is_step,
                        actual_layers=self._actual_layers,
                        exc=mesh_exc,
                    ):
                        raise
                    self._log.append_log(
                        "  ! layered STEP meshing failed for fine slices; "
                        "retrying standard unstructured mesh"
                    )
                    mesh, actual_layers = mesher.build_layered_mesh(
                        config, algo_type="standard"
                    )
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
            self._log.append_log(f"  · initial mesh {_format_bbox_mm(mesh.vertices)}")
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
    def _apply_solver_diagnostics_env_for_run(self) -> None:
        """Apply the GUI diagnostics checkbox to this simulation run."""
        if self._solver_diag_env_backup is None:
            self._solver_diag_env_backup = {
                "HYDROGEL_VBD_SOLVER_DIAG": os.environ.get(
                    "HYDROGEL_VBD_SOLVER_DIAG"
                )
            }
        enabled = (
            self._chk_solver_diag is not None
            and self._chk_solver_diag.isChecked()
        )
        if enabled:
            os.environ["HYDROGEL_VBD_SOLVER_DIAG"] = "1"
        else:
            os.environ.pop("HYDROGEL_VBD_SOLVER_DIAG", None)

    def _restore_solver_diagnostics_env_after_run(self) -> None:
        """Restore diagnostics-related environment variables after a run."""
        if self._solver_diag_env_backup is None:
            return
        for key, value in self._solver_diag_env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._solver_diag_env_backup = None

    def _on_run(self) -> None:
        """运行仿真按钮槽函数 —— 异步 Worker 线程版。

        创建 ``SimulationWorker`` 实例并移入 ``QThread`` 执行。
        所有密集的 VBD 物理计算在 Worker 线程中完成，主线程仅负责
        接收信号并刷新 3D 渲染，从根本上消除"未响应"问题。
        """
        # ── 防御性编程：检查 self._generated_mesh 数据完整性 ──
        if self._generated_mesh is None:
            self._log.append_log(
                "<span style='color:orange;font-weight:bold;'>"
                "[错误] 网格数据为空，请先点击「划分网格」</span>"
            )
            QtWidgets.QMessageBox.warning(
                self, "网格数据缺失",
                "错误：网格数据为空，请先点击【划分网格】生成网格后再运行仿真。"
            )
            return

        # ── 深拷贝关键数据，防止 get_config() 触发的 _on_params 清空 self._generated_mesh ──
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
            f"层厚: {self._layer_thickness_mm:.2f} mm | "
            f"顶点: {len(saved_generated_mesh.vertices)} | "
            f"四面体: {len(saved_generated_mesh.tets)}"
        )
        # ── 停止之前可能还在运行的动画回放定时器 ──
        self._stop_animation_timer()
        self.animation_frames.clear()
        self._anim_tets = None
        self.current_frame_idx = 0

        self._status.showMessage("仿真运行中 …")

        # get_config() 会触发 _on_params，其内部可能清空 self._generated_mesh
        config = self._param_panel.get_config()
        self._apply_debug_solver_toggles(config)
        solver_diag_enabled = (
            self._chk_solver_diag is not None
            and self._chk_solver_diag.isChecked()
        )
        field_debug_enabled = (
            self._chk_field_debug is not None
            and self._chk_field_debug.isChecked()
        )

        # ── 恢复网格数据，确保 Worker 构造函数获取到完整 MeshState ──
        self._generated_mesh = saved_generated_mesh
        self._actual_layers = saved_actual_layers

        # ── C++ 求解器可用性检查 / 自动编译 ──
        _user_wants_cpp = (
            self._chk_use_cpp is not None and self._chk_use_cpp.isChecked()
        )
        _cpp_available = is_cpp_available()
        _cpp_ready = _user_wants_cpp and _cpp_available and not field_debug_enabled
        _field_debug_cpp_ready = (
            _user_wants_cpp and _cpp_available and field_debug_enabled
        )
        if _user_wants_cpp and not _cpp_available:
            builder = CppBuilder()
            if builder.pyd_exists():
                self._log.append_log("  [build] 检测到旧版 .pyd，重新编译 ...")
            else:
                self._log.append_log("  [build] C++ 求解器未编译，开始自动编译 ...")

            missing = builder.check_prerequisites()
            if missing:
                self._log.append_log(
                    f"  [build] 缺少依赖: {', '.join(missing)}"
                )
                self._log.append_log("  [build] 回退到 Python 参考求解器")
            else:
                self._log.append_log("  [build] 前置条件就绪，正在后台编译 ...")
                self._start_cpp_build(builder, config)
                return
        if field_debug_enabled and _field_debug_cpp_ready:
            self._log.append_log(
                "  [field-debug] 电场调试对比将使用直接 C++ adapter；本次不走 C++ 子进程隔离"
            )
        if _cpp_ready:
            self._log.append_log("  [info] 使用 C++ 加速求解器")
        elif _field_debug_cpp_ready:
            self._log.append_log("  [info] 使用 Python 控制循环 + C++ adapter 分支求解")
        elif _user_wants_cpp and not _cpp_available:
            pass  # 已在上方打印回退消息
        else:
            self._log.append_log("  [info] 使用 Python 参考求解器（用户选择）")

        self._log.append_log(
            f"  [debug] enable_czm={config.enable_czm}, "
            f"rho_cheb={config.rho_cheb}"
        )

        output_dir = (Path(__file__).resolve().parents[3] / "outputs" / "gui")

        self._apply_solver_diagnostics_env_for_run()
        if solver_diag_enabled:
            diag_csv_path = output_dir / "reports" / "solver_diagnostics.csv"
            self._log.append_log(
                f"  [diag] 输出求解器诊断 CSV: {diag_csv_path}"
            )
        if field_debug_enabled:
            self._log.append_log(
                "  [field-debug] 已开启每层 no-field / with-field 指标对比"
            )

        try:
            # ── 创建异步 Worker 并移入 QThread ──
            self._worker = SimulationWorker(
                mesh=self._generated_mesh,
                config=config,
                n_layers=self._actual_layers,
                output_dir=output_dir,
                use_cpp=_cpp_ready,
                solver_diagnostics_enabled=solver_diag_enabled,
                field_debug_enabled=field_debug_enabled,
                field_debug_use_cpp=_field_debug_cpp_ready,
            )
            self._thread = QtCore.QThread(self)

            self._worker.moveToThread(self._thread)

            # ── 连接 7 个信号 ──
            self._worker.frame_ready.connect(self._on_worker_frame)
            self._worker.progress_update.connect(self._on_worker_progress)
            self._worker.log_message.connect(self._on_worker_log)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker.cancelled.connect(self._on_worker_cancelled)
            self._worker.error.connect(self._on_worker_error)
            self._worker.sub_progress.connect(self._on_worker_sub_progress)
            self._worker.layer_finished.connect(self._on_worker_layer_finished)

            # ── 线程生命周期管理 ──
            self._thread.started.connect(self._worker.run)
            self._worker.finished.connect(self._thread.quit)
            self._worker.cancelled.connect(self._thread.quit)
            self._worker.error.connect(self._thread.quit)
            self._worker.finished.connect(self._worker.deleteLater)
            self._worker.cancelled.connect(self._worker.deleteLater)
            self._thread.finished.connect(self._thread.deleteLater)
            self._thread.finished.connect(
                lambda: setattr(self, "_thread", None)
            )
            self._thread.finished.connect(
                lambda: setattr(self, "_worker", None)
            )

            # ── 仿真期间切换运行 / 中断按钮 ──
            self._simulation_stop_requested = False
            self._set_simulation_running(True)

            self._thread.start()

        except Exception as exc:
            self._restore_solver_diagnostics_env_after_run()
            import traceback
            self._log.append_log(f"\n[错误] 启动仿真线程失败: {exc}")
            self._log.append_log(traceback.format_exc())
            self._status.showMessage("仿真启动失败 ✗")

    def _on_stop_simulation(self) -> None:
        """Request the active simulation worker to stop as soon as possible."""
        worker = self._worker
        if worker is None:
            return

        self._simulation_stop_requested = True
        self._btn_stop_sim.setEnabled(False)
        self._btn_stop_sim.setText("正在中断...")
        self._btn_run.setText("中断中...")
        self._status.showMessage("正在中断仿真 …")
        self._log.append_log("  ⏹ 已请求中断仿真，正在停止求解器...")

        worker.request_stop()
        if self._thread is not None:
            self._thread.requestInterruption()

    # ========================================================================
    # C++ 自动编译（后台线程）
    # ========================================================================
    def _start_cpp_build(
        self, builder: "CppBuilder", config: "SimulationConfig"
    ) -> None:
        """在后台线程启动 C++ 编译器。"""
        self._pending_build_config = config
        self._btn_run.setEnabled(False)
        self._btn_run.setText("编译 C++...")
        self._status.showMessage("正在编译 C++ 求解器 …")

        self._build_worker = _BuildWorker(builder)
        self._build_thread = QtCore.QThread(self)
        self._build_worker.moveToThread(self._build_thread)

        self._build_thread.started.connect(self._build_worker.run)
        self._build_worker.build_finished.connect(self._on_cpp_build_finished)
        self._build_worker.build_output.connect(self._log.append_log)
        self._build_worker.build_finished.connect(self._build_thread.quit)

        self._build_thread.start()

    def _on_cpp_build_finished(self, result: "CppBuildResult") -> None:
        """编译线程完成回调（在主线程中执行）。"""
        if self._build_thread is not None:
            self._build_thread.wait(2000)
            self._build_thread = None
        self._build_worker = None

        config = self._pending_build_config
        self._pending_build_config = None

        self._btn_run.setEnabled(True)

        if result.success:
            self._log.append_log("  [build] [OK] 编译成功，加载 C++ 求解器")
            self._status.showMessage("C++ 求解器就绪")
            refresh_availability()
        else:
            self._log.append_log(
                "  [build] [FAIL] 编译失败，回退到 Python 参考求解器"
            )
            self._status.showMessage("编译失败，使用 Python 求解器")

        # 编译成功后，重新触发仿真（递归调用 _on_run）
        # config 仍然有效，网格未变，直接创建 Worker
        if config is not None:
            # 手动再次触发 _on_run（此时 C++ 已可用或已回退）
            self._on_run()

    # ========================================================================
    # Worker 信号槽（主线程 UI 更新入口）
    # ========================================================================

    @QtCore.Slot(dict)
    def _on_worker_frame(self, payload: dict) -> None:
        """接收 Worker 的 ``frame_ready`` 信号，更新 3D 视图。

        同时将帧数据缓存到 ``animation_frames``，供仿真结束后的
        动画回放使用。

        **DVR 时间轴门控**：当用户正在拖动滑块（``_dv_is_slider_down=True``）
        时，跳过实时 3D 渲染推送，仅保留帧缓存写入。避免拖动冲突。

        Parameters
        ----------
        payload : dict
            ``SimulationWorker.frame_ready`` 信号携带的帧数据，
            包含 vertices / tets / active_mask / title 四个键。
        """
        vertices: np.ndarray = payload["vertices"]
        tets: np.ndarray | None = payload.get("tets")
        active_mask: np.ndarray = payload["active_mask"]
        active_tet_mask: np.ndarray | None = payload.get("active_tet_mask")
        title: str = payload.get("title", "")

        if self._anim_tets is None and tets is not None:
            self._anim_tets = tets.copy()

        # ── DVR 滑块拖动门控：拖动期间抑制实时渲染 ──
        if not self._dv_is_slider_down and not self._is_rendering_worker_frame:
            self._is_rendering_worker_frame = True
            try:
                self._viewer.show_deformed_mesh(
                    vertices,
                    tets,
                    active_mask,
                    active_tet_mask=active_tet_mask,
                    title=title,
                )
            finally:
                self._is_rendering_worker_frame = False

        # ── 缓存动画帧 ──
        lid = -1
        try:
            import re
            m = re.search(r"第\s*(\d+)/", title)
            if m:
                lid = int(m.group(1)) - 1
        except Exception:
            pass

        lid = _frame_layer_id_from_title(title)
        self.animation_frames.append({
            "vertices": vertices.copy(),
            "active_mask": active_mask.copy(),
            "active_tet_mask": (
                active_tet_mask.copy() if active_tet_mask is not None else None
            ),
            "title": title,
            "layer_id": lid,
        })

    @QtCore.Slot(int, int, int, int)
    def _on_worker_progress(
        self, layer: int, n_layers: int, step: int, iteration: int
    ) -> None:
        """接收 Worker 的 ``progress_update`` 信号，更新进度条。

        Parameters
        ----------
        layer : int
            当前层序号（从 1 开始）。
        n_layers : int
            总层数。
        step : int
            当前步数计数器。
        iteration : int
            当前 VBD 迭代次数。
        """
        self._progress.set_layer(layer, n_layers)
        self._status.showMessage(
            f"仿真运行中 — 第 {layer}/{n_layers} 层 "
            f"(步 {step}, 迭代 {iteration})"
        )

    @QtCore.Slot(str)
    def _on_worker_log(self, message: str) -> None:
        """接收 Worker 的 ``log_message`` 信号，追加到日志面板。

        Parameters
        ----------
        message : str
            日志消息字符串（支持 HTML 富文本）。
        """
        self._log.append_log(message)

    @QtCore.Slot(list)
    def _on_worker_finished(self, results: list[LayerResult]) -> None:
        """接收 Worker 的 ``finished`` 信号，执行后处理。

        恢复运行按钮、清除线程引用、调用 ``_on_finish`` 生成汇总
        和启动动画回放。

        Parameters
        ----------
        results : list[LayerResult]
            仿真结果列表（来自 Worker）。
        """
        self._last_results = results
        self._restore_solver_diagnostics_env_after_run()

        # ── 恢复运行 / 中断按钮 ──
        self._set_simulation_running(False)

        # ── 若当前正在动画回放，先停止 ──
        self._stop_animation_timer()

        # ── 设置动画帧数据（Worker 已通过 frame_ready 填充）──
        if self._anim_tets is None and results:
            self._anim_tets = (
                self._generated_mesh.tets if self._generated_mesh else None
            )

        # ── 触发汇总逻辑 ──
        self._on_finish(results)

    @QtCore.Slot(list)
    def _on_worker_cancelled(self, results: list[LayerResult]) -> None:
        """Handle a user-requested simulation interruption."""
        self._last_results = results
        self._restore_solver_diagnostics_env_after_run()
        self._set_simulation_running(False)
        self._stop_animation_timer()
        self._progress.set_sub_progress(0)
        self._log.append_log("\n[已中断] 仿真已由用户手动停止")
        self._status.showMessage(f"仿真已中断 | 已完成 {len(results)} 层")

    @QtCore.Slot(str)
    def _on_worker_error(self, error_msg: str) -> None:
        """接收 Worker 的 ``error`` 信号，显示错误信息。

        Parameters
        ----------
        error_msg : str
            错误信息和回溯字符串。
        """
        self._restore_solver_diagnostics_env_after_run()
        self._set_simulation_running(False)
        self._log.append_log(f"\n[严重错误] {error_msg}")
        self._status.showMessage("仿真线程崩溃 ✗")
        QtWidgets.QMessageBox.critical(
            self, "仿真线程错误",
            f"仿真线程发生不可恢复的错误:\n\n{error_msg}",
        )

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
        播放启动时显示动画播放控制面板。
        """
        if not self.animation_frames:
            self._log.append_log(
                "  ⚠ 无动画帧可播放（animation_frames 为空）"
            )
            return

        # 停止已有的定时器（如果正在播放）
        self._stop_animation_timer()
        self._anim_paused = False
        self._anim_layer_filter = None
        self.current_frame_idx = 0

        # ── 填充层筛选下拉框 ──
        if self._anim_layer_combo is not None:
            self._anim_layer_combo.blockSignals(True)
            self._anim_layer_combo.clear()
            self._anim_layer_combo.addItem("全部层", -1)
            unique_layers = sorted({
                f["layer_id"]
                for f in self.animation_frames
                if "layer_id" in f
            })
            for lid in unique_layers:
                self._anim_layer_combo.addItem(f"第 {lid + 1} 层", lid)
            self._anim_layer_combo.setCurrentIndex(0)
            self._anim_layer_combo.blockSignals(False)

        # ── 重建帧索引映射 ──
        self._rebuild_frame_indices()

        # ── 显示控制面板 ──
        self._set_anim_panel_visible(True)

        # 创建定时器：约 33 ms → ~30 FPS
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._anim_timer.timeout.connect(self._update_anim_frame)
        self._anim_timer.start(33)

        # 更新顶部栏的停止回放按钮
        self._btn_stop_anim.setVisible(True)

        self._log.append_log(
            "  ▶ 动画回放开始 (循环播放, ~30 FPS, "
            f"{len(self.animation_frames)} 帧, "
            f"筛选后 {len(self._frame_indices)} 帧)"
        )
        self._status.showMessage(
            f"动画回放中 … [{len(self._frame_indices)} 帧]"
        )

    def _set_anim_panel_visible(self, visible: bool) -> None:
        """统一控制动画面板所有控件的可见性。"""
        if self._btn_anim_play is not None:
            self._btn_anim_play.setVisible(visible)
        if self._btn_anim_pause is not None:
            self._btn_anim_pause.setVisible(visible)
        if hasattr(self, '_btn_stop_anim_from_panel') and self._btn_stop_anim_from_panel is not None:
            self._btn_stop_anim_from_panel.setVisible(visible)
        if self._anim_frame_label is not None:
            self._anim_frame_label.setVisible(visible)
        if self._anim_layer_combo is not None:
            self._anim_layer_combo.setVisible(visible)
        if self._anim_slider is not None:
            self._anim_slider.setVisible(visible)

    def _rebuild_frame_indices(self) -> None:
        """根据当前层筛选重建帧索引映射列表。

        若 ``_anim_layer_filter`` 为 None，包含全部帧；
        否则仅保留 ``layer_id`` 匹配的帧。
        同时更新 Slider 范围和显示。
        """
        self._frame_indices.clear()
        for i, f in enumerate(self.animation_frames):
            lid = f.get("layer_id")
            if self._anim_layer_filter is None or lid == self._anim_layer_filter:
                self._frame_indices.append(i)

        # 更新 Slider 范围
        n_filt = len(self._frame_indices)
        if self._anim_slider is not None:
            self._anim_slider.blockSignals(True)
            self._anim_slider.setRange(0, max(n_filt - 1, 0))
            self._anim_slider.setValue(min(self.current_frame_idx, max(n_filt - 1, 0)))
            self._anim_slider.blockSignals(False)

        # 钳制当前帧索引
        if self.current_frame_idx >= n_filt:
            self.current_frame_idx = 0

        # 更新帧标签
        if self._anim_frame_label is not None:
            total = n_filt
            self._anim_frame_label.setText(
                f"帧: {self.current_frame_idx + 1} / {total}"
            )

    def _update_anim_frame(self) -> None:
        """逐帧渲染动画：每次定时器超时时调用。

        按 ``_frame_indices`` 筛选后的顺序读取帧数据，
        调用 ``_viewer.show_deformed_mesh`` 刷新 3D 视图。
        播放到最后一帧后自动回到第 0 帧实现无限循环。
        同步更新 Slider 位置和帧标签。
        """
        if not self._frame_indices or self._anim_tets is None:
            self._stop_animation_timer()
            return

        if self._anim_paused:
            return

        n_filt = len(self._frame_indices)
        if n_filt == 0:
            return

        real_idx = self._frame_indices[self.current_frame_idx]
        if real_idx >= len(self.animation_frames):
            real_idx = 0

        frame = self.animation_frames[real_idx]

        self._viewer.show_deformed_mesh(
            frame["vertices"],
            self._anim_tets,
            frame["active_mask"],
            active_tet_mask=frame.get("active_tet_mask"),
            title=f"🔁 回放 — {frame['title']}",
        )

        # 更新 Slider 和帧标签（双向绑定中的主动方向）
        if self._anim_slider is not None:
            self._anim_slider.blockSignals(True)
            self._anim_slider.setValue(self.current_frame_idx)
            self._anim_slider.blockSignals(False)

        if self._anim_frame_label is not None:
            self._anim_frame_label.setText(
                f"帧: {self.current_frame_idx + 1} / {n_filt}"
            )

        # 循环播放：到达末尾后回到第 0 帧
        self.current_frame_idx += 1
        if self.current_frame_idx >= n_filt:
            self.current_frame_idx = 0

    # ── 动画控制面板信号槽 ──

    def _on_anim_play_clicked(self) -> None:
        """「播放」按钮：从头开始播放动画。"""
        if not self.animation_frames:
            return
        self._anim_paused = False
        self.current_frame_idx = 0
        if self._anim_slider is not None:
            self._anim_slider.setValue(0)

    def _on_anim_pause_clicked(self) -> None:
        """「暂停」按钮：切换暂停/继续状态。"""
        self._anim_paused = not self._anim_paused
        if self._btn_anim_pause is not None:
            self._btn_anim_pause.setText("继续" if self._anim_paused else "暂停")
            self._btn_anim_pause.setToolTip(
                "继续动画回放" if self._anim_paused else "暂停动画回放"
            )

    def _on_anim_layer_filter_changed(self, _index: int) -> None:
        """层筛选下拉框变更：重建帧索引并重置到第 0 帧。"""
        if self._anim_layer_combo is None:
            return
        data = self._anim_layer_combo.currentData()
        if data == -1:
            self._anim_layer_filter = None
        else:
            self._anim_layer_filter = int(data)
        self.current_frame_idx = 0
        self._rebuild_frame_indices()

    def _on_anim_slider_changed(self, value: int) -> None:
        """Slider 值变更：跳转到指定筛选帧（用户拖动时触发）。

        注意：blockSignals 用于防止 _update_anim_frame 更新
        Slider 时触发递归调用。
        """
        if self._anim_slider is None:
            return
        # 仅当信号未被阻塞时处理（即用户主动拖动）
        if self._anim_slider.signalsBlocked():
            return
        n_filt = len(self._frame_indices)
        if 0 <= value < n_filt:
            self.current_frame_idx = value
            # 立即渲染该帧
            real_idx = self._frame_indices[value]
            if real_idx < len(self.animation_frames):
                frame = self.animation_frames[real_idx]
                if self._anim_tets is not None:
                    self._viewer.show_deformed_mesh(
                        frame["vertices"],
                        self._anim_tets,
                        frame["active_mask"],
                        active_tet_mask=frame.get("active_tet_mask"),
                        title=f"🔁 回放 — {frame['title']}",
                    )
            if self._anim_frame_label is not None:
                self._anim_frame_label.setText(
                    f"帧: {value + 1} / {n_filt}"
                )

    def _on_stop_animation(self) -> None:
        """手动停止动画回放，恢复初始网格视图。

        停止定时器，清空动画帧缓存，隐藏停止按钮和控制面板，
        并将 3D 视图恢复为划分网格后的初始状态。
        """
        # 停止定时器
        self._stop_animation_timer()

        # 清空动画帧
        self.animation_frames.clear()
        self._anim_tets = None
        self.current_frame_idx = 0
        self._frame_indices.clear()
        self._anim_paused = False
        self._anim_layer_filter = None

        # 隐藏停止按钮
        self._btn_stop_anim.setVisible(False)
        # 隐藏面板控件
        self._set_anim_panel_visible(False)

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
        """仿真完成汇总，并弹出电场-误差分析窗口。

        Parameters
        ----------
        results : list[LayerResult]
            全部层的仿真结果。
        """
        self._progress.set_done()
        success_count = sum(1 for r in results if r.success)
        plot_data = ElectricFieldPlotWindow._build_plot_data(results)
        self._log.append_log("\n===== 仿真结束 =====")
        if plot_data.mode == "shape":
            primary = plot_data.primary[np.isfinite(plot_data.primary)]
            secondary = plot_data.secondary[np.isfinite(plot_data.secondary)]
            max_e = float(np.max(primary)) if primary.size else 0.0
            rms_e = (
                float(np.sqrt(np.mean(secondary**2))) if secondary.size else 0.0
            )
            self._status.showMessage(
                f"完成 ✓ | 层数: {len(results)} | "
                f"最大形状误差: {max_e:.4e} m | RMS: {rms_e:.4e} m | "
                f"成功层: {success_count}/{len(results)}"
            )
            self._log.append_log(
                f"汇总: 共 {len(results)} 层, "
                f"最大形状误差 {max_e:.4e} m, RMS {rms_e:.4e} m, "
                f"成功 {success_count}/{len(results)} 层"
            )
        else:
            primary = plot_data.primary[np.isfinite(plot_data.primary)]
            secondary = plot_data.secondary[np.isfinite(plot_data.secondary)]
            max_dx = float(np.max(primary)) if primary.size else 0.0
            avg_call_ms = float(np.mean(secondary)) if secondary.size else 0.0
            self._status.showMessage(
                f"完成 ✓ | 层数: {len(results)} | "
                f"solver max_dx: {max_dx:.4e} m | "
                f"avg call: {avg_call_ms:.2f} ms | "
                f"成功层: {success_count}/{len(results)}"
            )
            self._log.append_log(
                f"汇总: 共 {len(results)} 层, "
                f"solver max_dx {max_dx:.4e} m, "
                f"平均单步 {avg_call_ms:.2f} ms, "
                f"成功 {success_count}/{len(results)} 层"
            )

        # ── 弹出电场-误差分析窗口 ──
        if _HAS_MPL and results:
            try:
                analysis_win = ElectricFieldPlotWindow(results, self)
                analysis_win.setWindowModality(QtCore.Qt.WindowModality.NonModal)
                analysis_win.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
                analysis_win.show()
                self._log.append_log("  📊 已弹出电场-误差分析图窗")
            except Exception as exc:
                self._log.append_log(f"  ⚠ 分析图窗创建失败: {exc}")
        elif not _HAS_MPL:
            self._log.append_log(
                "  ⚠ matplotlib 未安装，跳过分析图窗。"
                " 请运行: pip install matplotlib"
            )


    # ========================================================================
    # DVR 实时时间轴槽函数
    # ========================================================================

    @QtCore.Slot(int, int, int)
    def _on_worker_sub_progress(
        self, layer_idx: int, percentage: int, step_count: int
    ) -> None:
        """接收 Worker 的 ``sub_progress`` 信号，更新细粒度子进度条。

        Parameters
        ----------
        layer_idx : int
            当前层序号（从 0 开始）。
        percentage : int
            0–100 的整数，当前层内提升进度。
        step_count : int
            总步数计数器。
        """
        # ── 更新绿色子进度条 ──
        self._progress.set_sub_progress(percentage)

        # ── 更新 DVR 时间轴标签 ──
        if self._dv_label is not None:
            self._dv_label.setText(
                f"层 {layer_idx + 1}/{self._actual_layers} — {percentage}%"
            )

    @QtCore.Slot(object)
    def _on_worker_layer_finished(self, result: "LayerResult") -> None:
        """接收 Worker 的 ``layer_finished`` 信号。

        每层完成后更新 DVR 时间轴滑块范围和标签，
        并实时绘制电场-误差迷你图表。

        Parameters
        ----------
        result : LayerResult
            当前层的结果对象。
        """
        layer_id = int(getattr(result, "layer_id", 0))
        total_layers = self._actual_layers or 1

        # ── 更新 DVR 滑块范围 ──
        if self._dv_slider is not None:
            self._dv_slider.blockSignals(True)
            self._dv_slider.setRange(0, total_layers - 1)
            self._dv_slider.setValue(layer_id)
            self._dv_slider.blockSignals(False)

        # ── 实时电场数据采集 ──
        e_z = (
            result.error_metrics.get("E_z", 0.0)
            if hasattr(result, "error_metrics") and result.error_metrics
            else 0.0
        )
        max_err = getattr(result, "max_deformation", 0.0)
        rms_err = getattr(result, "rms_error", 0.0)
        self._dv_efield_layer_data.append(
            (layer_id, float(e_z), float(max_err), float(rms_err))
        )

        # ── 绘制迷你电场图表 ──
        self._draw_dv_efield_chart()

    def _draw_dv_efield_chart(self) -> None:
        """在迷你 matplotlib 嵌板中实时绘制 E_z 逐层折线图。

        仅在 ``matplotlib`` 可用且嵌板已创建时执行。
        """
        if self._dv_efield_fig is None or self._dv_efield_canvas is None:
            return
        if not self._dv_efield_layer_data:
            return

        fig = self._dv_efield_fig
        fig.clear()

        ax = fig.add_subplot(1, 1, 1)
        layer_ids = [d[0] for d in self._dv_efield_layer_data]
        e_z_vals = [d[1] for d in self._dv_efield_layer_data]

        ax.plot(
            layer_ids, e_z_vals,
            "b-o", markersize=3, linewidth=1.2,
            label="E_z (V/m)",
        )
        ax.set_xlabel("层", fontsize=7)
        ax.set_ylabel("E_z", fontsize=7, color="b")
        ax.tick_params(axis="both", labelsize=6)
        ax.set_title("实时 E_z 逐层", fontsize=8)
        ax.grid(True, alpha=0.25)

        fig.tight_layout(pad=0.8)
        fig.canvas.draw_idle()

    # ── DVR 滑块交互槽 ──

    def _on_dv_slider_pressed(self) -> None:
        """用户开始拖动 DVR 时间轴滑块。"""
        self._dv_is_slider_down = True

    def _on_dv_slider_released(self) -> None:
        """用户释放 DVR 时间轴滑块。

        关闭门控并立即渲染当前滑块位置对应的最新帧缓存。
        """
        self._dv_is_slider_down = False

        # ── 查找滑块位置 (layer_id) 对应的最新帧 ──
        if self._dv_slider is None:
            return
        target_layer = self._dv_slider.value()
        self._render_dv_layer_frame(target_layer)

    def _on_dv_slider_changed(self, value: int) -> None:
        """DVR 滑块值变化时更新标签。"""
        if self._dv_label is not None:
            self._dv_label.setText(
                f"查看: 第 {value + 1}/{self._actual_layers} 层"
            )

    def _render_dv_layer_frame(self, target_layer: int) -> None:
        """从 ``animation_frames`` 缓存中查找目标层的最后一帧并渲染。

        Parameters
        ----------
        target_layer : int
            目标层序号（从 0 开始）。
        """
        # ── 逆序查找目标层的最新帧 ──
        best_idx = -1
        for i in range(len(self.animation_frames) - 1, -1, -1):
            lid = self.animation_frames[i].get("layer_id", -1)
            if lid == target_layer:
                best_idx = i
                break

        if best_idx < 0:
            # ── 向前顺向查找最近的一帧 ──
            for i in range(len(self.animation_frames)):
                lid = self.animation_frames[i].get("layer_id", -1)
                if lid >= 0 and lid <= target_layer:
                    best_idx = i
                else:
                    break
            if best_idx < 0:
                return

        frame = self.animation_frames[best_idx]
        tets = self._anim_tets if self._anim_tets is not None else frame.get("tets")
        self._viewer.show_deformed_mesh(
            frame["vertices"],
            tets,
            frame["active_mask"],
            active_tet_mask=frame.get("active_tet_mask"),
            title=(
                f"📽 DVR 定位 — 第 {target_layer + 1} 层"
                f" — {frame['title']}"
            ),
        )

class ElectricFieldPlotWindow(QtWidgets.QDialog):
    """电场-误差追踪分析窗口。

    仿真完成后弹出，嵌入 matplotlib 双面板图表：
    - **左上**：E_z 电场强度 vs 层数折线图（右侧 y 轴为累计误差百分比填充）
    - **右下**：max_error 与 rms_error 双线对比图

    支持 PNG / SVG 格式保存到 outputs/gui/reports/ 目录。

    Parameters
    ----------
    results : list[LayerResult]
        各层仿真结果列表。
    parent : QWidget or None
        父窗口。
    """

    _DEFAULT_SAVE_DIR = Path("outputs/gui/reports")

    def __init__(
        self,
        results: list[LayerResult],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._results = results
        self.setWindowTitle("电场-误差追踪分析")
        self.resize(800, 500)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── 创建 matplotlib Figure 和 Canvas ──
        fig = Figure(figsize=(8, 4.5), dpi=100)
        self._fig = fig
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # ── 保存按钮行 ──
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        btn_save_png = QtWidgets.QPushButton("💾 保存 PNG")
        btn_save_png.setFixedHeight(28)
        btn_save_png.clicked.connect(lambda: self._save_figure("png"))
        btn_layout.addWidget(btn_save_png)

        btn_save_svg = QtWidgets.QPushButton("💾 保存 SVG")
        btn_save_svg.setFixedHeight(28)
        btn_save_svg.clicked.connect(lambda: self._save_figure("svg"))
        btn_layout.addWidget(btn_save_svg)

        layout.addLayout(btn_layout)

        # ── 绑制图表 ──
        self._draw_charts()

    # ────────────────────────────────────────────────────────
    # 数据提取
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _metrics(result: LayerResult) -> dict[str, float]:
        """安全取得 ``LayerResult.error_metrics``。"""
        metrics = getattr(result, "error_metrics", None)
        return metrics if isinstance(metrics, dict) else {}

    @staticmethod
    def _metric_value(
        result: LayerResult,
        keys: tuple[str, ...],
        fallback: float,
    ) -> float:
        """按优先级从 ``error_metrics`` 中取浮点值。"""
        metrics = ElectricFieldPlotWindow._metrics(result)
        for key in keys:
            value = metrics.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return float(fallback)

    @staticmethod
    def _build_plot_data(results: list[LayerResult]) -> _ElectricFieldPlotData:
        """将 LayerResult 转成报告图数据，并区分 shape / solver 指标。"""
        n = len(results)
        layers = np.arange(n, dtype=int)
        e_z_vals = np.array(
            [
                ElectricFieldPlotWindow._metric_value(r, ("E_z",), 0.0)
                for r in results
            ],
            dtype=float,
        )

        def _has_shape_metrics(result: LayerResult) -> bool:
            metrics = ElectricFieldPlotWindow._metrics(result)
            explicit = metrics.get("shape_error_available")
            if explicit is not None:
                try:
                    return float(explicit) > 0.0
                except (TypeError, ValueError):
                    return False
            return (
                "shape_max_error" in metrics
                or "shape_rms_error" in metrics
                or "max_error" in metrics
            )

        has_shape_metrics = any(_has_shape_metrics(r) for r in results)
        if has_shape_metrics:
            max_err = np.array(
                [
                    ElectricFieldPlotWindow._metric_value(
                        r, ("shape_max_error", "max_error"),
                        getattr(r, "max_deformation", np.nan),
                    )
                    for r in results
                ],
                dtype=float,
            )
            rms_err = np.array(
                [
                    ElectricFieldPlotWindow._metric_value(
                        r, ("shape_rms_error", "rms_error"),
                        getattr(r, "rms_error", np.nan),
                    )
                    for r in results
                ],
                dtype=float,
            )
            has_field_debug = any(
                "field_no_field_rms" in ElectricFieldPlotWindow._metrics(r)
                and "field_with_field_rms" in ElectricFieldPlotWindow._metrics(r)
                for r in results
            )
            if has_field_debug:
                aux_values: list[float] = []
                guard_passed: list[float] = []
                for result in results:
                    no_field_rms = ElectricFieldPlotWindow._metric_value(
                        result, ("field_no_field_rms",), np.nan
                    )
                    with_field_rms = ElectricFieldPlotWindow._metric_value(
                        result, ("field_with_field_rms",), np.nan
                    )
                    if (
                        np.isfinite(no_field_rms)
                        and np.isfinite(with_field_rms)
                        and no_field_rms > 1.0e-15
                    ):
                        aux_values.append(
                            (no_field_rms - with_field_rms)
                            / no_field_rms
                            * 100.0
                        )
                    else:
                        aux_values.append(np.nan)
                    guard_passed.append(
                        ElectricFieldPlotWindow._metric_value(
                            result, ("field_guard_passed",), np.nan
                        )
                    )
                aux_pct = np.array(aux_values, dtype=float)
                aux_label = "RMS 改善率 (%)"
                guard_status = np.array(guard_passed, dtype=float)
            else:
                cum_source = np.nan_to_num(
                    max_err, nan=0.0, posinf=0.0, neginf=0.0
                )
                cum_sum = np.cumsum(cum_source)
                total = cum_sum[-1] if cum_sum.size and cum_sum[-1] > 0 else 1.0
                aux_pct = cum_sum / total * 100.0
                aux_label = "累计误差 (%)"
                guard_status = None
            return _ElectricFieldPlotData(
                layers=layers,
                e_z=e_z_vals,
                primary=max_err,
                secondary=rms_err,
                aux_pct=aux_pct,
                mode="shape",
                primary_label="max_error (m)",
                secondary_label="rms_error (m)",
                aux_label=aux_label,
                primary_title="形状误差对比",
                primary_ylabel="误差 (m)",
                guard_passed=guard_status,
            )

        solver_max_dx = np.array(
            [
                ElectricFieldPlotWindow._metric_value(
                    r, ("solver_final_max_dx", "solver_max_dx", "final_max_dx"),
                    getattr(r, "max_deformation", np.nan),
                )
                for r in results
            ],
            dtype=float,
        )
        avg_call_ms = np.array(
            [
                ElectricFieldPlotWindow._metric_value(
                    r, ("solver_avg_call_ms",), np.nan
                )
                for r in results
            ],
            dtype=float,
        )
        max_iter_pct: list[float] = []
        for result in results:
            metrics = ElectricFieldPlotWindow._metrics(result)
            pct = metrics.get("solver_max_iter_hit_pct")
            if pct is None:
                hits = float(metrics.get("solver_max_iter_hits", 0.0))
                steps = float(metrics.get("solver_total_steps", 0.0))
                pct = (hits / steps * 100.0) if steps > 0.0 else 0.0
            max_iter_pct.append(float(pct))

        return _ElectricFieldPlotData(
            layers=layers,
            e_z=e_z_vals,
            primary=solver_max_dx,
            secondary=avg_call_ms,
            aux_pct=np.array(max_iter_pct, dtype=float),
            mode="solver",
            primary_label="solver max_dx (m)",
            secondary_label="avg call (ms)",
            aux_label="max_iter 命中率 (%)",
            primary_title="求解器诊断",
            primary_ylabel="solver max_dx (m)",
        )

    def _extract_data(self) -> _ElectricFieldPlotData:
        """从结果列表中提取绘制所需的数据列。"""
        return self._build_plot_data(self._results)

    # ────────────────────────────────────────────────────────
    # 绘图
    # ────────────────────────────────────────────────────────

    def _draw_charts(self) -> None:
        """绑制双面板图表。

        左上（axes[0]）：E_z 折线图 + 右侧 y 轴累计百分比填充
        右下（axes[1]）：max_error 与 rms_error 双线对比
        """
        fig = self._fig
        fig.clear()

        plot_data = self._extract_data()
        layers = plot_data.layers

        # ── 左上：E_z 电场 ──
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(layers, plot_data.e_z, "b-o", markersize=3, linewidth=1.3, label="E_z (V/m)")
        ax1.set_xlabel("层序号")
        ax1.set_ylabel("E_z (V/m)", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1.set_title("电场强度 E_z 逐层变化")
        ax1.grid(True, alpha=0.3)

        # 右侧 y 轴：shape 模式显示累计误差，solver 模式显示 max_iter 命中率。
        ax1b = ax1.twinx()
        ax1b.fill_between(
            layers,
            plot_data.aux_pct,
            0,
            alpha=0.08,
            color="orange",
            label=plot_data.aux_label,
        )
        ax1b.plot(
            layers,
            plot_data.aux_pct,
            "orange",
            linestyle="--",
            linewidth=1.0,
        )
        if plot_data.guard_passed is not None:
            finite_aux = np.isfinite(plot_data.aux_pct)
            failed = finite_aux & ~(plot_data.guard_passed > 0.5)
            if np.any(failed):
                ax1b.scatter(
                    layers[failed],
                    plot_data.aux_pct[failed],
                    color="red",
                    marker="x",
                    s=28,
                    label="守门失败",
                    zorder=4,
                )
        ax1b.set_ylabel(plot_data.aux_label, color="orange")
        ax1b.tick_params(axis="y", labelcolor="orange")
        if plot_data.guard_passed is None:
            ax1b.set_ylim(0, 105)
        else:
            finite_aux = plot_data.aux_pct[np.isfinite(plot_data.aux_pct)]
            if finite_aux.size:
                lo = min(0.0, float(np.min(finite_aux)))
                hi = max(0.0, float(np.max(finite_aux)))
                span = max(1.0, hi - lo)
                ax1b.set_ylim(lo - span * 0.15, hi + span * 0.15)
            else:
                ax1b.set_ylim(-1.0, 1.0)

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

        # ── 右下：shape error 或 solver diagnostic ──
        ax2 = fig.add_subplot(2, 2, 4)
        ax2.plot(
            layers,
            plot_data.primary,
            "r-s",
            markersize=4,
            linewidth=1.5,
            label=plot_data.primary_label,
        )
        if plot_data.mode == "shape":
            ax2.plot(
                layers,
                plot_data.secondary,
                "g-^",
                markersize=4,
                linewidth=1.5,
                label=plot_data.secondary_label,
            )
        elif np.any(np.isfinite(plot_data.secondary)):
            ax2b = ax2.twinx()
            ax2b.plot(
                layers,
                plot_data.secondary,
                "g-^",
                markersize=4,
                linewidth=1.5,
                label=plot_data.secondary_label,
            )
            ax2b.set_ylabel(plot_data.secondary_label, color="g")
            ax2b.tick_params(axis="y", labelcolor="g")
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2b.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        else:
            ax2.legend(loc="upper left", fontsize=8)
        ax2.set_xlabel("层序号")
        ax2.set_ylabel(plot_data.primary_ylabel)
        ax2.set_title(plot_data.primary_title)
        if plot_data.mode == "shape":
            ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.canvas.draw()

    # ────────────────────────────────────────────────────────
    # 保存
    # ────────────────────────────────────────────────────────

    def _save_figure(self, fmt: str) -> None:
        """将当前图表保存为 PNG 或 SVG 文件。

        Parameters
        ----------
        fmt : str
            文件格式扩展名 ("png" 或 "svg")。
        """
        self._DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        default_name = f"electric_field_error_analysis.{fmt}"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            f"保存图表为 {fmt.upper()}",
            str(self._DEFAULT_SAVE_DIR / default_name),
            f"{fmt.upper()} 文件 (*.{fmt})",
        )
        if not path:
            return

        try:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
            QtWidgets.QMessageBox.information(
                self,
                "保存成功",
                f"图表已保存至:\n{path}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "保存失败",
                f"无法保存图表:\n{exc}",
            )


# ============================================================================
# C++ 编译后台 Worker（QThread 模式）
# ============================================================================
class _BuildWorker(QtCore.QObject):
    """在后台线程中执行 C++ 编译的 worker 对象。"""

    build_output = QtCore.Signal(str)
    build_finished = QtCore.Signal(CppBuildResult)

    def __init__(self, builder: "CppBuilder") -> None:
        super().__init__()
        self._builder = builder

    @QtCore.Slot()
    def run(self) -> None:
        def on_line(line: str) -> None:
            self.build_output.emit(line)

        result = self._builder.build(on_line=on_line)
        self.build_finished.emit(result)


def launch_gui() -> None:
    """启动 GUI 应用程序（主入口）。

    创建 QApplication 实例，应用 Fusion 主题风格，
    显示主窗口并进入事件循环。
    """
    # ── 全局崩溃日志（闪退时唯一可用的诊断信息）──
    import traceback as _tb
    _crash_log = Path(__file__).resolve().parent.parent.parent.parent / "crash.log"

    def _global_excepthook(exc_type, exc_value, exc_tb):
        """将所有未捕获异常写入 crash.log 后再调用默认处理器。"""
        msg = "".join(_tb.format_exception(exc_type, exc_value, exc_tb))
        try:
            _crash_log.write_text(msg, encoding="utf-8")
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _global_excepthook

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # CJK 字体
    _qt_font = QtGui.QFont()
    _qt_font.setFamilies(["Microsoft YaHei", "SimHei", "Microsoft JhengHei", "SimSun"])
    app.setFont(_qt_font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
