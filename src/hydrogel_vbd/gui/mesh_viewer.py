# -*- coding: utf-8 -*-
"""3D 网格可视化组件 —— 基于 matplotlib + PySide6。

提供 ``MeshViewer`` 类，用于在 GUI 中实时展示：
- 初始网格（导入模型后）
- 逐层仿真变形结果
"""

from __future__ import annotations

import numpy as np

try:
    from PySide6 import QtWidgets, QtCore

    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError:
    raise ImportError(
        "PySide6 和 matplotlib 未安装。请运行:\n"
        "pip install pyside6 matplotlib"
    )


class MeshViewer(QtWidgets.QWidget):
    """3D 网格可视化器。

    内嵌 matplotlib 的三维坐标轴，可旋转/缩放/平移查看网格。
    支持展示初始网格和逐层变形结果。

    Parameters
    ----------
    parent : QWidget | None
        父级控件。
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._fig = Figure(figsize=(5, 4), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.set_xlabel("X (mm)")
        self._ax.set_ylabel("Y (mm)")
        self._ax.set_zlabel("Z (mm)")
        self._ax.set_title("网格预览")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        # ── 添加 matplotlib 导航工具栏（缩放/旋转/平移）──
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        self.setLayout(layout)

        # 存储当前绘图元素，方便清除
        self._surf = None
        self._scatter = None
        self._wireframe_lines: list = []

    def clear(self) -> None:
        """清除当前显示内容。"""
        self._ax.cla()
        self._ax.set_xlabel("X (mm)")
        self._ax.set_ylabel("Y (mm)")
        self._ax.set_zlabel("Z (mm)")
        self._surf = None
        self._scatter = None
        self._wireframe_lines.clear()
        self._boundary_faces = None  # 缓存的边界面索引
        self._canvas.draw_idle()

    def _extract_boundary_faces(self, tets: np.ndarray) -> np.ndarray:
        """提取四面体外表面（缓存结果，避免每帧重复 O(N log N) 计算）。

        拓扑不变时边界索引可复用。
        """
        faces = np.vstack([
            tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]], tets[:, [1, 2, 3]],
        ])
        faces_sorted = np.sort(faces, axis=1)
        _, inverse, counts = np.unique(
            faces_sorted, axis=0, return_inverse=True, return_counts=True
        )
        boundary_mask = counts[inverse] == 1
        return faces[boundary_mask]

    def show_initial_mesh(
        self,
        vertices: np.ndarray,
        tets: np.ndarray,
        title: str = "初始网格",
    ) -> None:
        """展示初始（未变形）网格。

        通过四面体的外表面（提取三角形面）来绘制表面。

        Parameters
        ----------
        vertices : ndarray (N, 3)
            节点坐标。
        tets : ndarray (M, 4)
            四面体单元（节点索引）。
        title : str
            图表标题。
        """
        self._ax.cla()
        self._ax.set_xlabel("X (mm)")
        self._ax.set_ylabel("Y (mm)")
        self._ax.set_zlabel("Z (mm)")
        self._ax.set_title(title)

        if len(vertices) == 0 or len(tets) == 0:
            self._canvas.draw_idle()
            return

        # 提取并缓存边界面（拓扑不变，后续帧复用）
        self._boundary_faces = self._extract_boundary_faces(tets)
        boundary_faces = self._boundary_faces

        # 用 trisurf 绘制外表面
        try:
            self._surf = self._ax.plot_trisurf(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                triangles=boundary_faces,
                cmap="viridis",
                alpha=0.75,
                edgecolor="k",
                linewidth=0.3,
            )
        except Exception:
            # 如果 trisurf 失败（如退化面），回退到散点图
            self._scatter = self._ax.scatter(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                c="steelblue",
                s=8,
                alpha=0.8,
            )

        # 自动缩放以包含所有点
        self._ax.set_xlim(
            float(np.min(vertices[:, 0])) - 0.01,
            float(np.max(vertices[:, 0])) + 0.01,
        )
        self._ax.set_ylim(
            float(np.min(vertices[:, 1])) - 0.01,
            float(np.max(vertices[:, 1])) + 0.01,
        )
        self._ax.set_zlim(
            float(np.min(vertices[:, 2])) - 0.01,
            float(np.max(vertices[:, 2])) + 0.01,
        )
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def show_deformed_mesh(
        self,
        vertices_deformed: np.ndarray,
        tets: np.ndarray,
        active_mask: np.ndarray | None = None,
        title: str = "变形网格",
        color_mode: str = "active",
        layer_id_per_vertex: np.ndarray | None = None,
        damage: np.ndarray | None = None,
        czm_state: np.ndarray | None = None,
        displacement: np.ndarray | None = None,
    ) -> None:
        """展示变形后的网格（每层仿真结果）。

        Parameters
        ----------
        vertices_deformed : ndarray (N, 3)
            变形后的节点坐标。
        tets : ndarray (M, 4)
            四面体单元（节点索引）。
        active_mask : ndarray (N,) of bool, optional
            已激活节点掩码。
        title : str
            图表标题。
        color_mode : str
            着色模式: "active", "layer", "damage", "displacement", "czm"
        layer_id_per_vertex : ndarray (N,), optional
            每顶点的层 ID（color_mode="layer" 时需要）。
        damage : ndarray (N,), optional
            损伤值（color_mode="damage" 时需要）。
        czm_state : ndarray (N,), optional
            CZM 状态（color_mode="czm" 时需要）。
        displacement : ndarray (N,), optional
            位移幅值（color_mode="displacement" 时需要）。
        """
        self._ax.cla()
        self._ax.set_xlabel("X (mm)")
        self._ax.set_ylabel("Y (mm)")
        self._ax.set_zlabel("Z (mm)")
        self._ax.set_title(title)

        if len(vertices_deformed) == 0 or len(tets) == 0:
            self._canvas.draw_idle()
            return

        # 复用缓存的边界面（拓扑不变，避免每帧 O(N log N) 的 unique 操作）
        if self._boundary_faces is not None:
            boundary_faces = self._boundary_faces
        else:
            boundary_faces = self._extract_boundary_faces(tets)
            self._boundary_faces = boundary_faces

        # ── 根据着色模式计算面颜色 ──
        cmap_name, face_values = self._compute_face_colors(
            boundary_faces, color_mode, active_mask, layer_id_per_vertex,
            damage, czm_state, displacement
        )

        try:
            self._surf = self._ax.plot_trisurf(
                vertices_deformed[:, 0],
                vertices_deformed[:, 1],
                vertices_deformed[:, 2],
                triangles=boundary_faces,
                cmap=cmap_name,
                alpha=0.75,
                edgecolor="k",
                linewidth=0.3,
            )
            if face_values is not None:
                self._surf.set_array(face_values)
        except Exception:
            if active_mask is not None and np.any(active_mask):
                colors = np.where(active_mask, "red", "gray")
            else:
                colors = "steelblue"
            self._scatter = self._ax.scatter(
                vertices_deformed[:, 0],
                vertices_deformed[:, 1],
                vertices_deformed[:, 2],
                c=colors,
                s=8,
                alpha=0.8,
            )

        # 自动缩放
        x_range = float(np.ptp(vertices_deformed[:, 0])) or 0.01
        y_range = float(np.ptp(vertices_deformed[:, 1])) or 0.01
        z_range = float(np.ptp(vertices_deformed[:, 2])) or 0.01
        pad = max(x_range, y_range, z_range) * 0.1
        self._ax.set_xlim(
            float(np.min(vertices_deformed[:, 0])) - pad,
            float(np.max(vertices_deformed[:, 0])) + pad,
        )
        self._ax.set_ylim(
            float(np.min(vertices_deformed[:, 1])) - pad,
            float(np.max(vertices_deformed[:, 1])) + pad,
        )
        self._ax.set_zlim(
            float(np.min(vertices_deformed[:, 2])) - pad,
            float(np.max(vertices_deformed[:, 2])) + pad,
        )
        self._fig.tight_layout()
        self._canvas.draw_idle()
        # 强制刷新事件循环
        QtWidgets.QApplication.processEvents()

    @staticmethod
    def _compute_face_colors(
        boundary_faces: np.ndarray,
        color_mode: str,
        active_mask: np.ndarray | None,
        layer_id_per_vertex: np.ndarray | None,
        damage: np.ndarray | None,
        czm_state: np.ndarray | None,
        displacement: np.ndarray | None,
    ) -> tuple[str, np.ndarray | None]:
        """根据着色模式计算每个三角面的标量值。

        Returns
        -------
        (cmap_name, face_values)
            cmap_name: 颜色映射名称
            face_values: shape (F,) 的面标量值，None 表示均匀着色
        """
        if boundary_faces is None or len(boundary_faces) == 0:
            return "viridis", None

        if color_mode == "layer" and layer_id_per_vertex is not None:
            face_layer = layer_id_per_vertex[boundary_faces].max(axis=1)
            return "tab20", face_layer.astype(float)

        if color_mode == "damage" and damage is not None:
            face_dmg = damage[boundary_faces].mean(axis=1)
            return "hot", face_dmg

        if color_mode == "displacement" and displacement is not None:
            face_disp = displacement[boundary_faces].mean(axis=1)
            return "plasma", face_disp

        if color_mode == "czm" and czm_state is not None:
            face_czm = czm_state[boundary_faces].max(axis=1)
            return "coolwarm", face_czm.astype(float)

        # default: active/inactive binary coloring
        if active_mask is not None and np.any(active_mask):
            face_active = active_mask[boundary_faces].any(axis=1).astype(float)
            return "coolwarm", face_active

        return "viridis", None

    def show_bounding_box(
        self,
        bbox_x: tuple[float, float],
        bbox_y: tuple[float, float],
        bbox_z: tuple[float, float],
        title: str = "模型包围盒",
    ) -> None:
        """绘制 3D 包围盒线框（用于 STEP 模型无三角面可渲染时）。

        Parameters
        ----------
        bbox_x : tuple[float, float]
            (x_min, x_max) 单位 m。
        bbox_y : tuple[float, float]
            (y_min, y_max) 单位 m。
        bbox_z : tuple[float, float]
            (z_min, z_max) 单位 m。
        title : str
            图表标题。
        """
        self._ax.cla()
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.set_zlabel("Z (m)")
        self._ax.set_title(title)

        x_min, x_max = bbox_x
        y_min, y_max = bbox_y
        z_min, z_max = bbox_z

        # 包围盒的 8 个角点
        corners = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ])

        # 12 条棱的索引对
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7),  # 竖直棱
        ]

        for i, j in edges:
            self._ax.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color="dodgerblue",
                linewidth=1.5,
            )

        # 绘制角点散点
        self._ax.scatter(
            corners[:, 0], corners[:, 1], corners[:, 2],
            c="dodgerblue", s=30, alpha=0.9,
        )

        # 标注包围盒尺寸
        dx = x_max - x_min
        dy = y_max - y_min
        dz = z_max - z_min
        self._ax.text(
            x_min + dx * 0.5, y_min + dy * 0.5, z_max + dz * 0.15,
            f"包围盒: {dx*1000:.1f} × {dy*1000:.1f} × {dz*1000:.1f} mm",
            color="gray", fontsize=9, ha="center",
        )

        pad = max(dx, dy, dz) * 0.15 or 0.005
        self._ax.set_xlim(x_min - pad, x_max + pad)
        self._ax.set_ylim(y_min - pad, y_max + pad)
        self._ax.set_zlim(z_min - pad, z_max + pad)
        self._ax.view_init(elev=25, azim=-60)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def show_stl_surface(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        title: str = "STL 模型",
    ) -> None:
        """直接展示 STL 模型的三角面（不依赖四面体网格）。

        Parameters
        ----------
        vertices : ndarray (N, 3)
            顶点坐标。
        faces : ndarray (M, 3)
            三角面索引。
        title : str
            图表标题。
        """
        self._ax.cla()
        self._ax.set_xlabel("X (mm)")
        self._ax.set_ylabel("Y (mm)")
        self._ax.set_zlabel("Z (mm)")
        self._ax.set_title(title)

        if len(vertices) == 0 or len(faces) == 0:
            self._canvas.draw_idle()
            return

        try:
            self._surf = self._ax.plot_trisurf(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                triangles=faces,
                cmap="viridis",
                alpha=0.8,
                edgecolor="k",
                linewidth=0.2,
            )
        except Exception:
            self._scatter = self._ax.scatter(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                c="steelblue",
                s=4,
                alpha=0.8,
            )

        # 自动缩放
        pad = (
            max(*[float(np.ptp(vertices[:, d])) or 0.01 for d in range(3)])
            * 0.1
        )
        for d, label in enumerate("XYZ"):
            axis = getattr(self._ax, f"set_{label.lower()}lim")
            axis(
                float(np.min(vertices[:, d])) - pad,
                float(np.max(vertices[:, d])) + pad,
            )
        self._fig.tight_layout()
        self._canvas.draw_idle()
        QtWidgets.QApplication.processEvents()
