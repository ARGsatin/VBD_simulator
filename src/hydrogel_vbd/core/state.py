# -*- coding: utf-8 -*-
"""网格状态与仿真状态数据结构模块。

本模块定义了整个仿真系统中流转的核心数据结构：

- **MeshState**：四面体网格的完整拓扑、几何、物理状态
- **MaterialState**：材料属性（固化度、弹性模量等）
- **ForceState**：各物理力场分量及合力
- **FieldCommand**：电场控制指令
- **LayerResult**：单层仿真结果

这些数据类贯穿整个仿真管线：从网格构建 → 力计算 → 求解器迭代 → 结果输出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         内部辅助：数组验证函数                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _as_float_array(name: str, value: Any, shape_tail: tuple[int, ...]) -> np.ndarray:
    """将输入值转换为浮点数组并验证形状。

    Parameters
    ----------
    name : str
        字段名称（用于错误消息）。
    value : Any
        输入值（列表、numpy 数组等）。
    shape_tail : tuple[int, ...]
        期望的尾部形状。例如 (3,) 表示形状为 (N, 3)。

    Returns
    -------
    np.ndarray
        转换后的二维浮点数组。

    Raises
    ------
    ValueError
        如果形状不匹配。
    """
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} 的形状必须为 (N, {', '.join(map(str, shape_tail))})")
    return array


def _as_int_array(name: str, value: Any, shape_tail: tuple[int, ...]) -> np.ndarray:
    """将输入值转换为整型数组并验证形状。

    与 `_as_float_array` 类似，但转换为 ``dtype=int``。
    """
    array = np.asarray(value, dtype=int)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} 的形状必须为 (N, {', '.join(map(str, shape_tail))})")
    return array


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      MeshState —— 网格状态（核心）                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class MeshState:
    """四面体网格的完整状态表示。

    这是整个仿真系统的核心数据结构，承载了网格的拓扑关系、
    几何信息、动力学状态、损伤状态、以及求解所需的预计算量。

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        顶点坐标数组。每一行是一个顶点的 (x, y, z) 坐标。
    tets : np.ndarray, shape (T, 4)
        四面体索引数组。每一行的 4 个整数是构成该四面体的顶点索引。
    layer_id_per_vertex : np.ndarray, shape (N,)
        每个顶点所属的层号。
    layer_id_per_tet : np.ndarray, shape (T,)
        每个四面体所属的层号。
    ideal_vertices : np.ndarray or None, shape (N, 3)
        理想（无变形）顶点坐标，用于计算参考形矩阵和弹性力。
    first_active_layer : np.ndarray or None, shape (N,)
        每个顶点首次被激活的层号。
    is_bottom_surface : np.ndarray or None, shape (N,)
        布尔数组，标记底面（Z 最小值处）节点。
    is_top_surface_of_layer : np.ndarray or None, shape (N,)
        整数数组，每个顶点所属的层顶面编号（-1 表示不属于任何层面）。
    is_top_fixed : np.ndarray or None, shape (N,)
        布尔数组，标记平台夹持的顶层节点（提升时刚性位移）。
    prev_vertices : np.ndarray or None, shape (N, 3)
        上一时间步的顶点坐标（用于 Verlet 积分和速度计算）。
    velocities : np.ndarray or None, shape (N, 3)
        顶点速度 (m/s)。
    vertex2tets : list[list[int]]
        顶点到四面体的邻接表：``vertex2tets[i]`` 是包含顶点 i 的所有四面体索引。
    tet_volumes : np.ndarray or None, shape (T,)
        每个四面体的体积 (m³)。
    dm_inv : np.ndarray or None, shape (T, 3, 3)
        参考形矩阵的逆矩阵（用于计算形变梯度）。
    dm : np.ndarray or None, shape (T, 3, 3)
        参考形矩阵（理想形态下的边向量矩阵）。
    neighbors : list[set[int]]
        每个顶点的邻居顶点集合。
    node_mass : np.ndarray or None, shape (N,)
        每个节点的集中质量 (kg)。
    czm_state : np.ndarray or None, shape (N,)
        CZM 内聚力模型状态码：0=完好，1=软化，2=失效。
    damage : np.ndarray or None, shape (N,)
        损伤变量数组（0=完好，1=完全失效）。
    time_free : np.ndarray or None, shape (N,)
        每个节点从 CZM 失效后经历的"自由时间"（用于流体阻尼计时）。
    colors : np.ndarray or None, shape (N,)
        图着色分组（用于并行化力计算）。
    color_ranges : list[tuple[int, int]]
        每种颜色的顶点索引范围。
    active_vertex_mask : np.ndarray or None, shape (N,)
        布尔数组，标记当前层激活的顶点。
    active_tet_mask : np.ndarray or None, shape (T,)
        布尔数组，标记当前层激活的四面体。
    boundary_flags : np.ndarray or None, shape (N,)
        布尔数组，标记边界节点（用于边界约束）。
    """

    # ── 几何拓扑字段 ──
    vertices: np.ndarray                     # (N, 3)  当前顶点坐标
    tets: np.ndarray                         # (T, 4)  四面体顶点索引
    layer_id_per_vertex: np.ndarray          # (N,)    顶点层号
    layer_id_per_tet: np.ndarray             # (T,)    四面体层号
    ideal_vertices: np.ndarray | None = None # (N, 3)  理想坐标（基准形）
    first_active_layer: np.ndarray | None = None  # (N,) 首次激活层号
    is_bottom_surface: np.ndarray | None = None   # (N,) 底面标记
    is_top_surface_of_layer: np.ndarray | None = None  # (N,) 顶面层号
    is_top_fixed: np.ndarray | None = None   # (N,) 平台夹持标记

    # ── 运动学字段 ──
    prev_vertices: np.ndarray | None = None  # (N, 3)  上一帧坐标
    velocities: np.ndarray | None = None     # (N, 3)  速度

    # ── 拓扑邻接 ──
    vertex2tets: list[list[int]] = field(default_factory=list)  # 顶点→四面体
    tet_volumes: np.ndarray | None = None    # (T,)    四面体体积
    dm_inv: np.ndarray | None = None         # (T,3,3) 参考形逆矩阵
    dm: np.ndarray | None = None             # (T,3,3) 参考形矩阵
    neighbors: list[set[int]] = field(default_factory=list)     # 邻居表

    # ── 质量与损伤 ──
    node_mass: np.ndarray | None = None      # (N,)    节点质量
    czm_state: np.ndarray | None = None      # (N,)    CZM 状态码
    damage: np.ndarray | None = None         # (N,)    损伤变量
    time_free: np.ndarray | None = None      # (N,)    自由时间

    # ── 并行化支持 ──
    colors: np.ndarray | None = None         # (N,)    图着色分组
    color_ranges: list[tuple[int, int]] = field(default_factory=list)  # 颜色索引范围

    # ── 激活掩码 ──
    active_vertex_mask: np.ndarray | None = None  # (N,) 激活顶点
    active_tet_mask: np.ndarray | None = None     # (T,) 激活四面体
    boundary_flags: np.ndarray | None = None      # (N,) 边界节点

    def __post_init__(self) -> None:
        """数据类初始化后自动验证和填充。

        对每个字段执行形状检查，对 ``None`` 字段赋合理默认值。
        这是 dataclass 的魔法方法，实例化后自动调用。
        """
        # ── 验证核心几何字段 ──
        self.vertices = _as_float_array("vertices", self.vertices, (3,))
        self.tets = _as_int_array("tets", self.tets, (4,))
        vertex_count = self.vertices.shape[0]
        tet_count = self.tets.shape[0]

        # ── 理想坐标（默认等于当前坐标） ──
        self.ideal_vertices = (
            self.vertices.copy()
            if self.ideal_vertices is None
            else _as_float_array("ideal_vertices", self.ideal_vertices, (3,))
        )
        if self.ideal_vertices.shape != self.vertices.shape:
            raise ValueError("ideal_vertices 的形状必须与 vertices 相同")

        # ── 层号数组 ──
        self.layer_id_per_vertex = np.asarray(self.layer_id_per_vertex, dtype=int)
        if self.layer_id_per_vertex.shape != (vertex_count,):
            raise ValueError("layer_id_per_vertex 的形状必须为 (N,)")

        self.layer_id_per_tet = np.asarray(self.layer_id_per_tet, dtype=int)
        if self.layer_id_per_tet.shape != (tet_count,):
            raise ValueError("layer_id_per_tet 的形状必须为 (T,)")

        self.first_active_layer = (
            self.layer_id_per_vertex.copy()
            if self.first_active_layer is None
            else np.asarray(self.first_active_layer, dtype=int)
        )
        if self.first_active_layer.shape != (vertex_count,):
            raise ValueError("first_active_layer 的形状必须为 (N,)")

        # ── 四面体索引合法性检查 ──
        if np.any(self.tets < 0) or np.any(self.tets >= vertex_count):
            raise ValueError("tets 包含越界的顶点索引")

        # ── 运动学默认值 ──
        self.prev_vertices = (
            self.vertices.copy()
            if self.prev_vertices is None
            else _as_float_array("prev_vertices", self.prev_vertices, (3,))
        )
        self.velocities = (
            np.zeros_like(self.vertices)
            if self.velocities is None
            else _as_float_array("velocities", self.velocities, (3,))
        )
        if self.prev_vertices.shape != self.vertices.shape or self.velocities.shape != self.vertices.shape:
            raise ValueError("prev_vertices 和 velocities 的形状必须与 vertices 相同")

        # ── 激活掩码默认值（全未激活） ──
        self.active_vertex_mask = (
            np.zeros(vertex_count, dtype=bool)
            if self.active_vertex_mask is None
            else np.asarray(self.active_vertex_mask, dtype=bool)
        )
        self.active_tet_mask = (
            np.zeros(tet_count, dtype=bool)
            if self.active_tet_mask is None
            else np.asarray(self.active_tet_mask, dtype=bool)
        )
        if self.active_vertex_mask.shape != (vertex_count,) or self.active_tet_mask.shape != (tet_count,):
            raise ValueError("激活掩码的形状必须与网格尺寸匹配")

        # ── 边界标记 ──
        self.boundary_flags = (
            np.zeros(vertex_count, dtype=bool)
            if self.boundary_flags is None
            else np.asarray(self.boundary_flags, dtype=bool)
        )
        if self.boundary_flags.shape != (vertex_count,):
            raise ValueError("boundary_flags 的形状必须为 (N,)")

        # ── 底面自动检测（Z 最小值处） ──
        z_min = float(np.min(self.ideal_vertices[:, 2])) if vertex_count else 0.0
        self.is_bottom_surface = (
            np.isclose(self.ideal_vertices[:, 2], z_min)
            if self.is_bottom_surface is None
            else np.asarray(self.is_bottom_surface, dtype=bool)
        )

        # ── 顶面层号（默认 -1=不属于任何层面） ──
        self.is_top_surface_of_layer = (
            np.full(vertex_count, -1, dtype=int)
            if self.is_top_surface_of_layer is None
            else np.asarray(self.is_top_surface_of_layer, dtype=int)
        )

        # ── 平台夹持标记（默认无） ──
        self.is_top_fixed = (
            np.zeros(vertex_count, dtype=bool)
            if self.is_top_fixed is None
            else np.asarray(self.is_top_fixed, dtype=bool)
        )

        # ── 统一检查布尔/整数面的形状 ──
        for name, array in (
            ("is_bottom_surface", self.is_bottom_surface),
            ("is_top_surface_of_layer", self.is_top_surface_of_layer),
            ("is_top_fixed", self.is_top_fixed),
        ):
            if array.shape != (vertex_count,):
                raise ValueError(f"{name} 的形状必须为 (N,)")

        # ── 构建拓扑邻接（若未提供） ──
        if not self.vertex2tets:
            self.vertex2tets = self._build_vertex2tets()
        if not self.neighbors:
            self.neighbors = self._build_neighbors()

        # ── 四面体体积 ──
        if self.tet_volumes is None:
            self.tet_volumes = self._compute_tet_volumes(self.ideal_vertices, self.tets)
        else:
            self.tet_volumes = np.asarray(self.tet_volumes, dtype=float)

        # ── 参考形矩阵预计算（若未提供） ──
        if self.dm is None or self.dm_inv is None:
            self.precompute_reference_matrices(c_shrink=1.0)

        # ── 节点质量 ──
        self.node_mass = (
            self._build_node_masses(density=1.0)
            if self.node_mass is None
            else np.asarray(self.node_mass, dtype=float)
        )

        # ── CZM 状态 / 损伤 / 自由时间（默认全零） ──
        self.czm_state = (
            np.zeros(vertex_count, dtype=int)
            if self.czm_state is None
            else np.asarray(self.czm_state, dtype=int)
        )
        self.damage = (
            np.zeros(vertex_count, dtype=float)
            if self.damage is None
            else np.asarray(self.damage, dtype=float)
        )
        self.time_free = (
            np.zeros(vertex_count, dtype=float)
            if self.time_free is None
            else np.asarray(self.time_free, dtype=float)
        )
        for name, array in (("node_mass", self.node_mass), ("czm_state", self.czm_state),
                            ("damage", self.damage), ("time_free", self.time_free)):
            if array.shape != (vertex_count,):
                raise ValueError(f"{name} 的形状必须为 (N,)")

    # ───────────────────────────────────────────────────────────────────────
    # 拓扑构建方法
    # ───────────────────────────────────────────────────────────────────────

    def _build_vertex2tets(self) -> list[list[int]]:
        """构建顶点→四面体的邻接表。

        遍历所有四面体，为每个顶点记录其所属的四面体索引列表。

        Returns
        -------
        list[list[int]]
            ``adjacency[i]`` 是包含顶点 i 的所有四面体索引。
        """
        adjacency = [[] for _ in range(self.vertices.shape[0])]
        for tet_id, tet in enumerate(self.tets):
            for vertex_id in tet:
                adjacency[int(vertex_id)].append(tet_id)
        return adjacency

    def _build_neighbors(self) -> list[set[int]]:
        """构建每个顶点的邻居集合。

        两个顶点如果在同一个四面体中则互为邻居。

        Returns
        -------
        list[set[int]]
            ``neighbors[i]`` 是顶点 i 的所有邻居顶点索引集合。
        """
        neighbors: list[set[int]] = [set() for _ in range(self.vertices.shape[0])]
        for tet in self.tets:
            for vertex_id in tet:
                neighbors[int(vertex_id)].update(
                    int(other) for other in tet if int(other) != int(vertex_id)
                )
        return neighbors

    # ───────────────────────────────────────────────────────────────────────
    # 几何预计算方法
    # ───────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_tet_volumes(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
        """计算所有四面体的体积。

        使用行列式公式：V = |det([p1-p0, p2-p0, p3-p0])| / 6

        Parameters
        ----------
        vertices : np.ndarray, shape (N, 3)
            顶点坐标。
        tets : np.ndarray, shape (T, 4)
            四面体索引。

        Returns
        -------
        np.ndarray, shape (T,)
            每个四面体的体积。
        """
        volumes = np.zeros(tets.shape[0], dtype=float)
        for tet_id, tet in enumerate(tets):
            p0, p1, p2, p3 = vertices[tet]
            volumes[tet_id] = abs(float(np.linalg.det(
                np.column_stack((p1 - p0, p2 - p0, p3 - p0))
            ))) / 6.0
        return volumes

    def precompute_reference_matrices(self, c_shrink: float) -> None:
        """预计算参考形矩阵 Dₘ 及其逆矩阵 Dₘ⁻¹。

        参考形矩阵由理想坐标的边向量构成：
        ``Dₘ = [p1-p0, p2-p0, p3-p0]``（乘以收缩因子）。

        其逆矩阵用于在求解器中快速计算形变梯度 F = (∂x/∂X)。

        Parameters
        ----------
        c_shrink : float
            固化收缩因子。理想坐标乘以该因子以模拟收缩效应。
        """
        dm = np.zeros((self.tets.shape[0], 3, 3), dtype=float)
        dm_inv = np.zeros_like(dm)
        volumes = np.zeros(self.tets.shape[0], dtype=float)
        # 应用收缩因子到理想坐标
        reference = self.ideal_vertices * float(c_shrink)

        for tet_id, tet in enumerate(self.tets):
            p0, p1, p2, p3 = reference[tet]
            matrix = np.column_stack((p1 - p0, p2 - p0, p3 - p0))  # Dₘ
            dm[tet_id] = matrix
            det = float(np.linalg.det(matrix))
            volumes[tet_id] = abs(det) / 6.0
            if abs(det) > 1e-12:
                dm_inv[tet_id] = np.linalg.inv(matrix)  # Dₘ⁻¹

        self.dm = dm
        self.dm_inv = dm_inv
        self.tet_volumes = volumes

    def _build_node_masses(self, density: float) -> np.ndarray:
        """计算集中节点质量。

        采用**均匀分配**：每个四面体的质量（密度×体积）平均分配给 4 个顶点。

        对质量为零的节点（如孤立点），赋最小值 1.0 以避免除零。

        Parameters
        ----------
        density : float
            材料密度 (kg/m³)。

        Returns
        -------
        np.ndarray, shape (N,)
            节点质量数组。
        """
        masses = np.zeros(self.vertices.shape[0], dtype=float)
        volumes = self.tet_volumes if self.tet_volumes is not None else np.ones(self.tets.shape[0], dtype=float)
        for tet_id, tet in enumerate(self.tets):
            masses[tet] += float(volumes[tet_id]) * float(density) / 4.0
        masses[masses <= 0.0] = 1.0
        return masses

    # ───────────────────────────────────────────────────────────────────────
    # 层激活与查询方法
    # ───────────────────────────────────────────────────────────────────────

    def activate_layer(self, current_layer: int) -> None:
        """激活指定层及之前所有层的顶点和四面体。

        将 ``first_active_layer <= current_layer`` 的顶点和四面体标记为激活状态，
        后续力计算和求解仅作用于激活单元。

        Parameters
        ----------
        current_layer : int
            当前打印层号（从 0 开始）。
        """
        self.active_vertex_mask = self.first_active_layer <= current_layer
        self.active_tet_mask = self.layer_id_per_tet <= current_layer

    def bottom_nodes(self, layer_id: int) -> np.ndarray:
        """获取指定层底面节点的索引。

        Parameters
        ----------
        layer_id : int
            层号。

        Returns
        -------
        np.ndarray
            底面节点索引数组。
        """
        return np.flatnonzero(self.is_top_surface_of_layer == layer_id)

    def top_nodes(self, layer_id: int) -> np.ndarray:
        """获取指定层顶面节点的索引。

        Parameters
        ----------
        layer_id : int
            层号。

        Returns
        -------
        np.ndarray
            顶面节点索引数组。
        """
        return np.flatnonzero(self.is_top_surface_of_layer == layer_id + 1)

    def layer_interface_nodes(self, interface_id: int) -> np.ndarray:
        """获取指定层面（接口）上的所有节点索引。

        Parameters
        ----------
        interface_id : int
            层面编号。

        Returns
        -------
        np.ndarray
            该层面上所有节点的索引。
        """
        return np.flatnonzero(self.is_top_surface_of_layer == interface_id)

    @property
    def masses(self) -> np.ndarray:
        """返回节点质量的副本（兼容性属性）。"""
        return self.node_mass.copy() if self.node_mass is not None else np.ones(self.vertices.shape[0], dtype=float)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      MaterialState —— 材料状态                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class MaterialState:
    """水凝胶材料的状态描述。

    包含固化度、弹性模量、拉梅常数等随工艺演化的材料属性。
    ``mu`` 和 ``lam`` 为派生量（由 Young 模量和 Poisson 比计算），
    若未显式提供则由水凝胶模型自动计算。

    Parameters
    ----------
    density : float
        材料密度 (kg/m³)。
    young_modulus : np.ndarray, shape (N,)
        每个节点的杨氏模量 (Pa)，随固化度变化。
    poisson_ratio : float
        泊松比（假设各向同性，全局常数）。
    damping : np.ndarray, shape (N,)
        每个节点的阻尼系数。
    curing_degree : np.ndarray, shape (N,)
        每个节点的固化度（0~1，0=液态，1=完全固化）。
    peel_stress_crit : float
        临界剥离应力 (Pa)，用于离型膜脱粘判断。
    electric_response_alpha : np.ndarray, shape (N,)
        电场响应系数 α。
    mu : np.ndarray or None, shape (N,)
        第一拉梅常数（剪切模量），可由 Young/Poisson 导出。
    lam : np.ndarray or None, shape (N,)
        第二拉梅常数，可由 Young/Poisson 导出。
    """
    density: float
    young_modulus: np.ndarray             # (N,) 杨氏模量
    poisson_ratio: float
    damping: np.ndarray                   # (N,) 阻尼系数
    curing_degree: np.ndarray             # (N,) 固化度 0~1
    peel_stress_crit: float               # 临界剥离应力 (Pa)
    electric_response_alpha: np.ndarray   # (N,) 电场响应系数
    mu: np.ndarray | None = None          # (N,) 第一拉梅常数（自动计算）
    lam: np.ndarray | None = None         # (N,) 第二拉梅常数（自动计算）


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      ForceState —— 力场状态                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class ForceState:
    """作用于每个节点的各物理力场分量。

    包含五种基本力：
    1. **重力** (gravity)
    2. **剥离力** (peel) —— 离型膜脱粘力
    3. **流体阻尼力** (fluid) —— 树脂流动阻力
    4. **表面张力** (surface)
    5. **电场力** (electric)

    ``total`` 属性返回所有分量的叠加合力。

    Parameters
    ----------
    gravity : np.ndarray, shape (N, 3)
        重力向量。
    peel : np.ndarray, shape (N, 3)
        剥离力向量（离型膜界面）。
    fluid : np.ndarray, shape (N, 3)
        流体阻尼力向量。
    surface : np.ndarray, shape (N, 3)
        表面张力向量。
    electric : np.ndarray, shape (N, 3)
        电场力向量。
    """
    gravity: np.ndarray   # (N, 3) 重力
    peel: np.ndarray      # (N, 3) 剥离力
    fluid: np.ndarray     # (N, 3) 流体阻尼力
    surface: np.ndarray   # (N, 3) 表面张力
    electric: np.ndarray  # (N, 3) 电场力

    @property
    def total(self) -> np.ndarray:
        """返回所有力场的叠加合力。

        这是求解器中实际使用的驱动力向量：
        ``F_total = F_gravity + F_peel + F_fluid + F_surface + F_electric``

        Returns
        -------
        np.ndarray, shape (N, 3)
            合力向量。
        """
        return self.gravity + self.peel + self.fluid + self.surface + self.electric


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      FieldCommand —— 电场控制指令                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class FieldCommand:
    """单次电场施加指令。

    由 PID 控制器根据形状误差计算得出，指定在特定时间窗口内
    对各电极施加的电压和极性。

    Parameters
    ----------
    voltage : np.ndarray
        电压幅值数组 (V)。
    polarity : np.ndarray or None
        极性数组（+1 / -1 / 0），None 时由电压符号自动推断。
    duration : float
        电场施加持续时间 (s)。
    start_time : float
        施加开始时间 (s)。
    electrode_ids : list[str]
        电极标识符列表。
    """
    voltage: np.ndarray                             # 电压幅值 (V)
    polarity: np.ndarray | None = None              # 极性 (±1/0)
    duration: float = 0.0                           # 持续时间 (s)
    start_time: float = 0.0                         # 开始时间 (s)
    electrode_ids: list[str] = field(default_factory=list)  # 电极 ID

    def __post_init__(self) -> None:
        """自动推断极性（若未提供）。"""
        self.voltage = np.asarray(self.voltage, dtype=float)
        if self.polarity is None:
            # 由电压符号自动推断：正→+1，负→-1，零→0
            self.polarity = np.where(self.voltage > 0.0, 1,
                                     np.where(self.voltage < 0.0, -1, 0))
        else:
            self.polarity = np.asarray(self.polarity, dtype=int)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      LayerResult —— 单层仿真结果                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class LayerResult:
    """单层打印仿真的完整结果。

    主循环每完成一层后生成此结构，包含变形后的网格坐标、
    形状误差度量、以及为下一层计算的电场指令。

    Parameters
    ----------
    layer_id : int
        层号。
    x_sim : np.ndarray, shape (N, 3)
        仿真结束后的顶点坐标。
    v_sim : np.ndarray, shape (N, 3)
        仿真结束后的顶点速度。
    error_metrics : dict[str, float]
        各类形状误差度量（如 RMS、最大偏差等）。
    field_command_next : FieldCommand
        为下一层计算的电场指令。
    max_deformation : float
        当前层的最大变形量 (m)。
    rms_error : float
        当前层的 RMS 形状误差。
    success : bool
        该层仿真是否成功完成。
    """
    layer_id: int
    x_sim: np.ndarray                    # (N, 3) 终点坐标
    v_sim: np.ndarray                    # (N, 3) 终点速度
    error_metrics: dict[str, float]      # 误差度量字典
    field_command_next: FieldCommand     # 下一层电场指令
    max_deformation: float               # 最大变形 (m)
    rms_error: float                     # RMS 误差
    success: bool                        # 是否成功
