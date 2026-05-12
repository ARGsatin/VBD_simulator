# -*- coding: utf-8 -*-
r"""非结构化 Delaunay 四面体网格剖分模块。

本模块负责从 STL / STEP 表面模型生成分层四面体网格，
完全移除旧版的体素化（Voxelization）规则网格方法。

.. attention::
   **格式升级建议**：强烈推荐导入 ``.step`` / ``.stp`` 等 **边界表示（B-Rep）**
   格式模型，以确保 Gmsh OpenCASCADE (OCC) 布尔运算的鲁棒性。
   STL 是离散三角面片格式，缺乏拓扑信息，OCC 布尔碎片化操作可能失败；
   STEP 格式包含完整的实体边界拓扑（Vertex → Edge → Wire → Face → Shell → Solid），
   布尔运算结果可预测且数值稳定。

   若仅有 STL 模型，本模块会自动回退到基于点云的 Delaunay 剖分方法
   （:class:`DelaunayTetMesher`），该方法在层切面插入约束点以保证层间
   连续性，但无法像 OCC Boolean Fragment 那样在几何层面杜绝四面体跨越
   打印层。

架构
----
.. code-block:: text

    STEP / STL 表面模型
        │
        ├── (优先) Gmsh OCC Boolean Fragment
        │       │
        │       ├── 包围盒 Z_min / Z_max
        │       ├── 每层高度生成巨型矩形切片平面
        │       ├── gmsh.model.occ.fragment(实体, 切片平面组)
        │       ├── gmsh.model.occ.synchronize()
        │       ├── 3D Delaunay 网格划分
        │       └── 逐四面体重心 Z 分配 layer_id
        │
        ├── (回退) scipy.spatial.Delaunay 四面体剖分
        │       │
        │       ├── 表面采样点（保持几何特征）
        │       ├── 内部填充点（Poisson 盘采样）
        │       ├── 层切面约束点（沿 Z 向等距采样）
        │       ├── α-shape / 内外判别过滤
        │       └── 层归属分类
        │
        ▼
    MeshState 兼容输出
        · vertices (N, 3)
        · tets (T, 4)
        · layer_id_per_vertex
        · first_active_layer
        · is_top_surface_of_layer
        · layer_id_per_tet

核心类
------
- **OCCFragmentMesher** : 基于 Gmsh OCC Boolean Fragment 的主网格构建器（推荐）
- **DelaunayTetMesher** : 基于 scipy Delaunay 的点云剖分构建器（回退方案）
- **STLMesher** : 旧版体素化构建器（保留兼容，仅用作 fallback）

依赖
----
- gmsh >= 4.11 (推荐，用于 OCC Boolean Fragment)
- scipy >= 1.9 (scipy.spatial.Delaunay，回退方案)
- numpy
- trimesh (用于 STL 加载与点云采样)
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any

import numpy as np

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.solver.graph_coloring import greedy_vertex_coloring
from hydrogel_vbd.core.state import MeshState

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
_STL_UNIT_SCALE = 0.001  # mm → m（trimesh 加载时自动应用）

# ---------------------------------------------------------------------------
# Gmsh 可用性检测
# ---------------------------------------------------------------------------
try:
    import gmsh
    _GMSH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GMSH_AVAILABLE = False
    gmsh = None  # type: ignore[assignment]


# ============================================================================
# 构建方向旋转工具
# ============================================================================

def _apply_build_axis(mesh: MeshState, build_axis: int) -> None:
    """原地交换顶点坐标列，使构建轴对齐到 Z。

    若 build_axis != 2，交换 ``vertices[:, build_axis]`` 和
    ``vertices[:, 2]``，使下游求解器统一以 Z 为构建轴工作。

    ``ideal_vertices``、``prev_vertices``、``velocities`` 同步旋转。
    """
    if build_axis == 2:
        return
    if build_axis not in (0, 1):
        raise ValueError(f"build_axis 必须为 0, 1, 2, 实际: {build_axis}")

    for arr_name in ("vertices", "ideal_vertices", "prev_vertices", "velocities"):
        arr = getattr(mesh, arr_name, None)
        if arr is not None and arr.ndim == 2 and arr.shape[1] == 3:
            arr[:, [build_axis, 2]] = arr[:, [2, build_axis]]


# ============================================================================
# Poisson 盘采样（Bridson 算法）—— DelaunayTetMesher 回退方案使用
# ============================================================================

def _poisson_disk_sampling_3d(
    bounds: np.ndarray,          # (2, 3) [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    min_dist: float,
    max_points: int = 20000,
    k_candidates: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """在三维包围盒内生成 Poisson 盘采样点。

    使用 Bridson 算法保证任意两点间距 ≥ ``min_dist``，
    在空间内均匀填充点，避免规则网格的锯齿效应。

    Parameters
    ----------
    bounds : np.ndarray, shape (2, 3)
        包围盒 [[xmin, ymin, zmin], [xmax, ymax, zmax]]。
    min_dist : float
        任意两点之间允许的最小距离 (m)。
    max_points : int
        最大采样点数（防止内存爆炸）。
    k_candidates : int
        每次尝试的候选点数。
    seed : int
        随机种子。

    Returns
    -------
    np.ndarray, shape (M, 3)
        采样点坐标。
    """
    rng = np.random.default_rng(seed)
    cell_size = min_dist / math.sqrt(3)
    lo = bounds[0] - 0.5 * min_dist
    hi = bounds[1] + 0.5 * min_dist
    size = hi - lo
    grid_shape = tuple(int(math.ceil(s / cell_size)) for s in size)

    # 空网格表示"尚未放置点"
    grid: np.ndarray = np.full(grid_shape, -1, dtype=int)
    points: list[list[float]] = []
    active: list[int] = []

    def _insert(pt: np.ndarray) -> int:
        idx = len(points)
        points.append(pt.tolist())
        gx = int((pt[0] - lo[0]) / cell_size)
        gy = int((pt[1] - lo[1]) / cell_size)
        gz = int((pt[2] - lo[2]) / cell_size)
        if 0 <= gx < grid_shape[0] and 0 <= gy < grid_shape[1] and 0 <= gz < grid_shape[2]:
            grid[gx, gy, gz] = idx
        return idx

    # 初始点：包围盒中心
    center = bounds.mean(axis=0)
    active.append(_insert(center))

    while active and len(points) < max_points:
        ai = rng.integers(len(active))
        p0 = np.asarray(points[active[ai]], dtype=float)
        found = False
        for _ in range(k_candidates):
            # 在球壳 (min_dist, 2*min_dist) 内随机采样
            theta = rng.random() * 2.0 * math.pi
            phi = math.acos(2.0 * rng.random() - 1.0)
            radius = min_dist * (1.0 + rng.random())
            candidate = p0 + radius * np.array([
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi),
            ])
            # 边界检查
            if np.any(candidate < lo + min_dist) or np.any(candidate > hi - min_dist):
                continue
            # 检查网格中邻居是否存在冲突
            gx = int((candidate[0] - lo[0]) / cell_size)
            gy = int((candidate[1] - lo[1]) / cell_size)
            gz = int((candidate[2] - lo[2]) / cell_size)
            conflict = False
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nx, ny, nz = gx + dx, gy + dy, gz + dz
                        if 0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1] and 0 <= nz < grid_shape[2]:
                            ng = grid[nx, ny, nz]
                            if ng >= 0:
                                np_pt = np.asarray(points[ng], dtype=float)
                                if float(np.linalg.norm(candidate - np_pt)) < min_dist:
                                    conflict = True
                                    break
                    if conflict:
                        break
                if conflict:
                    break
            if not conflict:
                active.append(_insert(candidate))
                found = True
                break
        if not found:
            # 该点周围填满，从活动列表中移除
            active.pop(ai)

    return np.asarray(points, dtype=float) if points else np.zeros((0, 3), dtype=float)


# ============================================================================
# 层归属分类工具
# ============================================================================

def _classify_points_to_layers(
    z_coords: np.ndarray,       # (N,) Z 坐标
    z_min: float,
    n_layers: int,
    layer_thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """将顶点按 Z 坐标分配到各层。

    Parameters
    ----------
    z_coords : np.ndarray, shape (N,)
        每个顶点的 Z 坐标。
    z_min : float
        模型底部 Z 坐标。
    n_layers : int
        总层数。
    layer_thickness : float
        每层厚度 (m)。

    Returns
    -------
    layer_id : np.ndarray, shape (N,)
        每个顶点的层号（首次激活层）。
    interface_id : np.ndarray, shape (N,)
        每个顶点最近的上层界面编号。
    """
    rel_z = z_coords - z_min
    # 首次激活层 = floor(rel_z / layer_thickness)
    layer_id = np.clip(
        np.floor(rel_z / layer_thickness).astype(int), 0, n_layers - 1
    )
    # 所属界面编号 = round(rel_z / layer_thickness) → 0..n_layers
    interface_id = np.clip(
        np.round(rel_z / layer_thickness).astype(int), 0, n_layers
    )
    return layer_id, interface_id


def _classify_tets_to_layers(
    tet_centroids_z: np.ndarray,  # (T,) 四面体中心 Z 坐标
    z_min: float,
    n_layers: int,
    layer_thickness: float,
) -> np.ndarray:
    """将四面体按中心 Z 坐标分配到各层。

    Parameters
    ----------
    tet_centroids_z : np.ndarray, shape (T,)
        四面体质心的 Z 坐标。
    z_min : float
        模型底部 Z 坐标。
    n_layers : int
        总层数。
    layer_thickness : float
        每层厚度 (m)。

    Returns
    -------
    np.ndarray, shape (T,)
        每个四面体的层号。
    """
    rel_z = tet_centroids_z - z_min
    layer_id = np.clip(
        np.floor(rel_z / layer_thickness).astype(int), 0, n_layers - 1
    )
    return layer_id


# ============================================================================
# OCC Boolean Fragment 顶点分类 —— 正确处理共享接口节点
# ============================================================================

def _classify_occ_vertices(
    point_z: np.ndarray,
    z_min_m: float,
    n_layers: int,
    layer_thickness_m: float,
    tol: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """对 OCC Boolean Fragment 输出的顶点进行层归属分类。

    与 :func:`_classify_points_to_layers` 不同，此函数正确处理
    布尔碎片化产生的共享接口节点：位于切割平面上的顶点被其
    上方和下方的层同时使用，因此其 ``first_active_layer`` 应为
    下方层的编号（即 ``max(interface_id - 1, 0)``）。

    .. code-block:: text

        切割平面示意（3 层示例）：

            z_max  ───── surface 3 (top)
                        │  tet 层 2
            z2     ───── surface 2 (切割平面) ← 节点 first_active=1
                        │  tet 层 1
            z1     ───── surface 1 (切割平面) ← 节点 first_active=0
                        │  tet 层 0
            z_min  ───── surface 0 (bottom)  ← 节点 first_active=0

    Parameters
    ----------
    point_z : np.ndarray, shape (N,)
        顶点 Z 坐标 (m)。
    z_min_m : float
        模型底部 Z 坐标 (m)。
    n_layers : int
        总层数。
    layer_thickness_m : float
        层厚 (m)。
    tol : float | None
        判定顶点位于切割平面的容差 (m)。默认为层厚的 1%。

    Returns
    -------
    first_active_layer : np.ndarray, shape (N,)
        每个顶点的首次激活层号。
    is_top_surface_of_layer : np.ndarray, shape (N,)
        每个顶点所属的表面编号（-1 = 内部顶点，不在任何表面上）。
    """
    if tol is None:
        tol = layer_thickness_m * 0.01

    rel_z = point_z - z_min_m
    n_vertices = len(point_z)

    first_active_layer = np.zeros(n_vertices, dtype=int)
    is_top_surface_of_layer = np.full(n_vertices, -1, dtype=int)

    # 遍历每个接口表面（从底面到顶面）
    classified = np.zeros(n_vertices, dtype=bool)

    for k in range(n_layers + 1):
        interface_z = float(k) * layer_thickness_m
        on_interface = (
            np.isclose(rel_z, interface_z, atol=tol) & ~classified
        )
        if not np.any(on_interface):
            continue

        if k == 0:
            # 底面（surface 0）—— 第一层即激活
            first_active_layer[on_interface] = 0
            is_top_surface_of_layer[on_interface] = 0
        elif k == n_layers:
            # 顶面（surface n_layers）—— 由最后一层激活
            first_active_layer[on_interface] = n_layers - 1
            is_top_surface_of_layer[on_interface] = n_layers
        else:
            # 内部切割平面（interface k）
            # 该平面上的顶点是 layer k-1 的顶面，也是 layer k 的底面
            # 因此当 layer k-1 被激活时即应被激活
            first_active_layer[on_interface] = k - 1
            is_top_surface_of_layer[on_interface] = k

        classified[on_interface] = True

    # 内部顶点（不在任何切割平面上）
    interior = ~classified
    if np.any(interior):
        interior_rel_z = rel_z[interior]
        layer_ids = np.clip(
            np.floor(interior_rel_z / layer_thickness_m).astype(int),
            0, n_layers - 1,
        )
        first_active_layer[interior] = layer_ids
        # 内部顶点的 is_top_surface_of_layer 保持 -1（不属于任何层面）

    return first_active_layer, is_top_surface_of_layer


# ============================================================================
# OCCFragmentMesher —— 基于 Gmsh OCC Boolean Fragment 的分层网格构建器
# ============================================================================

class OCCFragmentMesher:
    r"""基于 Gmsh OpenCASCADE (OCC) 布尔碎片化的分层四面体网格构建器。

    通过 Z 轴切片平面组与 3D 实体的 Boolean Fragment 操作，
    在**几何层面**将模型强制切分为"千层蛋糕"结构，确保：
    
    * 每个四面体完全位于单一打印层内，**杜绝跨层跨越**
    * 层间共享节点（共形网格），保证应力/变形继承的拓扑一致性
    * 每层可独立激活，支持 3D 打印逐层仿真

    求解流程
    --------
    1. 导入 STEP / STL 模型到 Gmsh OCC 几何内核
    2. 获取模型包围盒 :math:`(Z_{\min}, Z_{\max})`
    3. 根据 ``layer_thickness`` 生成一组 Z 轴切片平面（巨型矩形 Plane Surface）
    4. 调用 ``gmsh.model.occ.fragment()`` 对实体 + 切片平面组进行布尔碎片化
    5. 调用 ``gmsh.model.occ.synchronize()`` 同步几何实体
    6. 执行 3D Delaunay 网格划分（``gmsh.model.mesh.generate(3)``）
    7. 提取顶点与四面体，通过四面体重心 Z 坐标分配 ``layer_id``
    8. 组装 :class:`MeshState` 输出

    .. note::
       **格式建议**：优先使用 ``.step`` / ``.stp`` (B-Rep) 格式模型。
       OCC 布尔运算依赖精确的拓扑信息（Vertex → Edge → Wire → Face → Shell → Solid），
       STEP 格式天然携带这些信息，运算鲁棒性远优于 STL 离散三角面片。

       若 Gmsh 未安装或 OCC 操作失败，自动回退到 :class:`DelaunayTetMesher`。

    Parameters
    ----------
    stl_path : str
        模型文件路径（支持 .stl, .step, .stp, .igs, .iges 等 Gmsh 支持的格式）。
    layer_thickness : float
        每层厚度（Z 方向），单位 m。
    resolution : float
        目标网格边长（≈ 最大四面体棱长），单位 m。
    quality_factor : float
        质量因子（0.5~2.0），越小网格越密。默认为 1.0。
    max_points : int
        最大网格顶点数（用于回退方案 DelaunayTetMesher）。

    Raises
    ------
    ImportError
        如果 Gmsh 未安装且回退方案也失败。
    RuntimeError
        如果网格剖分在任何路径下均失败。
    """

    def __init__(
        self,
        stl_path: str,
        layer_thickness: float = 5e-5,
        resolution: float = 0.02,
        quality_factor: float = 1.0,
        max_points: int = 30000,
    ) -> None:
        self.stl_path = stl_path
        self.layer_thickness = float(layer_thickness)
        self.resolution = max(float(resolution), 0.001)
        self.quality_factor = float(quality_factor)
        self.max_points = int(max_points)

    # ------------------------------------------------------------------
    # 辅助：检测文件格式
    # ------------------------------------------------------------------
    @staticmethod
    def _is_step_file(filepath: str) -> bool:
        """检测是否为 STEP (B-Rep) 格式文件。

        Parameters
        ----------
        filepath : str
            文件路径。

        Returns
        -------
        bool
        """
        ext = os.path.splitext(filepath)[1].lower()
        return ext in ('.step', '.stp', '.igs', '.iges', '.brep')

    # ------------------------------------------------------------------
    # 主构建流程
    # ------------------------------------------------------------------
    def build_layered_mesh(
        self,
        config: SimulationConfig | None = None,
        algo_type: str = "layered",
    ) -> tuple[MeshState, int]:
        """构建分层四面体网格（双通道：STEP → OCC 切片 / STL → GEO 回退）。

        核心理念
        --------
        根据文件扩展名自动选择网格生成路径：

        * **通道 A (STEP/OCC)** — .step/.stp/.igs/.brep:
          使用 Gmsh OpenCASCADE 几何内核进行 Boolean Fragment 水平切片，
          在几何层面将模型切分为"千层蛋糕"结构，保证四面体不跨层。
        * **通道 B (STL/GEO)** — .stl:
          使用 ``gmsh.merge()`` 直接加载离散面片，跳过 OCC 实体操作，
          以整体非切层模式生成网格。层归属通过四面体重心 Z 坐标分配。

        网格算法选择
        ------------
        通过 ``algo_type`` 参数可在两种网格策略之间切换：

        * ``"layered"`` (默认): 通过 OCC Boolean Fragment 水平切片，
          保证四面体不跨层，适用于逐层 3D 打印仿真。
        * ``"standard"``: 跳过切片步骤，直接对模型进行
          自由四面体网格划分（标准非结构化网格），
          不保证层间拓扑连续性。

        Parameters
        ----------
        config : SimulationConfig | None
            仿真配置。若为 None，使用默认配置。
        algo_type : str
            网格算法类型：
            ``"layered"`` — 规整分层算法（OCC 切片）；
            ``"standard"`` — 标准非结构化算法（自由四面体）。

        Returns
        -------
        mesh : MeshState
            包含完整拓扑的网格状态。
        n_layers : int
            总打印层数。

        Raises
        ------
        ImportError
            如果 Gmsh 不可用。
        RuntimeError
            如果网格剖分失败。
        """
        if not _GMSH_AVAILABLE:
            raise ImportError(
                "Gmsh 未安装，无法使用 OCC Boolean Fragment 路径。"
                " 请运行: pip install gmsh 或 conda install -c conda-forge gmsh"
                "（将自动回退到 Delaunay 点云剖分方案）"
            )

        cfg = config or SimulationConfig(layer_thickness=self.layer_thickness)
        if cfg.layer_thickness <= 0:
            cfg.layer_thickness = self.layer_thickness

        layer_thickness_m = float(cfg.layer_thickness)

        # ── 标准非结构化网格：跳过切片，直接自由四面体剖分 ──
        if algo_type == "standard":
            mesh_state, n_layers = self._build_standard_unstructured(
                layer_thickness_m, cfg
            )
            _apply_build_axis(mesh_state, cfg.build_axis)
            return mesh_state, n_layers

        is_step = self._is_step_file(self.stl_path)

        # ── 初始化 Gmsh ──
        gmsh.initialize()
        try:
            # 抑制 Gmsh 终端输出（可选：设为 1 恢复输出）
            gmsh.option.setNumber("General.Terminal", 0)

            if is_step:
                # ── 通道 A：STEP B-Rep 实体 OCC 高保真切片 ──
                mesh_state, n_layers = self._build_step_occ(
                    layer_thickness_m, cfg
                )
            else:
                # ── 通道 B：STL 离散面片 GEO/Merge 安全回退 ──
                mesh_state, n_layers = self._build_stl_geo(
                    layer_thickness_m, cfg
                )

            _apply_build_axis(mesh_state, cfg.build_axis)
            return mesh_state, n_layers

        except Exception as exc:
            # ── 网格生成失败 ──
            # STEP 文件不能回退到 Delaunay（不支持 B-Rep），直接抛出
            if is_step:
                try:
                    gmsh.finalize()
                except Exception:
                    pass
                raise RuntimeError(
                    f"STEP 模型网格划分失败: {exc}"
                ) from exc

            # STL 文件可尝试回退到 Delaunay 点云剖分
            warnings.warn(
                f"Gmsh 网格生成失败: {exc}。"
                " 自动回退到 scipy.spatial.Delaunay 点云剖分方案。"
            )
            # 确保 Gmsh 已清理
            try:
                gmsh.finalize()
            except Exception:
                pass

            mesh_state, n_layers = self._fallback_delaunay(cfg)
            _apply_build_axis(mesh_state, cfg.build_axis)
            return mesh_state, n_layers

        finally:
            # 正常路径确保 Gmsh 已清理
            try:
                if gmsh.isInitialized():
                    gmsh.finalize()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 通道 A：STEP B-Rep 实体 OCC Boolean Fragment 高保真切片
    # ------------------------------------------------------------------
    def _build_step_occ(
        self,
        layer_thickness_m: float,
        cfg: SimulationConfig,
    ) -> tuple[MeshState, int]:
        r"""通道 A：使用 Gmsh OCC 几何内核进行 Boolean Fragment 水平切片。

        专为 STEP / STP / IGS / BREP 等 B-Rep 格式模型设计。
        通过 Z 轴切片平面组与 3D 实体的 Boolean Fragment 操作，
        在**几何层面**将模型强制切分为层结构，确保第四面体不跨层。

        流程
        ----
        1. ``gmsh.model.occ.importShapes()`` 导入 B-Rep 实体
        2. 动态筛选 3D 实体（volumes）
        3. ``gmsh.model.occ.addRectangle()`` 生成水平切片平面
        4. ``gmsh.model.occ.fragment()`` 执行布尔碎片化
        5. ``gmsh.model.occ.synchronize()`` 同步几何

        Parameters
        ----------
        layer_thickness_m : float
            层厚 (m)。
        cfg : SimulationConfig
            仿真配置。

        Returns
        -------
        tuple[MeshState, int]
        """
        # ── 1. 导入 B-Rep 模型到 OCC 几何内核 ──
        gmsh.model.add("STEP_OCC_Model")
        try:
            imported = gmsh.model.occ.importShapes(self.stl_path)
        except Exception as exc:
            raise RuntimeError(
                f"Gmsh OCC 无法导入 STEP 模型: {exc}\n"
                "  文件可能损坏、使用不支持的 STEP 协议，或包含非标准几何。\n"
                "  请尝试: (1) 在 CAD 中导出为 STEP AP214\n"
                "         (2) 导出为 STL 格式加载"
            ) from exc
        if not imported:
            raise RuntimeError(
                f"无法导入 STEP 模型: {self.stl_path}（格式不支持或文件损坏）"
            )

        gmsh.model.occ.synchronize()

        # ── 2. 获取 3D 实体（volumes）──
        #    先尝试 dim=3 直接查询（兼容性最好），再回退到 dim=-1
        try:
            volumes_entities = gmsh.model.occ.getEntities(dim=3)
        except Exception:
            try:
                all_entities = gmsh.model.occ.getEntities(dim=-1)
                volumes_entities = [e for e in all_entities if e[0] == 3]
            except Exception as exc2:
                raise RuntimeError(
                    f"Gmsh OCC 无法解析 STEP 模型中的几何实体: {exc2}\n"
                    "  模型可能包含不兼容的曲面类型或损坏的几何数据。\n"
                    "  请将模型重新导出为 STEP AP214 或 STL 格式。"
                ) from exc2

        volumes = [e[1] for e in volumes_entities]
        if not volumes:
            # 检查是否有 2D 曲面（非封闭模型）
            try:
                surfaces = gmsh.model.occ.getEntities(dim=2)
            except Exception:
                surfaces = []
            if surfaces:
                raise RuntimeError(
                    f"STEP 模型仅包含 {len(surfaces)} 个开放曲面，无 3D 实体。\n"
                    "  无法进行 Boolean Fragment 体网格划分。\n"
                    "  请将模型导出为封闭的 B-Rep 实体 (.step 格式)。"
                )
            raise RuntimeError(
                "STEP 模型中未检测到 3D 实体 (volumes)。"
                " 模型可能仅包含开放面片，无法进行 Boolean Fragment。"
                " 请将模型导出为封闭的 B-Rep 实体 (.step 格式)。"
            )

        # ── 3. 获取包围盒（mm 单位）──
        #    从已筛选的 3D 实体逐个计算，避免 dim=-1/tag=-1 的兼容性问题
        bbox = gmsh.model.occ.getBoundingBox(dim=3, tag=volumes[0])
        for tag in volumes[1:]:
            bb = gmsh.model.occ.getBoundingBox(dim=3, tag=tag)
            bbox = [
                min(bbox[0], bb[0]), min(bbox[1], bb[1]), min(bbox[2], bb[2]),
                max(bbox[3], bb[3]), max(bbox[4], bb[4]), max(bbox[5], bb[5]),
            ]
        if not bbox or len(bbox) < 6:
            raise RuntimeError("无法获取模型包围盒（模型可能为空）")

        x_min, y_min, z_min = bbox[0], bbox[1], bbox[2]
        x_max, y_max, z_max = bbox[3], bbox[4], bbox[5]
        extent_x = x_max - x_min
        extent_y = y_max - y_min
        extent_z = z_max - z_min

        # 转换为 m 单位（用于层数计算）
        layer_thickness_mm = layer_thickness_m * 1000.0  # m → mm
        z_min_m = z_min * _STL_UNIT_SCALE
        z_max_m = z_max * _STL_UNIT_SCALE
        extent_z_m = extent_z * _STL_UNIT_SCALE

        # ── 4. 计算层数 ──
        n_layers = max(1, int(math.ceil(extent_z_m / layer_thickness_m)))
        if n_layers <= 1:
            # 单层：无需碎片化，直接划分网格
            return self._mesh_and_extract(
                layer_thickness_m, z_min_m, n_layers, cfg
            )

        # ── 5. 生成切片平面组 ──
        margin_x = extent_x * 0.5
        margin_y = extent_y * 0.5
        plane_x0 = x_min - margin_x
        plane_y0 = y_min - margin_y
        plane_dx = extent_x + 2.0 * margin_x
        plane_dy = extent_y + 2.0 * margin_y

        cutting_plane_tags: list[int] = []
        for k in range(1, n_layers):
            z_mm = z_min + k * layer_thickness_mm
            if z_mm <= z_min + 1e-6 or z_mm >= z_max - 1e-6:
                continue
            plane_tag = gmsh.model.occ.addRectangle(
                plane_x0, plane_y0, z_mm, plane_dx, plane_dy
            )
            cutting_plane_tags.append(plane_tag)

        if not cutting_plane_tags:
            return self._mesh_and_extract(
                layer_thickness_m, z_min_m, n_layers, cfg
            )

        gmsh.model.occ.synchronize()

        # ── 6. Boolean Fragment：3D 实体 + 切片平面 → 碎片化实体 ──
        solids = gmsh.model.occ.getEntities(dim=3)
        if not solids:
            raise RuntimeError(
                "OCC 同步后丢失了 3D 实体。模型可能不封闭。"
            )

        tool_dim_tags = [(2, tag) for tag in cutting_plane_tags]

        try:
            out_dim_tags, _ = gmsh.model.occ.fragment(
                objectDimTags=solids,
                toolDimTags=tool_dim_tags,
                removeObject=True,
                removeTool=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"OCC Boolean Fragment 操作失败: {exc}。"
                " 可能原因：模型非流形、存在自交面、或几何退化。"
                " 请检查 STEP 模型的水密性。"
            ) from exc

        if not out_dim_tags:
            raise RuntimeError(
                "Boolean Fragment 未产生任何实体。"
                " 切片平面可能未正确贯穿模型截面。"
            )

        # ── 7. 同步与网格划分 ──
        gmsh.model.occ.synchronize()

        return self._mesh_and_extract(
            layer_thickness_m, z_min_m, n_layers, cfg
        )

    # ------------------------------------------------------------------
    # 通道 B：STL 离散面片 GEO/Merge 安全回退
    # ------------------------------------------------------------------
    def _build_stl_geo(
        self,
        layer_thickness_m: float,
        cfg: SimulationConfig,
    ) -> tuple[MeshState, int]:
        r"""通道 B：使用 Gmsh GEO 内核加载 STL 离散面片。

        **绝不使用 OCC 相关实体操作**。通过 ``gmsh.merge()``
        直接将 STL 文件加载为离散网格，跳过共形水平切片步骤。

        警告
        ----
        STL 是离散三角面片格式，缺乏拓扑信息，无法在几何层面
        进行精确的布尔切割。网格将以整体非切层模式生成，
        层归属通过四面体重心 Z 坐标后处理分配。

        流程
        ----
        1. ``gmsh.merge(path)`` 直接加载离散网格
        2. ``gmsh.model.geo.synchronize()`` 同步 GEO 模型
        3. (可选) ``gmsh.model.geo.addSurfaceLoop`` + ``addVolume``
           尝试将面片封闭为体，以获得 3D 网格元素

        Parameters
        ----------
        layer_thickness_m : float
            层厚 (m)。
        cfg : SimulationConfig
            仿真配置。

        Returns
        -------
        tuple[MeshState, int]
        """
        warnings.warn(
            "⚠ 检测到 STL 离散格式，已跳过共形水平切片。"
            " 网格将以整体非切层模式生成。"
            " 层归属将通过四面体重心 Z 坐标后处理分配。"
            " 强烈建议使用 .step B-Rep 格式以获得精确的分层网格。",
            UserWarning,
        )

        # ── 1. 使用 gmsh.merge() 直接加载 STL 离散网格 ──
        gmsh.model.add("STL_GEO_Model")
        try:
            gmsh.merge(self.stl_path)
        except Exception as exc:
            raise RuntimeError(
                f"无法加载 STL 文件: {exc}。"
                " 文件可能已损坏或格式不正确。"
            ) from exc

        # ── 2. 同步 GEO 模型 ──
        gmsh.model.geo.synchronize()

        # ── 3. 获取包围盒（用于层数计算）──
        try:
            bbox = gmsh.model.occ.getBoundingBox(
                dim=-1, tag=-1
            )
        except Exception:
            # GEO 模型可能不兼容 OCC getBoundingBox，
            # 回退：从 mesh 节点获取（先做一次粗网格或直接用 trimesh）
            bbox = None

        if bbox and len(bbox) >= 6:
            x_min, y_min, z_min_mm = bbox[0], bbox[1], bbox[2]
            _x_max, _y_max, z_max_mm = bbox[3], bbox[4], bbox[5]
            z_min_m = z_min_mm * _STL_UNIT_SCALE
            z_max_m = z_max_mm * _STL_UNIT_SCALE
            extent_z_m = z_max_m - z_min_m
        else:
            # 回退方案：使用 trimesh 加载 STL 获取包围盒
            try:
                import trimesh
                loaded = trimesh.load(self.stl_path)
                if isinstance(loaded, trimesh.Scene):
                    geometries = list(loaded.geometry.values())
                    if geometries:
                        loaded = trimesh.util.concatenate(geometries)
                    else:
                        raise ValueError("STL 场景中无几何体")
                if isinstance(loaded, trimesh.Trimesh):
                    loaded.vertices *= _STL_UNIT_SCALE
                    z_min_m = float(np.min(loaded.vertices[:, 2]))
                    z_max_m = float(np.max(loaded.vertices[:, 2]))
                    extent_z_m = z_max_m - z_min_m
                else:
                    raise TypeError("不支持的几何类型")
            except ImportError:
                # 没有 trimesh 时的默认高度
                extent_z_m = 0.001
                z_min_m = 0.0
                z_max_m = extent_z_m

        # ── 4. 计算层数 ──
        n_layers = max(1, int(math.ceil(extent_z_m / layer_thickness_m)))

        # ── 5. (可选) 尝试将 STL 表面封闭为体 ──
        # 这对于获得 3D 四面体网格至关重要。
        # 如果 STL 不封闭（有孔洞），此步骤可能失败，
        # 但不会中断流程——Gmsh 仍可从表面面片生成 2D/3D 网格。
        try:
            # 获取所有 2D 表面实体
            surfaces = gmsh.model.geo.getEntities(dim=2)
            if surfaces:
                surface_tags = [tag for _dim, tag in surfaces]

                # 尝试创建表面环（Surface Loop）并封闭为体
                sl_tag = gmsh.model.geo.addSurfaceLoop(
                    surface_tags
                )
                gmsh.model.geo.addVolume([sl_tag])
                gmsh.model.geo.synchronize()
        except Exception:
            # 表面非封闭（不可封闭为体），静默跳过
            # Gmsh 仍然可以从表面三角面片生成网格
            pass

        # ── 6. 统一网格生成 ──
        return self._mesh_and_extract(
            layer_thickness_m, z_min_m, n_layers, cfg
        )

    # ------------------------------------------------------------------
    # 网格划分与数据提取
    # ------------------------------------------------------------------
    def _mesh_and_extract(
        self,
        layer_thickness_m: float,
        z_min_m: float,
        n_layers: int,
        cfg: SimulationConfig,
    ) -> tuple[MeshState, int]:
        """对当前 Gmsh 模型执行网格划分，提取顶点和四面体，
        并构建 :class:`MeshState`。

        此方法假设 Gmsh 已初始化且模型已就绪（OCC 同步后）。

        Parameters
        ----------
        layer_thickness_m : float
            层厚 (m)。
        z_min_m : float
            模型底部 Z 坐标 (m)。
        n_layers : int
            总层数。
        cfg : SimulationConfig
            仿真配置。

        Returns
        -------
        tuple[MeshState, int]
        """
        # ── 设置网格尺寸 ──
        mesh_size_mm = self.resolution * 1000.0 / self.quality_factor  # m → mm
        mesh_size_mm = max(mesh_size_mm, layer_thickness_m * 1000.0 * 0.5)  # 确保不小于层厚的一半
        mesh_size_mm = min(mesh_size_mm, 500.0)  # 上限 500 mm

        gmsh.option.setNumber("Mesh.Algorithm3D", 1)        # 3D Delaunay
        gmsh.option.setNumber("Mesh.Algorithm", 5)           # 2D Delaunay
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_mm * 0.1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_mm)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        # 为所有点设置网格尺寸
        gmsh.model.mesh.setSize(
            gmsh.model.getEntities(0), mesh_size_mm
        )

        # ── 生成 3D 网格 ──
        try:
            gmsh.model.mesh.generate(3)
        except Exception as exc:
            raise RuntimeError(
                f"Gmsh 3D 网格生成失败: {exc}。"
                " 请尝试增大 resolution 或简化模型几何。"
            ) from exc

        # ── 提取节点 ──
        node_tags, node_coords_flat, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise RuntimeError("网格划分后无节点生成")

        # 坐标重塑为 (N, 3)，单位 mm → m
        vertices_mm = node_coords_flat.reshape(-1, 3)
        vertices = vertices_mm * _STL_UNIT_SCALE  # mm → m

        # ── 构建 Gmsh 节点标签 → 数组索引映射 ──
        tag_to_idx: dict[int, int] = {}
        for i, tag in enumerate(node_tags):
            tag_to_idx[int(tag)] = i

        # ── 提取四面体单元（Gmsh 单元类型 4 = 4-node tetrahedron）──
        try:
            elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(
                dim=3, tag=-1
            )
        except Exception:
            # 某些 Gmsh 版本可能返回不同格式
            elem_types, elem_tags_list, elem_node_tags_list = [], [], []

        # 在所有 3D 元素类型中查找四面体（类型 4）
        tet_node_connectivity: list[list[int]] = []
        _TET_ELEMENT_TYPE = 4  # Gmsh 中 4 节点四面体的类型码

        for etype, etags, enodes in zip(elem_types, elem_tags_list, elem_node_tags_list):
            if etype == _TET_ELEMENT_TYPE:
                n_nodes_per_elem = 4
                n_elems = len(etags)
                for i in range(n_elems):
                    start = i * n_nodes_per_elem
                    end = start + n_nodes_per_elem
                    elem_nodes_gmsh = enodes[start:end]
                    # 转换为 0 索引
                    elem_nodes = [tag_to_idx[int(tag)] for tag in elem_nodes_gmsh]
                    tet_node_connectivity.append(elem_nodes)

        if not tet_node_connectivity:
            # 回退：尝试使用 getElementsByType
            try:
                etags, enodes = gmsh.model.mesh.getElementsByType(_TET_ELEMENT_TYPE)
                n_elems = len(etags)
                for i in range(n_elems):
                    start = i * 4
                    end = start + 4
                    elem_nodes_gmsh = enodes[start:end]
                    elem_nodes = [tag_to_idx[int(tag)] for tag in elem_nodes_gmsh]
                    tet_node_connectivity.append(elem_nodes)
            except Exception:
                pass

        if not tet_node_connectivity:
            raise RuntimeError(
                "Gmsh 网格中未找到四面体单元（类型 4）。"
                " 请检查模型是否为封闭实体，或尝试使用 STEP 格式。"
            )

        tets = np.array(tet_node_connectivity, dtype=int)

        # ── 层归属分类 ──
        # 四面体层号：按重心 Z 坐标分配
        # OCC Boolean Fragment 保证四面体完全位于单层内，不跨层
        tet_centroids = np.mean(vertices[tets], axis=1)
        tet_z = tet_centroids[:, 2]
        eps_z = layer_thickness_m * 1e-6  # 微小偏移处理浮点边界情况
        layer_id_per_tet = np.clip(
            np.floor((tet_z - z_min_m + eps_z) / layer_thickness_m).astype(int),
            0, n_layers - 1,
        )

        # 顶点层号与表面编号（OCC Boolean Fragment 专用分类）
        # 正确处理共享接口节点：位于切割平面上的顶点被上下层共用，
        # 其 first_active_layer = max(interface_id - 1, 0)
        point_z = vertices[:, 2]
        first_active_layer, surface_ids = _classify_occ_vertices(
            point_z, z_min_m, n_layers, layer_thickness_m,
        )
        # layer_id_per_vertex 与 first_active_layer 保持一致
        layer_id_per_vertex_arr = first_active_layer.copy()

        # ── 组装 MeshState ──
        # is_bottom_surface 由 MeshState.__post_init__ 根据 Z_min 自动检测
        mesh = MeshState(
            vertices=np.asarray(vertices, dtype=float),
            tets=np.asarray(tets, dtype=int),
            layer_id_per_vertex=np.asarray(layer_id_per_vertex_arr, dtype=int),
            first_active_layer=np.asarray(first_active_layer, dtype=int),
            layer_id_per_tet=np.asarray(layer_id_per_tet, dtype=int),
            ideal_vertices=np.asarray(vertices, dtype=float).copy(),
            is_bottom_surface=None,  # 由 MeshState.__post_init__ 自动检测
            is_top_surface_of_layer=np.asarray(surface_ids, dtype=int),
        )
        mesh.precompute_reference_matrices(cfg.c_shrink)
        mesh.node_mass = mesh._build_node_masses(cfg.rho)
        mesh.colors = greedy_vertex_coloring(mesh)

        return mesh, n_layers

    # ------------------------------------------------------------------
    # 标准非结构化网格：跳过切片，直接自由四面体剖分
    # ------------------------------------------------------------------
    def _build_standard_unstructured(
        self,
        layer_thickness_m: float,
        cfg: SimulationConfig,
    ) -> tuple[MeshState, int]:
        """标准非结构化网格生成：跳过 OCC 切片，直接自由四面体剖分。

        流程
        ----
        1. 导入模型 (STEP → OCC / STL → GEO/Merge)
        2. 获取包围盒, 计算名义层数
        3. 直接调用 Gmsh 3D 自由网格划分 (无 Boolean Fragment)
        4. 后处理: 按四面体重心 Z 坐标分配 layer_id (仅用于层归属标识,
           不保证拓扑连续性)

        .. note::
           此方法生成的网格无层间拓扑约束，四面体可自由跨越多个
           打印层，适用于对分层刚性无要求的通用非结构化仿真场景。

        Parameters
        ----------
        layer_thickness_m : float
            层厚 (m)。
        cfg : SimulationConfig
            仿真配置。

        Returns
        -------
        tuple[MeshState, int]
            网格状态和名义总层数。
        """
        if not _GMSH_AVAILABLE:
            raise ImportError(
                "标准非结构化网格需要 Gmsh。请运行: pip install gmsh"
            )

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("Standard_Unstr_Model")

            is_step = self._is_step_file(self.stl_path)

            if is_step:
                # STEP B-Rep → OCC 导入
                imported = gmsh.model.occ.importShapes(self.stl_path)
                if not imported:
                    raise RuntimeError(
                        f"无法导入 STEP 模型: {self.stl_path}"
                    )
                gmsh.model.occ.synchronize()
            else:
                # STL → GEO/Merge 导入
                import warnings as _w
                _w.warn(
                    "标准非结构化网格: 使用 STL 格式（缺失拓扑信息）。"
                    " 建议使用 STEP B-Rep 格式以获得更高质量的四面体网格。",
                    UserWarning,
                )
                gmsh.merge(self.stl_path)
                gmsh.model.geo.synchronize()

                # 尝试获取包围盒（优先 OCC，失败回退 trimesh）
                try:
                    bbox = gmsh.model.occ.getBoundingBox(
                        dim=-1, tag=-1
                    )
                    if bbox and len(bbox) >= 6:
                        z_min_m = bbox[2] * _STL_UNIT_SCALE
                        z_max_m = bbox[5] * _STL_UNIT_SCALE
                        extent_z_m = z_max_m - z_min_m
                    else:
                        raise ValueError("BBox 不完整")
                except Exception:
                    try:
                        import trimesh

                        loaded = trimesh.load(self.stl_path)
                        if isinstance(loaded, trimesh.Scene):
                            geometries = list(loaded.geometry.values())
                            if geometries:
                                loaded = trimesh.util.concatenate(geometries)
                            else:
                                raise ValueError("STL 场景中无几何体")
                        if isinstance(loaded, trimesh.Trimesh):
                            loaded.vertices *= _STL_UNIT_SCALE
                            z_min_m = float(np.min(loaded.vertices[:, 2]))
                            z_max_m = float(np.max(loaded.vertices[:, 2]))
                            extent_z_m = z_max_m - z_min_m
                        else:
                            raise TypeError("不支持的几何类型")
                    except ImportError:
                        extent_z_m = 0.001
                        z_min_m = 0.0
                        z_max_m = extent_z_m

                # 计算层数
                n_layers = max(
                    1,
                    int(math.ceil(extent_z_m / layer_thickness_m)),
                )

                # 尝试封闭面片为体
                try:
                    surfaces = gmsh.model.geo.getEntities(dim=2)
                    if surfaces:
                        surface_tags = [tag for _, tag in surfaces]
                        sl_tag = gmsh.model.geo.addSurfaceLoop(
                            surface_tags
                        )
                        gmsh.model.geo.addVolume([sl_tag])
                        gmsh.model.geo.synchronize()
                except Exception:
                    pass

                # 直接跳过 Boolean Fragment，进入网格划分
                return self._mesh_and_extract(
                    layer_thickness_m,
                    z_min_m,
                    n_layers,
                    cfg,
                )

            # STEP 路径：获取包围盒
            bbox = gmsh.model.occ.getBoundingBox(dim=-1, tag=-1)
            if not bbox or len(bbox) < 6:
                raise RuntimeError("无法获取 STEP 模型包围盒")

            z_min_m = bbox[2] * _STL_UNIT_SCALE
            z_max_m = bbox[5] * _STL_UNIT_SCALE
            extent_z_m = z_max_m - z_min_m
            n_layers = max(
                1,
                int(math.ceil(extent_z_m / layer_thickness_m)),
            )

            # 直接自由四面体网格划分，无布尔切片
            return self._mesh_and_extract(
                layer_thickness_m,
                z_min_m,
                n_layers,
                cfg,
            )

        except Exception as exc:
            raise RuntimeError(
                f"标准非结构化网格生成失败: {exc}"
            ) from exc

        finally:
            try:
                if gmsh.isInitialized():
                    gmsh.finalize()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 回退方案：Delaunay 点云剖分
    # ------------------------------------------------------------------
    def _fallback_delaunay(
        self, cfg: SimulationConfig
    ) -> tuple[MeshState, int]:
        """回退到 Delaunay 点云剖分方案。

        Parameters
        ----------
        cfg : SimulationConfig
            仿真配置。

        Returns
        -------
        tuple[MeshState, int]
        """
        fallback = DelaunayTetMesher(
            stl_path=self.stl_path,
            layer_thickness=self.layer_thickness,
            resolution=self.resolution,
            quality_factor=self.quality_factor,
            max_points=self.max_points,
        )
        return fallback.build_layered_mesh(cfg)


# ============================================================================
# DelaunayTetMesher —— 非结构化四面体网格构建器（回退方案）
# ============================================================================

class DelaunayTetMesher:
    """基于 Delaunay 三角剖分的非结构化四面体网格构建器。

    彻底移除旧版的体素化（Voxelization）规则网格方法。
    通过 Poisson 盘采样、Delaunay 剖分和 α-shape 过滤，
    生成高质量、保形的不规则四面体网格。

    .. note::
       此方案是 OCC Boolean Fragment 的回退选项。
       推荐优先使用 :class:`OCCFragmentMesher`。

    求解流程
    --------
    1. 加载 STL 表面模型（trimesh）
    2. 表面采样 + 内部 Poisson 盘采样 + 层切面约束点
    3. scipy.spatial.Delaunay 四面体剖分
    4. 过滤外部四面体（inside / outside test）
    5. 层归属分类
    6. 组装 MeshState 输出

    Parameters
    ----------
    stl_path : str
        STL 文件路径（坐标单位: mm，内部自动转为 m）。
    layer_thickness : float
        每层厚度（Z 方向），单位 m。
    resolution : float
        目标网格边长（≈ 最大四面体外接球半径），单位 m。
    quality_factor : float
        质量因子（0.5~2.0），越小网格越密。
        默认为 1.0。
    max_points : int
        最大采样点数（防止内存爆炸）。
    """

    def __init__(
        self,
        stl_path: str,
        layer_thickness: float = 5e-5,
        resolution: float = 0.02,
        quality_factor: float = 1.0,
        max_points: int = 30000,
    ) -> None:
        self.stl_path = stl_path
        self.layer_thickness = float(layer_thickness)
        self.resolution = max(float(resolution), 0.001)
        self.quality_factor = float(quality_factor)
        self.max_points = int(max_points)
        self._mesh: Any = None  # trimesh 对象

    def _load(self) -> None:
        """加载 STL 并将坐标从 mm 转换为 m。

        STEP 文件不支持 —— Delaunay 回退路径不适用于 B-Rep 格式。
        """
        if self._mesh is not None:
            return

        import os as _os
        _ext = _os.path.splitext(str(self.stl_path))[1].lower()
        if _ext in ('.step', '.stp', '.igs', '.iges', '.brep'):
            raise RuntimeError(
                f"STEP/B-Rep 文件 ({_ext}) 需要 Gmsh OCC 几何内核。\n"
                "请安装: pip install gmsh\n"
                "或先将模型转换为 STL 格式再加载。"
            )

        try:
            import trimesh
        except ImportError as exc:
            raise ImportError(
                "trimesh is required for STL loading. Run: pip install trimesh"
            ) from exc

        with open(self.stl_path, 'rb') as fh:
            loaded = trimesh.load(fh, file_type='stl')

        if isinstance(loaded, trimesh.Scene):
            geometries = list(loaded.geometry.values())
            if not geometries:
                raise ValueError(f"No geometry found in STL scene: {self.stl_path}")
            loaded = trimesh.util.concatenate(geometries)
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(
                f"Unsupported type from STL: {type(loaded).__name__}. "
                "Expected trimesh.Trimesh."
            )
        loaded.vertices *= _STL_UNIT_SCALE
        self._mesh = loaded

    @property
    def tri_mesh(self):
        """返回 trimesh 对象（延迟加载，坐标已转为 m）。"""
        self._load()
        return self._mesh

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------

    def _sample_surface_points(self, n_surface: int) -> np.ndarray:
        """在 STL 三角面上均匀采样点。

        Parameters
        ----------
        n_surface : int
            表面采样点数。

        Returns
        -------
        np.ndarray, shape (n_surface, 3)
        """
        try:
            pts, _ = self.tri_mesh.sample(n_surface, return_index=True)
        except Exception:
            pts = self.tri_mesh.vertices.copy()
            if len(pts) > n_surface:
                idx = np.random.default_rng(42).choice(
                    len(pts), n_surface, replace=False
                )
                pts = pts[idx]
        return np.asarray(pts, dtype=float)

    def _sample_interior_points(
        self,
        min_dist: float,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """通过 Poisson 盘采样 + 内部过滤生成内部填充点。

        Parameters
        ----------
        min_dist : float
            Poisson 盘最小间距 (m)。
        bounds : np.ndarray, shape (2, 3)
            模型包围盒。

        Returns
        -------
        np.ndarray, shape (M, 3)
        """
        interior_pts = _poisson_disk_sampling_3d(
            bounds,
            min_dist=min_dist * 0.85,
            max_points=self.max_points,
            k_candidates=20,
        )
        if len(interior_pts) == 0:
            return interior_pts

        # 过滤：只保留 STL 内部的点
        try:
            inside = self.tri_mesh.contains(interior_pts)
        except ModuleNotFoundError as exc:
            if 'rtree' in str(exc).lower():
                raise ModuleNotFoundError(
                    "Delaunay 回退路径需要 rtree 包。请运行: pip install rtree"
                ) from exc
            raise
        return interior_pts[inside]

    def _sample_layer_interface_points(
        self,
        z_values: np.ndarray,   # (n_layers+1,) 每层接口 Z 坐标
        spacing: float,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """在每层接口平面上采样约束点。

        在每一 Z 接口平面生成规则网格点，过滤掉 STL 外部的点，
        作为"保面"约束点插入点云，保证 Delaunay 剖分后层间
        存在明确的接口面。

        Parameters
        ----------
        z_values : np.ndarray
            每层接口的 Z 坐标。
        spacing : float
            平面内采样间距 (m)。
        bounds : np.ndarray
            模型包围盒。

        Returns
        -------
        np.ndarray, shape (L, 3)
        """
        x_min, x_max = float(bounds[0, 0]), float(bounds[1, 0])
        y_min, y_max = float(bounds[0, 1]), float(bounds[1, 1])
        x_vals = np.arange(x_min, x_max + spacing, spacing)
        y_vals = np.arange(y_min, y_max + spacing, spacing)

        all_pts: list[np.ndarray] = []
        for z in z_values:
            xx, yy = np.meshgrid(x_vals, y_vals)
            grid_2d = np.column_stack((xx.ravel(), yy.ravel()))
            pts_3d = np.column_stack(
                (grid_2d, np.full(len(grid_2d), z, dtype=float))
            )
            # 过滤 STL 外部的点
            inside = self.tri_mesh.contains(pts_3d)
            if np.any(inside):
                all_pts.append(pts_3d[inside])

        if all_pts:
            return np.vstack(all_pts)
        return np.zeros((0, 3), dtype=float)

    # ------------------------------------------------------------------
    # Delaunay 剖分 + 过滤
    # ------------------------------------------------------------------

    @staticmethod
    def _delaunay_tetrahedralize(points: np.ndarray) -> np.ndarray:
        """执行 Delaunay 四面体剖分。

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)

        Returns
        -------
        np.ndarray, shape (T, 4)
            四面体顶点索引（索引引用 ``points``）。
        """
        from scipy.spatial import Delaunay

        tri = Delaunay(points)
        return tri.simplices

    def _filter_exterior_tets(
        self,
        points: np.ndarray,     # (N, 3)
        tets: np.ndarray,       # (T, 4)
        max_circumradius: float | None = None,
    ) -> np.ndarray:
        """过滤外部四面体。

        两个标准：
        1. **内外判据**：四面体质心在 STL 外部 → 剔除
        2. **α-shape 判据**：外接球半径过大（空洞填充） → 剔除

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
        tets : np.ndarray, shape (T_in, 4)
        max_circumradius : float | None
            最大外接球半径，None 则设为 resolution * 3。

        Returns
        -------
        np.ndarray, shape (T_out, 4)
        """
        centroids = np.mean(points[tets], axis=1)  # (T, 3)

        # 判据 1: 内部/外部
        inside = self.tri_mesh.contains(centroids)

        # 判据 2: α-shape 外接球半径
        if max_circumradius is None:
            max_circumradius = self.resolution * 3.0
        keep = np.zeros(len(tets), dtype=bool)

        for i in range(len(tets)):
            if not inside[i]:
                continue
            # 计算四面体外接球半径
            p0, p1, p2, p3 = points[tets[i]]
            # 使用底面的外接圆半径 + 高度估计
            # 简化版：计算顶点到质心的最大距离作为半对角线
            c = centroids[i]
            max_dist = max(
                float(np.linalg.norm(p0 - c)),
                float(np.linalg.norm(p1 - c)),
                float(np.linalg.norm(p2 - c)),
                float(np.linalg.norm(p3 - c)),
            )
            if max_dist <= max_circumradius:
                keep[i] = True

        return tets[keep]

    # ------------------------------------------------------------------
    # 主构建流程
    # ------------------------------------------------------------------

    def build_layered_mesh(
        self, config: SimulationConfig | None = None
    ) -> tuple[MeshState, int]:
        """构建非结构化分层四面体网格（Delaunay 剖分）。

        这是替代 :meth:`STLMesher.build_layered_mesh` 的核心方法。

        Parameters
        ----------
        config : SimulationConfig | None
            仿真配置。若为 None，使用默认配置。

        Returns
        -------
        mesh : MeshState
            包含完整拓扑的网格状态。
        n_layers : int
            总打印层数。
        """
        self._load()
        cfg = config or SimulationConfig(layer_thickness=self.layer_thickness)
        if cfg.layer_thickness <= 0:
            cfg.layer_thickness = self.layer_thickness

        tri_mesh = self.tri_mesh
        bounds = tri_mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        z_min = float(bounds[0, 2])
        z_max = float(bounds[1, 2])
        model_dims = bounds[1] - bounds[0]
        model_volume = float(np.prod(model_dims))

        # ── 1. 计算层数 ──
        n_layers = max(1, int(math.ceil((z_max - z_min) / cfg.layer_thickness)))

        # ── 2. 自适应采样预算分配 ──
        r = self.resolution   # 目标边长 (m)
        # 估计体积内的点数
        est_surface_points = int(model_volume / (r ** 3) * 0.15)
        est_surface_points = max(200, min(est_surface_points, self.max_points // 3))
        est_interior_points = min(
            int(self.max_points * 0.5),
            self.max_points - est_surface_points - 500,
        )
        est_layer_points = min(
            int(self.max_points * 0.2),
            self.max_points - est_surface_points - est_interior_points - 100,
        )

        # ── 3. 表面采样 ──
        surface_pts = self._sample_surface_points(est_surface_points)

        # ── 4. 内部填充 ──
        interior_pts = self._sample_interior_points(r * 1.2, bounds)

        # ── 5. 层切面约束点 ──
        z_interfaces = z_min + np.arange(n_layers + 1) * cfg.layer_thickness
        layer_pts = self._sample_layer_interface_points(
            z_interfaces, spacing=r * 1.5, bounds=bounds
        ) if n_layers >= 1 else np.zeros((0, 3), dtype=float)

        # ── 6. 合并点云 ──
        all_pts_parts = [surface_pts]
        if len(interior_pts) > 0:
            all_pts_parts.append(interior_pts)
        if len(layer_pts) > 0:
            all_pts_parts.append(layer_pts)
        all_points = np.vstack(all_pts_parts)

        # 去重（距离 < r * 0.01 的点合并）
        if len(all_points) > 1:
            from scipy.spatial import KDTree
            tree = KDTree(all_points)
            pairs = tree.query_pairs(r * 0.01, output_type='ndarray')
            if len(pairs) > 0:
                mask = np.ones(len(all_points), dtype=bool)
                # 保留每组中的第一个
                kept = set(range(len(all_points)))
                for i, j in pairs:
                    if j in kept and i in kept:
                        kept.discard(j)
                all_points = all_points[list(kept)]

        if len(all_points) < 4:
            raise ValueError(
                f"采样点数不足 ({len(all_points)} 个)。"
                f" 请减小 resolution（当前 {r*1000:.1f} mm）或增大 max_points。"
            )

        # ── 7. Delaunay 四面体剖分 ──
        try:
            tets_raw = self._delaunay_tetrahedralize(all_points)
        except Exception as exc:
            raise RuntimeError(
                f"Delaunay 剖分失败: {exc}。"
                " 请尝试增大 resolution 或简化 STL 模型。"
            ) from exc

        # ── 8. 过滤外部四面体 ──
        tets = self._filter_exterior_tets(all_points, tets_raw)

        if len(tets) < 1:
            raise ValueError(
                f"过滤后无有效四面体（原始 {len(tets_raw)} 个，全部被剔除）。"
                f" 请检查 STL 水密性，或减小 resolution（当前 {r*1000:.1f} mm）。"
            )

        # ── 9. 层归属分类 ──
        tet_centroids = np.mean(all_points[tets], axis=1)
        tet_z = tet_centroids[:, 2]
        layer_id_per_tet = _classify_tets_to_layers(
            tet_z, z_min, n_layers, cfg.layer_thickness
        )

        point_z = all_points[:, 2]
        layer_id_per_vertex, interface_id = _classify_points_to_layers(
            point_z, z_min, n_layers, cfg.layer_thickness
        )

        is_bottom_surface = np.isclose(point_z, z_min, atol=cfg.layer_thickness * 0.2)

        # ── 10. 组装 MeshState ──
        mesh = MeshState(
            vertices=np.asarray(all_points, dtype=float),
            tets=np.asarray(tets, dtype=int),
            layer_id_per_vertex=np.asarray(layer_id_per_vertex, dtype=int),
            first_active_layer=np.asarray(layer_id_per_vertex, dtype=int),
            layer_id_per_tet=np.asarray(layer_id_per_tet, dtype=int),
            ideal_vertices=np.asarray(all_points, dtype=float).copy(),
            is_bottom_surface=np.asarray(is_bottom_surface, dtype=bool),
            is_top_surface_of_layer=np.asarray(interface_id, dtype=int),
        )
        mesh.precompute_reference_matrices(cfg.c_shrink)
        mesh.node_mass = mesh._build_node_masses(cfg.rho)
        mesh.colors = greedy_vertex_coloring(mesh)

        return mesh, n_layers


# ============================================================================
# STLMesher —— 旧版体素化兼容包装
# ============================================================================

class STLMesher:
    """网格构建器统一包装（自动选择最优方案）。

    根据环境（Gmsh 是否可用、模型格式）自动选择：
    1. **优先**：:class:`OCCFragmentMesher` —— OCC Boolean Fragment 分层网格
    2. **回退**：:class:`DelaunayTetMesher` —— scipy Delaunay 点云剖分

    .. deprecated:: 2.0
        直接使用 :class:`OCCFragmentMesher` 或 :class:`DelaunayTetMesher`
        可获得更明确的控制。
    """

    def __init__(
        self,
        stl_path: str,
        layer_thickness: float = 5e-5,
        resolution: float = 0.02,
        max_points: int = 50000,
    ) -> None:
        self._stl_path = stl_path
        self._layer_thickness = float(layer_thickness)
        self._resolution = float(resolution)
        self._max_points = int(max_points)

    @property
    def stl_path(self) -> str:
        """模型文件路径（兼容性属性）。"""
        return self._stl_path

    @property
    def layer_thickness(self) -> float:
        """层厚 (m)。"""
        return self._layer_thickness

    @property
    def resolution(self) -> float:
        """网格分辨率 (m)。"""
        return self._resolution

    def build_layered_mesh(
        self, config: SimulationConfig | None = None
    ) -> tuple[MeshState, int]:
        """自动选择最优网格构建方案。

        优先尝试 OCC Boolean Fragment，失败时回退到 Delaunay 点云剖分。

        Parameters
        ----------
        config : SimulationConfig | None
            仿真配置。

        Returns
        -------
        tuple[MeshState, int]
        """
        # 优先尝试 OCC Boolean Fragment
        if _GMSH_AVAILABLE:
            try:
                occ_mesher = OCCFragmentMesher(
                    stl_path=self._stl_path,
                    layer_thickness=self._layer_thickness,
                    resolution=self._resolution,
                    max_points=self._max_points,
                )
                return occ_mesher.build_layered_mesh(config)
            except Exception:
                # OCC 失败，静默回退
                pass

        # 回退到 Delaunay 点云剖分
        return DelaunayTetMesher(
            stl_path=self._stl_path,
            layer_thickness=self._layer_thickness,
            resolution=self._resolution,
            max_points=self._max_points,
        ).build_layered_mesh(config)


# ============================================================================
# create_demo_or_stl —— 统一入口
# ============================================================================

def create_demo_or_stl(
    stl_path: str | None = None,
    layers: int = 1,
    layer_thickness: float = 5e-5,
    resolution: float = 0.02,
    config: SimulationConfig | None = None,
    **kwargs: Any,
) -> tuple[MeshState, int]:
    """统一接口：从 STL / STEP 加载或退回 demo 正方体。

    根据模型格式和 Gmsh 可用性自动选择最优网格构建方案：
    * STEP (.step / .stp) 等 B-Rep 格式 → OCC Boolean Fragment（推荐）
    * STL 格式 + Gmsh 可用 → 优先尝试 OCC，失败回退 Delaunay
    * STL 格式 + Gmsh 不可用 → Delaunay 点云剖分
    * Demo 模式 → ConformalMeshPipeline

    Parameters
    ----------
    stl_path : str | None
        模型文件路径（.stl / .step / .stp 等），None 时使用 demo 正方体。
    layers : int
        demo 模式下的层数（STL 模式下忽略）。
    layer_thickness : float
        层厚 (m)。
    resolution : float
        网格分辨率 (m)。
    config : SimulationConfig | None
        仿真配置。
    **kwargs
        透传参数（quality_factor, max_points 等）。

    Returns
    -------
    mesh : MeshState
    n_layers : int
    """
    if stl_path is not None:
        # ── STL / STEP 模式 ──
        ext = os.path.splitext(stl_path)[1].lower()
        is_step = ext in ('.step', '.stp', '.igs', '.iges', '.brep')

        if _GMSH_AVAILABLE:
            # Gmsh 可用：使用 OCC Fragment（自动带 Delaunay 回退）
            mesher = OCCFragmentMesher(
                stl_path=stl_path,
                layer_thickness=layer_thickness,
                resolution=resolution,
                max_points=kwargs.get('max_points', 30000),
            )
            return mesher.build_layered_mesh(config)
        else:
            # Gmsh 不可用：直接使用 Delaunay 点云剖分
            if is_step:
                warnings.warn(
                    "STEP 格式模型需要 Gmsh（OCC 几何内核）支持，"
                    "但当前环境未安装 Gmsh。请运行: pip install gmsh"
                    "（将尝试使用 trimesh 加载，可能失败）"
                )
            mesher = DelaunayTetMesher(
                stl_path=stl_path,
                layer_thickness=layer_thickness,
                resolution=resolution,
                max_points=kwargs.get('max_points', 30000),
            )
            return mesher.build_layered_mesh(config)

    # 回退到 demo 正方体
    from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline

    return ConformalMeshPipeline.create_demo(
        layers=layers,
        layer_thickness=layer_thickness,
        config=config,
    )
