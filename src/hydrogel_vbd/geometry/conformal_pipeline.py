# -*- coding: utf-8 -*-
"""共形四面体网格构建流水线。

本模块负责创建供仿真使用的分层四面体网格。实际生产中会调用支持
分段线性复合体（PLC）的全功能网格生成器（如 TetGen），此处提供
演示版实现便于快速启动和测试。

核心组件
--------
- **ConformalMeshPipeline**：网格构建工具类

   - ``create_demo()`` 静态工厂方法生成一个多层共形四面体网格，
     层间共享节点，保证应力继承和激活演化的拓扑一致性。
   - ``from_stl()`` 从 STL 文件构建共形分层网格（TetGen 路径）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.geometry.tet_mesher import from_stl as tet_mesh_from_stl
from hydrogel_vbd.solver.graph_coloring import greedy_vertex_coloring
from hydrogel_vbd.core.state import MeshState


def _compute_bottom_surface(vertices: np.ndarray, z_tol: float = 1e-9) -> np.ndarray:
    """检测全局底面顶点（Z 坐标接近 Z_min 的顶点）。"""
    z_min = float(np.min(vertices[:, 2]))
    return np.abs(vertices[:, 2] - z_min) < z_tol


class ConformalMeshPipeline:
    """构建全局共形的分层四面体网格。

    演示路径在每层分割出规则的四边形底面（4 个顶点），
    再用 5 个四面体填充每一层（Shephard 分割），
    这样层间共享接口节点，激活与应力继承在同一个拓扑约定下运作。

    生产环境中，此流水线应替换为对 PLC 网格生成器（如 TetGen）的调用。
    """

    @staticmethod
    def create_demo(
        layers: int,
        layer_thickness: float = 5e-5,
        config: SimulationConfig | None = None,
    ) -> tuple[MeshState, int]:
        """创建演示用分层四面体网格。

        为每一"表面"（layer + 1 个表面）生成 4 个顶点构成单位正方形
        （坐标范围 [0,1]×[0,1]），然后每相邻两层之间用 5 个四面体
        进行 Shephard 六面体分割。

        网格结构示意（2 层示例）：:

            Surface 2:  [v8, v9, v10, v11]   z = 2·h  ← 顶层
                         │ tetra layer 1     │
            Surface 1:  [v4, v5, v6, v7]     z = 1·h  ← 中间接口
                         │ tetra layer 0     │
            Surface 0:  [v0, v1, v2, v3]     z = 0     ← 底层（FEP 接触面）

        Parameters
        ----------
        layers : int
            打印层数（必须 ≥ 1）。
        layer_thickness : float, optional
            每层厚度（m），默认 5e-5（0.05 mm）。
        config : SimulationConfig or None, optional
            仿真配置对象。若为 None，使用默认配置。

        Returns
        -------
        tuple[MeshState, int]
            - **mesh** : 构建好的网格状态对象，已预计算参考矩阵、
              节点质量、图着色等。
            - **layer_count** : 层数（与输入 ``layers`` 相同）。

        Raises
        ------
        ValueError
            如果 ``layers < 1``。

        Notes
        -----
        * 每层由 5 个四面体组成（立方体到四面体的 Shephard 分割）。
        * ``is_bottom_surface`` 标记第一层底面的顶点（CZM 固定）。
        * ``interface_ids`` 将每个顶点关联到其所属表面编号，
          用于 ``layer_id_per_vertex`` 追踪。
        """
        if layers < 1:
            raise ValueError("层数必须为正整数")

        config = config or SimulationConfig(layer_thickness=layer_thickness)

        # ── 数据结构：逐层构建顶点和元数据 ──
        vertices: list[list[float]] = []       # [x, y, z] 坐标
        first_active: list[int] = []            # 每顶点首次激活的层号
        interface_ids: list[int] = []           # 每顶点所属表面编号
        is_bottom: list[bool] = []              # 是否属于底面（FEP 接触面）

        # ── 为每个表面生成 4 个顶点的单位正方形 ──
        for surface_id in range(layers + 1):
            z = surface_id * layer_thickness
            vertices.extend(
                [
                    [0.0, 0.0, z],   # v0: 左下
                    [1.0, 0.0, z],   # v1: 右下
                    [0.0, 1.0, z],   # v2: 左上
                    [1.0, 1.0, z],   # v3: 右上
                ]
            )
            # 每个顶点的首次激活层 = max(surface_id - 1, 0)
            #   surface 0 → layer 0（底层顶点在仿真开始时即激活）
            #   surface k → layer k-1（上层顶点在上一层曝光时激活）
            first = max(surface_id - 1, 0)
            first_active.extend([first] * 4)
            interface_ids.extend([surface_id] * 4)
            # 只有 surface 0 的顶点标记为"底面"
            is_bottom.extend([surface_id == 0] * 4)

        # ── 四面体生成（每层 5 个，Shephard 分割） ──
        # 将底层正方形(b0-b3)与顶层正方形(t0-t3)之间的三棱柱
        # 分割为 5 个非重叠四面体
        tets: list[list[int]] = []
        tet_layers: list[int] = []
        for layer in range(layers):
            b = layer * 4        # 底层顶点起始索引
            t = (layer + 1) * 4  # 顶层顶点起始索引
            layer_tets = [
                [b + 0, b + 1, b + 2, t + 0],  # 前下三棱柱
                [b + 1, b + 3, b + 2, t + 3],  # 后上三棱柱
                [b + 1, t + 1, t + 0, t + 3],  # 右前
                [b + 2, t + 0, t + 2, t + 3],  # 左后
                [b + 1, b + 2, t + 0, t + 3],  # 中心对角
            ]
            tets.extend(layer_tets)
            tet_layers.extend([layer] * len(layer_tets))

        # ── 构造 MeshState ──
        mesh = MeshState(
            vertices=np.asarray(vertices, dtype=float),
            tets=np.asarray(tets, dtype=int),
            layer_id_per_vertex=np.asarray(first_active, dtype=int),
            first_active_layer=np.asarray(first_active, dtype=int),
            layer_id_per_tet=np.asarray(tet_layers, dtype=int),
            is_bottom_surface=np.asarray(is_bottom, dtype=bool),
            is_top_surface_of_layer=np.asarray(interface_ids, dtype=int),
        )

        # ── 预计算网格不变量 ──
        mesh.precompute_reference_matrices(config.c_shrink)  # Dm^{-1}、参考体积
        mesh.node_mass = mesh._build_node_masses(config.rho)  # 节点集中质量
        mesh.colors = greedy_vertex_coloring(mesh)            # 图着色分组

        return mesh, layers

    # ------------------------------------------------------------------
    # STL 流水线
    # ------------------------------------------------------------------

    @staticmethod
    def from_stl(
        stl_path: str | Path,
        layer_height: float,
        config: SimulationConfig | None = None,
        quality: float = 1.0,
    ) -> tuple[MeshState, int]:
        """从 STL 文件构建共形分层四面体网格。

        步骤：
        1. 通过 TetGen 对 STL 进行四面体剖分
        2. 按重心 Z 坐标将每个四面体分配到对应打印层
        3. 计算顶点的首次激活层和顶层表面标记
        4. 检测底面、预计算形函数矩阵、节点质量和图着色

        Parameters
        ----------
        stl_path : str | Path
            输入 STL 文件路径。
        layer_height : float
            打印层厚（与 STL 同单位）。
        config : SimulationConfig or None
            仿真配置。None 则使用默认值。
        quality : float
            TetGen 网格细化因子（0.1 … 5.0，默认 1.0）。

        Returns
        -------
        tuple[MeshState, int]
            - mesh : 含完整层元数据的网格状态对象
            - num_layers : 总层数
        """
        config = config or SimulationConfig(layer_thickness=layer_height)

        # 1. TetGen 四面体剖分
        vertices, tets = tet_mesh_from_stl(stl_path, quality=quality)
        n_vertices = len(vertices)
        z_min = float(np.min(vertices[:, 2]))
        z_max = float(np.max(vertices[:, 2]))

        if z_max - z_min < 1e-12:
            raise ValueError("STL 在 Z 轴方向无厚度")

        num_layers = max(1, int(np.ceil((z_max - z_min) / layer_height)))

        # 2. 按重心 Z 将四面体分配到层
        centroids = vertices[tets].mean(axis=1)
        tet_layers = np.floor((centroids[:, 2] - z_min) / layer_height).astype(np.int32)
        tet_layers = np.clip(tet_layers, 0, num_layers - 1)

        # 3. 每个顶点的首次激活层
        first_active = np.full(n_vertices, num_layers, dtype=np.int32)
        for tid, layer in enumerate(tet_layers):
            first_active[tets[tid]] = np.minimum(first_active[tets[tid]], layer)

        # 4. 每个顶点涉及的层集合（用于检测层间界面顶点）
        vertex_layers: list[set[int]] = [set() for _ in range(n_vertices)]
        for tid, layer in enumerate(tet_layers):
            for v in tets[tid]:
                vertex_layers[int(v)].add(int(layer))

        # 5. is_top_surface_of_layer 标记
        #    跨多层 → 界面顶点（取最大层号）
        #    单一层 → Z 接近层边界时标记
        is_top = np.full(n_vertices, -1, dtype=np.int32)
        for vi, layers_set in enumerate(vertex_layers):
            if len(layers_set) >= 2:
                is_top[vi] = max(layers_set)
            elif len(layers_set) == 1:
                layer = list(layers_set)[0]
                vz = float(vertices[vi, 2])
                lo = z_min + layer * layer_height
                hi = lo + layer_height
                if abs(vz - z_max) < layer_height * 0.05:
                    is_top[vi] = layer + 1
                elif abs(vz - lo) < layer_height * 0.05:
                    is_top[vi] = layer
                elif abs(vz - hi) < layer_height * 0.05:
                    is_top[vi] = layer + 1

        # 6. 底面检测
        is_bottom = _compute_bottom_surface(vertices)

        # 7. 组装 MeshState
        mesh = MeshState(
            vertices=vertices,
            tets=tets,
            layer_id_per_vertex=first_active.copy(),
            first_active_layer=first_active.copy(),
            layer_id_per_tet=tet_layers,
            is_bottom_surface=is_bottom,
            is_top_surface_of_layer=is_top,
        )
        mesh.precompute_reference_matrices(config.c_shrink)
        mesh.node_mass = mesh._build_node_masses(config.rho)
        mesh.colors = greedy_vertex_coloring(mesh)
        return mesh, num_layers
