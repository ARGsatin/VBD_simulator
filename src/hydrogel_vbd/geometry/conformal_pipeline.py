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
"""

from __future__ import annotations

import numpy as np

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.solver.graph_coloring import greedy_vertex_coloring
from hydrogel_vbd.core.state import MeshState


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
            # Top-down printing: layer 0 is the geometric top layer attached to
            # the build platform, then later layers grow downward toward the FEP.
            first = max(layers - surface_id - 1, 0)
            first_active.extend([first] * 4)
            interface_ids.extend([layers - surface_id] * 4)
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
            tet_layers.extend([layers - 1 - layer] * len(layer_tets))

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
