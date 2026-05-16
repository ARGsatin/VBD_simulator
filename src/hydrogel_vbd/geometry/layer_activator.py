# -*- coding: utf-8 -*-
"""层激活器 —— 管理网格顶点的逐层激活与状态继承。

DLP 3D 打印是**逐层固化**的过程：每曝光一层，该层新固化的顶点
才被"激活"进入仿真，而之前各层的顶点已经处于活动状态。

本模块提供激活器（``LayerActivator``），负责在每层开始打印前：
1. 更新活动顶点掩码
2. 继承上一层底面变形信息
3. 重置新层顶点的速度、CZM 状态与损伤
4. 将顶层节点临时固定（模拟平台夹持）
"""

from __future__ import annotations

import numpy as np

from hydrogel_vbd.physics.czm import CZMState
from hydrogel_vbd.core.state import MeshState


class LayerActivator:
    """管理网格顶点逐层激活与边界状态重置。

    每次调用 ``activate()`` 或 ``activate_with_inheritance()``
    都会根据当前层号更新 mesh 的活动掩码，并设置新激活顶点的
    初始运动学和 CZM 状态。
    """

    @staticmethod
    def _geometric_layer_top_nodes(
        mesh: MeshState,
        current_layer: int,
    ) -> np.ndarray:
        """当接口编号缺失时，用当前层四面体的最高 Z 节点恢复顶面。"""
        if mesh.tets is None or mesh.layer_id_per_tet is None:
            return np.zeros(0, dtype=int)
        layer_tets = mesh.tets[mesh.layer_id_per_tet == current_layer]
        if layer_tets.size == 0:
            return np.zeros(0, dtype=int)

        layer_nodes = np.unique(layer_tets.ravel())
        if layer_nodes.size == 0:
            return np.zeros(0, dtype=int)

        z_values = mesh.vertices[layer_nodes, 2]
        z_max = float(np.max(z_values))
        z_span = float(np.ptp(mesh.vertices[:, 2])) if mesh.vertices.size else 0.0
        tol = max(1e-12, z_span * 1e-9)
        return np.asarray(
            layer_nodes[np.isclose(z_values, z_max, atol=tol)],
            dtype=int,
        )

    @staticmethod
    def _infer_layer_thickness(mesh: MeshState, current_layer: int) -> float:
        bottom = mesh.bottom_nodes(current_layer)
        top = mesh.top_nodes(current_layer)
        if len(bottom) and len(top):
            return float(
                np.median(mesh.ideal_vertices[top, 2])
                - np.median(mesh.ideal_vertices[bottom, 2])
            )
        surface_ids = np.unique(mesh.is_top_surface_of_layer)
        surface_ids = surface_ids[surface_ids >= 0]
        z_levels = []
        for sid in surface_ids:
            nodes = mesh.layer_interface_nodes(int(sid))
            if len(nodes):
                z_levels.append(float(np.median(mesh.ideal_vertices[nodes, 2])))
        if len(z_levels) < 2:
            return 0.0
        diffs = np.diff(np.sort(np.asarray(z_levels, dtype=float)))
        diffs = diffs[diffs > 1e-12]
        return float(np.median(diffs)) if len(diffs) else 0.0

    @staticmethod
    def _reset_reference_for_tets(mesh: MeshState, tet_ids: np.ndarray) -> None:
        if mesh.tets is None or mesh.dm_inv is None or mesh.tet_volumes is None:
            return
        for tet_id in np.asarray(tet_ids, dtype=int):
            if tet_id < 0 or tet_id >= mesh.tets.shape[0]:
                continue
            tet = mesh.tets[tet_id]
            p0, p1, p2, p3 = mesh.vertices[tet]
            matrix = np.column_stack((p1 - p0, p2 - p0, p3 - p0))
            det = float(np.linalg.det(matrix))
            if abs(det) <= 1e-12:
                continue
            if mesh.dm is not None:
                mesh.dm[tet_id] = matrix
            mesh.dm_inv[tet_id] = np.linalg.inv(matrix)
            mesh.tet_volumes[tet_id] = abs(det) / 6.0

    @staticmethod
    def _lower_active_stack_to_contact(
        mesh: MeshState,
        current_layer: int,
        z_fep: float,
    ) -> None:
        if current_layer <= 0:
            return
        active = np.asarray(mesh.active_vertex_mask, dtype=bool).copy()
        previous_bottom = mesh.bottom_nodes(current_layer - 1)
        previous_bottom = (
            previous_bottom[active[previous_bottom]]
            if len(previous_bottom)
            else previous_bottom
        )
        thickness = LayerActivator._infer_layer_thickness(mesh, current_layer)
        if len(previous_bottom) and thickness > 0.0:
            target_z = float(z_fep) + thickness
            dz = target_z - float(np.median(mesh.vertices[previous_bottom, 2]))
            mesh.vertices[active, 2] += dz
            mesh.velocities[active] = 0.0

    def activate(self, mesh: MeshState, current_layer: int) -> MeshState:
        """激活指定层（轻量版，不处理继承）。

        仅调用 ``mesh.activate_layer()`` 更新活动顶点掩码，
        使 ``current_layer`` 及之前所有层的顶点进入活动状态。

        Parameters
        ----------
        mesh : MeshState
            待更新的网格状态（将被原地修改）。
        current_layer : int
            当前打印层编号（从 0 开始）。

        Returns
        -------
        MeshState
            更新后的 mesh（原地返回，方便链式调用）。
        """
        mesh.activate_layer(current_layer)
        return mesh

    def activate_with_inheritance(
        self,
        mesh: MeshState,
        current_layer: int,
        z_fep: float = 0.0,
    ) -> MeshState:
        """带变形继承的层激活（完整版）。

        在激活新层之前处理关键的物理连续性：

        1. **碰撞修正**：对上一层的底面节点，
           若 z < z_fep（穿透离型膜）则修正到 z_fep 并清零速度。
        2. **激活新层**：调用 ``activate_layer()``。
        3. **坐标初始化**：
           - 顶层节点（平台夹持）：z 不超过理想值
           - 底面节点（新层底面）：z 锁定在 z_fep（离型膜面）
           - 内部节点：z 钳位在 [z_fep, ideal_z]
        4. **状态重置**：
           - 新激活节点速度归零
           - 顶层节点标记为固定（``is_top_fixed = True``）
           - 新激活节点 CZM 状态初始化为 FREE（顶/内部）或 FIXED（底面）
           - 损伤和时间计数器归零

        Parameters
        ----------
        mesh : MeshState
            待更新的网格状态（将被原地修改）。
        current_layer : int
            当前打印层编号（从 0 开始）。
        z_fep : float, optional
            离型膜平面（FEP）的高度，默认 0.0。

        Returns
        -------
        MeshState
            原地修改后的 mesh。

        Notes
        -----
        * ``current_layer == 0`` 时无"上一层"的继承步骤。
        * 底面节点初始化为 ``CZMState.FIXED`` 表示粘附在离型膜上，
          其余节点为 ``FREE``。
        * ``is_top_fixed`` 掩码在每个新层激活时被重新覆盖。
        """
        active_before = np.asarray(mesh.active_vertex_mask, dtype=bool).copy()
        self._lower_active_stack_to_contact(mesh, current_layer, z_fep)

        # ── 激活新层 ──
        mesh.activate_layer(current_layer)

        # ── 获取新激活节点的各子集 ──
        born_nodes = np.flatnonzero(
            mesh.first_active_layer == current_layer
        )
        new_nodes = born_nodes
        top = mesh.top_nodes(current_layer)       # 新层顶层（平台夹持面）
        bottom = mesh.bottom_nodes(current_layer)  # 新层底面（离型膜接触面）
        if len(top) == 0:
            top = self._geometric_layer_top_nodes(mesh, current_layer)

        # ── 顶层节点：z 不超过理想构型 ──
        # ── 底面节点：z 锁定在离型膜面 ──
        face_nodes = np.union1d(top, bottom)
        if len(face_nodes):
            mesh.active_vertex_mask[face_nodes] = True
            new_nodes = np.union1d(new_nodes, face_nodes)

        if len(bottom):
            dz_new = float(z_fep) - float(np.median(mesh.ideal_vertices[bottom, 2]))
            coordinate_nodes = np.union1d(born_nodes, bottom)
            if len(top):
                fresh_top = top[~active_before[top]]
                coordinate_nodes = np.union1d(coordinate_nodes, fresh_top)
            mesh.vertices[coordinate_nodes, 2] = (
                mesh.ideal_vertices[coordinate_nodes, 2] + dz_new
            )
            mesh.vertices[bottom, 2] = z_fep

        # ── 内部节点：z 钳位在 [z_fep, ideal_z] ──
        interior = np.setdiff1d(
            new_nodes, np.union1d(top, bottom), assume_unique=False
        )
        for idx in interior:
            mesh.vertices[idx, 2] = max(mesh.vertices[idx, 2], z_fep)

        layer_tets = np.flatnonzero(mesh.layer_id_per_tet == current_layer)
        self._reset_reference_for_tets(mesh, layer_tets)

        # ── 重置运动学状态 ──
        mesh.velocities[new_nodes] = 0.0

        # ── 固定顶层节点（模拟平台夹持） ──
        if current_layer == 0 or not np.any(mesh.is_top_fixed & mesh.active_vertex_mask):
            mesh.is_top_fixed[:] = False
            mesh.is_top_fixed[top] = True

        # ── 初始化 CZM 状态 ──
        mesh.czm_state[new_nodes] = CZMState.FREE       # 默认自由
        mesh.czm_state[bottom] = CZMState.FIXED          # 底面粘附

        # ── 重置损伤与时间计数器 ──
        mesh.damage[new_nodes] = 0.0
        mesh.time_free[new_nodes] = 0.0

        return mesh
