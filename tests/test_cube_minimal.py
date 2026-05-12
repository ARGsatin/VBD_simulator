# -*- coding: utf-8 -*-
"""正方体最小网格测试 —— tet 数量 > 1 时的弹性求解验证。

从单个四面体递进到正方体（5-6 tet），验证求解器在
较小但非平凡网格上的行为。

.. note::
    本文件不依赖 GUI、gmsh、IO 或完整仿真管道。
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.core.state import MeshState
from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver
from tests.validator import make_unit_cube_tets


def _make_cube_mesh(
    vertices: np.ndarray,
    tets: np.ndarray,
    fix_top: bool = True,
    fix_bottom: bool = False,
) -> MeshState:
    """构造正方体网格 MeshState。"""
    n = vertices.shape[0]
    layer_ids = np.zeros(n, dtype=np.int32)

    mesh = MeshState(
        vertices=vertices.copy(),
        tets=tets.copy(),
        layer_id_per_vertex=layer_ids,
        layer_id_per_tet=np.zeros(len(tets), dtype=np.int32),
    )
    mesh.ideal_vertices = vertices.copy()
    mesh.active_vertex_mask = np.ones(n, dtype=bool)
    mesh.active_tet_mask = np.ones(len(tets), dtype=bool)
    mesh.node_mass = np.full(n, 1.0)
    mesh.prev_vertices = vertices.copy()
    mesh.velocities = np.zeros_like(vertices)
    mesh.is_bottom_surface = np.zeros(n, dtype=bool)
    mesh.is_top_fixed = np.zeros(n, dtype=bool)

    # 顶面 = Z 最大的顶点，底面 = Z 最小的顶点
    z_max = vertices[:, 2].max()
    z_min = vertices[:, 2].min()
    if fix_top:
        mesh.is_top_fixed[vertices[:, 2] >= z_max - 0.01] = True
    if fix_bottom:
        mesh.is_bottom_surface[vertices[:, 2] <= z_min + 0.01] = True

    mesh.czm_state = np.full(n, 2, dtype=np.int32)  # all FREE (no CZM)
    mesh.damage = np.zeros(n, dtype=np.float64)
    mesh.time_free = np.zeros(n, dtype=np.float64)

    # 预计算 dm_inv 和体积
    mesh.precompute_reference_matrices(c_shrink=1.0)
    mesh.colors = _simple_coloring(mesh)

    return mesh


def _simple_coloring(mesh: MeshState) -> np.ndarray:
    """简化贪心着色（不导入 graph_coloring 模块）。"""
    n = mesh.vertices.shape[0]
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for tet in mesh.tets:
        for i in tet:
            i_int = int(i)
            neighbors[i_int].update(int(j) for j in tet if int(j) != i_int)
    colors = np.full(n, -1, dtype=np.int32)
    for v_id, adj in enumerate(neighbors):
        used = {colors[nb] for nb in adj if colors[nb] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[v_id] = c
    return colors


def _make_cube_config(**kwargs) -> SimulationConfig:
    """正方体测试用配置。"""
    defaults = dict(
        mu=50000.0, kappa=1.0e7, c_shrink=1.0,
        g=(0.0, 0.0, -9.81),
        q_ion=1.2e-3, v_lift=0.0, T_max=0.0,
        dt=0.01, max_iters=50, epsilon=1e-6, N_stable=5,
    )
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


class TestCubeGravity:
    """正方体受重力拉伸测试。"""

    def test_gravity_stretches_cube(self) -> None:
        """固定顶面，底面受重力向下 → 底部 Z 应 < 初始值。"""
        verts, tets = make_unit_cube_tets(n=2)
        mesh = _make_cube_mesh(verts, tets, fix_top=True)

        config = _make_cube_config()
        solver = PythonReferenceVBDSolver(config)

        z_bottom_initial = np.min(mesh.vertices[~mesh.is_top_fixed, 2])

        result = solver.solve_until_stable(mesh, layer_id=0, e_z=0.0)

        z_bottom_final = np.min(mesh.vertices[~mesh.is_top_fixed, 2])
        # 底部应被拉下
        assert z_bottom_final < z_bottom_initial, (
            f"重力未拉伸: z_initial={z_bottom_initial:.6f}, z_final={z_bottom_final:.6f}"
        )
        # 无 NaN
        assert not np.any(np.isnan(mesh.vertices)), "顶点含 NaN"

    def test_cube_converges(self) -> None:
        """求解应在 max_iters 内收敛。"""
        verts, tets = make_unit_cube_tets(n=2)
        mesh = _make_cube_mesh(verts, tets, fix_top=True)

        config = _make_cube_config(max_iters=50, epsilon=1e-4)
        solver = PythonReferenceVBDSolver(config)

        result = solver.solve_until_stable(mesh, layer_id=0, e_z=0.0)

        assert result.iterations <= config.max_iters, (
            f"未在 max_iters={config.max_iters} 内完成, 实际 {result.iterations}"
        )
        assert result.max_dx < 0.01, f"最终 max_dx={result.max_dx:.2e} 过大"


class TestCubeElectricField:
    """正方体 + 电场测试。"""

    def test_electric_field_lifts_bottom(self) -> None:
        """固定顶面，电场向上 → 底部 Z 应上移。"""
        verts, tets = make_unit_cube_tets(n=2)
        mesh_no_e = _make_cube_mesh(verts.copy(), tets, fix_top=True)

        # 无电场求解
        config = _make_cube_config()
        solver = PythonReferenceVBDSolver(config)
        solver.solve_until_stable(mesh_no_e, layer_id=0, e_z=0.0)
        z_bottom_no_e = np.min(mesh_no_e.vertices[~mesh_no_e.is_top_fixed, 2])

        # 有电场求解
        mesh_with_e = _make_cube_mesh(verts.copy(), tets, fix_top=True)
        solver2 = PythonReferenceVBDSolver(config)
        solver2.solve_until_stable(mesh_with_e, layer_id=0, e_z=500.0)
        z_bottom_with_e = np.min(mesh_with_e.vertices[~mesh_with_e.is_top_fixed, 2])

        # 电场应抬升底部
        assert z_bottom_with_e > z_bottom_no_e, (
            f"电场未抬升: no_e={z_bottom_no_e:.6f}, with_e={z_bottom_with_e:.6f}"
        )
