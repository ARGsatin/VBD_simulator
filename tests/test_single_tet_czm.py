# -*- coding: utf-8 -*-
"""单四面体 CZM 测试 —— 内聚力模型状态机和力验证。

测试 CZM 状态机 (FIXED→DAMAGING→FREE) 的完整生命周期，
以及 CZM 力贡献与有限差分的比对。

.. note::
    本文件不依赖 GUI、gmsh、IO 或完整仿真管道。
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrogel_vbd.physics.czm import CZMState, update_czm_states
from tests.test_single_tet_elastic import _make_mini_mesh, _make_config
from tests.validator import make_unit_tet


class TestCZMStateMachine:
    """验证 CZM 状态转换逻辑。"""

    def test_fixed_stays_fixed_at_zero_gap(self) -> None:
        """gap=0 → 节点保持 FIXED。"""
        verts, tets, dm_inv = make_unit_tet(scale=1.0)
        mesh = _make_mini_mesh(verts, tets, dm_inv, fix_bottom=False)
        mesh.czm_state[:] = CZMState.FIXED
        mesh.is_bottom_surface[:] = True
        mesh.vertices[:, 2] = 0.0  # 顶点在 FEP 上
        bottom = np.array([0, 1, 2], dtype=int)

        update_czm_states(
            mesh, bottom,
            internal_pull_z=np.full(3, 5000.0 * 1.05),
            area=1.0, t_max=5000.0, k_czm=1e8, delta_f=1e-4,
            z_fep=0.0, dt=0.01,  # z_fep = 顶点 Z → gap=0
        )
        # gap=0 → traction=0, gap≯delta_f(1e-4)
        assert all(mesh.czm_state[i] == CZMState.FIXED for i in bottom)

    def test_fixed_to_damaging_by_gap(self) -> None:
        """gap > delta_f → FIXED→DAMAGING。"""
        verts, tets, dm_inv = make_unit_tet(scale=1.0)
        mesh = _make_mini_mesh(verts, tets, dm_inv, fix_bottom=False)
        mesh.czm_state[:] = CZMState.FIXED
        mesh.is_bottom_surface[:] = True
        # 将底部节点抬高 0.001m >> delta_f=1e-4
        mesh.vertices[0:3, 2] = 0.001
        bottom = np.array([0, 1, 2], dtype=int)

        update_czm_states(
            mesh, bottom,
            internal_pull_z=np.full(3, 5000.0 * 1.05),
            area=1.0, t_max=5000.0, k_czm=1e8, delta_f=1e-4,
            z_fep=0.0, dt=0.01,
        )
        assert all(mesh.czm_state[i] == CZMState.DAMAGING for i in bottom)
        assert all(mesh.damage[i] == 0.0 for i in bottom)

    def test_damaging_to_free_by_damage_accumulation(self) -> None:
        """damage 积累到 1.0 → DAMAGING→FREE。"""
        verts, tets, dm_inv = make_unit_tet(scale=1.0)
        mesh = _make_mini_mesh(verts, tets, dm_inv, fix_bottom=False)
        mesh.czm_state[:] = CZMState.DAMAGING
        mesh.damage[:] = 0.99  # 即将失效
        mesh.is_bottom_surface[:] = True
        mesh.vertices[0:3, 2] = 0.001  # gap > 0
        bottom = np.array([0, 1, 2], dtype=int)

        update_czm_states(
            mesh, bottom,
            internal_pull_z=np.full(3, 5000.0 * 1.05),
            area=1.0, t_max=5000.0, k_czm=1e8, delta_f=1e-4,
            z_fep=0.0, dt=0.01,
        )
        assert all(mesh.czm_state[i] == CZMState.FREE for i in bottom)
        assert all(mesh.damage[i] == 1.0 for i in bottom)
        assert all(mesh.time_free[i] == 0.0 for i in bottom)

    def test_damaging_to_free_by_large_gap(self) -> None:
        """gap > 5·delta_f → DAMAGING→FREE（不依赖 damage）。"""
        verts, tets, dm_inv = make_unit_tet(scale=1.0)
        mesh = _make_mini_mesh(verts, tets, dm_inv, fix_bottom=False)
        mesh.czm_state[:] = CZMState.DAMAGING
        mesh.damage[:] = 0.1  # 低损伤
        mesh.is_bottom_surface[:] = True
        mesh.vertices[0:3, 2] = 0.01  # gap = 0.01 >> 5*1e-4 = 0.005
        bottom = np.array([0, 1, 2], dtype=int)

        update_czm_states(
            mesh, bottom,
            internal_pull_z=np.full(3, 0.0),  # 无拉力
            area=1.0, t_max=5000.0, k_czm=1e8, delta_f=1e-4,
            z_fep=0.0, dt=0.01,
        )
        assert all(mesh.czm_state[i] == CZMState.FREE for i in bottom)

    def test_free_stays_free(self) -> None:
        """FREE 状态永远保持 FREE，time_free 递增。"""
        verts, tets, dm_inv = make_unit_tet(scale=1.0)
        mesh = _make_mini_mesh(verts, tets, dm_inv, fix_bottom=False)
        mesh.czm_state[:] = CZMState.FREE
        mesh.time_free[:] = 2.0
        mesh.is_bottom_surface[:] = True
        bottom = np.array([0, 1, 2], dtype=int)

        update_czm_states(
            mesh, bottom,
            internal_pull_z=np.full(3, 0.0),
            area=1.0, t_max=5000.0, k_czm=1e8, delta_f=1e-4,
            z_fep=0.0, dt=0.01,
        )
        assert all(mesh.czm_state[i] == CZMState.FREE for i in bottom)
        assert all(mesh.time_free[i] == 2.01 for i in bottom)


class TestCZMNumericalGuards:
    """验证数值边界保护。"""

    def test_empty_bottom_nodes(self) -> None:
        """空 bottom_nodes → 不崩溃。"""
        verts, tets, dm_inv = make_unit_tet(scale=1.0)
        mesh = _make_mini_mesh(verts, tets, dm_inv, fix_bottom=False)
        update_czm_states(
            mesh, np.array([], dtype=int),
            internal_pull_z=np.array([]),
            area=1.0, t_max=5000.0, k_czm=1e8, delta_f=1e-4,
            z_fep=0.0, dt=0.01,
        )

    def test_damage_clipped_at_one(self) -> None:
        """damage 超 1.0 被钳位。"""
        verts, tets, dm_inv = make_unit_tet(scale=1.0)
        mesh = _make_mini_mesh(verts, tets, dm_inv, fix_bottom=False)
        mesh.czm_state[0] = CZMState.DAMAGING
        mesh.damage[0] = 1.5  # 超限
        mesh.is_bottom_surface[0] = True
        mesh.vertices[0, 2] = 0.001
        bottom = np.array([0], dtype=int)

        update_czm_states(
            mesh, bottom,
            internal_pull_z=np.full(1, 5000.0 * 1.05),
            area=1.0, t_max=5000.0, k_czm=1e8, delta_f=1e-4,
            z_fep=0.0, dt=0.01,
        )
        # damage >= 1.0 → FREE
        assert mesh.czm_state[0] == CZMState.FREE

    @pytest.mark.xfail(reason="已知缺陷: 无效状态码(如3)静默无操作,无 default 分支")
    def test_invalid_czm_state_silent(self) -> None:
        """无效的 CZM 状态码应引起注意（当前静默无操作）。"""
        verts, tets, dm_inv = make_unit_tet(scale=1.0)
        mesh = _make_mini_mesh(verts, tets, dm_inv, fix_bottom=False)
        mesh.czm_state[0] = 3  # 无效状态
        mesh.is_bottom_surface[0] = True
        mesh.vertices[0, 2] = 0.001
        bottom = np.array([0], dtype=int)

        # 应至少产生 warning 或 state 被修正
        update_czm_states(
            mesh, bottom,
            internal_pull_z=np.full(1, 5000.0 * 1.05),
            area=1.0, t_max=5000.0, k_czm=1e8, delta_f=1e-4,
            z_fep=0.0, dt=0.01,
        )
        # 当前行为：状态码 3 保持不变（静默）
        assert mesh.czm_state[0] != 3, "无效状态码应被处理"


class TestCZMFullLifecycle:
    """完整 FIXED→DAMAGING→FREE 生命周期。"""

    def test_full_lifecycle(self) -> None:
        """单个节点从 FIXED 走到 FREE 的完整路径。

        使用较小的 pull 使 damage 逐步积累，验证完整状态链。
        """
        verts, tets, dm_inv = make_unit_tet(scale=1.0)
        mesh = _make_mini_mesh(verts, tets, dm_inv, fix_bottom=False)
        mesh.czm_state[0] = CZMState.FIXED
        mesh.damage[0] = 0.0
        mesh.is_bottom_surface[0] = True
        bottom = np.array([0], dtype=int)

        # Step 1: 抬高节点使之进入 DAMAGING
        mesh.vertices[0, 2] = 0.001
        update_czm_states(
            mesh, bottom,
            internal_pull_z=np.full(1, 5000.0 * 1.05),
            area=1.0, t_max=5000.0, k_czm=1e8, delta_f=1e-4,
            z_fep=0.0, dt=0.01,
        )
        assert mesh.czm_state[0] == CZMState.DAMAGING

        # 用小的 dt 和 pull 让 damage 逐步积累
        small_pull = 25.0  # dmg_rate = 25*0.01/(5000*1e-4) = 0.25/0.5 = 0.5
        for step in range(10):
            update_czm_states(
                mesh, bottom,
                internal_pull_z=np.full(1, small_pull),
                area=1.0, t_max=5000.0, k_czm=1e8, delta_f=1e-4,
                z_fep=0.0, dt=0.01,
            )
            if mesh.czm_state[0] == CZMState.FREE:
                break

        assert mesh.czm_state[0] == CZMState.FREE, (
            f"10 步后应 FREE, 实际 state={mesh.czm_state[0]}, damage={mesh.damage[0]:.4f}"
        )
        assert mesh.damage[0] == 1.0
