# -*- coding: utf-8 -*-
"""数值防护与 CZM 对齐的单元测试。

验证 Phase 1.2 / 1.3 的改动不会引入退化或 NaN。
"""

import numpy as np
import pytest

from hydrogel_vbd.physics.elastic_energy import (
    neo_hookean_energy_density,
    neo_hookean_pk1_stress,
    neo_hookean_material_tangent_9x9,
    compute_tet_force_and_hessian_contributions,
)
from hydrogel_vbd.physics.czm import CZMState, update_czm_states
from hydrogel_vbd.core.state import MeshState


# ─────────────────────────────────────────────
# Neo-Hookean J 值防护测试
# ─────────────────────────────────────────────

@pytest.mark.parametrize("J", [0.0, 1e-15, 1e-12, 1e-10, 1e-9, 1e-6, 0.5, 1.0, 2.0])
def test_energy_finite_for_all_J(J: float) -> None:
    """能量密度在所有 J 值下应为有限值。"""
    F = np.eye(3) * (J ** (1.0 / 3.0))
    mu, lam = 1e5, 2e5
    psi = neo_hookean_energy_density(F, mu, lam)
    assert np.isfinite(psi), f"psi={psi} for J={J}"


@pytest.mark.parametrize("J", [1e-15, 1e-10, 1e-9, 1e-6, 0.5, 1.0, 2.0])
def test_pk1_finite_for_all_J(J: float) -> None:
    """PK1 应力在所有 J 值下应为有限值。"""
    F = np.eye(3) * (J ** (1.0 / 3.0))
    mu, lam = 1e5, 2e5
    P = neo_hookean_pk1_stress(F, mu, lam)
    assert np.all(np.isfinite(P)), f"PK1 has NaN/inf for J={J}"


@pytest.mark.parametrize("J", [1e-15, 1e-10, 1e-9, 1e-6, 0.5, 1.0, 2.0])
def test_tangent_finite_for_all_J(J: float) -> None:
    """9×9 材料切线模量在所有 J 值下应为有限值。"""
    F = np.eye(3) * (J ** (1.0 / 3.0))
    mu, lam = 1e5, 2e5
    C = neo_hookean_material_tangent_9x9(F, mu, lam)
    assert np.all(np.isfinite(C)), f"Tangent has NaN/inf for J={J}"


def test_tangent_vectorized_matches_analytic() -> None:
    """向量化 9×9 切线模量与已知对称性一致。"""
    F = np.array([[1.1, 0.02, 0.0], [0.0, 0.95, 0.01], [0.0, 0.0, 1.05]])
    mu, lam = 1e5, 2e5
    C = neo_hookean_material_tangent_9x9(F, mu, lam)
    # 切线模量应为 9×9 对称矩阵
    assert C.shape == (9, 9)
    assert np.allclose(C, C.T, atol=1e-10)


def test_tangent_positive_semidef_near_identity() -> None:
    """对接近单位阵的变形梯度，9×9 切线模量应半正定。"""
    F = np.eye(3) * 1.001
    C = neo_hookean_material_tangent_9x9(F, 1e5, 2e5)
    eigvals = np.linalg.eigvalsh(C)
    assert np.all(eigvals >= -1e-10), f"Negative eigenvalues: {eigvals.min()}"


# ─────────────────────────────────────────────
# Hessian 向量化一致性测试
# ─────────────────────────────────────────────

def test_hessian_diagonal_finite() -> None:
    """四面体 Hessian 对角块应对正常输入返回有限值。"""
    tet_verts = np.array([
        [0.0, 0.0, 0.0],
        [0.001, 0.0, 0.0],
        [0.0, 0.001, 0.0],
        [0.0, 0.0, 0.001],
    ], dtype=np.float64)
    dm_inv = np.linalg.inv(tet_verts[1:] - tet_verts[0])
    rest_vol = 1e-12 / 6.0
    forces, hessian = compute_tet_force_and_hessian_contributions(
        tet_verts, dm_inv, rest_vol, 1e5, 2e5
    )
    assert forces.shape == (4, 3)
    assert hessian.shape == (4, 3, 3)
    assert np.all(np.isfinite(forces))
    assert np.all(np.isfinite(hessian))


# ─────────────────────────────────────────────
# CZM 状态机对齐测试
# ─────────────────────────────────────────────

def _make_mesh_for_czm(n_verts: int = 10) -> MeshState:
    """创建用于 CZM 测试的最小网格。"""
    vertices = np.zeros((n_verts, 3), dtype=np.float64)
    tets = np.zeros((1, 4), dtype=int)  # 至少需要一个 tet
    tets[0] = [0, 1, 2, 3] if n_verts >= 4 else [0, 0, 0, 0]
    mesh = MeshState(
        vertices=vertices,
        tets=tets,
        layer_id_per_vertex=np.zeros(n_verts, dtype=int),
        layer_id_per_tet=np.zeros(1, dtype=int),
    )
    mesh.is_bottom_surface = np.zeros(n_verts, dtype=bool)
    mesh.is_top_surface_of_layer = -np.ones(n_verts, dtype=int)
    mesh.czm_state = np.full(n_verts, CZMState.FIXED, dtype=int)
    mesh.damage = np.zeros(n_verts)
    mesh.time_free = np.zeros(n_verts)
    mesh.active_vertex_mask = np.ones(n_verts, dtype=bool)
    return mesh


def test_czm_fixed_to_damaging_by_traction() -> None:
    """弹性牵引力超过 T_max 时应触发 FIXED→DAMAGING。"""
    mesh = _make_mesh_for_czm(5)
    bottom = np.array([0, 1, 2], dtype=int)
    mesh.is_bottom_surface[bottom] = True
    # 抬高节点使 gap 产生弹性牵引力
    mesh.vertices[0, 2] = 0.002  # gap = 2mm → traction = k_czm * 0.002
    mesh.vertices[1, 2] = 0.0
    mesh.vertices[2, 2] = 0.0

    t_max, k_czm = 1e4, 1e7  # traction(0.002) = 20000 > t_max
    update_czm_states(mesh, bottom, np.zeros(len(bottom)),
                      area=1e-8, t_max=t_max, k_czm=k_czm,
                      delta_f=1e-4, z_fep=0.0, dt=1e-4)
    assert mesh.czm_state[0] == CZMState.DAMAGING
    assert mesh.damage[0] == 0.0  # 初始损伤为 0
    assert mesh.czm_state[1] == CZMState.FIXED  # gap=0，不应触发


def test_czm_damage_rate_based_accumulation() -> None:
    """损伤应按率累积而非瞬时跳跃。"""
    mesh = _make_mesh_for_czm(3)
    bottom = np.array([0], dtype=int)
    mesh.is_bottom_surface[bottom] = True
    mesh.czm_state[0] = CZMState.DAMAGING
    mesh.vertices[0, 2] = 0.00005  # gap small
    mesh.damage[0] = 0.2

    t_max, delta_f, dt = 1e4, 1e-4, 1e-4
    # 损伤增量 ≈ pull * dt / (t_max * delta_f)
    update_czm_states(mesh, bottom,
                      internal_pull_z=np.array([1.0]),
                      area=1e-8, t_max=t_max, k_czm=1e7,
                      delta_f=delta_f, z_fep=0.0, dt=dt)
    dmg = mesh.damage[0]
    # 损伤应增加但不超过 1.0
    assert dmg > 0.2, f"损伤应增加, got {dmg}"
    assert dmg <= 1.0
    assert dmg < 1.0  # 单步不应到 1.0（小的 pull 累积分量）


def test_czm_free_threshold_5x_delta_f() -> None:
    """gap > 5·δ_f 时应触发 DAMAGING→FREE。"""
    mesh = _make_mesh_for_czm(3)
    bottom = np.array([0], dtype=int)
    mesh.is_bottom_surface[bottom] = True
    mesh.czm_state[0] = CZMState.DAMAGING
    mesh.vertices[0, 2] = 0.001  # gap = 1mm, 5*delta_f = 0.5mm

    update_czm_states(mesh, bottom, np.zeros(1),
                      area=1e-8, t_max=1e4, k_czm=1e7,
                      delta_f=1e-4, z_fep=0.0, dt=1e-4)
    assert mesh.czm_state[0] == CZMState.FREE
    assert mesh.damage[0] == 1.0
    assert mesh.time_free[0] == 0.0  # 刚脱粘，time_free 重置
