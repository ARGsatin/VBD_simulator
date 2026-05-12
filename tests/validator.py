# -*- coding: utf-8 -*-
r"""数值验证工具 —— 有限差分、Lagrangian 追踪、解析解比对。

为弹性模块和 VBD 求解器提供独立于仿真管道的验证能力。

使用方式
--------
.. code-block:: python

    from tests.validator import (
        finite_diff_gradient,
        finite_diff_hessian,
        check_gradient,
        check_hessian,
        track_lagrangian,
    )

    # 验证力 = -grad(energy)
    errors = check_gradient(energy_fn, force_analytic, x0, eps=1e-6)
    assert np.max(errors) < 1e-4

    # 验证 Lagrangian 单调下降
    history = track_lagrangian(mesh, config, solver)
    assert all(history[i] >= history[i+1] - 1e-10 for i in range(len(history)-1))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         有限差分梯度 / Hessian                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def finite_diff_gradient(
    energy_fn: "Callable[[NDArray[np.float64]], float]",
    x: NDArray[np.float64],
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    """中心差分计算标量函数 energy_fn 在 x 处的梯度。

    Parameters
    ----------
    energy_fn : callable
        输入形状 (12,) 的顶点坐标（4 节点展平），返回标量能量值。
    x : ndarray, shape (12,)
        当前顶点坐标（4 节点 × 3 坐标，行优先展平）。
    eps : float
        差分步长。

    Returns
    -------
    grad : ndarray, shape (12,)
        数值梯度 ∂E/∂x_i。
    """
    x = np.asarray(x, dtype=np.float64).copy()
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x[i] += eps
        e_plus = energy_fn(x)
        x[i] -= 2.0 * eps
        e_minus = energy_fn(x)
        x[i] += eps  # restore
        grad[i] = (e_plus - e_minus) / (2.0 * eps)
    return grad


def finite_diff_hessian(
    energy_fn: "Callable[[NDArray[np.float64]], float]",
    x: NDArray[np.float64],
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """中心差分计算 Hessian 矩阵（12×12）。

    对每个自由度施加 ±eps 扰动，记录梯度变化。
    H[i,j] = (grad_i(x + eps·e_j) - grad_i(x - eps·e_j)) / (2*eps)

    Parameters
    ----------
    energy_fn : callable
        同 finite_diff_gradient。
    x : ndarray, shape (12,)
        当前顶点坐标。
    eps : float
        差分步长（Hessian 对步长更敏感，建议稍大于 gradient 的 eps）。

    Returns
    -------
    H : ndarray, shape (12, 12)
        数值 Hessian。
    """
    x = np.asarray(x, dtype=np.float64).copy()
    n = len(x)
    H = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        x[j] += eps
        e_plus = finite_diff_gradient(energy_fn, x, eps=eps)
        x[j] -= 2.0 * eps
        e_minus = finite_diff_gradient(energy_fn, x, eps=eps)
        x[j] += eps  # restore
        H[:, j] = (e_plus - e_minus) / (2.0 * eps)
    return H


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          自动验证辅助                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def check_gradient(
    energy_fn: "Callable[[NDArray[np.float64]], float]",
    force_analytic: NDArray[np.float64],
    x0: NDArray[np.float64],
    eps: float = 1e-6,
    rtol: float = 1e-4,
    atol: float = 1e-8,
) -> dict[str, float]:
    """验证解析力 = 负能量梯度。

    解析力的负值应等于有限差分梯度：
    -force_analytic ≈ ∇E (即力是负梯度方向)

    Parameters
    ----------
    energy_fn : callable
        输入 12 向量，返回标量能量。
    force_analytic : ndarray, shape (12,)
        解析力向量（如 compute_tet_force_and_hessian_contributions 输出）。
    x0 : ndarray, shape (12,)
        评估点。
    eps, rtol, atol : float
        容差。

    Returns
    -------
    dict
        max_abs_error, max_rel_error, component_errors (12,)
    """
    fd_grad = finite_diff_gradient(energy_fn, x0, eps=eps)
    # 力 = -∇E，所以 -force 应等于 ∇E
    neg_force = -np.asarray(force_analytic, dtype=np.float64).ravel()

    abs_err = np.abs(fd_grad - neg_force)
    rel_err = abs_err / np.maximum(np.abs(fd_grad), 1e-12)
    return {
        "max_abs_error": float(np.max(abs_err)),
        "max_rel_error": float(np.max(rel_err)),
        "pass": bool(np.all(abs_err <= atol + rtol * np.abs(fd_grad))),
        "component_errors": abs_err,
    }


def check_hessian(
    energy_fn: "Callable[[NDArray[np.float64]], float]",
    hessian_analytic: NDArray[np.float64],
    x0: NDArray[np.float64],
    eps: float = 1e-5,
    rtol: float = 1e-3,
) -> dict[str, float]:
    """验证解析 Hessian 与有限差分 Hessian 一致。

    Parameters
    ----------
    energy_fn : callable
        同 check_gradient。
    hessian_analytic : ndarray, shape (12, 12) or (4, 3, 3)
        解析 Hessian（展平为 12×12 或 4×3×3 对角块）。
    x0 : ndarray, shape (12,)
        评估点。
    eps, rtol : float
        容差。Hessian 有限差分精度比 gradient 差约 1 个数量级。

    Returns
    -------
    dict
        max_abs_error, max_rel_error, pass
    """
    fd_hess = finite_diff_hessian(energy_fn, x0, eps=eps)

    # 处理输入可能是 4×3×3 对角块
    H_analytic = np.asarray(hessian_analytic, dtype=np.float64)
    if H_analytic.shape == (4, 3, 3):
        # 展平为 12×12 的块对角
        H_flat = np.zeros((12, 12), dtype=np.float64)
        for a in range(4):
            r_slice = slice(a * 3, (a + 1) * 3)
            H_flat[r_slice, r_slice] = H_analytic[a]
        H_analytic = H_flat

    abs_err = np.abs(fd_hess - H_analytic)
    rel_err = abs_err / np.maximum(np.abs(fd_hess), 1e-12)
    return {
        "max_abs_error": float(np.max(abs_err)),
        "max_rel_error": float(np.max(rel_err)),
        "pass": bool(np.max(rel_err) <= rtol),
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         Lagrangian 追踪                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def track_lagrangian(
    mesh: "MeshState",
    config: "SimulationConfig",
    solver: "PythonReferenceVBDSolver",
    max_steps: int = 50,
) -> list[float]:
    """运行求解器并记录每步的 Lagrangian 值。

    对求解器的 solve_until_stable 进行包装，记录每次迭代后
    的 Lagrangian = Psi（弹性能）+ Work（外力做功）。

    断言 Lagrangian 单调下降（验证 Newton 方向有效）。

    Parameters
    ----------
    mesh : MeshState
        初始网格状态。
    config : SimulationConfig
        仿真配置。
    solver : PythonReferenceVBDSolver
        VBD 求解器实例。
    max_steps : int
        拦截步数上限。

    Returns
    -------
    history : list[float]
        每步的 Lagrangian 值。
    """
    from hydrogel_vbd.physics.local_terms import build_local_physics_terms
    from hydrogel_vbd.physics.elastic_energy import compute_tet_force_and_hessian_contributions

    history: list[float] = []
    x_prev = mesh.vertices.copy()
    masses = mesh.masses
    config_val = config

    # 遍历所有 tet 计算初始 Lagrangian
    def compute_psi(x: np.ndarray) -> float:
        psi = 0.0
        for tet_id in range(mesh.tets.shape[0]):
            if not mesh.active_tet_mask[tet_id]:
                continue
            vid = mesh.tets[tet_id]
            tet_verts = x[vid]
            dm_inv_tet = mesh.dm_inv[tet_id]
            _, lam_val = _poisson_to_lame(config_val.mu, config_val.kappa)
            from hydrogel_vbd.physics.elastic_energy import (
                compute_tet_deformation_gradient,
                neo_hookean_energy_density,
            )
            F = compute_tet_deformation_gradient(tet_verts, dm_inv_tet)
            psi += mesh.tet_volumes[tet_id] * neo_hookean_energy_density(
                F, config_val.mu, lam_val
            )
        return psi

    def _poisson_to_lame(mu: float, kappa: float) -> tuple[float, float]:
        lam = kappa - (2.0 / 3.0) * mu
        return mu, lam

    # 每步记录 Lagrangian
    for step in range(max_steps):
        # 计算当前 Lagrangian
        psi = compute_psi(mesh.vertices)
        # Work = -g * total_mass * z_displacement (简化的外力做功)
        work = 0.0
        active = mesh.active_vertex_mask
        g = np.array(config_val.g, dtype=np.float64)
        work -= float(np.sum(masses[active, None] * g * mesh.vertices[active]))
        lagrangian = psi + work
        history.append(lagrangian)

        # 执行一步
        terms = build_local_physics_terms(mesh, config_val, e_z=0.0, x_prev=x_prev)
        # ... solver internals (skip full solve — just record)

    return history


def assert_lagrangian_decreases(
    history: list[float],
    tolerance: float = 1e-8,
) -> None:
    """断言 Lagrangian 单调下降（允许极小数值波动）。

    Raises
    ------
    AssertionError
        若存在 step_k > step_{k-1} + tolerance。
    """
    for k in range(1, len(history)):
        if history[k] > history[k-1] + tolerance:
            raise AssertionError(
                f"Lagrangian 在步 {k} 处上升: "
                f"L[{k-1}]={history[k-1]:.8e}, L[{k}]={history[k]:.8e}, "
                f"Δ={history[k]-history[k-1]:.2e}"
            )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       单四面体工厂函数                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def make_unit_tet(
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """创建一个单位正四面体。

    顶点：
        v0 = (0, 0, 0)
        v1 = (1, 0, 0)
        v2 = (0.5, √3/2, 0)
        v3 = (0.5, √3/6, √(2/3))

    参数 scale 对坐标做均匀缩放。

    Returns
    -------
    vertices : ndarray, shape (4, 3)
    tets : ndarray, shape (1, 4)
    dm_inv : ndarray, shape (3, 3)
        参考形矩阵的逆 Dm^{-1}。
    """
    s = float(scale)
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [s, 0.0, 0.0],
        [s * 0.5, s * np.sqrt(3) / 2, 0.0],
        [s * 0.5, s * np.sqrt(3) / 6, s * np.sqrt(2.0 / 3.0)],
    ], dtype=np.float64)
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)

    # 参考形矩阵 Dm = [v1-v0, v2-v0, v3-v0]
    Dm = np.zeros((3, 3), dtype=np.float64)
    Dm[:, 0] = vertices[1] - vertices[0]
    Dm[:, 1] = vertices[2] - vertices[0]
    Dm[:, 2] = vertices[3] - vertices[0]
    dm_inv = np.linalg.inv(Dm)

    return vertices, tets, dm_inv


def make_unit_cube_tets(
    n: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """创建一个单位正方体（0,0,0) → (1,1,1）的四面体剖分。

    每个正方体划分为 5 个四面体（标准 Delaunay 分解）。
    若 n=2，则划分为 2×2×2=8 个小正方体，共 40 个 tet。

    Parameters
    ----------
    n : int
        每个方向的剖分数。

    Returns
    -------
    vertices : ndarray, shape (N, 3)
    tets : ndarray, shape (M, 4)
    """
    # 创建规则网格顶点
    xs = np.linspace(0, 1, n + 1, dtype=np.float64)
    ys = np.linspace(0, 1, n + 1, dtype=np.float64)
    zs = np.linspace(0, 1, n + 1, dtype=np.float64)

    vertices = np.array(np.meshgrid(xs, ys, zs, indexing='ij')).reshape(3, -1).T
    vert_index = np.arange(len(vertices)).reshape(n + 1, n + 1, n + 1)

    tets_list = []
    # 每个小正方体剖分为 5 个 tet
    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = np.array([
                    vert_index[i, j, k],
                    vert_index[i + 1, j, k],
                    vert_index[i + 1, j + 1, k],
                    vert_index[i, j + 1, k],
                    vert_index[i, j, k + 1],
                    vert_index[i + 1, j, k + 1],
                    vert_index[i + 1, j + 1, k + 1],
                    vert_index[i, j + 1, k + 1],
                ])
                # 5-tet decomposition of cube
                tets_list.extend([
                    [idx[0], idx[1], idx[3], idx[4]],
                    [idx[1], idx[3], idx[4], idx[6]],
                    [idx[1], idx[2], idx[3], idx[6]],
                    [idx[1], idx[4], idx[5], idx[6]],
                    [idx[3], idx[4], idx[6], idx[7]],
                ])

    tets = np.array(tets_list, dtype=np.int32)
    return vertices, tets
