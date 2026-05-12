# -*- coding: utf-8 -*-
"""Neo-Hookean 弹性力学核心 —— 变形梯度、能量、应力与 Hessian。

本模块是 VBD 求解器中弹性力与刚度矩阵的数学基础，实现了：

1. **变形梯度计算**：从四面体当前坐标与参考形状矩阵的逆计算 **F**
2. **能量密度**：Neo-Hookean 超弹性本构 Ψ(F)
3. **第一 Piola-Kirchhoff 应力**：**P = ∂Ψ/∂F**
4. **材料切线模量**：∂²Ψ/∂F²，展开为 9×9 矩阵
5. **逐顶点组装**：将连续介质力与刚度投影到离散顶点上

所有函数均为纯函数（无状态），便于单元测试和向量化加速。

物理约定
--------
* 使用 **第一 Piola-Kirchhoff (PK1)** 应力/切线模量框架
* 参考（初始）坐标系中定义形函数梯度
* 力 = 负能量梯度；Hessian 只保留对角 3×3 块（VBD 局部求解器所需）
* 翻转单元（J ≤ 1e-12）处以二次惩罚项代替物理能量
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     变形梯度 F = Ds · Dm^{-1}                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_tet_deformation_gradient(
    vertices: NDArray[np.float64],
    dm_inv: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""计算四面体的变形梯度。

    变形梯度 F 描述了从参考构型（Dm）到当前构型（Ds）的仿射映射：

    .. math::
        F = D_s \cdot D_m^{-1}

    其中：
    - **Dm** = [v1-v0, v2-v0, v3-v0]（参考构型中的边矩阵，3×3）
    - **Ds** = [p1-p0, p2-p0, p3-p0]（当前构型中的边矩阵，3×3）

    Parameters
    ----------
    vertices : ndarray, shape (4, 3)
        四面体当前顶点坐标 (v0, v1, v2, v3)。
    dm_inv : ndarray, shape (3, 3)
        参考形状矩阵的逆 Dm⁻¹（在初始化时预计算，避免每步求逆）。

    Returns
    -------
    F : ndarray, shape (3, 3)
        变形梯度张量。满足 dx_current = F · dx_reference。
    """
    p0, p1, p2, p3 = vertices
    # 当前构型的边矩阵 Ds = [v1-v0, v2-v0, v3-v0]，沿列方向堆叠
    ds = np.column_stack((p1 - p0, p2 - p0, p3 - p0))
    return ds @ dm_inv


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                 Neo-Hookean 能量密度 Ψ(F)                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def neo_hookean_energy_density(
    F: NDArray[np.float64],
    mu: float,
    lam: float,
    inverted_penalty: float = 1e8,
) -> float:
    r"""Neo-Hookean 超弹性应变能密度。

    采用经典的 Neo-Hookean 本构模型（可压缩形式）：

    .. math::
        \Psi(\mathbf{F}) = \frac{\mu}{2}\left(I_c - 3\right)
                         - \mu \ln(J)
                         + \frac{\lambda}{2} \ln^2(J)

    其中：
    - :math:`I_c = \operatorname{tr}(\mathbf{F}^T \mathbf{F})` 为右 Cauchy-Green 不变量
    - :math:`J = \det(\mathbf{F})` 为体积比
    - **μ** 为剪切模量（第一 Lamé 参数）
    - **λ** 为第二 Lamé 参数

    当单元发生翻转（J ≤ 1e-12）时，物理能量被二次惩罚项替代，
    以避免对数/除零奇异并驱动单元恢复正体积。

    Parameters
    ----------
    F : ndarray, shape (3, 3)
        变形梯度张量。
    mu : float
        剪切模量（第一 Lamé 参数，Pa）。
    lam : float
        第二 Lamé 参数（Pa）。
    inverted_penalty : float, optional
        翻转单元的惩罚刚度系数，默认为 1e8。

    Returns
    -------
    psi : float
        应变能密度（J/m³ 原始构型）。
    """
    I_c = float(np.trace(F.T @ F))  # 右 Cauchy-Green 第一不变量
    J = float(np.linalg.det(F))     # 体积比

    # 翻转/近零体积 → 二次惩罚（提高阈值到 1e-10 提前捕获）
    if J <= 1e-10:
        return inverted_penalty * (1.0 - J) ** 2

    # 低 J 区域 smooth blend，消除能量梯度不连续
    if J < 1e-8:
        t = J / 1e-8  # 0→1 的 blend 因子
        log_J_safe = float(np.log(1e-8))
        psi_physical = 0.5 * mu * (I_c - 3.0) - mu * log_J_safe + 0.5 * lam * log_J_safe ** 2
        psi_penalty = inverted_penalty * (1.0 - J) ** 2
        return t * psi_physical + (1.0 - t) * psi_penalty

    log_J = float(np.log(J))
    return 0.5 * mu * (I_c - 3.0) - mu * log_J + 0.5 * lam * log_J ** 2


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              第一 Piola-Kirchhoff 应力 P = ∂Ψ/∂F                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def neo_hookean_pk1_stress(
    F: NDArray[np.float64],
    mu: float,
    lam: float,
    inverted_penalty: float = 1e8,
) -> NDArray[np.float64]:
    r"""计算第一 Piola-Kirchhoff (PK1) 应力张量。

    PK1 应力定义为能量密度对变形梯度的偏导：

    .. math::
        \mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}
                   = \mu \mathbf{F}
                     + \left(\lambda \ln(J) - \mu\right) \mathbf{F}^{-T}

    物理含义：
    - **μF** 为剪切响应
    - **(λ ln(J) - μ) F^{-T}** 为体积变化响应
    - 与 Cauchy 应力的关系：**σ = (1/J) P F^T**

    Parameters
    ----------
    F : ndarray, shape (3, 3)
        变形梯度张量。
    mu : float
        剪切模量（Pa）。
    lam : float
        第二 Lamé 参数（Pa）。
    inverted_penalty : float, optional
        翻转单元的惩罚刚度系数，默认为 1e8。

    Returns
    -------
    P : ndarray, shape (3, 3)
        PK1 应力张量（单位：Pa，参考构型测量）。
    """
    J = float(np.linalg.det(F))

    # 翻转单元 → 惩罚应力
    if J <= 1e-12:
        FinvT = np.linalg.inv(F).T
        penalty_factor = -2.0 * inverted_penalty * (1.0 - J) * J
        return penalty_factor * FinvT

    FinvT = np.linalg.inv(F).T
    log_J = float(np.log(J))
    return mu * F + (lam * log_J - mu) * FinvT


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                材料切线模量 ∂²Ψ/∂F² (9×9)                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def neo_hookean_material_tangent_9x9(
    F: NDArray[np.float64],
    mu: float,
    lam: float,
    inverted_penalty: float = 1e8,
) -> NDArray[np.float64]:
    r"""计算 Neo-Hookean 材料切线模量（二阶导），展开为 9×9 矩阵。

    将 **F** 的 9 个分量按行优先顺序排列：
    (F_00, F_01, F_02, F_10, F_11, F_12, F_20, F_21, F_22)。

    解析二阶导公式：

    .. math::
        \frac{\partial^2 \Psi}{\partial F_{ij} \partial F_{kl}}
        = \mu \cdot \delta_{ik} \delta_{jl}
          + (\lambda \ln J - \mu) \cdot F^{-T}_{il} F^{-T}_{kj}
          + \lambda \cdot F^{-T}_{ij} F^{-T}_{kl}

    该矩阵用于 VBD 局部求解器中组装顶点的局部 Hessian。

    Parameters
    ----------
    F : ndarray, shape (3, 3)
        变形梯度张量。
    mu : float
        剪切模量（Pa）。
    lam : float
        第二 Lamé 参数（Pa）。
    inverted_penalty : float, optional
        翻转单元的惩罚刚度系数，默认为 1e8。

    Returns
    -------
    C : ndarray, shape (9, 9)
        材料切线模量矩阵（单位：Pa，参考构型测量）。
    """
    J = float(np.linalg.det(F))
    if J <= 1e-12:
        # 翻转单元 → 对角惩罚（保证正定）
        return inverted_penalty * np.eye(9, dtype=np.float64)

    FinvT = np.linalg.inv(F).T   # F^{-T}，形状 (3,3)
    log_J = float(np.log(J))
    coeff = lam * log_J - mu     # (λ ln(J) - μ)

    # ── 向量化 9×9 材料切线模量 ──
    # C_{(i,j),(k,l)} = term1 + term2 + term3
    #   term1: μ · δ_{ik} · δ_{jl}
    #   term2: (λ ln(J) - μ) · F^{-T}_{il} · F^{-T}_{kj}
    #   term3: λ · F^{-T}_{ij} · F^{-T}_{kl}
    C = mu * np.eye(9, dtype=np.float64)

    # term2: coeff * FinvT[i,l] * FinvT[k,j] = coeff * FinvT[i,l] * FinvT^T[j,k]
    # 使用 broadcasting 构建 4D tensor [i,j,k,l] 后 reshape 为 9×9
    C_4d_t2 = coeff * FinvT[:, None, None, :] * FinvT.T[None, :, :, None]
    C += C_4d_t2.reshape(9, 9)

    # term3: lam * FinvT[i,j] * FinvT[k,l] = lam * outer(vec(F^{-T}), vec(F^{-T}))
    a_flat = FinvT.ravel()
    C += lam * np.outer(a_flat, a_flat)

    return C


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║           四面体力与 Hessian 组装（逐顶点对角块）                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_tet_force_and_hessian_contributions(
    tet_vertices: NDArray[np.float64],
    dm_inv: NDArray[np.float64],
    rest_volume: float,
    mu: float,
    lam: float,
    inverted_penalty: float = 1e8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """计算单个四面体的 Neo-Hookean 力与逐顶点 Hessian 对角块。

    这是 VBD 求解器在每个时间步调用的**核心组装函数**：

    1. 计算变形梯度 **F** → PK1 应力 **P** → 材料切线模量 **C** (9×9)
    2. 通过形函数梯度将连续介质推导到离散顶点：
       - **力 f_a = -V₀ · P · g_a**（负能量梯度）
       - **局部 Hessian H_aa = V₀ · C : (g_a ⊗ g_a)**（只保留对角 3×3 块）
    3. 形函数梯度在参考构型中计算，可预存复用

    .. note::
       VBD 不组装全局稀疏矩阵，只保留每个顶点的 3×3 对角块，
       因此本函数只返回 4×3×3 的对角 Hessian（不包含交叉项）。

    Parameters
    ----------
    tet_vertices : ndarray, shape (4, 3)
        四面体当前顶点坐标 (v0, v1, v2, v3)。
    dm_inv : ndarray, shape (3, 3)
        参考形状矩阵的逆 Dm⁻¹。
    rest_volume : float
        四面体参考体积 V₀（m³）。
    mu : float
        剪切模量（Pa）。
    lam : float
        第二 Lamé 参数（Pa）。
    inverted_penalty : float, optional
        翻转单元的惩罚刚度系数，默认为 1e8。

    Returns
    -------
    forces_per_vertex : ndarray, shape (4, 3)
        每个顶点的力向量（负能量梯度，单位 N）。
    hessian_per_vertex : ndarray, shape (4, 3, 3)
        每个顶点的对角 Hessian 块（3×3，单位 N/m）。
        用于 VBD 局部 Newton 迭代。
    """
    # ── 步骤 1：连续介质物理量 ──
    F = compute_tet_deformation_gradient(tet_vertices, dm_inv)
    P = neo_hookean_pk1_stress(F, mu, lam, inverted_penalty)
    C_9x9 = neo_hookean_material_tangent_9x9(F, mu, lam, inverted_penalty)

    # ── 步骤 2：形函数梯度（在参考构型中） ──
    # Dm^{-T} (3×3)，记 B = Dm^{-T}
    # 4 个顶点的形函数梯度：
    #   g_v0 = -sum(B 的列)   (满足 Σ g_a = 0 的约束)
    #   g_v1 = B[:,0]
    #   g_v2 = B[:,1]
    #   g_v3 = B[:,2]
    B = dm_inv.T  # Dm^{-T}, shape (3, 3)
    g = np.zeros((4, 3), dtype=np.float64)
    g[0] = -(B[:, 0] + B[:, 1] + B[:, 2])  # g_v0 = -col_sum(B)
    g[1] = B[:, 0]   # g_v1
    g[2] = B[:, 1]   # g_v2
    g[3] = B[:, 2]   # g_v3

    # ── 步骤 3：组装力（负 PK1 应力在形函数梯度上的投影） ──
    # f_a = -V₀ · P · g_a    (3×1)
    forces = np.zeros((4, 3), dtype=np.float64)
    for a in range(4):
        forces[a] = -rest_volume * (P @ g[a])

    # ── 步骤 4：组装局部 Hessian 对角块 ──
    # H_aa[p,q] = V₀ · Σ_{n,s} C_{(p,n),(q,s)} · g_a^n · g_a^s
    # 将 C_9x9 重塑为 (3,3,3,3) [p, n, q, s]，用 einsum 向量化
    C_reshaped = C_9x9.reshape(3, 3, 3, 3)  # [p, n, q, s]
    hessian = np.zeros((4, 3, 3), dtype=np.float64)
    for a in range(4):
        hessian[a] = rest_volume * np.einsum(
            'n,pnsq,s->pq', g[a], C_reshaped, g[a]
        )

    return forces, hessian


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              向后兼容的线性弹簧占位符（已弃用）                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def linear_tetrahedral_placeholder_force(
    vertices: np.ndarray,
    rest_vertices: np.ndarray,
    stiffness: float,
) -> np.ndarray:
    """向后兼容的线性弹簧占位符力模型。

    这是一个简化的力模型，仅用于测试：将每个顶点当作连接其
    参考位置的线性弹簧处理。

    .. deprecated::
       请使用 ``compute_tet_force_and_hessian_contributions``
       进行正确的 Neo-Hookean 能量计算。

    Parameters
    ----------
    vertices : ndarray
        当前顶点坐标。
    rest_vertices : ndarray
        参考顶点坐标（必须与 vertices 形状相同）。
    stiffness : float
        弹簧刚度系数。

    Returns
    -------
    forces : ndarray
        线性弹簧力 = -stiffness · (current - rest)。
    """
    vertices = np.asarray(vertices, dtype=float)
    rest_vertices = np.asarray(rest_vertices, dtype=float)
    if vertices.shape != rest_vertices.shape:
        raise ValueError("vertices 和 rest_vertices 必须具有相同形状")
    return -float(stiffness) * (vertices - rest_vertices)
