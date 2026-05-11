# -*- coding: utf-8 -*-
"""水凝胶材料模型 —— 固化度依赖的非线性弹性参数计算。

``HydrogelMaterialModel`` 类封装了 DLP 光固化水凝胶的本构行为。
其核心功能是根据空间分布的固化度 φ (0~1) 计算各位置的
Lamé 常数（μ, λ），供有限元弹性刚度矩阵使用。

原理
----
水凝胶在固化过程中经历了从**液态**（φ ≈ 0，杨氏模量 ≈ E_min）
到**弹性固态**（φ = 1，杨氏模量 = E_max）的连续过渡。
本模型采用以下假设：

1. **幂律固化**：杨氏模量按幂律从 E_min 演化到 E_max
2. **泊松比恒定**：假设 ν 不随固化度变化
3. **各向同性**：材料在任意位置均为各向同性线弹性

Lamé 常数由下式计算：

.. math::
    \\mu = \\frac{E}{2(1+\\nu)}, \\quad
    \\lambda = \\frac{E\\nu}{(1+\\nu)(1-2\\nu)}
"""

from __future__ import annotations

import numpy as np

from hydrogel_vbd.core.state import MaterialState


class HydrogelMaterialModel:
    """水凝胶材料模型。

    根据输入的全局材料参数和逐点固化度 φ，
    为每个节点/单元计算对应的 Lamé 常数。

    Attributes
    ----------
    density : float
        材料密度 (kg/m³)，用于惯性和重力项。
    young_modulus_min : float
        未固化（φ=0）时的杨氏模量 (Pa)。
    young_modulus_max : float
        完全固化（φ=1）时的杨氏模量 (Pa)。
    poisson_ratio : float
        泊松比 ν（0 < ν < 0.5 for compressible）。
    damping : float
        质量比例阻尼系数（用于 Rayleigh 阻尼）。
    curing_exponent : float
        固化幂律指数 p，控制 E(φ) 曲线的陡峭程度。
    peel_stress_crit : float
        剥离临界应力 (Pa)，用于判断层是否从离型膜脱落。
    electric_response_alpha : float
        电场响应系数 α (m²/(V·s) 或类似单位)，
        描述单位电场强度下材料的力电耦合响应强度。
    """

    def __init__(
        self,
        density: float,
        young_modulus_min: float,
        young_modulus_max: float,
        poisson_ratio: float,
        damping: float,
        curing_exponent: float,
        peel_stress_crit: float,
        electric_response_alpha: float,
    ) -> None:
        """初始化水凝胶材料模型。

        Parameters
        ----------
        density : float
            材料密度 (kg/m³)。
        young_modulus_min : float
            最小杨氏模量 (Pa)，对应 φ=0（液态）。
        young_modulus_max : float
            最大杨氏模量 (Pa)，对应 φ=1（完全固化）。
        poisson_ratio : float
            泊松比，范围为 (0, 0.5)。
        damping : float
            质量比例阻尼系数。
        curing_exponent : float
            固化幂律指数（通常 1.0~3.0）。
        peel_stress_crit : float
            剥离临界应力 (Pa)。
        electric_response_alpha : float
            电场力电耦合响应系数。
        """
        self.density = float(density)
        self.young_modulus_min = float(young_modulus_min)
        self.young_modulus_max = float(young_modulus_max)
        self.poisson_ratio = float(poisson_ratio)
        self.damping = float(damping)
        self.curing_exponent = float(curing_exponent)
        self.peel_stress_crit = float(peel_stress_crit)
        self.electric_response_alpha = float(electric_response_alpha)

    def create_state(
        self, curing_degree: np.ndarray
    ) -> MaterialState:
        """根据固化度数组计算空间依赖的材料状态。

        给定每个节点/单元的固化度 φ ∈ [0, 1]，计算对应的
        Lamé 常数（μ, λ）及其他场量，汇总为 ``MaterialState``。

        Parameters
        ----------
        curing_degree : np.ndarray
            逐点固化度数组，shape 为 (N,) 或 (N_elements,)，
            值域 [0, 1]。

        Returns
        -------
        MaterialState
            包含所有空间依赖材料参数的容器对象。
            关键字段：
            - ``mu``：第一 Lamé 常数（剪切模量）
            - ``lam``：第二 Lamé 常数
            - ``young_modulus``：逐点杨氏模量
            - ``damping``：逐点阻尼系数
            - ``electric_response_alpha``：逐点电场响应系数
        """
        phi = np.clip(
            np.asarray(curing_degree, dtype=float), 0.0, 1.0
        )
        # 幂律模型：E(φ) = E_min + (E_max - E_min) * φ^p
        young = self.young_modulus_min + (
            self.young_modulus_max - self.young_modulus_min
        ) * (phi**self.curing_exponent)
        nu = self.poisson_ratio
        # 各向同性线弹性 Lamé 常数
        mu = young / (2.0 * (1.0 + nu))  # 剪切模量
        lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # 第二 Lamé 常数
        return MaterialState(
            density=self.density,
            young_modulus=young,
            poisson_ratio=nu,
            mu=mu,
            lam=lam,
            damping=np.full_like(phi, self.damping, dtype=float),
            curing_degree=phi,
            peel_stress_crit=self.peel_stress_crit,
            electric_response_alpha=np.full_like(
                phi, self.electric_response_alpha, dtype=float
            ),
        )
