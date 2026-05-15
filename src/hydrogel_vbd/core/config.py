# -*- coding: utf-8 -*-
"""仿真配置模块 —— 所有物理参数、求解器参数、控制参数的集中定义。

本模块定义了 `SimulationConfig` 数据类，它是整个仿真系统的"参数中枢"。
所有模块（力模型、求解器、控制器、IO 等）都通过该配置获取参数，
从而保证参数来源统一、易于调参和复现实验。

支持从 YAML 配置文件加载参数（`from_yaml` 类方法），
也支持直接在代码中实例化并覆盖默认值。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 辅助函数：解析 YAML 中的标量值（支持 int / float / tuple）
# ---------------------------------------------------------------------------
def _parse_scalar(raw: str) -> Any:
    """将 YAML 行中的原始字符串值解析为 Python 类型。

    支持三种格式：
    - 元组：``[0.0, 0.0, -9.81]`` → ``(0.0, 0.0, -9.81)``
    - 浮点数：含小数点或科学计数法 → ``float``
    - 整数：纯数字 → ``int``
    - 其他 → 原字符串

    Parameters
    ----------
    raw : str
        YAML 值部分的原始字符串（已去除键名和冒号）。

    Returns
    -------
    Any
        解析后的 Python 对象。
    """
    text = raw.strip()
    # ── 方括号包裹的元组 ──
    if text.startswith("[") and text.endswith("]"):
        return tuple(float(part.strip()) for part in text[1:-1].split(",") if part.strip())
    # ── 数值类型 ──
    try:
        if any(marker in text.lower() for marker in (".", "e")):
            return float(text)
        return int(text)
    except ValueError:
        return text


# ---------------------------------------------------------------------------
# SimulationConfig：全局仿真参数数据类
# ---------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """水凝胶 DLP VBD 仿真全局配置。

    所有字段均有默认值，可通过关键字参数覆盖，也可从 YAML 文件加载。
    参数按功能分为以下几组：

    **物理 / 材料参数**
    - `g`：重力加速度向量 (m/s²)
    - `rho`：水凝胶密度 (kg/m³)
    - `mu`：第一拉梅常数（剪切模量，Pa）
    - `kappa`：第二拉梅常数（体积模量，Pa）
    - `k_d`：阻尼系数
    - `c_shrink`：固化收缩因子（<1 表示收缩）

    **损伤与断裂参数（CZM 内聚力模型）**
    - `T_max`：最大内聚强度 (Pa)
    - `K_czm`：内聚刚度 (Pa/m)
    - `delta_f`：失效分离距离 (m)
    - `eta`：损伤演化指数
    - `d_min`：最小损伤阈值

    **流体参数**
    - `d_fluid_max`：最大流体阻尼距离 (m)
    - `t_fluid_max`：流体阻尼最大持续时间 (s)
    - `fluid_radius`：流体作用半径 (m)
    - `node_area`：节点有效面积 (m²)

    **求解器参数**
    - `dt`：时间步长 (s)
    - `epsilon`：收敛容差
    - `max_iters`：每时间步最大迭代次数
    - `N_stable`：判定稳定的连续收敛步数
    - `rho_cheb`：Chebyshev 半隐式方法的谱半径

    **PID 控制器参数**
    - `c_init`：初始固化度
    - `err_target`：目标形状误差
    - `K_p`、`K_i`、`K_d`：PID 比例 / 积分 / 微分增益

    **电场参数**
    - `q_ion`：离子电荷密度 (C/m³)
    - `E_max`：最大电场强度 (V/m)

    **打印工艺参数**
    - `layer_thickness`：每层厚度 / 提升高度 (m)
    - `z_fep`：离型膜（FEP）Z 坐标 (m)
    - `v_lift`：平台提升速度 (m/s)，为 0 时跳过提升阶段
    - `C_0`：初始固化度比例常数
    """

    # ────────── 物理 / 材料 ──────────
    g: tuple[float, float, float] = (0.0, 0.0, -9.81)  # 重力加速度 (m/s²)
    rho: float = 1050.0                                 # 密度 (kg/m³)
    mu: float = 50000.0                                  # 第一拉梅常数 / 剪切模量 (Pa)
    kappa: float = 1.0e7                                 # 第二拉梅常数 / 体积模量 (Pa)
    k_d: float = 0.5                                     # 阻尼系数 (无量纲)
    c_shrink: float = 0.98                               # 固化收缩因子

    # ────────── 损伤 / CZM ──────────
    T_max: float = 5000.0       # 最大内聚强度 (Pa)
    K_czm: float = 1.0e8        # 内聚刚度 (Pa/m)
    delta_f: float = 1.0e-4     # 失效分离距离 (m)
    eta: float = 0.8            # 损伤演化指数
    d_min: float = 1.0e-6       # 最小损伤阈值

    # ────────── 流体 ──────────
    d_fluid_max: float = 2.0e-3     # 最大流体阻尼距离 (m)
    t_fluid_max: float = 0.5        # 流体阻尼最大持续时间 (s)
    fluid_radius: float = 0.001     # 流体作用半径 (m)
    node_area: float = 1.0          # 节点有效面积 (m²)
    enable_czm: bool = True         # 是否启用 CZM / 流体脱粘影响

    # ────────── 求解器 ──────────
    dt: float = 0.01             # 时间步长 (s)
    epsilon: float = 1.0e-6      # 收敛容差
    max_iters: int = 20          # 最大迭代次数
    N_stable: int = 10           # 稳定判据：连续收敛步数
    rho_cheb: float = 0.95       # Chebyshev 谱半径

    # ────────── PID 控制 ──────────
    c_init: float = 0.1          # 初始固化度
    err_target: float = 5.0e-4   # 目标形状误差
    K_p: float = 150.0           # 比例增益
    K_i: float = 20.0            # 积分增益
    K_d: float = 5.0             # 微分增益

    # ────────── 电场 ──────────
    q_ion: float = 1.2e-3        # 离子电荷密度 (C/m³)
    E_max: float = 500.0         # 最大电场强度 (V/m)

    # ────────── 打印工艺 ──────────
    layer_thickness: float = 5e-5    # 层厚 / 平台提升距离 (m) = 0.05mm
    lift_multiplier: float = 1.5    # 提升距离倍数（相对于 layer_thickness），用于计算实际提升高度
    z_fep: float = 0.0              # 离型膜 Z 坐标 (m)（Solver 内部 Z = 构建轴）
    v_lift: float = 0.001           # 提升速度 (m/s)，0 则跳过提升
    C_0: float = 1.0                # 固化度比例常数

    # ────────── 构建方向 ──────────
    build_axis: int = 2             # 构建轴: 0=X, 1=Y, 2=Z（默认 Z）

    # ────────── 生命周期 ──────────

    def __post_init__(self) -> None:
        """若 build_axis ≠ Z，自动旋转重力向量使物理方向一致。

        网格生成层会做坐标交换使构建轴对齐到 Z，
        此处提前旋转重力等向量，保持物理一致性。
        """
        if self.build_axis == 2:
            return
        if self.build_axis not in (0, 1):
            raise ValueError(f"build_axis 必须为 0(X), 1(Y), 2(Z), 实际: {self.build_axis}")

        # 交换 gravity 的 build_axis 和 Z 分量
        g = list(self.g)
        g[self.build_axis], g[2] = g[2], g[self.build_axis]
        self.g = tuple(g)

    # ────────── 类方法 ──────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationConfig":
        """从 YAML 配置文件加载参数并构造实例。

        忽略 YAML 中不存在于 `SimulationConfig` 字段中的键，
        因此配置文件可以包含注释或其他工具的额外键。

        Parameters
        ----------
        path : str | Path
            YAML 文件路径。

        Returns
        -------
        SimulationConfig
            使用 YAML 参数填充的配置实例。
        """
        values: dict[str, Any] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            # 跳过空行、注释、无冒号行
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            # 只加载类中存在的字段
            if not hasattr(cls, key):
                continue
            # 去除行内注释后解析值
            values[key] = _parse_scalar(raw_value.split("#", 1)[0])
        return cls(**values)
