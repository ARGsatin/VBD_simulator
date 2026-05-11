# -*- coding: utf-8 -*-
"""内聚力区模型 (CZM) —— 离型膜剥离/脱粘过程的损伤演化。

在 DLP 水凝胶打印中，固化后的水凝胶附着在离型膜（FEP）上。
随着打印平台每次抬升，附着界面受力拉开，经历从**粘附**到
**损伤**再到**完全脱粘**的三阶段演化。

本模块实现了基于牵引-分离法则（Traction-Separation Law）的
内聚力区模型，用于描述这一剥离过程的力学行为。

状态机
------
每个底面节点处于以下三种状态之一（``CZMState`` 枚举）：

.. code-block:: text

            ┌──────────┐      拉力 > T_max       ┌───────────┐
            │  FIXED   │ ──────────────────────→  │ DAMAGING  │
            │ (粘附)   │                           │ (损伤)    │
            └──────────┘                           └─────┬─────┘
                                                        │
                                            gap ≥ δ_f  │  或 damage ≥ 1
                                                        │
                                                        ↓
                                                 ┌────────────┐
                                                 │   FREE     │
                                                 │  (完全脱粘) │
                                                 └────────────┘

物理含义
--------
* **FIXED**：节点完全粘附在离型膜上，位移为零
* **DAMAGING**：界面拉力超过强度极限 T_max，损伤开始累积，
  残余牵引力随损伤软化
* **FREE**：完全脱粘，节点可自由移动（但仍受流体拖曳力）
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from hydrogel_vbd.core.state import MeshState


class CZMState(IntEnum):
    """内聚力区 (CZM) 节点状态的枚举类型。

    继承自 ``IntEnum``，便于存储在 numpy 整数数组中。

    Values
    ------
    FIXED = 0
        完全粘附状态 —— 节点附着在离型膜表面，
        位移被锁定为零（或离型膜面）。
    DAMAGING = 1
        损伤演化状态 —— 界面拉力已超过强度极限，
        损伤变量从 0 逐步增长到 1，残余牵引力线性软化。
    FREE = 2
        完全脱粘状态 —— 损伤已达最大值，节点不再受
        内聚力约束，仅受流体拖曳和弹性恢复力。
    """
    FIXED = 0
    DAMAGING = 1
    FREE = 2


def update_czm_states(
    mesh: MeshState,
    bottom_nodes: np.ndarray,
    internal_pull_z: np.ndarray,
    area: float,
    t_max: float,
    k_czm: float,
    delta_f: float,
    z_fep: float,
    dt: float,
) -> None:
    """推进底面节点的 CZM 状态演化。

    在每个时间步调用，根据当前内力和几何状态更新
    底面节点的粘附/损伤/脱粘状态机。

    **状态转换逻辑**：

    1. **FIXED → DAMAGING**：
       当 ``pull_z / area ≥ T_max``（界面拉应力超过强度）时触发。
    2. **DAMAGING 内部演化**：
       - 损伤变量 ``D = clip((gap / δ_f) · (T_max / k_czm), 0, 1)``
       - 若 ``gap ≥ δ_f`` 则 D 直接设为 1.0
    3. **DAMAGING → FREE**：
       当 ``gap ≥ δ_f`` 或 ``damage ≥ 1.0`` 时完全脱粘。
    4. **FREE**：
       累计自由状态时间 ``time_free += dt``（用于流体拖曳衰减）。

    Parameters
    ----------
    mesh : MeshState
        网格状态（原地修改 ``czm_state``、``damage``、``time_free``）。
    bottom_nodes : np.ndarray, shape (K,)
        当前活动层底面顶点的索引数组。
    internal_pull_z : np.ndarray, shape (K,)
        每个底面节点受到的内部法向拉力（N）—— 通常来自弹性恢复力。
    area : float
        每个节点的近似承载面积（m²），将力转换为应力。
    t_max : float
        界面强度（Pa）—— 粘附界面可承受的最大法向拉应力。
    k_czm : float
        CZM 罚刚度（Pa/m）—— 粘附界面的弹性刚度。
    delta_f : float
        完全破坏位移（m）—— 法向间隙超过此值即判定脱粘。
    z_fep : float
        离型膜 FEP 平面的 Z 坐标（m）。
    dt : float
        当前时间步长（s），用于累计 FREE 状态时间。

    Returns
    -------
    None
        原地修改 ``mesh.czm_state``、``mesh.damage``、``mesh.time_free``。

    Notes
    -----
    * 函数在原地修改 mesh 状态的三个数组，无返回值。
    * ``pull_z / area`` 计算的是界面法向拉应力。
    * 损伤演化采用线性软化法则，实际用于 ``local_terms.py``
      中的 CZM 力计算。
    """
    bottom_nodes = np.asarray(bottom_nodes, dtype=int)
    pulls = np.asarray(internal_pull_z, dtype=float)

    for local_idx, node_id in enumerate(bottom_nodes):
        state = CZMState(int(mesh.czm_state[node_id]))

        if state == CZMState.FIXED:
            # ── 检查是否达到强度极限 ──
            # 应力 = pull_z / area，超过 T_max 进入损伤
            if pulls[local_idx] / max(float(area), 1e-12) >= float(t_max):
                mesh.czm_state[node_id] = CZMState.DAMAGING

        elif state == CZMState.DAMAGING:
            # ── 损伤演化 ──
            gap = max(float(mesh.vertices[node_id, 2] - z_fep), 0.0)
            # 线性损伤累积：D ∝ gap · T_max / (δ_f · k_czm)
            damage = max(
                float(mesh.damage[node_id]),
                (gap / max(delta_f, 1e-12)) * (t_max / max(k_czm, 1e-12)),
            )
            if gap >= delta_f:
                damage = 1.0  # 超过完全破坏位移，直接完全损伤
            mesh.damage[node_id] = float(np.clip(damage, 0.0, 1.0))

            # ── 完全脱粘判定 ──
            if gap >= delta_f or mesh.damage[node_id] >= 1.0:
                mesh.czm_state[node_id] = CZMState.FREE

        elif state == CZMState.FREE:
            # ── 累计自由状态存活时间 ──
            mesh.time_free[node_id] += float(dt)
