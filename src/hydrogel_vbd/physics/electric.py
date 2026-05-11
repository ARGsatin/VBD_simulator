# -*- coding: utf-8 -*-
"""电场力模型 —— DLP 曝光期间固液界面的电动力学驱动。

在 DLP 光聚合过程中，电极间施加的电压在水凝胶-树脂界面
产生电场，引起离子迁移和电渗流，对固化界面施加附加作用力。

本模块提供两类接口：
1. **ElectricForceModel** 类：基于电压-力转换系数（α）的
   简化电驱动力模型。
2. **from_mapping()** 静态方法：基于预计算的电压-力映射矩阵
   的通用线性映射。
"""

from __future__ import annotations

import numpy as np

from hydrogel_vbd.core.state import FieldCommand, MeshState


class ElectricForceModel:
    """简化的电驱动体力模型。

    将活性电极电压线性映射为节点力：

    .. math::
        f_i = α_i · (Σ V_electrodes) · d

    其中：
    - **α_i** 是每个顶点的电场力转换系数（N/V）
    - **ΣV_electrodes** 是所有电极电压之和
    - **d** 是指定的单位方向向量（默认 (0,0,1)，即 +Z 方向）

    这种简化假设电场力与总电压成正比、沿固定方向均匀施加。
    对于更复杂的空间分布，请使用 ``from_mapping()`` 静态方法。
    """

    def __init__(
        self,
        alpha: float | np.ndarray,
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        """初始化电驱动模型。

        Parameters
        ----------
        alpha : float or np.ndarray
            电场力转换系数。
            - 若为标量：所有节点使用相同系数
            - 若为数组：必须为 shape (N,) 的逐节点系数
        direction : tuple[float, float, float], optional
            力的方向向量（将被归一化）。默认为 (0,0,1) 即 +Z 方向。
        """
        self.alpha = alpha
        direction_array = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction_array)
        if norm == 0.0:
            raise ValueError("direction must be nonzero")
        self.direction = direction_array / norm

    def compute(
        self,
        mesh: MeshState,
        command: FieldCommand,
    ) -> np.ndarray:
        """计算当前电极电压下的电驱动力。

        Parameters
        ----------
        mesh : MeshState
            网格状态（用于确定顶点数、活动掩码）。
        command : FieldCommand
            当前电极控制命令，包含 ``voltage`` 数组。

        Returns
        -------
        np.ndarray, shape (N, 3)
            每个顶点的电驱动力向量（N）。非活动节点力为零。

        Raises
        ------
        ValueError
            若 ``alpha`` 的维度与顶点数不匹配。
        """
        forces = np.zeros_like(mesh.vertices)
        alpha = np.asarray(self.alpha, dtype=float)

        # ── 展开 alpha ──
        if alpha.ndim == 0:
            alpha_per_vertex = np.full(mesh.vertices.shape[0], float(alpha))
        elif alpha.shape == (mesh.vertices.shape[0],):
            alpha_per_vertex = alpha
        else:
            raise ValueError(
                "alpha must be scalar or one value per vertex"
            )

        # ── 力 = α · (Σ V) · d ──
        voltage_sum = float(np.sum(command.voltage))
        forces[mesh.active_vertex_mask] = (
            alpha_per_vertex[mesh.active_vertex_mask, None]
            * voltage_sum
            * self.direction
        )

        return forces

    @staticmethod
    def from_mapping(
        mapping: np.ndarray,
        voltage: np.ndarray,
        vertex_count: int,
    ) -> np.ndarray:
        """从预计算电压-力映射矩阵生成节点力。

        通用接口，适用于电场分布由外部求解器（如 FEM 电场仿真）
        预计算好的场景。

        .. math::
            F_{flat} = M · V

        其中 **M** 是 (3N, K) 的映射矩阵，**V** 是 (K,) 的电极电压。

        Parameters
        ----------
        mapping : np.ndarray, shape (3N, K)
            电压到力的映射矩阵。
        voltage : np.ndarray, shape (K,)
            各电极电压值（V）。
        vertex_count : int
            顶点总数 N，用于将平坦结果 reshape 为 (N, 3)。

        Returns
        -------
        np.ndarray, shape (N, 3)
            顶点力向量。
        """
        flat_force = (
            np.asarray(mapping, dtype=float)
            @ np.asarray(voltage, dtype=float)
        )
        return flat_force.reshape(vertex_count, 3)
