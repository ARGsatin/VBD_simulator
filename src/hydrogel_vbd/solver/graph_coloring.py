# -*- coding: utf-8 -*-
"""图着色 —— 为 VBD 并行 Gauss-Seidel 提供安全的分组策略。

在 VBD 求解器中，每个顶点的局部 Newton 更新是 **Gauss-Seidel** 式的，
即使用最新的邻居位置。为了让同色顶点可以安全地并行更新
（互相不依赖对方的增量），需要对网格的对偶图进行顶点着色。

着色规则
--------
* **对偶图**：顶点 = 原网格的顶点，边 = 共享同一个四面体的两个顶点
* **目标**：用最少的颜色数给顶点着色，使**任意相邻顶点颜色不同**
* **算法**：贪心着色（Greedy Coloring），按顶点 ID 顺序依次分配
  —— 将当前顶点的颜色设为相邻顶点未使用的最小非负整数

时间复杂度
----------
* O(|V|·d) 其中 d 为平均度（四面体网格中 d ≈ 10-15）
* 颜色数通常 ≤ 最大度 + 1，在四面体网格中约 10-20

对求解精度的影响
-----------------
图着色**不影响**求解结果的数学精度，只影响：
1. **并行效率**：颜色数越少，每个颜色的独立顶点越多，并行度越高
2. **收敛速度**：串行 (Gauss-Seidel) 本质上比 Jacobi 收敛快，
   着色分组是两者之间的折中
"""

from __future__ import annotations

import numpy as np

from hydrogel_vbd.core.state import MeshState


def greedy_vertex_coloring(mesh: MeshState) -> np.ndarray:
    """对网格顶点执行贪心图着色。

    构建网格的对偶图（顶点 → 邻居集合），
    然后按顶点 ID 升序遍历，为每个顶点分配其邻居集合中
    尚未使用的最小非负整数颜色。

    算法伪代码
    ----------
    1. 初始化 ``colors = -1``（未着色）
    2. 构建邻接表 ``neighbors[i] = { 共享四面体的其他顶点索引 }``
    3. For vertex_id from 0 to N-1:
       a. 收集 ``neighbors[vertex_id]`` 中已使用的颜色集合
       b. 从 0 开始递增，找到第一个不在该集合中的整数
       c. 赋值 ``colors[vertex_id] = color``

    Parameters
    ----------
    mesh : MeshState
        网格状态对象，必须包含 ``tets``（四面体单元列表）
        和 ``vertices``（用于确定顶点总数 N）。

    Returns
    -------
    np.ndarray, shape (N,), dtype int
        每个顶点的颜色编号，从 0 开始连续。

    Side Effects
    ------------
    无。此函数为纯函数，不修改传入的 mesh 对象。

    Notes
    -----
    * 颜色编号的绝对大小无实际意义，只有同色/不同色之分。
    * 四面体网格中最大度通常不超过 20，因此颜色数可控。
    * 此算法是确定性的（给定同一网格，输出相同）。

    Examples
    --------
    >>> colors = greedy_vertex_coloring(mesh)
    >>> unique_colors = np.unique(colors)
    >>> print(f"使用了 {len(unique_colors)} 种颜色")
    """
    N = mesh.vertices.shape[0]
    colors = np.full(N, -1, dtype=int)  # -1 表示未着色

    # ── 步骤 1：构建邻接表 ──
    # 遍历所有四面体，对每个四元组 (i, j, k, l)，
    # 将彼此加入各自的邻居集合（自环自动排除）
    neighbors: list[set[int]] = [set() for _ in range(N)]
    for tet in mesh.tets:
        for i in tet:
            i_int = int(i)
            # 将四面体中所有**其他**顶点加入当前顶点的邻居集合
            neighbors[i_int].update(int(j) for j in tet if int(j) != i_int)

    # ── 步骤 2：贪心着色 ──
    for vertex_id, adjacent in enumerate(neighbors):
        # 收集所有已着色邻居的颜色
        used = {colors[n] for n in adjacent if colors[n] >= 0}

        # 从 0 开始找第一个未被使用的颜色
        color = 0
        while color in used:
            color += 1

        colors[vertex_id] = color

    return colors
