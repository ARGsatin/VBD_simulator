# -*- coding: utf-8 -*-
"""VTK 输出器 —— 将四面体网格和场数据导出为 ParaView 可读的 VTU 格式。

本模块生成符合 VTK (Visualization Toolkit) 标准的非结构化网格（Unstructured Grid）
XML 格式文件（``.vtu``），可在 **ParaView** 等专业可视化工具中打开，
用于观察仿真结果的三维变形、应力分布等物理场。

输出内容
--------
* **网格拓扑**：仅输出活动四面体单元及其顶点
* **点属性**：可附加任意标量或向量场数据（如位移、损伤、CZM 状态等）

.. note::
   生成的 VTU 文件采用 ASCII 编码，便于人类阅读和脚本解析；
   对大网格（>10⁵ 节点）建议改用二进制格式以减小文件体积。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hydrogel_vbd.core.state import MeshState


def write_vtu(
    path: str | Path,
    mesh: MeshState,
    point_data: dict[str, np.ndarray] | None = None,
    field_types: dict[str, str] | None = None,
) -> Path:
    """将网格和场数据写入 VTU 文件。

    以非结构化网格格式输出活动四面体单元，并可附加
    逐顶点的标量或向量属性数据。

    Parameters
    ----------
    path : str or Path
        输出路径（建议使用 ``.vtu`` 扩展名）。
    mesh : MeshState
        包含顶点位置、四面体拓扑和活动掩码的网格状态。
    point_data : dict[str, np.ndarray] | None, optional
        逐顶点的附加数据字典。
    field_types : dict[str, str] | None, optional
        每个字段的 VTK 数据类型（"Float32" / "Float64" / "Int32"），
        未指定则默认 "Float32"。

    Returns
    -------
    Path
        写入的 VTU 文件路径。
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    active_tets = mesh.tets[mesh.active_tet_mask]
    point_data = point_data or {}
    field_types = field_types or {}
    with output.open("w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0"?>\n')
        handle.write(
            '<VTKFile type="UnstructuredGrid" version="0.1"'
            ' byte_order="LittleEndian">\n'
        )
        handle.write("  <UnstructuredGrid>\n")
        handle.write(
            f'    <Piece NumberOfPoints="{len(mesh.vertices)}"'
            f' NumberOfCells="{len(active_tets)}">\n'
        )
        # ── 顶点坐标 ──
        handle.write("      <Points>\n")
        handle.write(
            '        <DataArray type="Float32"'
            ' NumberOfComponents="3" format="ascii">\n'
        )
        for vertex in mesh.vertices:
            handle.write(
                f"          {float(vertex[0])} {float(vertex[1])} {float(vertex[2])}\n"
            )
        handle.write("        </DataArray>\n")
        handle.write("      </Points>\n")

        # ── 单元拓扑 ──
        handle.write("      <Cells>\n")
        handle.write(
            '        <DataArray type="Int32" Name="connectivity"'
            ' format="ascii">\n'
        )
        for tet in active_tets:
            handle.write(
                f"          {tet[0]} {tet[1]} {tet[2]} {tet[3]}\n"
            )
        handle.write("        </DataArray>\n")
        handle.write(
            '        <DataArray type="Int32" Name="offsets"'
            ' format="ascii">\n'
        )
        handle.write(
            "          "
            + " ".join(
                str(4 * (i + 1)) for i in range(len(active_tets))
            )
            + "\n"
        )
        handle.write(
            '        <DataArray type="UInt8" Name="types"'
            ' format="ascii">\n'
        )
        handle.write(
            "          "
            + " ".join("10" for _ in range(len(active_tets)))
            + "\n"
        )
        handle.write("        </DataArray>\n")
        handle.write("      </Cells>\n")

        # ── 点属性数据 ──
        if point_data:
            handle.write("      <PointData>\n")
            for name, values in point_data.items():
                array = np.asarray(values)
                components = (
                    1 if array.ndim == 1 else array.shape[1]
                )
                vtk_type = field_types.get(name, "Float32")
                handle.write(
                    f'        <DataArray type="{vtk_type}" Name="{name}"'
                    f' NumberOfComponents="{components}"'
                    ' format="ascii">\n'
                )
                if vtk_type.startswith("Int"):
                    fmt_val = lambda v: str(int(v))
                else:
                    fmt_val = lambda v: str(float(v))
                for value in array:
                    if np.isscalar(value) or isinstance(value, (int, float)):
                        handle.write(f"          {fmt_val(value)}\n")
                    else:
                        handle.write(
                            "          "
                            + " ".join(fmt_val(item) for item in value)
                            + "\n"
                        )
                handle.write("        </DataArray>\n")
            handle.write("      </PointData>\n")
        handle.write("    </Piece>\n")
        handle.write("  </UnstructuredGrid>\n")
        handle.write("</VTKFile>\n")
    return output
