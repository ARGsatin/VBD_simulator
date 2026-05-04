from __future__ import annotations

from pathlib import Path

import numpy as np

from hydrogel_vbd.state import MeshState


def write_vtu(path: str | Path, mesh: MeshState, point_data: dict[str, np.ndarray] | None = None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    active_tets = mesh.tets[mesh.active_tet_mask]
    point_data = point_data or {}
    with output.open("w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0"?>\n')
        handle.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        handle.write("  <UnstructuredGrid>\n")
        handle.write(f'    <Piece NumberOfPoints="{len(mesh.vertices)}" NumberOfCells="{len(active_tets)}">\n')
        handle.write("      <Points>\n")
        handle.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for vertex in mesh.vertices:
            handle.write(f"          {vertex[0]} {vertex[1]} {vertex[2]}\n")
        handle.write("        </DataArray>\n")
        handle.write("      </Points>\n")
        handle.write("      <Cells>\n")
        handle.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for tet in active_tets:
            handle.write(f"          {tet[0]} {tet[1]} {tet[2]} {tet[3]}\n")
        handle.write("        </DataArray>\n")
        handle.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        handle.write("          " + " ".join(str(4 * (i + 1)) for i in range(len(active_tets))) + "\n")
        handle.write("        </DataArray>\n")
        handle.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        handle.write("          " + " ".join("10" for _ in range(len(active_tets))) + "\n")
        handle.write("        </DataArray>\n")
        handle.write("      </Cells>\n")
        if point_data:
            handle.write("      <PointData>\n")
            for name, values in point_data.items():
                array = np.asarray(values)
                components = 1 if array.ndim == 1 else array.shape[1]
                handle.write(
                    f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="{components}" format="ascii">\n'
                )
                for value in array:
                    if np.isscalar(value):
                        handle.write(f"          {float(value)}\n")
                    else:
                        handle.write("          " + " ".join(str(float(item)) for item in value) + "\n")
                handle.write("        </DataArray>\n")
            handle.write("      </PointData>\n")
        handle.write("    </Piece>\n")
        handle.write("  </UnstructuredGrid>\n")
        handle.write("</VTKFile>\n")
    return output
