from __future__ import annotations

import numpy as np

from hydrogel_vbd.state import MeshState


def greedy_vertex_coloring(mesh: MeshState) -> np.ndarray:
    colors = np.full(mesh.vertices.shape[0], -1, dtype=int)
    neighbors: list[set[int]] = [set() for _ in range(mesh.vertices.shape[0])]
    for tet in mesh.tets:
        for i in tet:
            neighbors[int(i)].update(int(j) for j in tet if int(j) != int(i))
    for vertex_id, adjacent in enumerate(neighbors):
        used = {colors[n] for n in adjacent if colors[n] >= 0}
        color = 0
        while color in used:
            color += 1
        colors[vertex_id] = color
    return colors
