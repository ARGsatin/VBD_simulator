from __future__ import annotations

import numpy as np

from hydrogel_vbd.config import SimulationConfig
from hydrogel_vbd.solver.graph_coloring import greedy_vertex_coloring
from hydrogel_vbd.state import MeshState


class ConformalMeshPipeline:
    """Builds a globally conformal layered tetrahedral mesh.

    The production hook will call a PLC-capable mesher. The demo path below
    creates shared layer-interface nodes so activation and stress inheritance
    exercise the same topology contract.
    """

    @staticmethod
    def create_demo(layers: int, layer_thickness: float = 0.05, config: SimulationConfig | None = None) -> tuple[MeshState, int]:
        if layers < 1:
            raise ValueError("layers must be positive")
        config = config or SimulationConfig(layer_thickness=layer_thickness)
        vertices: list[list[float]] = []
        first_active: list[int] = []
        interface_ids: list[int] = []
        is_bottom: list[bool] = []

        for surface_id in range(layers + 1):
            z = surface_id * layer_thickness
            vertices.extend(
                [
                    [0.0, 0.0, z],
                    [1.0, 0.0, z],
                    [0.0, 1.0, z],
                    [1.0, 1.0, z],
                ]
            )
            first = max(surface_id - 1, 0)
            first_active.extend([first] * 4)
            interface_ids.extend([surface_id] * 4)
            is_bottom.extend([surface_id == 0] * 4)

        tets: list[list[int]] = []
        tet_layers: list[int] = []
        for layer in range(layers):
            b = layer * 4
            t = (layer + 1) * 4
            layer_tets = [
                [b + 0, b + 1, b + 2, t + 0],
                [b + 1, b + 3, b + 2, t + 3],
                [b + 1, t + 1, t + 0, t + 3],
                [b + 2, t + 0, t + 2, t + 3],
                [b + 1, b + 2, t + 0, t + 3],
            ]
            tets.extend(layer_tets)
            tet_layers.extend([layer] * len(layer_tets))

        mesh = MeshState(
            vertices=np.asarray(vertices, dtype=float),
            tets=np.asarray(tets, dtype=int),
            layer_id_per_vertex=np.asarray(first_active, dtype=int),
            first_active_layer=np.asarray(first_active, dtype=int),
            layer_id_per_tet=np.asarray(tet_layers, dtype=int),
            is_bottom_surface=np.asarray(is_bottom, dtype=bool),
            is_top_surface_of_layer=np.asarray(interface_ids, dtype=int),
        )
        mesh.precompute_reference_matrices(config.c_shrink)
        mesh.node_mass = mesh._build_node_masses(config.rho)
        mesh.colors = greedy_vertex_coloring(mesh)
        return mesh, layers
