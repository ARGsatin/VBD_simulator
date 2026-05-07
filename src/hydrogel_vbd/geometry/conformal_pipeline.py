"""Build a globally conformal layered tetrahedral mesh.

Two construction paths are available:

* :meth:`create_demo` — synthetic rectangular column (original test mesh).
* :meth:`from_stl` — real geometry: read STL → TetGen → assign layer IDs.

Both return a :class:`~hydrogel_vbd.state.MeshState` whose layer metadata
is compatible with :class:`~hydrogel_vbd.geometry.layer_activator.LayerActivator`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hydrogel_vbd.config import SimulationConfig
from hydrogel_vbd.geometry.tet_mesher import from_stl as tet_mesh_from_stl
from hydrogel_vbd.solver.graph_coloring import greedy_vertex_coloring
from hydrogel_vbd.state import MeshState


def _build_vertex2tets(tets: np.ndarray, n_vertices: int) -> list[list[int]]:
    adj = [[] for _ in range(n_vertices)]
    for tid, tet in enumerate(tets):
        for v in tet:
            adj[int(v)].append(tid)
    return adj


def _compute_bottom_surface(vertices: np.ndarray, z_tol: float = 1e-6) -> np.ndarray:
    z_min = float(np.min(vertices[:, 2]))
    return np.abs(vertices[:, 2] - z_min) < z_tol


class ConformalMeshPipeline:
    """Builds a globally conformal layered tetrahedral mesh.

    The production hook will call a PLC-capable mesher. The demo path below
    creates shared layer-interface nodes so activation and stress inheritance
    exercise the same topology contract.
    """

    @staticmethod
    def create_demo(
        layers: int, layer_thickness: float = 0.05, config: SimulationConfig | None = None
    ) -> tuple[MeshState, int]:
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

    # ------------------------------------------------------------------
    # STL-based pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def from_stl(
        stl_path: str | Path,
        layer_height: float,
        config: SimulationConfig | None = None,
        quality: float = 1.0,
    ) -> tuple[MeshState, int]:
        """Build a conformal layered tet mesh from an STL file.

        Parameters
        ----------
        stl_path:
            Path to the input STL file.
        layer_height:
            Print layer thickness (same unit as the STL).
        config:
            Simulation configuration (material & solver parameters).
        quality:
            TetGen mesh refinement factor (0.1 … 5.0, default 1.0).

        Returns
        -------
        mesh:
            A fully initialized :class:`MeshState` with layer metadata.
        num_layers:
            The total number of layers.
        """
        config = config or SimulationConfig(layer_thickness=layer_height)

        # ---------- step 1 : tetrahedralise the full STL ----------
        vertices, tets = tet_mesh_from_stl(stl_path, quality=quality)
        n_vertices = len(vertices)
        z_min = float(np.min(vertices[:, 2]))
        z_max = float(np.max(vertices[:, 2]))

        if z_max - z_min < 1e-12:
            raise ValueError("STL has no thickness along the Z axis")

        num_layers = max(1, int(np.ceil((z_max - z_min) / layer_height)))

        # ---------- step 2 : assign every tet to a layer ----------
        centroids = vertices[tets].mean(axis=1)  # (T, 3)
        tet_layers = np.floor((centroids[:, 2] - z_min) / layer_height).astype(np.int32)
        tet_layers = np.clip(tet_layers, 0, num_layers - 1)

        # ---------- step 3 : per-vertex first-active layer ----------
        first_active = np.full(n_vertices, num_layers, dtype=np.int32)
        for tid, layer in enumerate(tet_layers):
            first_active[tets[tid]] = np.minimum(first_active[tets[tid]], layer)

        # ---------- step 4 : which layers each vertex touches ----------
        vertex_layers: list[set[int]] = [set() for _ in range(n_vertices)]
        for tid, layer in enumerate(tet_layers):
            for v in tets[tid]:
                vertex_layers[int(v)].add(int(layer))

        # ---------- step 5 : is_top_surface_of_layer ----------
        # A vertex at interface N (shared between layer N-1 and N tets)
        # gets is_top_surface_of_layer = N.
        is_top = np.full(n_vertices, -1, dtype=np.int32)
        for vi, layers in enumerate(vertex_layers):
            if len(layers) >= 2:
                # shared between multiple layers → it is an interface vertex
                is_top[vi] = max(layers)  # the upper interface
            elif len(layers) == 1:
                layer = list(layers)[0]
                vz = float(vertices[vi, 2])
                lo = z_min + layer * layer_height
                hi = lo + layer_height
                # top-most surface of the whole model
                if abs(vz - z_max) < layer_height * 0.05:
                    is_top[vi] = layer + 1
                elif abs(vz - lo) < layer_height * 0.05:
                    is_top[vi] = layer
                elif abs(vz - hi) < layer_height * 0.05:
                    is_top[vi] = layer + 1

        # ---------- step 6 : bottom surface ----------
        is_bottom = _compute_bottom_surface(vertices)

        # ---------- step 7 : assemble MeshState ----------
        mesh = MeshState(
            vertices=vertices,
            tets=tets,
            layer_id_per_vertex=first_active.copy(),
            first_active_layer=first_active.copy(),
            layer_id_per_tet=tet_layers,
            is_bottom_surface=is_bottom,
            is_top_surface_of_layer=is_top,
        )
        mesh.precompute_reference_matrices(config.c_shrink)
        mesh.node_mass = mesh._build_node_masses(config.rho)
        mesh.colors = greedy_vertex_coloring(mesh)
        return mesh, num_layers
