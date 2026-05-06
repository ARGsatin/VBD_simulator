from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _as_float_array(name: str, value: Any, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape (N, {', '.join(map(str, shape_tail))})")
    return array


def _as_int_array(name: str, value: Any, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=int)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape (N, {', '.join(map(str, shape_tail))})")
    return array


@dataclass
class MeshState:
    vertices: np.ndarray
    tets: np.ndarray
    layer_id_per_vertex: np.ndarray
    layer_id_per_tet: np.ndarray
    ideal_vertices: np.ndarray | None = None
    first_active_layer: np.ndarray | None = None
    is_bottom_surface: np.ndarray | None = None
    is_top_surface_of_layer: np.ndarray | None = None
    is_top_fixed: np.ndarray | None = None
    prev_vertices: np.ndarray | None = None
    velocities: np.ndarray | None = None
    vertex2tets: list[list[int]] = field(default_factory=list)
    tet_volumes: np.ndarray | None = None
    dm_inv: np.ndarray | None = None
    dm: np.ndarray | None = None
    neighbors: list[set[int]] = field(default_factory=list)
    node_mass: np.ndarray | None = None
    czm_state: np.ndarray | None = None
    damage: np.ndarray | None = None
    time_free: np.ndarray | None = None
    colors: np.ndarray | None = None
    color_ranges: list[tuple[int, int]] = field(default_factory=list)
    active_vertex_mask: np.ndarray | None = None
    active_tet_mask: np.ndarray | None = None
    boundary_flags: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.vertices = _as_float_array("vertices", self.vertices, (3,))
        self.tets = _as_int_array("tets", self.tets, (4,))
        vertex_count = self.vertices.shape[0]
        tet_count = self.tets.shape[0]
        self.ideal_vertices = (
            self.vertices.copy()
            if self.ideal_vertices is None
            else _as_float_array("ideal_vertices", self.ideal_vertices, (3,))
        )
        if self.ideal_vertices.shape != self.vertices.shape:
            raise ValueError("ideal_vertices must match vertices shape")

        self.layer_id_per_vertex = np.asarray(self.layer_id_per_vertex, dtype=int)
        if self.layer_id_per_vertex.shape != (vertex_count,):
            raise ValueError("layer_id_per_vertex must have shape (N,)")

        self.layer_id_per_tet = np.asarray(self.layer_id_per_tet, dtype=int)
        if self.layer_id_per_tet.shape != (tet_count,):
            raise ValueError("layer_id_per_tet must have shape (T,)")
        self.first_active_layer = (
            self.layer_id_per_vertex.copy()
            if self.first_active_layer is None
            else np.asarray(self.first_active_layer, dtype=int)
        )
        if self.first_active_layer.shape != (vertex_count,):
            raise ValueError("first_active_layer must have shape (N,)")

        if np.any(self.tets < 0) or np.any(self.tets >= vertex_count):
            raise ValueError("tets contains vertex indices outside vertices")

        self.prev_vertices = (
            self.vertices.copy()
            if self.prev_vertices is None
            else _as_float_array("prev_vertices", self.prev_vertices, (3,))
        )
        self.velocities = (
            np.zeros_like(self.vertices)
            if self.velocities is None
            else _as_float_array("velocities", self.velocities, (3,))
        )
        if self.prev_vertices.shape != self.vertices.shape or self.velocities.shape != self.vertices.shape:
            raise ValueError("prev_vertices and velocities must match vertices shape")

        self.active_vertex_mask = (
            np.zeros(vertex_count, dtype=bool)
            if self.active_vertex_mask is None
            else np.asarray(self.active_vertex_mask, dtype=bool)
        )
        self.active_tet_mask = (
            np.zeros(tet_count, dtype=bool)
            if self.active_tet_mask is None
            else np.asarray(self.active_tet_mask, dtype=bool)
        )
        if self.active_vertex_mask.shape != (vertex_count,) or self.active_tet_mask.shape != (tet_count,):
            raise ValueError("active masks must match mesh sizes")

        self.boundary_flags = (
            np.zeros(vertex_count, dtype=bool)
            if self.boundary_flags is None
            else np.asarray(self.boundary_flags, dtype=bool)
        )
        if self.boundary_flags.shape != (vertex_count,):
            raise ValueError("boundary_flags must have shape (N,)")
        z_min = float(np.min(self.ideal_vertices[:, 2])) if vertex_count else 0.0
        self.is_bottom_surface = (
            np.isclose(self.ideal_vertices[:, 2], z_min)
            if self.is_bottom_surface is None
            else np.asarray(self.is_bottom_surface, dtype=bool)
        )
        self.is_top_surface_of_layer = (
            np.full(vertex_count, -1, dtype=int)
            if self.is_top_surface_of_layer is None
            else np.asarray(self.is_top_surface_of_layer, dtype=int)
        )
        self.is_top_fixed = (
            np.zeros(vertex_count, dtype=bool)
            if self.is_top_fixed is None
            else np.asarray(self.is_top_fixed, dtype=bool)
        )
        for name, array in (
            ("is_bottom_surface", self.is_bottom_surface),
            ("is_top_surface_of_layer", self.is_top_surface_of_layer),
            ("is_top_fixed", self.is_top_fixed),
        ):
            if array.shape != (vertex_count,):
                raise ValueError(f"{name} must have shape (N,)")

        if not self.vertex2tets:
            self.vertex2tets = self._build_vertex2tets()
        if not self.neighbors:
            self.neighbors = self._build_neighbors()
        if self.tet_volumes is None:
            self.tet_volumes = self._compute_tet_volumes(self.ideal_vertices, self.tets)
        else:
            self.tet_volumes = np.asarray(self.tet_volumes, dtype=float)
        if self.dm is None or self.dm_inv is None:
            self.precompute_reference_matrices(c_shrink=1.0)
        self.node_mass = (
            self._build_node_masses(density=1.0)
            if self.node_mass is None
            else np.asarray(self.node_mass, dtype=float)
        )
        self.czm_state = (
            np.zeros(vertex_count, dtype=int)
            if self.czm_state is None
            else np.asarray(self.czm_state, dtype=int)
        )
        self.damage = (
            np.zeros(vertex_count, dtype=float)
            if self.damage is None
            else np.asarray(self.damage, dtype=float)
        )
        self.time_free = (
            np.zeros(vertex_count, dtype=float)
            if self.time_free is None
            else np.asarray(self.time_free, dtype=float)
        )
        for name, array in (("node_mass", self.node_mass), ("czm_state", self.czm_state), ("damage", self.damage), ("time_free", self.time_free)):
            if array.shape != (vertex_count,):
                raise ValueError(f"{name} must have shape (N,)")

    def _build_vertex2tets(self) -> list[list[int]]:
        adjacency = [[] for _ in range(self.vertices.shape[0])]
        for tet_id, tet in enumerate(self.tets):
            for vertex_id in tet:
                adjacency[int(vertex_id)].append(tet_id)
        return adjacency

    def _build_neighbors(self) -> list[set[int]]:
        neighbors: list[set[int]] = [set() for _ in range(self.vertices.shape[0])]
        for tet in self.tets:
            for vertex_id in tet:
                neighbors[int(vertex_id)].update(int(other) for other in tet if int(other) != int(vertex_id))
        return neighbors

    @staticmethod
    def _compute_tet_volumes(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
        volumes = np.zeros(tets.shape[0], dtype=float)
        for tet_id, tet in enumerate(tets):
            p0, p1, p2, p3 = vertices[tet]
            volumes[tet_id] = abs(float(np.linalg.det(np.column_stack((p1 - p0, p2 - p0, p3 - p0))))) / 6.0
        return volumes

    def precompute_reference_matrices(self, c_shrink: float) -> None:
        dm = np.zeros((self.tets.shape[0], 3, 3), dtype=float)
        dm_inv = np.zeros_like(dm)
        volumes = np.zeros(self.tets.shape[0], dtype=float)
        reference = self.ideal_vertices * float(c_shrink)
        for tet_id, tet in enumerate(self.tets):
            p0, p1, p2, p3 = reference[tet]
            matrix = np.column_stack((p1 - p0, p2 - p0, p3 - p0))
            dm[tet_id] = matrix
            det = float(np.linalg.det(matrix))
            volumes[tet_id] = abs(det) / 6.0
            if abs(det) > 1e-12:
                dm_inv[tet_id] = np.linalg.inv(matrix)
        self.dm = dm
        self.dm_inv = dm_inv
        self.tet_volumes = volumes

    def _build_node_masses(self, density: float) -> np.ndarray:
        masses = np.zeros(self.vertices.shape[0], dtype=float)
        volumes = self.tet_volumes if self.tet_volumes is not None else np.ones(self.tets.shape[0], dtype=float)
        for tet_id, tet in enumerate(self.tets):
            masses[tet] += float(volumes[tet_id]) * float(density) / 4.0
        masses[masses <= 0.0] = 1.0
        return masses

    def activate_layer(self, current_layer: int) -> None:
        self.active_vertex_mask = self.first_active_layer <= current_layer
        self.active_tet_mask = self.layer_id_per_tet <= current_layer

    def bottom_nodes(self, layer_id: int) -> np.ndarray:
        return np.flatnonzero(self.is_top_surface_of_layer == layer_id)

    def top_nodes(self, layer_id: int) -> np.ndarray:
        return np.flatnonzero(self.is_top_surface_of_layer == layer_id + 1)

    def layer_interface_nodes(self, interface_id: int) -> np.ndarray:
        return np.flatnonzero(self.is_top_surface_of_layer == interface_id)

    @property
    def masses(self) -> np.ndarray:
        return self.node_mass.copy() if self.node_mass is not None else np.ones(self.vertices.shape[0], dtype=float)


@dataclass
class MaterialState:
    density: float
    young_modulus: np.ndarray
    poisson_ratio: float
    damping: np.ndarray
    curing_degree: np.ndarray
    peel_stress_crit: float
    electric_response_alpha: np.ndarray
    mu: np.ndarray | None = None
    lam: np.ndarray | None = None


@dataclass
class ForceState:
    gravity: np.ndarray
    peel: np.ndarray
    fluid: np.ndarray
    surface: np.ndarray
    electric: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.gravity + self.peel + self.fluid + self.surface + self.electric


@dataclass
class FieldCommand:
    voltage: np.ndarray
    polarity: np.ndarray | None = None
    duration: float = 0.0
    start_time: float = 0.0
    electrode_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.voltage = np.asarray(self.voltage, dtype=float)
        if self.polarity is None:
            self.polarity = np.where(self.voltage > 0.0, 1, np.where(self.voltage < 0.0, -1, 0))
        else:
            self.polarity = np.asarray(self.polarity, dtype=int)


@dataclass
class LayerResult:
    layer_id: int
    x_sim: np.ndarray
    v_sim: np.ndarray
    error_metrics: dict[str, float]
    field_command_next: FieldCommand
    max_deformation: float
    rms_error: float
    success: bool
