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
    prev_vertices: np.ndarray | None = None
    velocities: np.ndarray | None = None
    vertex2tets: list[list[int]] = field(default_factory=list)
    tet_volumes: np.ndarray | None = None
    dm_inv: np.ndarray | None = None
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

        self.layer_id_per_vertex = np.asarray(self.layer_id_per_vertex, dtype=int)
        if self.layer_id_per_vertex.shape != (vertex_count,):
            raise ValueError("layer_id_per_vertex must have shape (N,)")

        self.layer_id_per_tet = np.asarray(self.layer_id_per_tet, dtype=int)
        if self.layer_id_per_tet.shape != (tet_count,):
            raise ValueError("layer_id_per_tet must have shape (T,)")

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

        if not self.vertex2tets:
            self.vertex2tets = self._build_vertex2tets()

    def _build_vertex2tets(self) -> list[list[int]]:
        adjacency = [[] for _ in range(self.vertices.shape[0])]
        for tet_id, tet in enumerate(self.tets):
            for vertex_id in tet:
                adjacency[int(vertex_id)].append(tet_id)
        return adjacency

    def activate_layer(self, current_layer: int) -> None:
        self.active_vertex_mask = self.layer_id_per_vertex <= current_layer
        self.active_tet_mask = self.layer_id_per_tet <= current_layer

    @property
    def masses(self) -> np.ndarray:
        if self.tet_volumes is None or len(self.tet_volumes) != self.tets.shape[0]:
            return np.ones(self.vertices.shape[0], dtype=float)
        masses = np.zeros(self.vertices.shape[0], dtype=float)
        for tet_id, tet in enumerate(self.tets):
            masses[tet] += float(self.tet_volumes[tet_id]) / 4.0
        masses[masses <= 0.0] = 1.0
        return masses


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
