# -*- coding: utf-8 -*-
"""Read-only diagnostics for solver performance and convergence issues."""

from __future__ import annotations

import copy
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.core.state import MeshState
from hydrogel_vbd.physics.czm import CZMState
from hydrogel_vbd.physics.local_terms import build_local_physics_terms


DX_CLIP_DIAGNOSTIC = 0.002


@dataclass(frozen=True)
class LiftPlanDiagnostics:
    lift_max: float
    lift_step: float
    expected_steps: int
    estimated_wall_s: float


def diagnostics_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether solver diagnostics are enabled by environment."""
    source = os.environ if env is None else env
    value = str(source.get("HYDROGEL_VBD_SOLVER_DIAG", "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _expected_lift_steps(lift_max: float, lift_step: float) -> int:
    if lift_max <= 0.0:
        return 0
    if lift_step <= 0.0:
        raise ValueError("lift_step must be positive")
    ratio = lift_max / lift_step
    return int(math.ceil(ratio - max(1e-12, abs(ratio) * 1e-12)))


def compute_lift_plan(
    *,
    layer_thickness: float,
    v_lift: float,
    dt: float,
    lift_multiplier: float = 1.5,
    avg_call_ms: float | None = None,
) -> LiftPlanDiagnostics:
    """Compute the outer lift-loop plan without mutating solver state."""
    lift_max = float(lift_multiplier) * float(layer_thickness)
    lift_step = float(v_lift) * float(dt)
    expected_steps = (
        _expected_lift_steps(lift_max, lift_step) if lift_step > 0.0 else 0
    )
    estimated_wall_s = (
        expected_steps * float(avg_call_ms) / 1000.0
        if avg_call_ms is not None
        else 0.0
    )
    return LiftPlanDiagnostics(
        lift_max=lift_max,
        lift_step=lift_step,
        expected_steps=expected_steps,
        estimated_wall_s=estimated_wall_s,
    )


@dataclass(frozen=True)
class SolverStepDiagnostics:
    layer_id: int
    step: int
    lift_max: float
    lift_step: float
    expected_steps: int
    iterations: int
    stable_steps: int
    max_dx: float
    clipped: bool
    active_vertices: int
    active_tets: int
    top_count: int
    bottom_count: int
    czm_counts: str
    z_fep: float
    current_bottom_z_min: float
    current_bottom_z_max: float
    previous_bottom_z_min: float
    previous_bottom_z_max: float
    global_bottom_z_min: float
    global_bottom_z_max: float
    max_move_node: int
    max_move_kind: str
    max_move_czm_state: str
    max_move_dx: float
    max_move_dy: float
    max_move_dz: float
    max_move_norm: float
    max_move_z: float
    call_ms: float

    @classmethod
    def csv_fields(cls) -> list[str]:
        return [
            "layer_id",
            "step",
            "lift_max",
            "lift_step",
            "expected_steps",
            "iterations",
            "stable_steps",
            "max_dx",
            "clipped",
            "active_vertices",
            "active_tets",
            "top_count",
            "bottom_count",
            "czm_counts",
            "z_fep",
            "current_bottom_z_min",
            "current_bottom_z_max",
            "previous_bottom_z_min",
            "previous_bottom_z_max",
            "global_bottom_z_min",
            "global_bottom_z_max",
            "max_move_node",
            "max_move_kind",
            "max_move_czm_state",
            "max_move_dx",
            "max_move_dy",
            "max_move_dz",
            "max_move_norm",
            "max_move_z",
            "call_ms",
        ]

    @classmethod
    def from_mesh(
        cls,
        mesh: MeshState,
        *,
        layer_id: int,
        step: int,
        lift_max: float,
        lift_step: float,
        expected_steps: int,
        result: Any,
        call_ms: float,
        dx_clip: float = DX_CLIP_DIAGNOSTIC,
        z_fep: float | None = None,
        x_before: np.ndarray | None = None,
    ) -> "SolverStepDiagnostics":
        active = np.asarray(mesh.active_vertex_mask, dtype=bool)
        active_tets = np.asarray(mesh.active_tet_mask, dtype=bool)
        top_mask = np.asarray(mesh.is_top_fixed, dtype=bool) & active
        bottom_nodes = mesh.bottom_nodes(layer_id)
        czm_mask = np.zeros_like(active, dtype=bool)
        if bottom_nodes.size:
            czm_mask[bottom_nodes] = True
        elif not np.any(mesh.is_top_surface_of_layer >= 0):
            czm_mask = np.asarray(mesh.is_bottom_surface, dtype=bool).copy()
        czm_mask &= active
        previous_bottom = (
            mesh.bottom_nodes(layer_id - 1)
            if int(layer_id) > 0
            else np.zeros(0, dtype=int)
        )
        global_bottom = np.flatnonzero(np.asarray(mesh.is_bottom_surface, dtype=bool) & active)
        current_z_min, current_z_max = _z_stats(mesh, bottom_nodes)
        previous_z_min, previous_z_max = _z_stats(mesh, previous_bottom)
        global_z_min, global_z_max = _z_stats(mesh, global_bottom)
        czm_state = np.asarray(mesh.czm_state, dtype=int)
        current_bottom_mask = np.zeros_like(active, dtype=bool)
        if bottom_nodes.size:
            current_bottom_mask[bottom_nodes] = True
        elif not np.any(mesh.is_top_surface_of_layer >= 0):
            current_bottom_mask = np.asarray(mesh.is_bottom_surface, dtype=bool).copy()
        current_bottom_mask &= active
        (
            max_move_node,
            max_move_kind,
            max_move_state,
            max_move_dx,
            max_move_dy,
            max_move_dz,
            max_move_norm,
            max_move_z,
        ) = (
            _max_move_info(
                mesh, layer_id, current_bottom=current_bottom_mask, x_before=x_before
            )
        )
        counts = {
            "fixed": int(np.sum(czm_mask & (czm_state == int(CZMState.FIXED)))),
            "damaging": int(np.sum(czm_mask & (czm_state == int(CZMState.DAMAGING)))),
            "free": int(np.sum(czm_mask & (czm_state == int(CZMState.FREE)))),
        }
        max_dx = float(getattr(result, "max_dx", 0.0))
        return cls(
            layer_id=int(layer_id),
            step=int(step),
            lift_max=float(lift_max),
            lift_step=float(lift_step),
            expected_steps=int(expected_steps),
            iterations=int(getattr(result, "iterations", 0)),
            stable_steps=int(getattr(result, "stable_steps", 0)),
            max_dx=max_dx,
            clipped=bool(max_dx >= dx_clip * (1.0 - 1e-9)),
            active_vertices=int(np.sum(active)),
            active_tets=int(np.sum(active_tets)),
            top_count=int(np.sum(top_mask)),
            bottom_count=int(len(bottom_nodes)),
            czm_counts=(
                f"fixed:{counts['fixed']},"
                f"damaging:{counts['damaging']},"
                f"free:{counts['free']}"
            ),
            z_fep=float("nan") if z_fep is None else float(z_fep),
            current_bottom_z_min=current_z_min,
            current_bottom_z_max=current_z_max,
            previous_bottom_z_min=previous_z_min,
            previous_bottom_z_max=previous_z_max,
            global_bottom_z_min=global_z_min,
            global_bottom_z_max=global_z_max,
            max_move_node=max_move_node,
            max_move_kind=max_move_kind,
            max_move_czm_state=max_move_state,
            max_move_dx=max_move_dx,
            max_move_dy=max_move_dy,
            max_move_dz=max_move_dz,
            max_move_norm=max_move_norm,
            max_move_z=max_move_z,
            call_ms=float(call_ms),
        )

    def as_csv_row(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.csv_fields()}


def _z_stats(mesh: MeshState, nodes: np.ndarray) -> tuple[float, float]:
    if len(nodes) == 0:
        return float("nan"), float("nan")
    z = np.asarray(mesh.vertices[nodes, 2], dtype=float)
    return float(np.min(z)), float(np.max(z))


def _max_move_info(
    mesh: MeshState,
    layer_id: int,
    *,
    current_bottom: np.ndarray,
    x_before: np.ndarray | None,
) -> tuple[int, str, str, float, float, float, float, float]:
    if x_before is None or x_before.shape != mesh.vertices.shape:
        nan = float("nan")
        return -1, "", "", nan, nan, nan, nan, nan
    active = np.asarray(mesh.active_vertex_mask, dtype=bool)
    if not np.any(active):
        nan = float("nan")
        return -1, "", "", nan, nan, nan, nan, nan
    delta = np.asarray(mesh.vertices - x_before, dtype=float)
    norms = np.linalg.norm(delta, axis=1)
    norms[~active] = -1.0
    node_id = int(np.argmax(norms))
    move = delta[node_id]
    return (
        node_id,
        _node_kind(mesh, layer_id, node_id, current_bottom=current_bottom),
        _czm_state_name(mesh, node_id),
        float(move[0]),
        float(move[1]),
        float(move[2]),
        float(norms[node_id]),
        float(mesh.vertices[node_id, 2]),
    )


def _node_kind(
    mesh: MeshState,
    layer_id: int,
    node_id: int,
    *,
    current_bottom: np.ndarray,
) -> str:
    previous_bottom = (
        mesh.bottom_nodes(layer_id - 1)
        if int(layer_id) > 0
        else np.zeros(0, dtype=int)
    )
    if bool(mesh.is_top_fixed[node_id]):
        return "top_fixed"
    if bool(current_bottom[node_id]):
        return "current_bottom"
    if np.any(previous_bottom == node_id):
        return "previous_bottom"
    if bool(mesh.is_bottom_surface[node_id]):
        return "global_bottom"
    return "interior"


def _czm_state_name(mesh: MeshState, node_id: int) -> str:
    value = int(mesh.czm_state[node_id])
    try:
        return CZMState(value).name.lower()
    except ValueError:
        return str(value)


def write_solver_diagnostics_csv(
    path: str | Path,
    rows: list[SolverStepDiagnostics],
) -> None:
    """Append solver diagnostics rows to CSV, writing a stable header first."""
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SolverStepDiagnostics.csv_fields())
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


class SolverRunawayGuard:
    """Detect repeated clipped non-converged solver steps in diagnostic mode."""

    def __init__(
        self,
        *,
        limit: int = 50,
        max_iters: int,
        dx_clip: float = DX_CLIP_DIAGNOSTIC,
    ) -> None:
        self.limit = int(limit)
        self.max_iters = int(max_iters)
        self.dx_clip = float(dx_clip)
        self.consecutive_bad_steps = 0

    def observe(self, diag: SolverStepDiagnostics) -> bool:
        bad = (
            int(diag.iterations) >= self.max_iters
            and float(diag.max_dx) >= self.dx_clip * (1.0 - 1e-9)
        )
        if bad:
            self.consecutive_bad_steps += 1
        else:
            self.consecutive_bad_steps = 0
        return self.consecutive_bad_steps >= self.limit


@dataclass(frozen=True)
class PhysicsAblationDiagnostic:
    case: str
    force_norm: float
    hessian_min_eig: float
    hessian_max_eig: float
    iterations: int
    stable_steps: int
    max_dx: float


def _active_hessian_eig_range(mesh: MeshState, hessian: np.ndarray) -> tuple[float, float]:
    active_ids = np.flatnonzero(mesh.active_vertex_mask)
    if len(active_ids) == 0:
        return 0.0, 0.0
    eigs = [np.linalg.eigvalsh(hessian[i]) for i in active_ids]
    all_eigs = np.concatenate(eigs)
    return float(np.min(all_eigs)), float(np.max(all_eigs))


def _configure_ablation_case(
    mesh: MeshState,
    config: SimulationConfig,
    layer_id: int,
    case: str,
) -> tuple[MeshState, SimulationConfig, float]:
    m = copy.deepcopy(mesh)
    c = copy.deepcopy(config)
    e_z_scale = 0.0

    c.g = (0.0, 0.0, 0.0)
    c.q_ion = 0.0
    c.c_shrink = 1.0
    c.d_fluid_max = 0.0
    c.t_fluid_max = 0.0

    if case in {"plus_shrink", "plus_czm", "plus_fluid", "plus_electric"}:
        c.c_shrink = config.c_shrink

    bottom = m.bottom_nodes(layer_id)
    if case in {"plus_czm", "plus_fluid", "plus_electric"} and len(bottom):
        m.czm_state[bottom] = int(CZMState.DAMAGING)
        m.damage[bottom] = 0.0

    if case in {"plus_fluid", "plus_electric"} and len(bottom):
        m.czm_state[bottom] = int(CZMState.FREE)
        m.time_free[bottom] = 0.0
        c.d_fluid_max = config.d_fluid_max
        c.t_fluid_max = config.t_fluid_max

    if case == "plus_electric":
        c.q_ion = config.q_ion
        e_z_scale = 1.0

    return m, c, e_z_scale


def collect_physics_ablation_diagnostics(
    mesh: MeshState,
    config: SimulationConfig,
    *,
    layer_id: int,
    e_z: float,
    lifting_top: np.ndarray,
    solve_step: Callable[[MeshState, SimulationConfig, float, int, np.ndarray], Any] | None = None,
) -> list[PhysicsAblationDiagnostic]:
    """Run isolated local-term cases and record force/Hessian/solver metrics."""
    if solve_step is None:
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver

        def solve_step(m: MeshState, c: SimulationConfig, ez: float, lid: int, top: np.ndarray) -> Any:
            return PythonReferenceVBDSolver(c).solve_with_lift(
                m, layer_id=lid, e_z=ez, lifting_top=top
            )

    rows: list[PhysicsAblationDiagnostic] = []
    for case in [
        "elastic_only",
        "plus_shrink",
        "plus_czm",
        "plus_fluid",
        "plus_electric",
    ]:
        case_mesh, case_config, e_z_scale = _configure_ablation_case(
            mesh, config, layer_id, case
        )
        x_prev = (
            case_mesh.prev_vertices
            if case_mesh.prev_vertices is not None
            else case_mesh.vertices.copy()
        )
        case_e_z = float(e_z) * e_z_scale
        terms = build_local_physics_terms(
            case_mesh, case_config, e_z=case_e_z, x_prev=x_prev
        )
        force_norm = float(np.linalg.norm(terms.force[case_mesh.active_vertex_mask]))
        hmin, hmax = _active_hessian_eig_range(case_mesh, terms.hessian)
        result = solve_step(
            case_mesh,
            case_config,
            case_e_z,
            layer_id,
            np.asarray(lifting_top, dtype=int),
        )
        rows.append(
            PhysicsAblationDiagnostic(
                case=case,
                force_norm=force_norm,
                hessian_min_eig=hmin,
                hessian_max_eig=hmax,
                iterations=int(getattr(result, "iterations", 0)),
                stable_steps=int(getattr(result, "stable_steps", 0)),
                max_dx=float(getattr(result, "max_dx", 0.0)),
            )
        )
    return rows


@dataclass(frozen=True)
class CppAdapterPreparationProfile:
    tets_copied: bool
    colors_copied: bool
    bottom_surface_copied: bool
    elapsed_ms: float


def _ascontiguous_copy_flag(arr: np.ndarray, dtype: Any) -> tuple[np.ndarray, bool]:
    prepared = np.ascontiguousarray(arr, dtype=dtype)
    return prepared, prepared is not arr


def profile_cpp_adapter_preparation(mesh: MeshState) -> CppAdapterPreparationProfile:
    """Measure whether the C++ adapter preparation would allocate copies."""
    t0 = time.perf_counter()
    _, tets_copied = _ascontiguous_copy_flag(mesh.tets, np.int32)
    _, colors_copied = _ascontiguous_copy_flag(mesh.colors, np.int32)
    _, bottom_surface_copied = _ascontiguous_copy_flag(
        mesh.is_bottom_surface, bool
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return CppAdapterPreparationProfile(
        tets_copied=bool(tets_copied),
        colors_copied=bool(colors_copied),
        bottom_surface_copied=bool(bottom_surface_copied),
        elapsed_ms=float(elapsed_ms),
    )
