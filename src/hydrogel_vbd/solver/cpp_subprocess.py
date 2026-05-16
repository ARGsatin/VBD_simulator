# -*- coding: utf-8 -*-
"""C++ 求解器子进程隔离层。

将 C++ 求解器放到独立子进程中运行，确保 segfault 只杀死子进程，
GUI 主进程不受影响。子进程崩溃时自动回退到 Python 求解器。

架构
----
主进程 (QThread)                      子进程
    │                                   │
    ├─ spawn multiprocessing.Process ──→│
    ├─ send (mesh, config) ────────────→│ 导入 C++ 模块
    │                                   │ 逐层运行仿真
    ├─ ←── recv (progress/log/frame) ───┤
    │                                   │
    ├─ ←── recv (done + results) ───────│
    │                                   │ (若 C++ segfault)
    │   detect process died ───────────→│ ✗ 进程终止
    │   回退 Python 求解器              │
"""

from __future__ import annotations

import multiprocessing as mp
import math
import os
import pickle
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

MAX_EXPECTED_LIFT_STEPS = 100_000
DX_CLIP_DIAGNOSTIC = 0.002
LIFT_EPSILON_STEP_FRACTION = 0.12
LIFT_EPSILON_LAYER_FRACTION = 1.5e-3


def _configure_cpp_runtime_for_subprocess() -> dict[str, int]:
    """配置 C++ 子进程运行时，默认在隔离进程中启用 OpenMP。"""
    raw_threads = os.environ.get("HYDROGEL_VBD_SUBPROCESS_THREADS")
    if raw_threads is not None:
        try:
            threads = max(1, int(raw_threads))
        except ValueError:
            threads = 1
    else:
        threads = max(1, min(8, os.cpu_count() or 1))

    if threads > 1:
        os.environ["HYDROGEL_VBD_OMP"] = "1"
        os.environ["OMP_NUM_THREADS"] = str(threads)
    else:
        os.environ.pop("HYDROGEL_VBD_OMP", None)
        os.environ["OMP_NUM_THREADS"] = "1"

    return {"threads": threads}


# ── 消息类型 ──
@dataclass
class _ProgressMsg:
    layer: int       # 当前层号 (1-indexed)
    percentage: int  # 当前层提升进度 0-100
    step: int        # 累计步数


@dataclass
class _LogMsg:
    text: str


@dataclass
class _FrameMsg:
    vertices: np.ndarray
    tets: np.ndarray
    active_mask: np.ndarray
    active_tet_mask: np.ndarray
    title: str


@dataclass
class _DoneMsg:
    results: list[dict[str, Any]]


@dataclass
class _ErrorMsg:
    error: str


def _current_lifting_top(mesh: Any) -> np.ndarray:
    """返回当前激活层的夹持顶面节点。"""
    if mesh.is_top_fixed is None or mesh.active_vertex_mask is None:
        return np.zeros(0, dtype=np.int32)
    return np.asarray(
        np.flatnonzero(mesh.is_top_fixed & mesh.active_vertex_mask),
        dtype=np.int32,
    )


def _expected_lift_steps(lift_max: float, lift_step: float) -> int:
    """计算完成本层提升所需的外层时间步数。"""
    if lift_max <= 0.0:
        return 0
    if lift_step <= 0.0:
        raise ValueError("lift_step 必须为正数；请检查 v_lift 和 dt")
    ratio = lift_max / lift_step
    return int(math.ceil(ratio - max(1e-12, abs(ratio) * 1e-12)))


def _platform_return_distance(actual_lift: float, next_gap: float) -> float:
    actual_lift = max(0.0, float(actual_lift))
    _ = next_gap
    return actual_lift


def _positive_step_distances(total: float, max_step: float) -> list[float]:
    total = max(0.0, float(total))
    max_step = max(0.0, float(max_step))
    if total <= 0.0 or max_step <= 0.0:
        return []
    steps: list[float] = []
    remaining = total
    while remaining > max(1e-15, total * 1e-12):
        step = min(max_step, remaining)
        steps.append(step)
        remaining -= step
    return steps


def _layer_contact_z(config: Any, layer_id: int) -> float:
    return float(config.z_fep)


def _format_z_stats(label: str, values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return f"{label}=empty"
    return (
        f"{label}=n:{values.size},"
        f"min:{float(np.min(values)):.6e},"
        f"median:{float(np.median(values)):.6e},"
        f"max:{float(np.max(values)):.6e}"
    )


def _lift_convergence_epsilon(config: Any, lift_step: float, target_gap: float) -> float:
    base = max(0.0, float(getattr(config, "epsilon", 0.0)))
    lift_step = abs(float(lift_step))
    target_gap = abs(float(target_gap))
    if lift_step <= 0.0 or target_gap <= 0.0:
        return base
    scaled = min(
        LIFT_EPSILON_STEP_FRACTION * lift_step,
        LIFT_EPSILON_LAYER_FRACTION * target_gap,
    )
    return max(base, scaled)


def _active_tet_quality_values(mesh: Any) -> np.ndarray:
    vertices = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))), dtype=float)
    tets = np.asarray(getattr(mesh, "tets", np.zeros((0, 4), dtype=int)), dtype=int)
    active_mask = np.asarray(
        getattr(mesh, "active_tet_mask", np.zeros(tets.shape[0], dtype=bool)),
        dtype=bool,
    )
    if vertices.size == 0 or tets.size == 0 or active_mask.size == 0:
        return np.zeros(0, dtype=float)
    active_tets = tets[active_mask]
    if active_tets.size == 0:
        return np.zeros(0, dtype=float)

    p = vertices[active_tets]
    edge_vectors = np.stack(
        (
            p[:, 1] - p[:, 0],
            p[:, 2] - p[:, 0],
            p[:, 3] - p[:, 0],
            p[:, 2] - p[:, 1],
            p[:, 3] - p[:, 1],
            p[:, 3] - p[:, 2],
        ),
        axis=1,
    )
    max_edge = np.max(np.linalg.norm(edge_vectors, axis=2), axis=1)
    dm = np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
    volume = np.abs(np.linalg.det(dm)) / 6.0
    quality = np.zeros(active_tets.shape[0], dtype=float)
    valid = max_edge > 0.0
    quality[valid] = 6.0 * math.sqrt(2.0) * volume[valid] / (max_edge[valid] ** 3)
    return np.clip(quality, 0.0, 1.0)


def _format_tet_quality_stats(label: str, mesh: Any) -> str:
    quality = _active_tet_quality_values(mesh)
    quality = quality[np.isfinite(quality)]
    if quality.size == 0:
        return f"{label}=empty"
    thin_count = int(np.count_nonzero(quality < 1.0e-3))
    return (
        f"{label}=n:{quality.size},"
        f"min:{float(np.min(quality)):.6e},"
        f"median:{float(np.median(quality)):.6e},"
        f"max:{float(np.max(quality)):.6e},"
        f"thin_lt1e-3:{thin_count}"
    )


def _validate_lift_plan(
    layer_id: int, lift_max: float, lift_step: float, expected_steps: int
) -> None:
    """拒绝明显异常的提升计划，避免子进程静默长跑。"""
    if expected_steps > MAX_EXPECTED_LIFT_STEPS:
        raise RuntimeError(
            "提升步数异常过大: "
            f"layer={layer_id}, lift_max={lift_max:.6e} m, "
            f"lift_step={lift_step:.6e} m, steps={expected_steps}. "
            "请检查 GUI 层厚单位是否为 mm。"
        )


# ── 子进程入口 ──

def _solver_step_converged(result: Any, config: Any) -> bool:
    """Return whether a single solver call reached the configured tolerance."""
    max_dx = float(getattr(result, "max_dx", math.inf))
    stable_steps = int(getattr(result, "stable_steps", 0))
    epsilon = float(getattr(config, "epsilon", 0.0))
    n_stable = max(1, int(getattr(config, "N_stable", 1)))
    return math.isfinite(max_dx) and max_dx < epsilon and stable_steps >= n_stable


def _raise_if_detached_before_convergence(
    layer_id: int,
    layer_steps: int,
    result: Any,
    config: Any,
) -> None:
    """Reject all-free CZM states produced by an unconverged solver step."""
    if not bool(getattr(result, "all_free", False)):
        return
    if _solver_step_converged(result, config):
        return

    max_dx = float(getattr(result, "max_dx", math.nan))
    iterations = int(getattr(result, "iterations", 0))
    stable_steps = int(getattr(result, "stable_steps", 0))
    raise RuntimeError(
        f"layer {layer_id} detached before solver convergence: "
        "solver did not converge "
        f"(lift_step={layer_steps}, iterations={iterations}/"
        f"{int(getattr(config, 'max_iters', 0))}, "
        f"stable_steps={stable_steps}/{int(getattr(config, 'N_stable', 0))}, "
        f"max_dx={max_dx:.6e}, epsilon={float(getattr(config, 'epsilon', 0.0)):.6e}). "
        "Refusing to accept this layer result because it would produce invalid geometry."
    )


def _worker_run(
    conn: mp.connection.Connection,
    mesh_dict: dict[str, np.ndarray],
    config_dict: dict[str, Any],
    n_layers: int,
    output_dir: str,
    diag_enabled_override: bool | None = None,
    diag_stride_override: int | None = None,
) -> None:
    """子进程主函数：加载 C++ 模块并执行完整仿真。

    通过 *conn* 向主进程发送进度、帧和最终结果。
    任何未捕获异常（包括 segfault → 进程直接被 OS 杀死）都会
    导致管道断开，主进程检测到后回退到 Python 求解器。
    """
    try:
        _run_simulation(
            conn,
            mesh_dict,
            config_dict,
            n_layers,
            output_dir,
            diag_enabled_override=diag_enabled_override,
            diag_stride_override=diag_stride_override,
        )
    except Exception as exc:
        try:
            conn.send(_ErrorMsg(error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
        except Exception:
            pass
    finally:
        conn.close()


def _run_simulation(
    conn: mp.connection.Connection,
    mesh_dict: dict[str, np.ndarray],
    config_dict: dict[str, Any],
    n_layers: int,
    output_dir: str,
    diag_enabled_override: bool | None = None,
    diag_stride_override: int | None = None,
) -> None:
    """在子进程中执行完整的多层仿真。"""
    # ── 子进程崩溃诊断 ──
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _crash_log = output_path / "cpp_subprocess_crash.log"
    import faulthandler as _fh
    _fh.enable(file=open(str(_crash_log), "a", encoding="utf-8"))
    runtime_info = _configure_cpp_runtime_for_subprocess()

    from hydrogel_vbd.core.config import SimulationConfig
    from hydrogel_vbd.core.state import MeshState
    from hydrogel_vbd.geometry.layer_activator import LayerActivator
    from hydrogel_vbd.control.field_controller import PIDFieldController
    from hydrogel_vbd.physics.czm import CZMState, update_czm_states
    from hydrogel_vbd.physics.local_terms import build_local_physics_terms
    from hydrogel_vbd.solver.cpp_adapter import (
        cpp_module_info,
        solve_lift_and_relax as cpp_solve_lift_and_relax,
    )
    from hydrogel_vbd.solver.diagnostics import (
        SolverRunawayGuard,
        SolverStepDiagnostics,
        diagnostics_enabled,
        prepare_solver_diagnostics_csv,
        write_solver_diagnostics_csv,
    )
    from hydrogel_vbd.solver.vbd_solver import VBDSolveResult, _normal_pull_from_terms

    # ── 重建对象 ──
    config = SimulationConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # 用必需字段正常构造 MeshState，让 __post_init__ 自动填充默认值，
    # 然后覆盖为实际数据。
    _required = {k: mesh_dict.pop(k) for k in (
        "vertices", "tets", "layer_id_per_vertex", "layer_id_per_tet",
    )}
    mesh = MeshState(**_required)
    for key, value in mesh_dict.items():
        try:
            setattr(mesh, key, value)
        except AttributeError:
            pass  # 只读 property（如 masses）忽略，底层数据已在 mesh_dict 中

    activator = LayerActivator()
    pid = PIDFieldController(config)

    results: list[dict[str, Any]] = []
    step_counter = 0
    render_interval = 50

    trace_path = Path(output_dir) / "worker_trace.log"
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    def _trace(msg: str) -> None:
        try:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(f"{time.perf_counter():.3f} {msg}\n")
                f.flush()
        except Exception:
            pass

    _trace("subprocess_simulation_start")
    _trace(f"subprocess_cpp_available=True n_layers={n_layers}")
    _trace(
        f"config enable_czm={getattr(config, 'enable_czm', True)} "
        f"rho_cheb={getattr(config, 'rho_cheb', None)}"
    )
    _trace(f"subprocess_runtime threads={runtime_info['threads']}")
    _trace("cpp_line_search=True")
    conn.send(_LogMsg(text=f"  [C++] module {cpp_module_info()}"))
    conn.send(_LogMsg(text="  [debug] cpp_line_search=True"))
    conn.send(
        _LogMsg(
            text=(
                f"  [debug] enable_czm={getattr(config, 'enable_czm', True)}, "
                f"rho_cheb={getattr(config, 'rho_cheb', None)}"
            )
        )
    )

    diag_enabled = (
        diagnostics_enabled()
        if diag_enabled_override is None
        else bool(diag_enabled_override)
    )
    diag_path = output_path / "reports" / "solver_diagnostics.csv"
    diag_stride = (
        max(1, int(os.environ.get("HYDROGEL_VBD_SOLVER_DIAG_STRIDE", "250")))
        if diag_stride_override is None
        else max(1, int(diag_stride_override))
    )
    conn.send(_LogMsg(text=f"  [diag] solver CSV enabled={diag_enabled} path={diag_path}"))
    if diag_enabled:
        prepare_solver_diagnostics_csv(diag_path)
        _trace(f"diagnostic_csv_prepared path={diag_path}")
    diag_guard = SolverRunawayGuard(
        limit=50, max_iters=int(config.max_iters), dx_clip=DX_CLIP_DIAGNOSTIC
    )
    diag_stopped = False

    def _should_record_diag(step: int, expected_steps: int, result: Any) -> bool:
        return (
            step == 0
            or step == 1
            or step == expected_steps
            or step % diag_stride == 0
            or int(getattr(result, "iterations", 0)) >= int(config.max_iters)
            or float(getattr(result, "max_dx", 0.0))
            >= DX_CLIP_DIAGNOSTIC * (1.0 - 1e-9)
        )

    def _record_diag(
        layer_id: int,
        step: int,
        lift_max: float,
        lift_step: float,
        expected_steps: int,
        result: Any,
        call_ms: float,
        x_before: np.ndarray | None = None,
        czm_pull: np.ndarray | None = None,
    ) -> bool:
        nonlocal diag_stopped
        if not diag_enabled:
            return False
        diag = SolverStepDiagnostics.from_mesh(
            mesh,
            layer_id=layer_id,
            step=step,
            lift_max=lift_max,
            lift_step=lift_step,
            expected_steps=expected_steps,
            result=result,
            call_ms=call_ms,
            dx_clip=DX_CLIP_DIAGNOSTIC,
            z_fep=_layer_contact_z(config, layer_id),
            x_before=x_before,
            czm_pull=czm_pull,
            czm_area=config.node_area,
            czm_t_max=config.T_max,
            czm_delta_f=config.delta_f,
        )
        write_solver_diagnostics_csv(diag_path, [diag])
        if diag_guard.observe(diag):
            diag_stopped = True
            _trace(
                f"diagnostic_runaway_guard layer={layer_id} step={step} "
                f"iterations={diag.iterations} max_dx={diag.max_dx:.6e}"
            )
            conn.send(_LogMsg(text=(
                "  [diag] 求解器诊断已停止仿真: "
                f"layer={layer_id}, step={step}, "
                f"连续 {diag_guard.consecutive_bad_steps} 步 max_iter 且 clipped"
            )))
            return True
        return False

    for layer_id in range(n_layers):
        layer_start = time.perf_counter()
        layer_total_iterations = 0
        layer_max_iter_hits = 0
        layer_clipped_steps = 0
        layer_call_elapsed_s = 0.0
        _trace(f"layer_{layer_id}_start")
        conn.send(_LogMsg(text=f"  [C++] 第 {layer_id + 1}/{n_layers} 层 ← VBD 求解"))

        layer_z_fep = _layer_contact_z(config, layer_id)
        layer_config = replace(config, z_fep=layer_z_fep)
        activator.activate_with_inheritance(mesh, layer_id, z_fep=layer_z_fep)
        previous_bottom = (
            mesh.bottom_nodes(layer_id - 1)
            if layer_id > 0
            else np.zeros(0, dtype=int)
        )
        current_bottom = mesh.bottom_nodes(layer_id)
        target_gap = LayerActivator._infer_layer_thickness(mesh, layer_id)
        top_after_activation = _current_lifting_top(mesh)
        active_nodes = np.flatnonzero(mesh.active_vertex_mask)
        active_tets = int(np.count_nonzero(mesh.active_tet_mask))
        tet_quality_stats = _format_tet_quality_stats("tet_quality", mesh)
        _trace(
            f"layer_{layer_id}_activation_state "
            f"target_gap={target_gap:.6e} "
            f"active_nodes={len(active_nodes)} active_tets={active_tets} "
            f"{tet_quality_stats} "
            f"{_format_z_stats('previous_bottom_z', mesh.vertices[previous_bottom, 2])} "
            f"{_format_z_stats('current_bottom_z', mesh.vertices[current_bottom, 2])} "
            f"{_format_z_stats('top_fixed_z', mesh.vertices[top_after_activation, 2])} "
            f"{_format_z_stats('active_z', mesh.vertices[active_nodes, 2])}"
        )
        conn.send(_FrameMsg(
            vertices=mesh.vertices.copy(),
            tets=mesh.tets.copy(),
            active_mask=mesh.active_vertex_mask.copy(),
            active_tet_mask=mesh.active_tet_mask.copy(),
            title=f"第 {layer_id + 1} 层 — 激活后/上提前",
        ))

        pid_state = pid.update(0.0)
        e_z = pid_state.E_z

        lift_max = config.lift_multiplier * config.layer_thickness
        lift_distance = 0.0
        v_lift = config.v_lift
        dt = config.dt
        lift_step = v_lift * dt
        expected_lift_steps = (
            _expected_lift_steps(lift_max, lift_step)
            if lift_step > 0.0
            else 0
        )
        solver_epsilon = _lift_convergence_epsilon(config, lift_step, target_gap)
        layer_config = replace(layer_config, epsilon=solver_epsilon)
        _trace(
            f"layer_{layer_id}_solver_tolerance "
            f"base_epsilon={float(config.epsilon):.6e} "
            f"solver_epsilon={solver_epsilon:.6e} "
            f"target_gap={target_gap:.6e} lift_step={lift_step:.6e}"
        )
        top_ids = _current_lifting_top(mesh)

        layer_steps = 0
        layer_return_steps = 0
        platform_return_distance = 0.0
        _record_diag(
            layer_id,
            0,
            lift_max,
            lift_step,
            expected_lift_steps,
            SimpleNamespace(iterations=0, stable_steps=0, max_dx=0.0),
            0.0,
        )

        # ── 求解分支：有提升 vs 无提升 ──
        _has_lift = expected_lift_steps > 0 and len(top_ids) > 0
        if not _has_lift:
            if expected_lift_steps > 0 and len(top_ids) == 0:
                msg = (
                    f"layer_{layer_id}_no_lift_top_nodes=0 "
                    f"expected_steps={expected_lift_steps}"
                )
                _trace(msg)
                conn.send(_LogMsg(
                    text=(
                        f"  [warn] 第 {layer_id + 1} 层没有可提升顶面节点，"
                        "已跳过平台提升；请检查网格层面分类。"
                    )
                ))
            from hydrogel_vbd.solver.cpp_adapter import (
                solve_until_stable as cpp_solve_until_stable,
            )
            _trace(f"layer_{layer_id}_solve_start no_lift")
            x_before_solve = mesh.vertices.copy()
            call_start = time.perf_counter()
            result = cpp_solve_until_stable(mesh, layer_config, e_z, layer_id)
            call_elapsed = time.perf_counter() - call_start
            layer_call_elapsed_s += call_elapsed
            layer_total_iterations += int(result.iterations)
            if result.iterations >= config.max_iters:
                layer_max_iter_hits += 1
            if result.max_dx >= DX_CLIP_DIAGNOSTIC * (1.0 - 1e-9):
                layer_clipped_steps += 1
            step_counter += result.iterations
            _record_diag(
                layer_id,
                0,
                lift_max,
                lift_step,
                expected_lift_steps,
                result,
                call_elapsed * 1000.0,
                x_before_solve,
            )
            # CZM 更新（提升后）
            bottom = mesh.bottom_nodes(layer_id)
            if bool(getattr(layer_config, "enable_czm", True)) and len(bottom) > 0:
                terms_after = build_local_physics_terms(
                    mesh, layer_config, e_z=e_z, x_prev=x_before_solve,
                    layer_id=layer_id,
                )
                update_czm_states(
                    mesh, bottom,
                    internal_pull_z=_normal_pull_from_terms(terms_after.force, bottom),
                    area=layer_config.node_area, t_max=layer_config.T_max,
                    k_czm=layer_config.K_czm, delta_f=layer_config.delta_f,
                    z_fep=layer_config.z_fep, dt=dt,
                )
        else:
            # 校验 top_ids 合法性
            nV = mesh.vertices.shape[0]
            if np.any(top_ids < 0) or np.any(top_ids >= nV):
                raise ValueError(f"top_ids 包含越界索引 (nV={nV}, min={top_ids.min()}, max={top_ids.max()})")

            _validate_lift_plan(
                layer_id, lift_max, lift_step, expected_lift_steps
            )
            _trace(
                f"layer_{layer_id}_lift_start top_ids={len(top_ids)} "
                f"v_lift={v_lift} dt={dt} lift_max={lift_max} "
                f"lift_step={lift_step} expected_steps={expected_lift_steps} "
                f"solver_epsilon={solver_epsilon:.6e}"
            )
            trace_every_step = os.environ.get("HYDROGEL_VBD_TRACE_STEPS") == "1"
            trace_stride = int(os.environ.get("HYDROGEL_VBD_TRACE_STRIDE", "250"))
            while layer_steps < expected_lift_steps:
                layer_steps += 1
                should_trace_step = (
                    trace_every_step
                    or layer_steps == 1
                    or layer_steps == expected_lift_steps
                    or layer_steps % max(trace_stride, 1) == 0
                )
                if should_trace_step:
                    _trace(
                        f"layer_{layer_id}_step_{layer_steps}_pre_call "
                        f"lift={lift_distance:.6e}"
                    )

                bottom = mesh.bottom_nodes(layer_id)
                bottom_state = mesh.czm_state[bottom].copy()
                bottom_damage = mesh.damage[bottom].copy()
                bottom_time_free = mesh.time_free[bottom].copy()
                x_before_solve = mesh.vertices.copy()
                call_start = time.perf_counter()
                result = cpp_solve_lift_and_relax(
                    mesh, layer_config, e_z, layer_id, top_ids,
                )
                pull_after = None
                if bool(getattr(layer_config, "enable_czm", True)) and len(bottom) > 0:
                    mesh.czm_state[bottom] = bottom_state
                    mesh.damage[bottom] = bottom_damage
                    mesh.time_free[bottom] = bottom_time_free
                    terms_after = build_local_physics_terms(
                        mesh, layer_config, e_z=e_z, x_prev=x_before_solve,
                        layer_id=layer_id,
                    )
                    pull_after = _normal_pull_from_terms(terms_after.force, bottom)
                    update_czm_states(
                        mesh,
                        bottom,
                        internal_pull_z=pull_after,
                        area=layer_config.node_area,
                        t_max=layer_config.T_max,
                        k_czm=layer_config.K_czm,
                        delta_f=layer_config.delta_f,
                        z_fep=layer_config.z_fep,
                        dt=dt,
                    )
                    result.all_free = bool(
                        np.all(mesh.czm_state[bottom] == int(CZMState.FREE))
                    )
                call_elapsed = time.perf_counter() - call_start
                layer_call_elapsed_s += call_elapsed
                layer_total_iterations += int(result.iterations)
                if result.iterations >= config.max_iters:
                    layer_max_iter_hits += 1
                if result.max_dx >= DX_CLIP_DIAGNOSTIC * (1.0 - 1e-9):
                    layer_clipped_steps += 1
                if should_trace_step:
                    _trace(
                        f"layer_{layer_id}_step_{layer_steps}_post_call "
                        f"max_dx={result.max_dx:.4e} "
                        f"iters={result.iterations} "
                        f"call_ms={call_elapsed * 1000.0:.3f}"
                    )
                if _should_record_diag(layer_steps, expected_lift_steps, result):
                    if _record_diag(
                        layer_id,
                        layer_steps,
                        lift_max,
                        lift_step,
                        expected_lift_steps,
                        result,
                        call_elapsed * 1000.0,
                        x_before_solve,
                        pull_after,
                    ):
                        break
                step_counter += result.iterations
                lift_distance = min(layer_steps * lift_step, lift_max)

                if diag_stopped:
                    break

                # 全部脱膜则退出提升循环
                if result.all_free and bool(getattr(layer_config, "enable_czm", True)):
                    _raise_if_detached_before_convergence(
                        layer_id, layer_steps, result, layer_config
                    )
                    break

                # 每 20 步检查主进程是否发来停止信号
                if layer_steps % 20 == 0 and conn.poll():
                    try:
                        ctl = conn.recv()
                        if isinstance(ctl, str) and ctl == "stop":
                            _trace(f"layer_{layer_id}_stopped_by_user")
                            break
                    except Exception:
                        pass

                # 帧推送
                if step_counter % render_interval == 0:
                    v = mesh.vertices
                    t = mesh.tets
                    conn.send(_FrameMsg(
                        vertices=v.copy() if v is not None else np.zeros((0, 3)),
                        tets=t.copy() if t is not None else np.zeros((0, 4), dtype=np.int32),
                        active_mask=(
                            mesh.active_vertex_mask.copy()
                            if mesh.active_vertex_mask is not None
                            else np.zeros(v.shape[0] if v is not None else 0, dtype=bool)
                        ),
                        active_tet_mask=(
                            mesh.active_tet_mask.copy()
                            if mesh.active_tet_mask is not None
                            else np.zeros(t.shape[0] if t is not None else 0, dtype=bool)
                        ),
                        title=f"第 {layer_id + 1} 层 — 步 {layer_steps}",
                    ))

                lift_pct = min(int(lift_distance / lift_max * 100), 100)
                conn.send(_ProgressMsg(
                    layer=layer_id + 1, percentage=lift_pct, step=step_counter,
                ))

        if (
            _has_lift
            and not diag_stopped
            and layer_id + 1 < n_layers
            and len(top_ids) > 0
            and abs(lift_step) > 0.0
        ):
            next_gap = LayerActivator._infer_layer_thickness(mesh, layer_id + 1)
            actual_lift = layer_steps * abs(lift_step)
            platform_return_distance = _platform_return_distance(
                actual_lift, next_gap
            )
            return_step_distances = _positive_step_distances(
                platform_return_distance, abs(lift_step)
            )
            if return_step_distances:
                _trace(
                    f"layer_{layer_id}_platform_return_start "
                    f"distance={platform_return_distance:.6e} "
                    f"next_gap={next_gap:.6e} "
                    f"steps={len(return_step_distances)}"
                )
            for return_step_index, return_step_distance in enumerate(
                return_step_distances, start=1
            ):
                bottom = mesh.bottom_nodes(layer_id)
                bottom_state = mesh.czm_state[bottom].copy()
                bottom_damage = mesh.damage[bottom].copy()
                bottom_time_free = mesh.time_free[bottom].copy()
                down_config = replace(
                    layer_config,
                    v_lift=-return_step_distance / max(abs(dt), 1e-12),
                )
                x_before_solve = mesh.vertices.copy()
                call_start = time.perf_counter()
                result = cpp_solve_lift_and_relax(
                    mesh, down_config, e_z, layer_id, top_ids,
                )
                call_elapsed = time.perf_counter() - call_start
                if bool(getattr(layer_config, "enable_czm", True)) and len(bottom) > 0:
                    mesh.czm_state[bottom] = bottom_state
                    mesh.damage[bottom] = bottom_damage
                    mesh.time_free[bottom] = bottom_time_free
                layer_return_steps += 1
                layer_call_elapsed_s += call_elapsed
                layer_total_iterations += int(result.iterations)
                if result.iterations >= config.max_iters:
                    layer_max_iter_hits += 1
                if result.max_dx >= DX_CLIP_DIAGNOSTIC * (1.0 - 1e-9):
                    layer_clipped_steps += 1
                if (
                    return_step_index == 1
                    or return_step_index == len(return_step_distances)
                ):
                    _trace(
                        f"layer_{layer_id}_platform_return_step_"
                        f"{return_step_index}_post_call "
                        f"max_dx={result.max_dx:.4e} "
                        f"iters={result.iterations} "
                        f"call_ms={call_elapsed * 1000.0:.3f}"
                    )

        elapsed_s = time.perf_counter() - layer_start
        layer_call_count = layer_steps + layer_return_steps
        avg_call_ms = (
            layer_call_elapsed_s / max(layer_call_count, 1) * 1000.0
            if layer_steps > 0
            else layer_call_elapsed_s * 1000.0
        )
        max_iter_hit_rate = (
            layer_max_iter_hits / layer_call_count * 100.0
            if layer_call_count > 0
            else 0.0
        )
        _trace(
            f"layer_{layer_id}_done steps={layer_steps} "
            f"return_steps={layer_return_steps} "
            f"elapsed_s={elapsed_s:.3f} total_iters={layer_total_iterations} "
            f"max_iter_hits={layer_max_iter_hits} "
            f"max_iter_hit_rate={max_iter_hit_rate:.2f}% "
            f"clipped_steps={layer_clipped_steps} "
            f"avg_call_ms={avg_call_ms:.3f} "
            f"active_nodes={len(active_nodes)} active_tets={active_tets} "
            f"solver_epsilon={solver_epsilon:.6e}"
        )
        if max_iter_hit_rate >= 20.0:
            conn.send(_LogMsg(text=(
                f"  [perf] layer {layer_id + 1}: "
                f"max_iter_hit_rate={max_iter_hit_rate:.1f}%, "
                f"avg_call={avg_call_ms:.1f} ms, "
                f"active_nodes={len(active_nodes)}, active_tets={active_tets}"
            )))
        results.append({
            "layer_id": layer_id,
            "total_steps": layer_steps,
            "final_max_dx": float(result.max_dx),
            "total_iterations": layer_total_iterations,
            "max_iter_hits": layer_max_iter_hits,
            "max_iter_hit_rate": max_iter_hit_rate,
            "clipped_steps": layer_clipped_steps,
            "elapsed_s": elapsed_s,
            "avg_call_ms": avg_call_ms,
            "lift_max": lift_max,
            "lift_step": lift_step,
            "platform_return_distance": platform_return_distance,
            "platform_return_steps": layer_return_steps,
            "solver_epsilon": solver_epsilon,
            "expected_steps": expected_lift_steps,
            "active_nodes": len(active_nodes),
            "active_tets": active_tets,
            "top_nodes": len(top_ids),
            "E_z": float(e_z),
            "success": not diag_stopped,
        })
        if diag_stopped:
            break

    _trace("subprocess_simulation_done")
    conn.send(_DoneMsg(results=results))


# ── 主进程侧管理器 ──

class CppSubprocessSolver:
    """在子进程中运行 C++ 求解器。

    用法
    ----
    solver = CppSubprocessSolver(mesh, config, n_layers, output_dir)
    solver.start()  # 启动子进程
    for msg in solver.iter_messages():  # 迭代接收消息
        if isinstance(msg, _FrameMsg):
            frame_ready.emit(...)
        ...
    results = solver.results  # 最终结果
    """

    def __init__(
        self,
        mesh_dict: dict[str, np.ndarray],
        config_dict: dict[str, Any],
        n_layers: int,
        output_dir: str,
        diagnostics_enabled: bool | None = None,
        diagnostics_stride: int | None = None,
    ):
        self._mesh_dict = mesh_dict
        self._config_dict = config_dict
        self._n_layers = n_layers
        self._output_dir = output_dir
        self._diagnostics_enabled = diagnostics_enabled
        self._diagnostics_stride = diagnostics_stride
        self._conn: mp.connection.Connection | None = None
        self._proc: mp.Process | None = None
        self._results: list[dict[str, Any]] | None = None
        self._crashed = False
        self._error: str | None = None

    @property
    def crashed(self) -> bool:
        return self._crashed

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def results(self) -> list[dict[str, Any]] | None:
        return self._results

    def start(self) -> None:
        """启动子进程。"""
        parent_conn, child_conn = mp.Pipe(duplex=True)
        self._conn = parent_conn

        self._proc = mp.Process(
            target=_worker_run,
            args=(
                child_conn,
                self._mesh_dict,
                self._config_dict,
                self._n_layers,
                self._output_dir,
                self._diagnostics_enabled,
                self._diagnostics_stride,
            ),
            name="cpp-solver-subprocess",
        )
        self._proc.start()
        child_conn.close()  # 子进程持有 child_conn，主进程关闭自己这边

    def iter_messages(self, timeout: float = 0.1):
        """迭代器：从子进程接收消息直到完成或崩溃。

        Yields
        ------
        _ProgressMsg | _LogMsg | _FrameMsg | _DoneMsg | _ErrorMsg
        """
        if self._conn is None:
            raise RuntimeError("CppSubprocessSolver not started")

        while True:
            # 检查子进程是否存活
            if self._proc is not None and not self._proc.is_alive():
                exitcode = self._proc.exitcode
                self._crashed = True
                self._error = f"C++ 子进程异常退出 (exitcode={exitcode})"
                yield _ErrorMsg(error=self._error)
                return

            # 轮询消息
            if self._conn.poll(timeout):
                try:
                    msg = self._conn.recv()
                except (EOFError, ConnectionResetError, pickle.UnpicklingError) as exc:
                    self._crashed = True
                    self._error = f"子进程通信失败: {exc}"
                    yield _ErrorMsg(error=self._error)
                    return

                if isinstance(msg, _DoneMsg):
                    self._results = msg.results
                    yield msg
                    return
                yield msg
            # 超时后继续循环，检查进程存活性

    def terminate(self) -> None:
        """强制终止子进程。"""
        if self._conn is not None:
            try:
                self._conn.send("stop")
            except Exception:
                pass
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=2)
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
