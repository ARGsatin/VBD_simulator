# -*- coding: utf-8 -*-
"""仿真工作线程 —— 将密集的 VBD 物理循环移入 QThread。

通过 pyqtSignal 向主线程推送顶点数据，主线程仅负责渲染，
从根本上消除"未响应"问题。

架构
----
- **SimulationWorker**：在 QThread 中执行逐层 VBD 求解
- **信号**：
  - ``frame_ready``：每 ``render_interval`` 步推送一帧 (vertices, tets, title)
  - ``progress_update``：推送进度 (当前层, 总层数, 步数, 迭代次数)
  - ``log_message``：向 GUI 日志面板推送消息
  - ``finished``：仿真完成 → 推送结果列表
  - ``error``：异常信号

控制反转（Inversion of Control）
---------------------------------
C++ 求解器函数 ``solve_lift_and_relax`` 已降级为"单步求解器"
（仅执行一次微提升 + 一次静平衡松弛），不包含任何 while 循环。
时间流逝由 Python 侧接管：本 Worker 在层循环内部显式 while 循环中
反复调用该单步函数，每步之间 GIL 自动释放，Worker 线程可及时向主线程
发射信号，彻底解决界面"未响应"问题。
"""

from __future__ import annotations

import atexit
import copy
import math
import os
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PySide6 import QtCore

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.core.state import FieldCommand, LayerResult, MeshState
from hydrogel_vbd.physics.czm import CZMState
from hydrogel_vbd.solver.cpp_adapter import is_cpp_available


class _SimulationCancelled(Exception):
    """Internal control-flow exception for user-requested simulation stops."""


@dataclass
class _FieldDebugBranchRun:
    """Field-debug branch state split into commit and guard snapshots."""

    commit_mesh: MeshState
    guard_mesh: MeshState
    commit_result: Any
    guard_result: Any
    commit_steps: int
    executed_steps: int
    total_iterations: int
    max_iter_hits: int
    clipped_steps: int
    lift_max: float
    info: dict[str, float | str]


class SimulationWorker(QtCore.QObject):
    """VBD 仿真工作线程（QThread Worker）。

    持有网格数据的完全深拷贝，确保工作线程与主线程的内存绝对隔离。
    所有被 C++ Eigen 库通过指针映射的 Numpy 数组均调用 ``.copy()``。

    Parameters
    ----------
    mesh : MeshState
        初始网格状态（将被深拷贝到工作线程）。
    config : SimulationConfig
        仿真配置。
    n_layers : int
        打印总层数。
    output_dir : str or Path
        输出目录路径。
    """

    # ── 信号定义 ──────────────────────────────────────────────
    frame_ready = QtCore.Signal(dict)
    progress_update = QtCore.Signal(int, int, int, int)  # layer, n_layers, step, iteration
    layer_finished = QtCore.Signal(object)  # LayerResult — 每层结束时推送
    sub_progress = QtCore.Signal(int, int, int)  # layer_idx, percentage_0_to_100, step_count
    log_message = QtCore.Signal(str)
    finished = QtCore.Signal(list)  # list[LayerResult]
    cancelled = QtCore.Signal(list)  # partial list[LayerResult]
    error = QtCore.Signal(str)

    MAX_EXPECTED_LIFT_STEPS = 100_000
    DX_CLIP_DIAGNOSTIC = 0.002

    def __init__(
        self,
        mesh: MeshState,
        config: SimulationConfig,
        n_layers: int,
        output_dir: str | Path,
        use_cpp: bool = True,
        solver_diagnostics_enabled: bool | None = None,
        solver_diagnostics_stride: int | None = None,
        field_debug_enabled: bool = False,
        field_debug_use_cpp: bool | None = None,
    ) -> None:
        super().__init__()
        self._mesh_original = mesh
        self._mesh = self._deep_copy_mesh(mesh)
        self._config = config
        self._n_layers = int(n_layers)
        self._output_dir = Path(output_dir)
        self._stop_flag = False
        self._use_cpp = use_cpp and is_cpp_available()
        self._cpp_solver: Any | None = None
        self._solver_diagnostics_enabled = solver_diagnostics_enabled
        self._solver_diagnostics_stride = solver_diagnostics_stride
        self._field_debug_enabled = bool(field_debug_enabled)
        if field_debug_use_cpp is None:
            field_debug_use_cpp = bool(field_debug_enabled and use_cpp)
        self._field_debug_use_cpp = (
            bool(field_debug_enabled)
            and bool(field_debug_use_cpp)
            and is_cpp_available()
        )
        self._field_debug_cpp_fallback_count = 0

    # ───────────────────────────────────────────────────────────
    # 网格深拷贝（避免数据竞争 → 杜绝 Segfault）
    # ───────────────────────────────────────────────────────────

    @staticmethod
    def _deep_copy_mesh(mesh: MeshState) -> MeshState:
        """创建 MeshState 的完全深拷贝。

        所有被 C++ Eigen 库通过 ``Eigen::Map`` 零拷贝映射的
        Numpy 数组均需执行 ``.copy()``，确保工作线程与主线程的
        内存绝对隔离。主线程对原始网格的任何清理/修改操作都不会
        引发 C++ 底层的段错误（Segfault）。

        Parameters
        ----------
        mesh : MeshState
            主线程中的原始网格状态。

        Returns
        -------
        MeshState
            完全独立的深拷贝副本。
        """
        import copy

        copy_mesh = copy.copy(mesh)

        # ── 几何与坐标（已被 C++ Eigen::Map 映射的数组）──
        copy_mesh.vertices = mesh.vertices.copy()
        copy_mesh.velocities = mesh.velocities.copy()
        copy_mesh.prev_vertices = (
            mesh.prev_vertices.copy()
            if mesh.prev_vertices is not None
            else mesh.vertices.copy()
        )
        copy_mesh.ideal_vertices = (
            mesh.ideal_vertices.copy()
            if mesh.ideal_vertices is not None
            else mesh.vertices.copy()
        )

        # ── 节点质量（通过 Eigen::Map<Eigen::VectorXd> 映射）──
        copy_mesh.node_mass = (
            mesh.node_mass.copy()
            if mesh.node_mass is not None
            else mesh.masses.copy()
        )

        # ── 布尔掩码（Eigen::Map<Eigen::Matrix<bool, ...>> 映射）──
        copy_mesh.is_top_fixed = (
            mesh.is_top_fixed.copy()
            if mesh.is_top_fixed is not None
            else np.zeros(mesh.vertices.shape[0], dtype=bool)
        )
        copy_mesh.is_bottom_surface = (
            mesh.is_bottom_surface.copy()
            if mesh.is_bottom_surface is not None
            else np.zeros(mesh.vertices.shape[0], dtype=bool)
        )
        copy_mesh.active_vertex_mask = (
            mesh.active_vertex_mask.copy()
            if mesh.active_vertex_mask is not None
            else np.zeros(mesh.vertices.shape[0], dtype=bool)
        )
        copy_mesh.active_tet_mask = (
            mesh.active_tet_mask.copy()
            if mesh.active_tet_mask is not None
            and mesh.tets is not None
            else np.zeros(
                mesh.tets.shape[0] if mesh.tets is not None else 0,
                dtype=bool,
            )
        )
        copy_mesh.boundary_flags = (
            mesh.boundary_flags.copy()
            if mesh.boundary_flags is not None
            else np.zeros(mesh.vertices.shape[0], dtype=bool)
        )

        # ── CZM 状态（Eigen::Map<Eigen::VectorXi> 映射，可读写）──
        copy_mesh.czm_state = (
            mesh.czm_state.copy()
            if mesh.czm_state is not None
            else np.zeros(mesh.vertices.shape[0], dtype=int)
        )
        copy_mesh.damage = (
            mesh.damage.copy()
            if mesh.damage is not None
            else np.zeros(mesh.vertices.shape[0], dtype=float)
        )
        copy_mesh.time_free = (
            mesh.time_free.copy()
            if mesh.time_free is not None
            else np.zeros(mesh.vertices.shape[0], dtype=float)
        )

        # ── 预计算张量（Eigen::Map 映射，被遗漏的关键数组）──
        # dm_inv: (T, 3, 3) → C++ 侧映射为 (T, 9) RowMajor
        copy_mesh.dm_inv = (
            mesh.dm_inv.copy()
            if mesh.dm_inv is not None
            else None
        )
        # dm: (T, 3, 3)  参考形矩阵
        copy_mesh.dm = (
            mesh.dm.copy()
            if mesh.dm is not None
            else None
        )
        # tet_volumes: (T,) 四面体体积
        copy_mesh.tet_volumes = (
            mesh.tet_volumes.copy()
            if mesh.tet_volumes is not None
            else None
        )
        # colors: (N,) 图着色分组
        copy_mesh.colors = (
            mesh.colors.copy()
            if mesh.colors is not None
            else None
        )

        # ── 层号数组（只读但需要隔离，防止主线程修改）──
        copy_mesh.layer_id_per_vertex = mesh.layer_id_per_vertex.copy()
        copy_mesh.layer_id_per_tet = (
            mesh.layer_id_per_tet.copy()
            if mesh.layer_id_per_tet is not None
            else None
        )
        copy_mesh.first_active_layer = (
            mesh.first_active_layer.copy()
            if mesh.first_active_layer is not None
            else mesh.layer_id_per_vertex.copy()
        )
        copy_mesh.is_top_surface_of_layer = (
            mesh.is_top_surface_of_layer.copy()
            if mesh.is_top_surface_of_layer is not None
            else np.full(mesh.vertices.shape[0], -1, dtype=int)
        )

        # ── tets 只读（通常不变，但为安全也拷贝）──
        # （如果 tets 已经通过 copy.copy 共享引用，此处显式拷贝）
        if mesh.tets is not None:
            copy_mesh.tets = mesh.tets.copy()

        # ── Python 对象引用（vertex2tets, neighbors）──
        # 这些是 list[list[int]] / list[set[int]]，元素为不可变的 int，
        # 深拷贝外层容器即可确保线程隔离
        if mesh.vertex2tets:
            copy_mesh.vertex2tets = [list(v) for v in mesh.vertex2tets]
        if mesh.neighbors:
            copy_mesh.neighbors = [set(n) for n in mesh.neighbors]

        # ── color_ranges 是 list[tuple[int, int]]，不可变元素 ──
        copy_mesh.color_ranges = list(mesh.color_ranges)

        return copy_mesh

    # ───────────────────────────────────────────────────────────
    # 停止控制
    # ───────────────────────────────────────────────────────────

    @QtCore.Slot()
    def request_stop(self) -> None:
        """Request simulation interruption and terminate any active C++ child."""
        self._stop_flag = True
        cpp_solver = self._cpp_solver
        if cpp_solver is not None:
            try:
                cpp_solver.terminate()
            except Exception:
                pass

    @staticmethod
    def _current_lifting_top(mesh: MeshState) -> np.ndarray:
        """返回当前激活层的夹持顶面节点。"""
        if mesh.is_top_fixed is None or mesh.active_vertex_mask is None:
            return np.zeros(0, dtype=np.int32)
        lifting_top = np.flatnonzero(mesh.is_top_fixed & mesh.active_vertex_mask)
        return np.asarray(lifting_top, dtype=np.int32)

    @staticmethod
    def _expected_lift_steps(lift_max: float, lift_step: float) -> int:
        """计算完成本层提升所需的外层时间步数。"""
        if lift_max <= 0.0:
            return 0
        if lift_step <= 0.0:
            raise ValueError("lift_step 必须为正数；请检查 v_lift 和 dt")
        ratio = lift_max / lift_step
        return int(math.ceil(ratio - max(1e-12, abs(ratio) * 1e-12)))

    @staticmethod
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

    @staticmethod
    def _layer_contact_z(config: SimulationConfig, layer_id: int) -> float:
        return float(config.z_fep)

    @staticmethod
    def _update_czm_from_current_terms(
        mesh: MeshState,
        config: SimulationConfig,
        layer_id: int,
        e_z: float,
        x_prev: np.ndarray,
    ) -> np.ndarray | None:
        """Update current layer CZM using the assembled post-solve normal pull."""
        if not bool(getattr(config, "enable_czm", True)):
            return None
        bottom = mesh.bottom_nodes(layer_id)
        if len(bottom) == 0:
            return None

        from hydrogel_vbd.physics.czm import update_czm_states
        from hydrogel_vbd.physics.local_terms import build_local_physics_terms
        from hydrogel_vbd.solver.vbd_solver import _normal_pull_from_terms

        terms = build_local_physics_terms(mesh, config, e_z=e_z, x_prev=x_prev)
        pull = _normal_pull_from_terms(terms.force, bottom)
        update_czm_states(
            mesh,
            bottom,
            internal_pull_z=pull,
            area=config.node_area,
            t_max=config.T_max,
            k_czm=config.K_czm,
            delta_f=config.delta_f,
            z_fep=config.z_fep,
            dt=config.dt,
        )
        return pull

    @staticmethod
    def _sync_cpp_lift_czm_state(
        mesh: MeshState,
        config: SimulationConfig,
        layer_id: int,
        e_z: float,
        x_prev: np.ndarray,
        bottom: np.ndarray,
        bottom_state: np.ndarray,
        bottom_damage: np.ndarray,
        bottom_time_free: np.ndarray,
        result: Any,
    ) -> None:
        """Mirror subprocess CZM handling for direct C++ field-debug steps."""
        if not bool(getattr(config, "enable_czm", True)) or len(bottom) == 0:
            return

        from hydrogel_vbd.physics.czm import CZMState, update_czm_states
        from hydrogel_vbd.physics.local_terms import build_local_physics_terms
        from hydrogel_vbd.solver.vbd_solver import _normal_pull_from_terms

        mesh.czm_state[bottom] = bottom_state
        mesh.damage[bottom] = bottom_damage
        mesh.time_free[bottom] = bottom_time_free
        terms = build_local_physics_terms(
            mesh, config, e_z=e_z, x_prev=x_prev, layer_id=layer_id
        )
        update_czm_states(
            mesh,
            bottom,
            internal_pull_z=_normal_pull_from_terms(terms.force, bottom),
            area=config.node_area,
            t_max=config.T_max,
            k_czm=config.K_czm,
            delta_f=config.delta_f,
            z_fep=config.z_fep,
            dt=config.dt,
        )
        result.all_free = bool(
            np.all(mesh.czm_state[bottom] == int(CZMState.FREE))
        )

    @staticmethod
    def _shape_debug_metrics(mesh: MeshState, layer_id: int) -> dict[str, float]:
        """Return global RMS and current-bottom Z metrics for a mesh snapshot."""
        target = np.asarray(mesh.ideal_vertices, dtype=float)
        vertices = np.asarray(mesh.vertices, dtype=float)
        if target.shape != vertices.shape or vertices.size == 0:
            rms = 0.0
            max_error = 0.0
        else:
            error = target - vertices
            rms = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
            max_error = float(np.max(np.linalg.norm(error, axis=1)))

        bottom = mesh.bottom_nodes(layer_id)
        if len(bottom):
            z_error = target[bottom, 2] - vertices[bottom, 2]
            bottom_mean = float(np.mean(z_error))
            bottom_max = float(np.max(z_error))
        else:
            bottom_mean = 0.0
            bottom_max = 0.0

        return {
            "rms": rms,
            "max_error": max_error,
            "bottom_z_mean": bottom_mean,
            "bottom_z_max": bottom_max,
            "bottom_count": float(len(bottom)),
        }

    @staticmethod
    def _field_debug_guard(
        no_field_metrics: dict[str, float],
        with_field_metrics: dict[str, float],
        tolerance: float,
    ) -> dict[str, float | str]:
        """Return conservative field-debug acceptance diagnostics."""
        eps = 1.0e-12
        rms_limit = no_field_metrics["rms"] * (1.0 + float(tolerance)) + eps
        max_limit = (
            no_field_metrics["max_error"] * (1.0 + float(tolerance)) + eps
        )
        rms_ok = with_field_metrics["rms"] <= rms_limit
        max_ok = with_field_metrics["max_error"] <= max_limit

        rms_improvement_eps = max(eps, abs(no_field_metrics["rms"]) * 1.0e-6)
        bottom_mean_eps = max(
            eps, abs(no_field_metrics["bottom_z_mean"]) * 1.0e-6
        )
        bottom_max_eps = max(
            eps, abs(no_field_metrics["bottom_z_max"]) * 1.0e-6
        )
        no_mean_sag = max(no_field_metrics["bottom_z_mean"], 0.0)
        with_mean_sag = max(with_field_metrics["bottom_z_mean"], 0.0)
        no_max_sag = max(no_field_metrics["bottom_z_max"], 0.0)
        with_max_sag = max(with_field_metrics["bottom_z_max"], 0.0)
        improved = (
            with_field_metrics["rms"]
            < no_field_metrics["rms"] - rms_improvement_eps
            or with_mean_sag < no_mean_sag - bottom_mean_eps
            or with_max_sag < no_max_sag - bottom_max_eps
        )

        if not rms_ok:
            reason = "rms_error_worse"
        elif not max_ok:
            reason = "max_error_worse"
        elif not improved:
            reason = "no_improvement"
        else:
            reason = "pass"

        return {
            "passed": float(rms_ok and max_ok and improved),
            "reason": reason,
            "rms_limit": float(rms_limit),
            "max_error_limit": float(max_limit),
            "rms_passed": float(rms_ok),
            "max_error_passed": float(max_ok),
            "improvement_passed": float(improved),
        }

    @staticmethod
    def _field_event_window_e_z(
        step: int,
        expected_steps: int,
        detach_e_z: float,
        detach_step: int,
        config: SimulationConfig,
        peak_e_z: float | None = None,
    ) -> tuple[float, bool, int]:
        """Return the event-window E_z for a 1-based lift step."""
        detach_e_z = float(detach_e_z)
        peak_e_z = detach_e_z if peak_e_z is None else float(peak_e_z)
        if (detach_e_z <= 0.0 and peak_e_z <= 0.0) or expected_steps <= 0:
            return 0.0, False, 0

        detach_pre = max(
            0, int(getattr(config, "field_detach_pre_steps", 0))
        )
        detach_post = max(
            0, int(getattr(config, "field_detach_post_steps", 1))
        )
        peak_steps = max(
            0, int(getattr(config, "field_peak_window_steps", 1))
        )
        peak_start = (
            max(1, expected_steps - peak_steps + 1)
            if peak_steps > 0
            else 0
        )

        in_detach_window = (
            detach_step > 0
            and (detach_step - detach_pre) <= step <= (detach_step + detach_post)
        )
        in_peak_window = peak_start > 0 and step >= peak_start
        candidates: list[float] = []
        if in_detach_window:
            candidates.append(max(detach_e_z, 0.0))
        if in_peak_window:
            candidates.append(max(peak_e_z, 0.0))
        value = max(candidates) if candidates else 0.0
        return value, value > 0.0, peak_start

    def _run_field_debug_branch(
        self,
        source_mesh: MeshState,
        config: SimulationConfig,
        layer_id: int,
        e_z: float,
        use_cpp: bool = False,
        event_window_detach_step: int | None = None,
        peak_e_z: float | None = None,
        continue_to_peak: bool = False,
    ) -> _FieldDebugBranchRun:
        """Run a cloned layer branch for field-debug comparison."""
        try:
            return self._run_field_debug_branch_once(
                source_mesh,
                config,
                layer_id,
                e_z,
                use_cpp=use_cpp,
                event_window_detach_step=event_window_detach_step,
                peak_e_z=peak_e_z,
                continue_to_peak=continue_to_peak,
            )
        except Exception as exc:
            if not use_cpp:
                raise
            self._field_debug_cpp_fallback_count += 1
            self._field_debug_use_cpp = False
            self.log_message.emit(
                f"  [field-debug] C++ adapter 分支求解失败，回退 Python: {exc}"
            )
            return self._run_field_debug_branch_once(
                source_mesh,
                config,
                layer_id,
                e_z,
                use_cpp=False,
                event_window_detach_step=event_window_detach_step,
                peak_e_z=peak_e_z,
                continue_to_peak=continue_to_peak,
            )

    def _run_field_debug_branch_once(
        self,
        source_mesh: MeshState,
        config: SimulationConfig,
        layer_id: int,
        e_z: float,
        use_cpp: bool = False,
        event_window_detach_step: int | None = None,
        peak_e_z: float | None = None,
        continue_to_peak: bool = False,
    ) -> _FieldDebugBranchRun:
        """Run one cloned field-debug branch with the selected solver backend."""
        from hydrogel_vbd.geometry.layer_activator import LayerActivator

        if use_cpp:
            from hydrogel_vbd.solver.cpp_adapter import (
                solve_until_stable as cpp_solve_until_stable,
                solve_lift_and_relax as cpp_solve_lift_and_relax,
            )
            solver = None
        else:
            from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver
            solver = PythonReferenceVBDSolver

        mesh = copy.deepcopy(source_mesh)
        layer_z_fep = self._layer_contact_z(config, layer_id)
        layer_config = replace(config, z_fep=layer_z_fep)
        activator = LayerActivator()
        activator.activate_with_inheritance(mesh, layer_id, z_fep=layer_z_fep)
        python_solver = None if use_cpp else solver(layer_config)

        lift_max = config.lift_multiplier * config.layer_thickness
        lift_step = config.v_lift * config.dt
        expected_lift_steps = (
            self._expected_lift_steps(lift_max, lift_step)
            if lift_step > 0.0
            else 0
        )
        top_ids = self._current_lifting_top(mesh)

        total_iterations = 0
        max_iter_hits = 0
        clipped_steps = 0
        layer_steps = 0
        return_steps = 0
        platform_return_distance = 0.0
        detach_step = 0
        commit_steps = 0
        commit_mesh: MeshState | None = None
        commit_result: Any | None = None
        field_applied_steps = 0
        peak_start_step = (
            max(
                1,
                expected_lift_steps
                - max(0, int(getattr(config, "field_peak_window_steps", 1)))
                + 1,
            )
            if expected_lift_steps > 0
            and int(getattr(config, "field_peak_window_steps", 1)) > 0
            else 0
        )
        timing_mode = (
            "event_windows_v2"
            if event_window_detach_step is not None and expected_lift_steps > 0
            else "constant"
        )
        if expected_lift_steps > 0 and len(top_ids) > 0:
            self._validate_lift_plan(
                layer_id, lift_max, lift_step, expected_lift_steps
            )
            result = None
            for _ in range(expected_lift_steps):
                step_index = layer_steps + 1
                if event_window_detach_step is None:
                    step_e_z = float(e_z)
                else:
                    step_e_z, active_window, peak_start_step = (
                        self._field_event_window_e_z(
                            step_index,
                            expected_lift_steps,
                            float(e_z),
                            int(event_window_detach_step or 0),
                            config,
                            peak_e_z=peak_e_z,
                        )
                    )
                    if active_window:
                        field_applied_steps += 1
                if use_cpp:
                    bottom = mesh.bottom_nodes(layer_id)
                    bottom_state = mesh.czm_state[bottom].copy()
                    bottom_damage = mesh.damage[bottom].copy()
                    bottom_time_free = mesh.time_free[bottom].copy()
                    x_before_solve = mesh.vertices.copy()
                    result = cpp_solve_lift_and_relax(
                        mesh, layer_config, step_e_z, layer_id, top_ids
                    )
                    self._sync_cpp_lift_czm_state(
                        mesh,
                        layer_config,
                        layer_id,
                        step_e_z,
                        x_before_solve,
                        bottom,
                        bottom_state,
                        bottom_damage,
                        bottom_time_free,
                        result,
                    )
                else:
                    result = python_solver.solve_with_lift(
                        mesh, layer_id=layer_id, e_z=step_e_z, lifting_top=top_ids
                    )
                layer_steps += 1
                total_iterations += int(getattr(result, "iterations", 0))
                if getattr(result, "iterations", 0) >= config.max_iters:
                    max_iter_hits += 1
                if (
                    getattr(result, "max_dx", 0.0)
                    >= self.DX_CLIP_DIAGNOSTIC * (1.0 - 1e-9)
                ):
                    clipped_steps += 1
                if result.all_free:
                    if detach_step <= 0:
                        detach_step = layer_steps
                    if commit_mesh is None:
                        commit_mesh = copy.deepcopy(mesh)
                        commit_result = result
                        commit_steps = layer_steps
                    if bool(getattr(config, "enable_czm", True)):
                        self._raise_if_detached_before_convergence(
                            layer_id, layer_steps, result, layer_config
                        )
                    if not continue_to_peak:
                        break
            if result is None:
                if use_cpp:
                    result = cpp_solve_until_stable(
                        mesh, layer_config, e_z, layer_id
                    )
                else:
                    result = python_solver.solve_until_stable(
                        mesh, layer_id=layer_id, e_z=e_z
                    )
        else:
            x_before_solve = mesh.vertices.copy()
            if e_z > 0.0:
                field_applied_steps = 1
            if use_cpp:
                result = cpp_solve_until_stable(mesh, layer_config, e_z, layer_id)
            else:
                result = python_solver.solve_until_stable(
                    mesh, layer_id=layer_id, e_z=e_z
                )
            total_iterations += int(getattr(result, "iterations", 0))
            if getattr(result, "iterations", 0) >= config.max_iters:
                max_iter_hits += 1
            if (
                getattr(result, "max_dx", 0.0)
                >= self.DX_CLIP_DIAGNOSTIC * (1.0 - 1e-9)
            ):
                clipped_steps += 1
            self._update_czm_from_current_terms(
                mesh, layer_config, layer_id, e_z, x_before_solve
            )

        guard_mesh = copy.deepcopy(mesh)
        guard_result = result
        if commit_mesh is None:
            commit_mesh = copy.deepcopy(guard_mesh)
            commit_result = guard_result
            commit_steps = layer_steps
        if (
            expected_lift_steps > 0
            and len(top_ids) > 0
            and layer_id + 1 < self._n_layers
            and abs(lift_step) > 0.0
            and commit_steps > 0
        ):
            platform_return_distance = float(commit_steps) * abs(float(lift_step))
            return_top_ids = self._current_lifting_top(commit_mesh)
            for return_step_distance in self._positive_step_distances(
                platform_return_distance, abs(lift_step)
            ):
                if len(return_top_ids) == 0:
                    break
                down_config = replace(
                    layer_config,
                    v_lift=-return_step_distance / max(abs(config.dt), 1e-12),
                    enable_czm=False,
                )
                bottom = commit_mesh.bottom_nodes(layer_id)
                bottom_state = commit_mesh.czm_state[bottom].copy()
                bottom_damage = commit_mesh.damage[bottom].copy()
                bottom_time_free = commit_mesh.time_free[bottom].copy()
                if use_cpp:
                    commit_result = cpp_solve_lift_and_relax(
                        commit_mesh,
                        down_config,
                        0.0,
                        layer_id,
                        return_top_ids,
                    )
                else:
                    return_solver = solver(down_config)
                    commit_result = return_solver.solve_with_lift(
                        commit_mesh,
                        layer_id=layer_id,
                        e_z=0.0,
                        lifting_top=return_top_ids,
                    )
                if bool(getattr(layer_config, "enable_czm", True)) and len(bottom) > 0:
                    commit_mesh.czm_state[bottom] = bottom_state
                    commit_mesh.damage[bottom] = bottom_damage
                    commit_mesh.time_free[bottom] = bottom_time_free
                return_steps += 1
                total_iterations += int(getattr(commit_result, "iterations", 0))
                if getattr(commit_result, "iterations", 0) >= config.max_iters:
                    max_iter_hits += 1
                if (
                    getattr(commit_result, "max_dx", 0.0)
                    >= self.DX_CLIP_DIAGNOSTIC * (1.0 - 1e-9)
                ):
                    clipped_steps += 1
        info: dict[str, float | str] = {
            "timing_mode": timing_mode,
            "expected_steps": float(expected_lift_steps),
            "detach_step": float(detach_step),
            "commit_step": float(commit_steps),
            "guard_step": float(layer_steps),
            "return_steps": float(return_steps),
            "platform_return_distance": float(platform_return_distance),
            "peak_start_step": float(peak_start_step),
            "applied_steps": float(field_applied_steps),
            "detach_E_z": float(e_z),
            "peak_E_z": float(e_z if peak_e_z is None else peak_e_z),
        }
        return _FieldDebugBranchRun(
            commit_mesh=commit_mesh,
            guard_mesh=guard_mesh,
            commit_result=commit_result,
            guard_result=guard_result,
            commit_steps=commit_steps,
            executed_steps=layer_steps,
            total_iterations=total_iterations,
            max_iter_hits=max_iter_hits,
            clipped_steps=clipped_steps,
            lift_max=lift_max,
            info=info,
        )

    def _validate_lift_plan(
        self, layer_id: int, lift_max: float, lift_step: float, expected_steps: int
    ) -> None:
        """拒绝明显异常的提升计划，避免 GUI 看似卡死。"""
        if expected_steps > self.MAX_EXPECTED_LIFT_STEPS:
            raise RuntimeError(
                "提升步数异常过大: "
                f"layer={layer_id}, lift_max={lift_max:.6e} m, "
                f"lift_step={lift_step:.6e} m, steps={expected_steps}. "
                "请检查 GUI 层厚单位是否为 mm。"
            )

    @staticmethod
    def _solver_step_converged(result: Any, config: SimulationConfig) -> bool:
        """Return whether a single solver call reached the configured tolerance."""
        max_dx = float(getattr(result, "max_dx", math.inf))
        stable_steps = int(getattr(result, "stable_steps", 0))
        n_stable = max(1, int(config.N_stable))
        return (
            math.isfinite(max_dx)
            and max_dx < float(config.epsilon)
            and stable_steps >= n_stable
        )

    @staticmethod
    def _raise_if_detached_before_convergence(
        layer_id: int,
        layer_steps: int,
        result: Any,
        config: SimulationConfig,
    ) -> None:
        """Reject all-free CZM states produced by an unconverged solver step."""
        if not bool(getattr(result, "all_free", False)):
            return
        if SimulationWorker._solver_step_converged(result, config):
            return

        max_dx = float(getattr(result, "max_dx", math.nan))
        iterations = int(getattr(result, "iterations", 0))
        stable_steps = int(getattr(result, "stable_steps", 0))
        raise RuntimeError(
            f"layer {layer_id} detached before solver convergence: "
            "solver did not converge "
            f"(lift_step={layer_steps}, iterations={iterations}/"
            f"{int(config.max_iters)}, stable_steps={stable_steps}/"
            f"{int(config.N_stable)}, max_dx={max_dx:.6e}, "
            f"epsilon={float(config.epsilon):.6e}). "
            "Refusing to accept this layer result because it would produce invalid geometry."
        )

    @staticmethod
    def _cpp_payload_to_layer_result(payload: dict[str, Any]) -> LayerResult:
        """将 C++ 子进程完成消息转换为 GUI 使用的 ``LayerResult``。"""
        total_steps = int(payload.get("total_steps", 0))
        final_max_dx = float(payload.get("final_max_dx", 0.0))
        metrics: dict[str, float] = {
            "total_steps": float(total_steps),  # Backward-compatible key.
            "solver_total_steps": float(total_steps),
            "solver_final_max_dx": final_max_dx,
            "solver_total_iterations": float(payload.get("total_iterations", 0)),
            "solver_max_iter_hits": float(payload.get("max_iter_hits", 0)),
            "solver_clipped_steps": float(payload.get("clipped_steps", 0)),
            "solver_elapsed_s": float(payload.get("elapsed_s", 0.0)),
            "solver_avg_call_ms": float(payload.get("avg_call_ms", 0.0)),
            "solver_lift_max": float(payload.get("lift_max", 0.0)),
            "solver_lift_step": float(payload.get("lift_step", 0.0)),
            "solver_expected_steps": float(payload.get("expected_steps", 0)),
            "solver_top_nodes": float(payload.get("top_nodes", 0)),
            "shape_error_available": 0.0,
        }
        if total_steps > 0:
            metrics["solver_max_iter_hit_pct"] = (
                metrics["solver_max_iter_hits"] / total_steps * 100.0
            )
            metrics["solver_clipped_pct"] = (
                metrics["solver_clipped_steps"] / total_steps * 100.0
            )
        metrics["E_z"] = float(payload.get("E_z", 0.0))
        return LayerResult(
            layer_id=int(payload["layer_id"]),
            x_sim=np.zeros((0, 3), dtype=float),
            v_sim=np.zeros((0, 3), dtype=float),
            error_metrics=metrics,
            field_command_next=FieldCommand(np.array([])),
            max_deformation=final_max_dx,
            rms_error=0.0,
            success=bool(payload.get("success", True)),
        )

    # ───────────────────────────────────────────────────────────
    # 主运行方法（在 QThread 中执行）
    # ───────────────────────────────────────────────────────────

    @QtCore.Slot()
    def run(self) -> None:
        """在 QThread 中执行完整仿真（不阻塞主线程事件循环）。

        分三阶段：
        1. **前处理**：构建合规网格管线
        2. **逐层仿真**：VBD 稳定求解 + 平台提升（Python 侧控制时间循环）
        3. **后处理**：最终结果写入 + 完成信号
        """
        # ── 文件级崩溃追踪（线程内 Qt 信号可能在 crash 前未送达）──
        _trace_path = Path(self._output_dir) / "worker_trace.log"
        _trace_path.parent.mkdir(parents=True, exist_ok=True)

        def _write_trace(msg: str) -> None:
            try:
                with open(_trace_path, "a", encoding="utf-8") as f:
                    f.write(f"{time.perf_counter():.3f} {msg}\n")
                    f.flush()
            except Exception:
                pass

        self._trace = _write_trace  # 供 _run_layers 等方法访问
        self._trace("worker_run_start")

        try:
            results: list[LayerResult] = []
            t_start = time.perf_counter()

            self.log_message.emit("⚙️ [Worker] 仿真线程已启动…")
            self._trace("worker_log_started")

            # ════════════════ 阶段 1：前处理 ════════════════
            self._preprocess()
            self._trace("preprocess_done")

            # ════════════════ 阶段 2：逐层仿真 ════════════════
            results = self._run_layers()

            if self._stop_flag:
                self._trace("worker_cancelled")
                self.log_message.emit("⏹ [Worker] 仿真已中断")
                self.cancelled.emit(results)
                return

            # ════════════════ 阶段 3：后处理 ════════════════
            elapsed = time.perf_counter() - t_start
            self.log_message.emit(
                f"✅ [Worker] 仿真完成 — {self._n_layers} 层，"
                f"耗时 {elapsed:.1f}s"
            )
            self.finished.emit(results)

        except _SimulationCancelled:
            self._trace("worker_cancelled")
            self.log_message.emit("⏹ [Worker] 仿真已中断")
            self.cancelled.emit(results)

        except Exception as exc:
            tb = traceback.format_exc()
            self.error.emit(f"仿真线程崩溃:\n{tb}")

    # ───────────────────────────────────────────────────────────
    # 前处理
    # ───────────────────────────────────────────────────────────

    def _preprocess(self) -> None:
        """前处理：网格已由 OCC 完美切片为最终形态，无需旧管线重建。

        直接使用传入的 ``self._mesh``，仅需自动识别最顶层节点
        作为拉拔夹持平台（若网格未预设 top_fixed 标志）。
        """
        mesh = self._mesh

        # ── 动态识别拉拔夹持平台（最顶层节点）──
        # 若 is_top_fixed 全为 False，说明网格生成时未设置顶部固定面，
        # 此时自动将 Z 坐标最高的顶点标记为拉拔夹持节点
        if mesh.is_top_fixed is None or not mesh.is_top_fixed.any():
            z_max = float(np.max(mesh.vertices[:, 2]))
            mesh.is_top_fixed = np.isclose(
                mesh.vertices[:, 2], z_max, atol=1e-4
            )

        # 仅用于前处理日志；每层实际提升节点由 LayerActivator 后的
        # is_top_fixed 动态计算，避免复用全模型最高面的陈旧缓存。
        top_ids_preview = np.flatnonzero(mesh.is_top_fixed)
        n_top_fixed = len(top_ids_preview)

        self.log_message.emit(
            f"📐 [Worker] 前处理完成 — "
            f"{mesh.vertices.shape[0]} 顶点，"
            f"{mesh.tets.shape[0] if mesh.tets is not None else 0} 四面体"
            f" | 拉拔夹持节点: {n_top_fixed}"
        )

    # ───────────────────────────────────────────────────────────
    # 逐层仿真
    # ───────────────────────────────────────────────────────────

    def _run_cpp_subprocess(self) -> list[LayerResult]:
        """在子进程中运行 C++ 仿真，逐层返回结果。"""
        from hydrogel_vbd.solver.cpp_subprocess import (
            CppSubprocessSolver, _ProgressMsg, _LogMsg,
            _FrameMsg, _DoneMsg, _ErrorMsg,
        )

        mesh_dict: dict[str, Any] = {}
        for attr in (
            "vertices", "velocities", "prev_vertices", "ideal_vertices",
            "node_mass", "active_vertex_mask", "is_top_fixed",
            "is_bottom_surface", "czm_state", "damage", "time_free",
            "tets", "active_tet_mask", "dm_inv", "tet_volumes", "colors",
            "layer_id_per_vertex", "layer_id_per_tet", "first_active_layer",
            "is_top_surface_of_layer",
        ):
            val = getattr(self._mesh, attr, None)
            if val is not None:
                mesh_dict[attr] = val

        config_dict: dict[str, Any] = {}
        for attr in (
            "dt", "max_iters", "N_stable", "epsilon", "k_d", "rho_cheb",
            "mu", "kappa", "c_shrink", "q_ion", "g",
            "T_max", "K_czm", "delta_f", "node_area",
            "enable_czm",
            "z_fep", "C_0", "eta", "fluid_radius",
            "d_min", "d_fluid_max", "t_fluid_max",
            "v_lift", "layer_thickness", "lift_multiplier", "c_init",
        ):
            val = getattr(self._config, attr, None)
            if val is not None:
                config_dict[attr] = val

        proc = CppSubprocessSolver(
            mesh_dict, config_dict,
            self._n_layers, str(self._output_dir),
            diagnostics_enabled=self._solver_diagnostics_enabled,
            diagnostics_stride=self._solver_diagnostics_stride,
        )
        self._cpp_solver = proc
        proc.start()
        self._trace("cpp_subprocess_started")

        results: list[LayerResult] = []
        last_layer = 0
        try:
            for msg in proc.iter_messages(timeout=0.2):
                if self._stop_flag:
                    self._trace("cpp_subprocess_stopped_by_user")
                    proc.terminate()
                    raise _SimulationCancelled()
                if isinstance(msg, _LogMsg):
                    self.log_message.emit(msg.text)
                elif isinstance(msg, _ProgressMsg):
                    self.sub_progress.emit(msg.layer, msg.percentage, msg.step)
                    last_layer = msg.layer
                elif isinstance(msg, _FrameMsg):
                    self.frame_ready.emit({
                        "vertices": msg.vertices,
                        "tets": msg.tets,
                        "active_mask": msg.active_mask,
                        "active_tet_mask": msg.active_tet_mask,
                        "title": msg.title,
                    })
                elif isinstance(msg, _ErrorMsg):
                    if self._stop_flag:
                        self._trace("cpp_subprocess_stopped_by_user")
                        raise _SimulationCancelled()
                    raise RuntimeError(msg.error)
                elif isinstance(msg, _DoneMsg):
                    self._trace(f"cpp_done layers={len(msg.results)}")
                    for r in msg.results:
                        results.append(self._cpp_payload_to_layer_result(r))
                    return results
        except Exception:
            self._trace(f"cpp_crash_at_layer_{last_layer}")
            if self._stop_flag:
                raise _SimulationCancelled()
            raise
        finally:
            proc.terminate()
            self._cpp_solver = None

        raise RuntimeError("C++ 子进程意外结束")

    def _restore_mesh_copy(self) -> None:
        self._mesh = self._deep_copy_mesh(self._mesh_original)

    def _run_layers(self) -> list[LayerResult]:
        """逐层执行 VBD 仿真，渲染降频推送帧数据。

        控制反转架构：
        C++ 求解器已降级为单步函数 ``solve_lift_and_relax``，
        本方法接管时间流逝：在层循环内通过 Python while 循环
        反复调用 C++ 单步函数，每次调用后 GIL 自动释放，
        Worker 线程可向主线程发射信号。

        Returns
        -------
        list[LayerResult]
            每层的结果列表。
        """
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver
        from hydrogel_vbd.geometry.layer_activator import LayerActivator
        from hydrogel_vbd.control.field_controller import (
            BottomZFieldController,
            PIDFieldController,
        )
        from hydrogel_vbd.solver.cpp_adapter import (
            solve_until_stable as cpp_solve_until_stable,
            solve_lift_and_relax as cpp_solve_lift_and_relax,
        )
        from hydrogel_vbd.solver.diagnostics import (
            SolverRunawayGuard,
            SolverStepDiagnostics,
            diagnostics_enabled,
            prepare_solver_diagnostics_csv,
            write_solver_diagnostics_csv,
        )

        if self._field_debug_enabled and self._field_debug_use_cpp:
            self.log_message.emit(
                "  [field-debug] 使用直接 C++ adapter 求解 no-field / with-field 分支"
            )

        if self._use_cpp and self._field_debug_enabled and not self._field_debug_use_cpp:
            self.log_message.emit(
                "  [field-debug] 电场对比需要克隆同层状态，已切换到 Python 求解器"
            )
            self._use_cpp = False

        if self._use_cpp and not self._field_debug_enabled:
            self.log_message.emit("  [info] 使用 C++ 加速求解器 (子进程)")
            self._trace("cpp_subprocess_start")
            try:
                results = self._run_cpp_subprocess()
                self._trace("cpp_subprocess_done")
                return results
            except _SimulationCancelled:
                raise
            except Exception as exc:
                self._trace(f"cpp_subprocess_failed: {exc}")
                self.log_message.emit(f"  [warn] C++ 回退 Python: {exc}")
                self._use_cpp = False
                self._restore_mesh_copy()

        if self._field_debug_enabled and self._field_debug_use_cpp:
            self.log_message.emit("  [info] 使用 Python 控制循环 + C++ adapter 分支求解")
        else:
            self.log_message.emit("  [info] 使用 Python 参考求解器")
        activator = LayerActivator()
        pid = PIDFieldController(self._config)
        bottom_z_detach_debug = BottomZFieldController(self._config)
        bottom_z_peak_debug = BottomZFieldController(self._config)
        self._trace(
            f"solvers_ready cpp={self._use_cpp} n_layers={self._n_layers} "
            f"enable_czm={getattr(self._config, 'enable_czm', True)} "
            f"rho_cheb={getattr(self._config, 'rho_cheb', None)}"
        )

        results: list[LayerResult] = []
        step_counter = 0
        render_interval = 50
        diag_enabled = (
            diagnostics_enabled()
            if self._solver_diagnostics_enabled is None
            else bool(self._solver_diagnostics_enabled)
        )
        diag_path = self._output_dir / "reports" / "solver_diagnostics.csv"
        diag_stride = (
            max(1, int(os.environ.get("HYDROGEL_VBD_SOLVER_DIAG_STRIDE", "250")))
            if self._solver_diagnostics_stride is None
            else max(1, int(self._solver_diagnostics_stride))
        )
        self.log_message.emit(
            f"  [diag] solver CSV enabled={diag_enabled} path={diag_path}"
        )
        if diag_enabled:
            prepare_solver_diagnostics_csv(diag_path)
            self._trace(f"diagnostic_csv_prepared path={diag_path}")
        diag_guard = SolverRunawayGuard(
            limit=50,
            max_iters=int(self._config.max_iters),
            dx_clip=self.DX_CLIP_DIAGNOSTIC,
        )
        diag_stop_reason: str | None = None

        def _should_record_diag(step: int, expected_steps: int, result: Any) -> bool:
            return (
                step == 0
                or step == 1
                or step == expected_steps
                or step % diag_stride == 0
                or int(getattr(result, "iterations", 0)) >= int(self._config.max_iters)
                or float(getattr(result, "max_dx", 0.0))
                >= self.DX_CLIP_DIAGNOSTIC * (1.0 - 1e-9)
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
            nonlocal diag_stop_reason
            if not diag_enabled:
                return False
            diag = SolverStepDiagnostics.from_mesh(
                self._mesh,
                layer_id=layer_id,
                step=step,
                lift_max=lift_max,
                lift_step=lift_step,
                expected_steps=expected_steps,
                result=result,
                call_ms=call_ms,
                dx_clip=self.DX_CLIP_DIAGNOSTIC,
                z_fep=self._layer_contact_z(self._config, layer_id),
                x_before=x_before,
                czm_pull=czm_pull,
                czm_area=self._config.node_area,
                czm_t_max=self._config.T_max,
                czm_delta_f=self._config.delta_f,
            )
            write_solver_diagnostics_csv(diag_path, [diag])
            if diag_guard.observe(diag):
                diag_stop_reason = (
                    f"layer={layer_id}, step={step}, "
                    f"连续 {diag_guard.consecutive_bad_steps} 步 max_iter 且 clipped"
                )
                self._trace(
                    f"diagnostic_runaway_guard layer={layer_id} step={step} "
                    f"iterations={diag.iterations} max_dx={diag.max_dx:.6e}"
                )
                self.log_message.emit(
                    f"  [diag] 求解器诊断已停止仿真: {diag_stop_reason}"
                )
                self._stop_flag = True
                return True
            return False

        for layer_id in range(self._n_layers):
            if self._stop_flag:
                self._trace(f"layer_{layer_id}_skipped_after_stop")
                break
            layer_start = time.perf_counter()
            layer_steps = 0
            layer_call_elapsed_s = 0.0
            layer_total_iterations = 0
            layer_max_iter_hits = 0
            layer_clipped_steps = 0
            self.log_message.emit(
                f"  🔹 第 {layer_id + 1}/{self._n_layers} 层 ← 开始 VBD 求解"
            )

            if self._field_debug_enabled:
                layer_start_mesh = copy.deepcopy(self._mesh)
                call_start = time.perf_counter()
                field_cpp_fallbacks_before = self._field_debug_cpp_fallback_count
                field_branch_requested_cpp = bool(self._field_debug_use_cpp)
                no_field_run = self._run_field_debug_branch(
                    layer_start_mesh,
                    self._config,
                    layer_id,
                    e_z=0.0,
                    use_cpp=field_branch_requested_cpp,
                    continue_to_peak=True,
                )
                no_field_metrics = self._shape_debug_metrics(
                    no_field_run.guard_mesh, layer_id
                )
                no_field_commit_metrics = self._shape_debug_metrics(
                    no_field_run.commit_mesh, layer_id
                )
                self._push_frame(
                    vertices=no_field_run.commit_mesh.vertices.copy(),
                    tets=no_field_run.commit_mesh.tets,
                    active_mask=no_field_run.commit_mesh.active_vertex_mask.copy(),
                    active_tet_mask=no_field_run.commit_mesh.active_tet_mask.copy(),
                    title=(
                        f"field-debug layer {layer_id + 1}/{self._n_layers} "
                        "no-field baseline (commit)"
                    ),
                )
                detach_state = bottom_z_detach_debug.update(
                    bottom_nodes=no_field_run.commit_mesh.bottom_nodes(layer_id),
                    target_vertices=no_field_run.commit_mesh.ideal_vertices,
                    simulated_vertices=no_field_run.commit_mesh.vertices,
                )
                peak_state = bottom_z_peak_debug.update(
                    bottom_nodes=no_field_run.guard_mesh.bottom_nodes(layer_id),
                    target_vertices=no_field_run.guard_mesh.ideal_vertices,
                    simulated_vertices=no_field_run.guard_mesh.vertices,
                )
                derived_e_z = max(float(detach_state.E_z), float(peak_state.E_z))
                derived_unclipped_e_z = max(
                    float(detach_state.unclipped_E_z),
                    float(peak_state.unclipped_E_z),
                )
                candidate_skipped = derived_e_z <= 1.0e-12
                if candidate_skipped:
                    with_field_run = _FieldDebugBranchRun(
                        commit_mesh=copy.deepcopy(no_field_run.commit_mesh),
                        guard_mesh=copy.deepcopy(no_field_run.guard_mesh),
                        commit_result=no_field_run.commit_result,
                        guard_result=no_field_run.guard_result,
                        commit_steps=0,
                        executed_steps=0,
                        total_iterations=0,
                        max_iter_hits=0,
                        clipped_steps=0,
                        lift_max=no_field_run.lift_max,
                        info={
                            "timing_mode": "event_windows_v2",
                            "expected_steps": no_field_run.info["expected_steps"],
                            "detach_step": no_field_run.info["detach_step"],
                            "commit_step": no_field_run.info["commit_step"],
                            "guard_step": no_field_run.info["guard_step"],
                            "return_steps": no_field_run.info["return_steps"],
                            "platform_return_distance": no_field_run.info[
                                "platform_return_distance"
                            ],
                            "peak_start_step": no_field_run.info["peak_start_step"],
                            "applied_steps": 0.0,
                            "detach_E_z": float(detach_state.E_z),
                            "peak_E_z": float(peak_state.E_z),
                        },
                    )
                    with_field_info = {
                        "timing_mode": "event_windows_v2",
                        "expected_steps": no_field_run.info["expected_steps"],
                        "detach_step": no_field_run.info["detach_step"],
                        "commit_step": no_field_run.info["commit_step"],
                        "guard_step": no_field_run.info["guard_step"],
                        "return_steps": no_field_run.info["return_steps"],
                        "platform_return_distance": no_field_run.info[
                            "platform_return_distance"
                        ],
                        "peak_start_step": no_field_run.info["peak_start_step"],
                        "applied_steps": 0.0,
                        "detach_E_z": float(detach_state.E_z),
                        "peak_E_z": float(peak_state.E_z),
                    }
                else:
                    with_field_run = self._run_field_debug_branch(
                        layer_start_mesh,
                        self._config,
                        layer_id,
                        e_z=detach_state.E_z,
                        use_cpp=bool(self._field_debug_use_cpp),
                        event_window_detach_step=int(
                            float(no_field_run.info["detach_step"])
                        ),
                        peak_e_z=peak_state.E_z,
                        continue_to_peak=True,
                    )
                    with_field_info = with_field_run.info
                with_field_metrics = self._shape_debug_metrics(
                    with_field_run.guard_mesh, layer_id
                )
                with_field_commit_metrics = self._shape_debug_metrics(
                    with_field_run.commit_mesh, layer_id
                )
                field_cpp_fallbacks = (
                    self._field_debug_cpp_fallback_count
                    - field_cpp_fallbacks_before
                )
                field_solver_backend = (
                    "cpp_adapter"
                    if field_branch_requested_cpp and field_cpp_fallbacks == 0
                    else "python"
                )
                guard = self._field_debug_guard(
                    no_field_metrics,
                    with_field_metrics,
                    float(getattr(self._config, "rms_guard_tolerance", 0.01)),
                )
                guard_passed = bool(guard["passed"])
                if guard_passed:
                    self._mesh = with_field_run.commit_mesh
                    selected_result = with_field_run.commit_result
                    selected_metrics = with_field_commit_metrics
                    effective_mode = "with_field"
                    selected_e_z = derived_e_z
                    layer_steps = with_field_run.executed_steps
                    layer_total_iterations = with_field_run.total_iterations
                    layer_max_iter_hits = with_field_run.max_iter_hits
                    layer_clipped_steps = with_field_run.clipped_steps
                else:
                    self._mesh = no_field_run.commit_mesh
                    selected_result = no_field_run.commit_result
                    selected_metrics = no_field_commit_metrics
                    effective_mode = "no_field"
                    selected_e_z = 0.0
                    layer_steps = no_field_run.executed_steps
                    layer_total_iterations = no_field_run.total_iterations
                    layer_max_iter_hits = no_field_run.max_iter_hits
                    layer_clipped_steps = no_field_run.clipped_steps

                self._push_frame(
                    vertices=self._mesh.vertices.copy(),
                    tets=self._mesh.tets,
                    active_mask=self._mesh.active_vertex_mask.copy(),
                    active_tet_mask=self._mesh.active_tet_mask.copy(),
                    title=(
                        f"field-debug layer {layer_id + 1}/{self._n_layers} "
                        f"selected {effective_mode} "
                        f"(commit {int(float(with_field_info['commit_step']))}/"
                        f"guard {int(float(with_field_info['guard_step']))}, "
                        f"return {int(float(with_field_info['return_steps']))})"
                    ),
                )

                x_final = self._mesh.vertices.copy()
                v_final = (
                    self._mesh.velocities.copy()
                    if self._mesh.velocities is not None
                    else np.zeros_like(x_final)
                )
                layer_elapsed_s = time.perf_counter() - layer_start
                branch_elapsed_s = time.perf_counter() - call_start
                avg_call_ms = (
                    branch_elapsed_s / 2.0 * 1000.0
                )
                layer_metrics: dict[str, float | str] = {
                    "E_z": selected_e_z,
                    "solver_total_steps": float(layer_steps),
                    "solver_final_max_dx": float(
                        getattr(selected_result, "max_dx", 0.0)
                    ),
                    "solver_total_iterations": float(layer_total_iterations),
                    "solver_max_iter_hits": float(layer_max_iter_hits),
                    "solver_clipped_steps": float(layer_clipped_steps),
                    "solver_elapsed_s": float(layer_elapsed_s),
                    "solver_avg_call_ms": float(avg_call_ms),
                    "solver_lift_max": float(no_field_run.lift_max),
                    "solver_lift_step": float(
                        self._config.v_lift * self._config.dt
                    ),
                    "solver_expected_steps": float(
                        self._expected_lift_steps(
                            no_field_run.lift_max,
                            self._config.v_lift * self._config.dt,
                        )
                        if self._config.v_lift * self._config.dt > 0.0
                        else 0
                    ),
                    "solver_top_nodes": float(
                        len(self._current_lifting_top(self._mesh))
                    ),
                    "shape_error_available": 1.0,
                    "field_debug_enabled": 1.0,
                    "field_debug_solver_backend": field_solver_backend,
                    "field_debug_cpp_fallbacks": float(field_cpp_fallbacks),
                    "field_candidate_skipped": float(candidate_skipped),
                    "field_timing_mode": with_field_info["timing_mode"],
                    "field_window_expected_steps": with_field_info[
                        "expected_steps"
                    ],
                    "field_window_detach_step": with_field_info[
                        "detach_step"
                    ],
                    "field_window_peak_start_step": with_field_info[
                        "peak_start_step"
                    ],
                    "field_window_applied_steps": with_field_info[
                        "applied_steps"
                    ],
                    "field_detach_E_z": with_field_info["detach_E_z"],
                    "field_peak_E_z": with_field_info["peak_E_z"],
                    "field_commit_step": with_field_info["commit_step"],
                    "field_guard_step": with_field_info["guard_step"],
                    "field_platform_return_steps": with_field_info[
                        "return_steps"
                    ],
                    "field_platform_return_distance": with_field_info[
                        "platform_return_distance"
                    ],
                    "field_bottom_node_count": no_field_metrics["bottom_count"],
                    "field_no_field_rms": no_field_metrics["rms"],
                    "field_with_field_rms": with_field_metrics["rms"],
                    "field_no_field_commit_rms": no_field_commit_metrics["rms"],
                    "field_with_field_commit_rms": with_field_commit_metrics["rms"],
                    "field_no_field_max_error": no_field_metrics["max_error"],
                    "field_with_field_max_error": with_field_metrics["max_error"],
                    "field_no_field_commit_max_error": no_field_commit_metrics[
                        "max_error"
                    ],
                    "field_with_field_commit_max_error": with_field_commit_metrics[
                        "max_error"
                    ],
                    "field_rms_improvement": (
                        no_field_metrics["rms"] - with_field_metrics["rms"]
                    ),
                    "field_no_field_bottom_z_mean": no_field_metrics[
                        "bottom_z_mean"
                    ],
                    "field_no_field_bottom_z_max": no_field_metrics[
                        "bottom_z_max"
                    ],
                    "field_with_field_bottom_z_mean": with_field_metrics[
                        "bottom_z_mean"
                    ],
                    "field_with_field_bottom_z_max": with_field_metrics[
                        "bottom_z_max"
                    ],
                    "field_derived_E_z": derived_e_z,
                    "field_unclipped_E_z": derived_unclipped_e_z,
                    "field_detach_unclipped_E_z": float(
                        detach_state.unclipped_E_z
                    ),
                    "field_peak_unclipped_E_z": float(
                        peak_state.unclipped_E_z
                    ),
                    "field_guard_passed": float(guard_passed),
                    "field_guard_reason": guard["reason"],
                    "field_rms_guard_passed": guard["rms_passed"],
                    "field_max_error_guard_passed": guard["max_error_passed"],
                    "field_improvement_guard_passed": guard[
                        "improvement_passed"
                    ],
                    "field_rms_guard_limit": guard["rms_limit"],
                    "field_max_error_guard_limit": guard["max_error_limit"],
                    "field_effective_mode": effective_mode,
                    "rms_error": selected_metrics["rms"],
                    "max_error": selected_metrics["max_error"],
                }
                self.log_message.emit(
                    "  [field-debug] "
                    f"layer={layer_id + 1}, bottom={int(no_field_metrics['bottom_count'])}, "
                    f"z_mean(no/with)="
                    f"{no_field_metrics['bottom_z_mean']:.6e}/"
                    f"{with_field_metrics['bottom_z_mean']:.6e}, "
                    f"rms(no/with)="
                    f"{no_field_metrics['rms']:.6e}/"
                    f"{with_field_metrics['rms']:.6e}, "
                    f"max(no/with)="
                    f"{no_field_metrics['max_error']:.6e}/"
                    f"{with_field_metrics['max_error']:.6e}, "
                    f"E(detach/peak)="
                    f"{float(with_field_info['detach_E_z']):.6e}/"
                    f"{float(with_field_info['peak_E_z']):.6e}, "
                    f"E_z={derived_e_z:.6e} "
                    f"(raw={derived_unclipped_e_z:.6e}), "
                    f"backend={field_solver_backend}, "
                    f"steps(commit/guard)="
                    f"{int(float(with_field_info['commit_step']))}/"
                    f"{int(float(with_field_info['guard_step']))}, "
                    f"return_steps={int(float(with_field_info['return_steps']))}, "
                    f"guard={guard['reason']}"
                )
                layer_result = LayerResult(
                    layer_id=layer_id,
                    x_sim=x_final,
                    v_sim=v_final,
                    error_metrics=layer_metrics,
                    field_command_next=FieldCommand(
                        np.array([selected_e_z], dtype=float),
                        electrode_ids=["E_z"],
                    ),
                    max_deformation=selected_metrics["max_error"],
                    rms_error=selected_metrics["rms"],
                    success=(
                        diag_stop_reason is None
                        and getattr(selected_result, "all_free", True)
                    ),
                )
                results.append(layer_result)
                self.layer_finished.emit(layer_result)
                self.progress_update.emit(
                    layer_id + 1,
                    self._n_layers,
                    step_counter + int(layer_total_iterations),
                    int(getattr(selected_result, "iterations", 0)),
                )
                continue

            # ── 激活当前层（继承版：处理激活传播 + FEP 阈值）──
            layer_z_fep = self._layer_contact_z(self._config, layer_id)
            layer_config = replace(self._config, z_fep=layer_z_fep)
            activator.activate_with_inheritance(
                self._mesh, layer_id, z_fep=layer_z_fep
            )
            layer_solver = (
                PythonReferenceVBDSolver(layer_config)
                if not self._use_cpp else None
            )

            # ── PID 电场（首步误差默认为 0，后续由 metrics 模块更新）──
            pid_state = pid.update(0.0)  # FIXME: 接入形状误差计算
            e_z = pid_state.E_z

            # ══════════════════════════════════════════════════
            # 控制反转：Python 侧接管时间循环
            #
            # 若存在提升需求（当前层夹持顶面非空），按预估步数反复调用
            # 单步求解器，每步执行：提升 → 静平衡 → CZM 更新
            # ══════════════════════════════════════════════════
            lift_max = self._config.lift_multiplier * self._config.layer_thickness
            lift_step = self._config.v_lift * self._config.dt
            expected_lift_steps = (
                self._expected_lift_steps(lift_max, lift_step)
                if lift_step > 0.0
                else 0
            )
            lift_distance = 0.0
            top_ids = self._current_lifting_top(self._mesh)
            _record_diag(
                layer_id,
                0,
                lift_max,
                lift_step,
                expected_lift_steps,
                SimpleNamespace(iterations=0, stable_steps=0, max_dx=0.0),
                0.0,
            )

            if expected_lift_steps > 0 and len(top_ids) > 0:
                # ── 带提升的控制反转循环 ──
                self._validate_lift_plan(
                    layer_id, lift_max, lift_step, expected_lift_steps
                )
                self._trace(
                    f"layer_{layer_id}_lift_start top_ids={len(top_ids)} "
                    f"lift_max={lift_max:.6e} lift_step={lift_step:.6e} "
                    f"expected_steps={expected_lift_steps}"
                )
                for lift_step_index in range(expected_lift_steps):
                    if self._stop_flag:
                        break
                    x_before_call = self._mesh.vertices.copy()
                    call_start = time.perf_counter()
                    if self._use_cpp:
                        result = cpp_solve_lift_and_relax(
                            self._mesh, layer_config, e_z, layer_id, top_ids
                        )
                    else:
                        result = layer_solver.solve_with_lift(
                            self._mesh,
                            layer_id=layer_id,
                            e_z=e_z,
                            lifting_top=top_ids,
                            on_iteration=lambda it, dx: self._on_solver_iteration(
                                layer_id, it, dx, step_counter
                            ),
                        )
                    call_elapsed = time.perf_counter() - call_start
                    layer_call_elapsed_s += call_elapsed
                    layer_steps += 1
                    layer_total_iterations += int(getattr(result, "iterations", 0))
                    if getattr(result, "iterations", 0) >= self._config.max_iters:
                        layer_max_iter_hits += 1
                    if (
                        getattr(result, "max_dx", 0.0)
                        >= self.DX_CLIP_DIAGNOSTIC * (1.0 - 1e-9)
                    ):
                        layer_clipped_steps += 1
                    if _should_record_diag(layer_steps, expected_lift_steps, result):
                        if _record_diag(
                            layer_id,
                            layer_steps,
                            lift_max,
                            lift_step,
                            expected_lift_steps,
                            result,
                            call_elapsed * 1000.0,
                            x_before_call,
                        ):
                            break

                    step_counter += result.iterations
                    lift_distance = min(
                        (lift_step_index + 1) * lift_step, lift_max
                    )

                    # ── 渲染降频推送（在 Python 循环中执行，非 C++ 内部）──
                    if step_counter % render_interval == 0:
                        self._push_frame(
                            vertices=self._mesh.vertices.copy(),
                            tets=self._mesh.tets,
                            active_mask=self._mesh.active_vertex_mask.copy(),
                            active_tet_mask=self._mesh.active_tet_mask.copy(),
                            title=(
                                f"第 {layer_id + 1}/{self._n_layers} 层"
                                f" — 提升 {lift_distance:.3e} m"
                                f" — max_dx={result.max_dx:.4e}"
                            ),
                        )

                    # ── 每 20 步检查停止标志 ──
                    if step_counter % 20 == 0 and self._stop_flag:
                        break

                    # ── 细粒度子进度（提升百分比）──
                    lift_pct = min(100, int(lift_distance / lift_max * 100))
                    self.sub_progress.emit(layer_id, lift_pct, step_counter)

                    # ── 全部脱膜则退出提升循环 ──
                    if result.all_free and bool(
                        getattr(self._config, "enable_czm", True)
                    ):
                        self._raise_if_detached_before_convergence(
                            layer_id, layer_steps, result, self._config
                        )
                        break

                    # ── 进度更新 ──
                    self.progress_update.emit(
                        layer_id + 1, self._n_layers, step_counter,
                        result.iterations,
                    )

            else:
                # ── 无提升：直接静平衡求解 ──
                if expected_lift_steps > 0 and len(top_ids) == 0:
                    self._trace(
                        f"layer_{layer_id}_no_lift_top_nodes=0 "
                        f"expected_steps={expected_lift_steps}"
                    )
                    self.log_message.emit(
                        f"  [warn] 第 {layer_id + 1} 层没有可提升顶面节点，"
                        "已跳过平台提升；请检查网格层面分类。"
                    )
                self._trace(f"layer_{layer_id}_solve_start cpp={self._use_cpp}")
                x_before_solve = self._mesh.vertices.copy()
                x_before_call = self._mesh.vertices.copy()
                call_start = time.perf_counter()
                if self._use_cpp:
                    result = cpp_solve_until_stable(
                        self._mesh, layer_config, e_z, layer_id
                    )
                else:
                    result = layer_solver.solve_until_stable(
                        self._mesh,
                        layer_id=layer_id,
                        e_z=e_z,
                        on_iteration=lambda it, dx: self._on_solver_iteration(
                            layer_id, it, dx, step_counter
                        ),
                    )
                call_elapsed = time.perf_counter() - call_start
                layer_call_elapsed_s += call_elapsed
                layer_total_iterations += int(getattr(result, "iterations", 0))
                if getattr(result, "iterations", 0) >= self._config.max_iters:
                    layer_max_iter_hits += 1
                if (
                    getattr(result, "max_dx", 0.0)
                    >= self.DX_CLIP_DIAGNOSTIC * (1.0 - 1e-9)
                ):
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
                    x_before_call,
                )

                self._update_czm_from_current_terms(
                    self._mesh,
                    layer_config,
                    layer_id,
                    e_z,
                    x_before_solve,
                )

            if self._stop_flag and layer_steps == 0:
                break

            # ── 收集结果 ──
            x_final = self._mesh.vertices.copy()
            v_final = (
                self._mesh.velocities.copy()
                if self._mesh.velocities is not None
                else np.zeros_like(x_final)
            )
            layer_elapsed_s = time.perf_counter() - layer_start
            avg_call_ms = (
                layer_call_elapsed_s / max(layer_steps, 1) * 1000.0
                if layer_steps > 0
                else layer_call_elapsed_s * 1000.0
            )
            layer_metrics: dict[str, float] = {
                "E_z": float(e_z),
                "solver_total_steps": float(layer_steps),
                "solver_final_max_dx": float(getattr(result, "max_dx", 0.0)),
                "solver_total_iterations": float(layer_total_iterations),
                "solver_max_iter_hits": float(layer_max_iter_hits),
                "solver_clipped_steps": float(layer_clipped_steps),
                "solver_elapsed_s": float(layer_elapsed_s),
                "solver_avg_call_ms": float(avg_call_ms),
                "solver_lift_max": float(lift_max),
                "solver_lift_step": float(lift_step),
                "solver_expected_steps": float(expected_lift_steps),
                "solver_top_nodes": float(len(top_ids)),
                "shape_error_available": 0.0,
            }
            if layer_steps > 0:
                layer_metrics["solver_max_iter_hit_pct"] = (
                    layer_max_iter_hits / layer_steps * 100.0
                )
                layer_metrics["solver_clipped_pct"] = (
                    layer_clipped_steps / layer_steps * 100.0
                )
            self._trace(
                f"layer_{layer_id}_done steps={layer_steps} "
                f"elapsed_s={layer_elapsed_s:.3f} "
                f"total_iters={layer_total_iterations} "
                f"max_iter_hits={layer_max_iter_hits} "
                f"clipped_steps={layer_clipped_steps} "
                f"avg_call_ms={avg_call_ms:.3f}"
            )
            layer_result = LayerResult(
                layer_id=layer_id,
                x_sim=x_final,
                v_sim=v_final,
                error_metrics=layer_metrics,
                field_command_next=FieldCommand(np.array([])),
                max_deformation=getattr(result, "max_dx", 0.0),
                rms_error=0.0,
                success=(diag_stop_reason is None and getattr(result, "all_free", True)),
            )
            results.append(layer_result)

            # ── 每层结束时推送 layer_finished（主线程用于实时误差分析、DVR 等）──
            self.layer_finished.emit(layer_result)

            self.progress_update.emit(
                layer_id + 1,
                self._n_layers,
                step_counter,
                getattr(result, "iterations", 0),
            )

            # ── 检查停止标志 ──
            if self._stop_flag:
                self.log_message.emit("🛑 [Worker] 用户请求停止仿真")
                break

        # ── 强制推送末帧 ──
        self._push_frame(
            vertices=self._mesh.vertices.copy(),
            tets=self._mesh.tets,
            active_mask=self._mesh.active_vertex_mask.copy(),
            active_tet_mask=self._mesh.active_tet_mask.copy(),
            title=f"仿真完成 — {self._n_layers} 层",
        )

        return results

    # ───────────────────────────────────────────────────────────
    # 求解器迭代回调（渲染降频控制）
    # ───────────────────────────────────────────────────────────

    def _on_solver_iteration(
        self, layer_id: int, iteration: int, max_dx: float, step_counter: int
    ) -> None:
        """VBD 求解器每次迭代后的回调，负责渲染降频推送。

        仅当 ``step_counter % render_interval == 0`` 时推送帧数据。
        非渲染周期：每 20 步检查一次停止标志。

        .. note::
           渲染降频逻辑已同时存在于 Python 控制循环中
           （``_run_layers`` 的 while 循环），此处作为备用回调。
        """
        render_interval = 50

        if step_counter % render_interval == 0:
            # ── 推送当前帧到主线程 ──
            self._push_frame(
                vertices=self._mesh.vertices.copy(),
                tets=self._mesh.tets,
                active_mask=self._mesh.active_vertex_mask.copy(),
                active_tet_mask=self._mesh.active_tet_mask.copy(),
                title=(
                    f"第 {layer_id + 1}/{self._n_layers} 层"
                    f" — 迭代 {iteration} — max_dx={max_dx:.4e}"
                ),
            )

        # 非渲染周期仍检查停止标志
        if self._stop_flag:
            self.log_message.emit("⏹ [Worker] 求解器回调检测到停止标志")
            raise _SimulationCancelled()

    # ───────────────────────────────────────────────────────────
    # 帧推送
    # ───────────────────────────────────────────────────────────

    def _push_frame(
        self,
        vertices: np.ndarray,
        tets: np.ndarray,
        active_mask: np.ndarray,
        active_tet_mask: np.ndarray,
        title: str,
    ) -> None:
        """通过 signal 推送一帧 3D 渲染数据到主线程。

        Parameters
        ----------
        vertices : np.ndarray, shape (N, 3)
            顶点坐标（深拷贝）。
        tets : np.ndarray, shape (M, 4)
            四面体索引。
        active_mask : np.ndarray, shape (N,), bool
            激活顶点掩码。
        title : str
            帧标题（显示在 GUI 标题栏或图例中）。
        """
        payload = {
            "vertices": vertices,
            "tets": tets,
            "active_mask": active_mask,
            "active_tet_mask": active_tet_mask,
            "title": title,
        }
        self.frame_ready.emit(payload)
