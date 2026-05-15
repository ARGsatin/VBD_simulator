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
import math
import os
import time
import traceback
from dataclasses import replace
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
        from hydrogel_vbd.control.field_controller import PIDFieldController
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

        if self._use_cpp:
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

        self.log_message.emit("  [info] 使用 Python 参考求解器")
        activator = LayerActivator()
        pid = PIDFieldController(self._config)
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
            "title": title,
        }
        self.frame_ready.emit(payload)
