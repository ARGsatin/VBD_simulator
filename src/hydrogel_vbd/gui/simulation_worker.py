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

import time
import traceback
from pathlib import Path

import numpy as np
from PySide6 import QtCore

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.core.state import FieldCommand, LayerResult, MeshState
from hydrogel_vbd.physics.czm import CZMState


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
    error = QtCore.Signal(str)

    def __init__(
        self,
        mesh: MeshState,
        config: SimulationConfig,
        n_layers: int,
        output_dir: str | Path,
    ) -> None:
        super().__init__()
        # 完全深拷贝网格状态，确保线程安全
        self._mesh = self._deep_copy_mesh(mesh)
        self._config = config
        self._n_layers = int(n_layers)
        self._output_dir = Path(output_dir)
        self._stop_flag = False

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

        # ── top_ids 缓存（前处理阶段生成）──
        if hasattr(mesh, "top_ids") and mesh.top_ids is not None:
            copy_mesh.top_ids = mesh.top_ids.copy()

        return copy_mesh

    # ───────────────────────────────────────────────────────────
    # 停止控制
    # ───────────────────────────────────────────────────────────

    def request_stop(self) -> None:
        """请求安全停止仿真（在下一次迭代判别时生效）。"""
        self._stop_flag = True

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
        try:
            results: list[LayerResult] = []
            t_start = time.perf_counter()

            self.log_message.emit("⚙️ [Worker] 仿真线程已启动…")

            # ════════════════ 阶段 1：前处理 ════════════════
            self._preprocess()

            # ════════════════ 阶段 2：逐层仿真 ════════════════
            results = self._run_layers()

            # ════════════════ 阶段 3：后处理 ════════════════
            elapsed = time.perf_counter() - t_start
            self.log_message.emit(
                f"✅ [Worker] 仿真完成 — {self._n_layers} 层，"
                f"耗时 {elapsed:.1f}s"
            )
            self.finished.emit(results)

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

        # ── 建立顶部索引缓存 ──
        mesh.top_ids = np.flatnonzero(mesh.is_top_fixed)
        n_top_fixed = len(mesh.top_ids)

        self.log_message.emit(
            f"📐 [Worker] 前处理完成 — "
            f"{mesh.vertices.shape[0]} 顶点，"
            f"{mesh.tets.shape[0] if mesh.tets is not None else 0} 四面体"
            f" | 拉拔夹持节点: {n_top_fixed}"
        )

    # ───────────────────────────────────────────────────────────
    # 逐层仿真
    # ───────────────────────────────────────────────────────────

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
        from hydrogel_vbd.physics.czm import update_czm_states
        from hydrogel_vbd.solver.cpp_adapter import (
            is_cpp_available,
            solve_until_stable as cpp_solve_until_stable,
            solve_lift_and_relax as cpp_solve_lift_and_relax,
        )

        _use_cpp = is_cpp_available()
        if _use_cpp:
            self.log_message.emit("  [info] 使用 C++ 加速求解器")
        else:
            self.log_message.emit("  [info] 使用 Python 参考求解器")
        solver = PythonReferenceVBDSolver(self._config) if not _use_cpp else None
        activator = LayerActivator()
        pid = PIDFieldController(self._config)

        results: list[LayerResult] = []
        step_counter = 0
        render_interval = 50

        for layer_id in range(self._n_layers):
            self.log_message.emit(
                f"  🔹 第 {layer_id + 1}/{self._n_layers} 层 ← 开始 VBD 求解"
            )

            # ── 激活当前层（继承版：处理激活传播 + FEP 阈值）──
            activator.activate_with_inheritance(
                self._mesh, layer_id, z_fep=self._config.z_fep
            )

            # ── PID 电场（首步误差默认为 0，后续由 metrics 模块更新）──
            pid_state = pid.update(0.0)  # FIXME: 接入形状误差计算
            e_z = pid_state.E_z

            # ══════════════════════════════════════════════════
            # 控制反转：Python 侧接管时间循环
            #
            # 若存在提升需求（top_ids 非空），通过 while 循环反复调用
            # 单步求解器，每步执行：提升 → 静平衡 → CZM 更新
            # ══════════════════════════════════════════════════
            lift_max = 5.0 * self._config.layer_thickness
            lift_distance = 0.0
            top_ids = getattr(self._mesh, "top_ids", None)

            if top_ids is not None and len(top_ids) > 0:
                # ── 带提升的控制反转循环 ──
                while lift_distance < lift_max and not self._stop_flag:
                    if _use_cpp:
                        result = cpp_solve_lift_and_relax(
                            self._mesh, self._config, e_z, layer_id, top_ids
                        )
                    else:
                        result = solver.solve_with_lift(
                            self._mesh,
                            layer_id=layer_id,
                            e_z=e_z,
                            lifting_top=top_ids,
                            on_iteration=lambda it, dx: self._on_solver_iteration(
                                layer_id, it, dx, step_counter
                            ),
                        )

                    step_counter += result.iterations
                    lift_distance += self._config.v_lift * self._config.dt

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
                    if result.all_free:
                        break

                    # ── 进度更新 ──
                    self.progress_update.emit(
                        layer_id + 1, self._n_layers, step_counter,
                        result.iterations,
                    )

                # ── 提升完成后更新 CZM ──
                bottom = self._mesh.bottom_nodes(layer_id)
                if len(bottom):
                    update_czm_states(
                        self._mesh,
                        bottom,
                        internal_pull_z=np.full(
                            len(bottom), self._config.T_max
                        ),
                        area=self._config.node_area,
                        t_max=self._config.T_max,
                        k_czm=self._config.K_czm,
                        delta_f=self._config.delta_f,
                        z_fep=self._config.z_fep,
                        dt=self._config.dt,
                    )
            else:
                # ── 无提升：直接静平衡求解 ──
                if _use_cpp:
                    result = cpp_solve_until_stable(
                        self._mesh, self._config, e_z, layer_id
                    )
                else:
                    result = solver.solve_until_stable(
                        self._mesh,
                        layer_id=layer_id,
                        e_z=e_z,
                        on_iteration=lambda it, dx: self._on_solver_iteration(
                            layer_id, it, dx, step_counter
                        ),
                    )
                step_counter += result.iterations

                # ── 更新 CZM ──
                bottom = self._mesh.bottom_nodes(layer_id)
                if len(bottom):
                    update_czm_states(
                        self._mesh,
                        bottom,
                        internal_pull_z=np.full(
                            len(bottom), self._config.T_max
                        ),
                        area=self._config.node_area,
                        t_max=self._config.T_max,
                        k_czm=self._config.K_czm,
                        delta_f=self._config.delta_f,
                        z_fep=self._config.z_fep,
                        dt=self._config.dt,
                    )

            # ── 收集结果 ──
            x_final = self._mesh.vertices.copy()
            v_final = (
                self._mesh.velocities.copy()
                if self._mesh.velocities is not None
                else np.zeros_like(x_final)
            )
            layer_result = LayerResult(
                layer_id=layer_id,
                x_sim=x_final,
                v_sim=v_final,
                error_metrics={},
                field_command_next=FieldCommand(np.array([])),
                max_deformation=getattr(result, "max_dx", 0.0),
                rms_error=0.0,
                success=getattr(result, "all_free", True),
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
        if step_counter % 20 == 0 and self._stop_flag:
            self.log_message.emit("🛑 [Worker] 求解器回调检测到停止标志")

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
