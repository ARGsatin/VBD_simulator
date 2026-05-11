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
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
from PySide6 import QtCore

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.core.state import LayerResult, MeshState


class SimulationWorker(QtCore.QObject):
    """VBD 仿真工作线程（QThread Worker）。

    持有网格数据的深拷贝，避免与主线程共享引用导致数据竞争。

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
        # 深拷贝网格状态，确保线程安全
        self._mesh = self._deep_copy_mesh(mesh)
        self._config = config
        self._n_layers = int(n_layers)
        self._output_dir = Path(output_dir)
        self._stop_flag = False

    # ───────────────────────────────────────────────────────────
    # 网格深拷贝（避免数据竞争）
    # ───────────────────────────────────────────────────────────

    @staticmethod
    def _deep_copy_mesh(mesh: MeshState) -> MeshState:
        """创建 MeshState 的深拷贝，写入独立的 numpy 数组。

        所有可变属性均调用 ``.copy()`` 确保线程隔离。
        """
        import copy

        copy_mesh = copy.copy(mesh)
        copy_mesh.vertices = mesh.vertices.copy()
        copy_mesh.velocities = mesh.velocities.copy()
        copy_mesh.prev_vertices = mesh.prev_vertices.copy() if mesh.prev_vertices is not None else mesh.vertices.copy()
        copy_mesh.masses = mesh.masses.copy()
        copy_mesh.is_top_fixed = mesh.is_top_fixed.copy()
        copy_mesh.is_bottom_surface = mesh.is_bottom_surface.copy()
        copy_mesh.active_vertex_mask = mesh.active_vertex_mask.copy()
        copy_mesh.czm_damage = mesh.czm_damage.copy() if mesh.czm_damage is not None else np.zeros_like(mesh.vertices[:, 0])
        # czm_state 是 object array，直接 copy 即可
        copy_mesh.czm_state = mesh.czm_state.copy() if mesh.czm_state is not None else None
        # tets 只读，无需深拷贝
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
        2. **逐层仿真**：VBD 稳定求解 + 平台提升
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
        """构建合规网格管线 (ConformalMeshPipeline)。"""
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline

        pipeline = ConformalMeshPipeline(self._config, self._output_dir)
        mesh = pipeline.run(self._mesh)  # 原地修改（返回同一引用）
        # 建立顶部索引缓存
        mesh.top_ids = np.flatnonzero(mesh.is_top_fixed)
        self.log_message.emit(
            f"📐 [Worker] 前处理完成 — "
            f"{mesh.vertices.shape[0]} 顶点，"
            f"{mesh.tets.shape[0] if mesh.tets is not None else 0} 四面体"
        )

    # ───────────────────────────────────────────────────────────
    # 逐层仿真
    # ───────────────────────────────────────────────────────────

    def _run_layers(self) -> list[LayerResult]:
        """逐层执行 VBD 仿真，渲染降频推送帧数据。

        Returns
        -------
        list[LayerResult]
            每层的结果列表。
        """
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver
        from hydrogel_vbd.geometry.layer_activator import LayerActivator
        from hydrogel_vbd.control.field_controller import PIDFieldController
        from hydrogel_vbd.physics.czm import update_czm_states

        solver = PythonReferenceVBDSolver(self._config)
        activator = LayerActivator()
        pid = PIDFieldController(self._config)

        results: list[LayerResult] = []
        step_counter = 0
        render_interval = 50

        for layer_id in range(self._n_layers):
            self.log_message.emit(f"  🔹 第 {layer_id + 1}/{self._n_layers} 层 ← 开始 VBD 求解")

            # ── 激活当前层 ──
            activator.activate_layer(self._mesh, layer_id)

            # ── PID 电场 ──
            e_z = pid.update(self._mesh, layer_id, None)

            # ── VBD 隐式求解 ──
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
                    internal_pull_z=np.full(len(bottom), self._config.T_max),
                    area=self._config.node_area,
                    t_max=self._config.T_max,
                    k_czm=self._config.K_czm,
                    delta_f=self._config.delta_f,
                    z_fep=self._config.z_fep,
                    dt=self._config.dt,
                )

            # ── 收集结果 ──
            layer_result = LayerResult(
                layer_id=layer_id,
                x=self._mesh.vertices.copy(),
                iterations=result.iterations,
                max_dx=result.max_dx,
                kinetic_energy=result.kinetic_energy,
                all_free=result.all_free,
            )
            results.append(layer_result)

            self.progress_update.emit(
                layer_id + 1, self._n_layers, step_counter, result.iterations
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
