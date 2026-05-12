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
import os
import pickle
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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
    title: str


@dataclass
class _DoneMsg:
    results: list[dict[str, Any]]


@dataclass
class _ErrorMsg:
    error: str


# ── 子进程入口 ──

def _worker_run(
    conn: mp.connection.Connection,
    mesh_dict: dict[str, np.ndarray],
    config_dict: dict[str, Any],
    n_layers: int,
    output_dir: str,
) -> None:
    """子进程主函数：加载 C++ 模块并执行完整仿真。

    通过 *conn* 向主进程发送进度、帧和最终结果。
    任何未捕获异常（包括 segfault → 进程直接被 OS 杀死）都会
    导致管道断开，主进程检测到后回退到 Python 求解器。
    """
    try:
        _run_simulation(conn, mesh_dict, config_dict, n_layers, output_dir)
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
) -> None:
    """在子进程中执行完整的多层仿真。"""
    # ── 子进程崩溃诊断 ──
    _crash_log = Path(output_dir) / "cpp_subprocess_crash.log"
    import faulthandler as _fh
    _fh.enable(file=open(str(_crash_log), "a", encoding="utf-8"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from hydrogel_vbd.core.config import SimulationConfig
    from hydrogel_vbd.core.state import MeshState
    from hydrogel_vbd.geometry.layer_activator import LayerActivator
    from hydrogel_vbd.control.field_controller import PIDFieldController
    from hydrogel_vbd.physics.czm import update_czm_states
    from hydrogel_vbd.solver.cpp_adapter import (
        solve_lift_and_relax as cpp_solve_lift_and_relax,
    )
    from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

    # ── 重建对象 ──
    config = SimulationConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # 用必需字段正常构造 MeshState，让 __post_init__ 自动填充默认值，
    # 然后覆盖为实际数据。非 dataclass 字段（如 top_ids）跳过构造函数。
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

    for layer_id in range(n_layers):
        _trace(f"layer_{layer_id}_start")
        conn.send(_LogMsg(text=f"  [C++] 第 {layer_id + 1}/{n_layers} 层 ← VBD 求解"))

        activator.activate_with_inheritance(mesh, layer_id, z_fep=config.z_fep)

        pid_state = pid.update(0.0)
        e_z = pid_state.E_z

        lift_max = 5.0 * config.layer_thickness
        lift_distance = 0.0
        top_ids = getattr(mesh, "top_ids", None)
        v_lift = config.v_lift
        dt = config.dt

        # 安全上限：每层最多 5000 步（防止无限循环）
        MAX_STEPS_PER_LAYER = 5000
        layer_steps = 0

        # ── 求解分支：有提升 vs 无提升 ──
        _has_lift = (v_lift > 0.0 and top_ids is not None and len(top_ids) > 0)
        if not _has_lift:
            from hydrogel_vbd.solver.cpp_adapter import (
                solve_until_stable as cpp_solve_until_stable,
            )
            _trace(f"layer_{layer_id}_solve_start no_lift")
            result = cpp_solve_until_stable(mesh, config, e_z, layer_id)
            step_counter += result.iterations
            # CZM 更新（提升后）
            bottom = mesh.bottom_nodes(layer_id)
            if len(bottom) > 0:
                update_czm_states(
                    mesh, bottom,
                    internal_pull_z=np.full(len(bottom), config.T_max),
                    area=config.node_area, t_max=config.T_max,
                    k_czm=config.K_czm, delta_f=config.delta_f,
                    z_fep=config.z_fep, dt=dt,
                )
        else:
            # 校验 top_ids 合法性
            nV = mesh.vertices.shape[0]
            if np.any(top_ids < 0) or np.any(top_ids >= nV):
                raise ValueError(f"top_ids 包含越界索引 (nV={nV}, min={top_ids.min()}, max={top_ids.max()})")

            _trace(f"layer_{layer_id}_lift_start top_ids={len(top_ids)} v_lift={v_lift} dt={dt} lift_max={lift_max}")
            while lift_distance < lift_max and layer_steps < MAX_STEPS_PER_LAYER:
                layer_steps += 1
                _trace(f"layer_{layer_id}_step_{layer_steps}_pre_call lift={lift_distance:.6e}")

                result = cpp_solve_lift_and_relax(
                    mesh, config, e_z, layer_id, top_ids,
                )
                _trace(f"layer_{layer_id}_step_{layer_steps}_post_call max_dx={result.max_dx:.4e} iters={result.iterations}")
                step_counter += result.iterations
                lift_distance += v_lift * dt

                # ── CZM 状态更新 ──
                bottom = mesh.bottom_nodes(layer_id)
                if len(bottom) > 0:
                    update_czm_states(
                        mesh, bottom,
                        internal_pull_z=np.full(len(bottom), config.T_max),
                        area=config.node_area, t_max=config.T_max,
                        k_czm=config.K_czm, delta_f=config.delta_f,
                        z_fep=config.z_fep, dt=dt,
                    )

                # 全部脱膜则退出提升循环
                if result.all_free:
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
                        title=f"第 {layer_id + 1} 层 — 步 {layer_steps}",
                    ))

                lift_pct = min(int(lift_distance / lift_max * 100), 100)
                conn.send(_ProgressMsg(
                    layer=layer_id + 1, percentage=lift_pct, step=step_counter,
                ))

        _trace(f"layer_{layer_id}_done steps={layer_steps}")
        results.append({
            "layer_id": layer_id,
            "total_steps": layer_steps,
            "final_max_dx": float(result.max_dx),
        })

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
    ):
        self._mesh_dict = mesh_dict
        self._config_dict = config_dict
        self._n_layers = n_layers
        self._output_dir = output_dir
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
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2)
        if self._conn is not None:
            self._conn.close()
