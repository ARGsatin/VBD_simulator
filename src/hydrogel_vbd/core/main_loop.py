# -*- coding: utf-8 -*-
"""
主仿真循环模块
================

本模块实现了水凝胶 DLP VBD 仿真的**主循环逻辑**，是框架的"导演"模块。
它协调以下组件完成逐层仿真：

1. **网格生成** (`ConformalMeshPipeline`) — 构建多层共形四面体网格
2. **逐层激活** (`LayerActivator`) — 激活新层并继承保形变形
3. **CZM 状态更新** — 更新层间界面节点的内聚力损伤状态
4. **VBD 求解** (`PythonReferenceVBDSolver`) — 执行 Chebyshev 半隐式迭代
5. **PID 控制** (`PIDFieldController`) — 根据底部垂度自动调节电场
6. **结果输出** — NPZ 状态快照、VTU 可视化、CSV 报告、G-code 补偿

使用方式
--------
.. code-block:: python

    from hydrogel_vbd.main_loop import run_demo
    results = run_demo(layers=5, output="outputs/my_sim")

.. code-block:: bash

    python -m hydrogel_vbd.main_loop --layers 5 --output outputs/my_sim
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.control.field_controller import (
    BottomZFieldController,
    BottomZFieldState,
    PIDFieldController,
    PIDFieldState,
)
from hydrogel_vbd.physics.czm import update_czm_states
from hydrogel_vbd.physics.local_terms import build_local_physics_terms
from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
from hydrogel_vbd.geometry.layer_activator import LayerActivator
from hydrogel_vbd.io.gcode_exporter import insert_pid_field_commands
from hydrogel_vbd.io.npz_state import save_layer_state
from hydrogel_vbd.io.report_writer import write_metrics_csv
from hydrogel_vbd.io.vtk_writer import write_vtu
from hydrogel_vbd.solver.vbd_solver import (
    PythonReferenceVBDSolver,
    _normal_pull_from_terms,
)
from hydrogel_vbd.core.state import FieldCommand, LayerResult, MeshState


FIELD_CONTROL_SCALAR_PID = "scalar_pid"
FIELD_CONTROL_BOTTOM_Z = "bottom_z"
FIELD_CONTROL_BOTTOM_Z_GUARDED = "bottom_z_guarded"
FIELD_CONTROL_MODES = {
    FIELD_CONTROL_SCALAR_PID,
    FIELD_CONTROL_BOTTOM_Z,
    FIELD_CONTROL_BOTTOM_Z_GUARDED,
}
DEFAULT_RMS_GUARD_TOLERANCE = 0.01


@dataclass
class _LayerRunData:
    """一次单层求解后的可复用结果快照。"""

    x_sim: np.ndarray
    v_sim: np.ndarray
    bottom: np.ndarray
    err_avg: float
    bottom_z_mean_error: float
    bottom_z_max_error: float
    max_error: float
    rms_error: float
    base_metrics: dict[str, float]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def create_demo_mesh(layers: int) -> MeshState:
    """快速创建演示用多层四面体网格。

    此函数是 `ConformalMeshPipeline.create_demo` 的便捷包装，
    用于测试和演示场景。

    Parameters
    ----------
    layers : int
        打印层数（每层对应一个独立的四面体区域）。

    Returns
    -------
    MeshState
        包含全局共形网格、理想顶点、掩码等完整状态的对象。
    """
    mesh, _ = ConformalMeshPipeline.create_demo(layers=layers, layer_thickness=5e-5)
    return mesh


def _command_json(layer_id: int, command: PIDFieldState | BottomZFieldState) -> dict:
    """将 PID 控制器状态序列化为 JSON 友好的字典。

    用于生成 `simulation_field_commands.json` 回放文件，
    供后续离线分析或 G-code 补偿使用。

    Parameters
    ----------
    layer_id : int
        当前层编号（从 0 开始）。
    command : PIDFieldState or BottomZFieldState
        控制器的当前状态快照。

    Returns
    -------
    dict
        含层号、电场强度、误差、PID 积分项等信息的字典。
    """
    payload = {
        "layer_id": layer_id,
        "E_z": float(command.E_z),
        "err_avg": float(command.err_avg),
        "PID_integral": float(command.PID_integral),
        "prev_error": float(command.prev_error),
        "delta_E": float(command.delta_E),
    }
    if hasattr(command, "bottom_z_mean_error"):
        payload.update(
            {
                "bottom_z_mean_error": float(command.bottom_z_mean_error),
                "bottom_z_max_error": float(command.bottom_z_max_error),
                "bottom_z_E_z": float(command.E_z),
            }
        )
    return payload


def _layer_contact_z(config: SimulationConfig, layer_id: int) -> float:
    """Return the layer-local FEP contact plane in world coordinates."""
    return float(config.z_fep)


def _validate_field_control_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in FIELD_CONTROL_MODES:
        allowed = ", ".join(sorted(FIELD_CONTROL_MODES))
        raise ValueError(f"field_control_mode must be one of: {allowed}")
    return normalized


def _select_rms_guarded_result(
    baseline: LayerResult,
    candidate: LayerResult,
    *,
    tolerance: float = DEFAULT_RMS_GUARD_TOLERANCE,
) -> tuple[LayerResult, bool]:
    """Select candidate only when its global RMS is within the tolerance band."""
    limit = float(baseline.rms_error) * (1.0 + float(tolerance)) + 1e-12
    passed = float(candidate.rms_error) <= limit
    return (candidate if passed else baseline), passed


def _simulate_layer_once(
    mesh: MeshState,
    config: SimulationConfig,
    layer_id: int,
    target_vertices: np.ndarray,
    e_z: float,
) -> _LayerRunData:
    """Run one layer on ``mesh`` in place using a scalar ``e_z`` command."""
    layer_z_fep = _layer_contact_z(config, layer_id)
    layer_config = replace(config, z_fep=layer_z_fep)
    solver = PythonReferenceVBDSolver(layer_config)
    activator = LayerActivator()
    activator.activate_with_inheritance(mesh, layer_id, z_fep=layer_z_fep)

    bottom = mesh.bottom_nodes(layer_id)
    x_before_solve = mesh.vertices.copy()
    did_lift = config.v_lift > 0 and np.any(mesh.is_top_fixed)
    if did_lift:
        lifting_top = np.flatnonzero(mesh.is_top_fixed & mesh.active_vertex_mask)
        solve_result = solver.solve_with_lift(
            mesh,
            layer_id=layer_id,
            e_z=e_z,
            lifting_top=lifting_top,
        )
    else:
        solve_result = solver.solve_until_stable(
            mesh, layer_id=layer_id, e_z=e_z
        )
        if len(bottom):
            terms_after = build_local_physics_terms(
                mesh, layer_config, e_z=e_z, x_prev=x_before_solve
            )
            update_czm_states(
                mesh,
                bottom,
                internal_pull_z=_normal_pull_from_terms(
                    terms_after.force, bottom
                ),
                area=layer_config.node_area,
                t_max=layer_config.T_max,
                k_czm=layer_config.K_czm,
                delta_f=layer_config.delta_f,
                z_fep=layer_config.z_fep,
                dt=layer_config.dt,
            )

    x_sim, v_sim = solve_result.x, solve_result.v
    if len(bottom):
        bottom_z_error = target_vertices[bottom, 2] - x_sim[bottom, 2]
        err_avg = float(np.mean(bottom_z_error))
        bottom_z_max_error = float(np.max(bottom_z_error))
    else:
        err_avg = 0.0
        bottom_z_max_error = 0.0

    max_error = float(np.max(np.linalg.norm(target_vertices - x_sim, axis=1)))
    rms_error = float(
        np.sqrt(np.mean(np.sum((target_vertices - x_sim) ** 2, axis=1)))
    )
    return _LayerRunData(
        x_sim=x_sim.copy(),
        v_sim=v_sim.copy(),
        bottom=bottom.copy(),
        err_avg=err_avg,
        bottom_z_mean_error=err_avg,
        bottom_z_max_error=bottom_z_max_error,
        max_error=max_error,
        rms_error=rms_error,
        base_metrics={
            "err_avg": err_avg,
            "kinetic_energy": solve_result.kinetic_energy,
            "stable_steps": float(solve_result.stable_steps),
            "max_dx": solve_result.max_dx,
            "all_free": float(solve_result.all_free),
            "max_error": max_error,
        },
    )


def _build_layer_result(
    layer_id: int,
    data: _LayerRunData,
    command: PIDFieldState | BottomZFieldState,
    *,
    requested_mode: str,
    effective_mode: str,
    rms_guard_passed: bool | None = None,
    rms_guard_baseline: float | None = None,
    rms_guard_candidate: float | None = None,
    bottom_state: BottomZFieldState | None = None,
) -> LayerResult:
    """Convert a layer run snapshot and chosen command into a public result."""
    metrics: dict[str, float | str] = dict(data.base_metrics)
    metrics.update(
        {
            "E_z": float(command.E_z),
            "PID_integral": float(command.PID_integral),
            "field_control_requested": requested_mode,
            "field_control_effective": effective_mode,
            "rms_guard_passed": (
                float(rms_guard_passed)
                if rms_guard_passed is not None
                else float("nan")
            ),
            "rms_guard_baseline": (
                float(rms_guard_baseline)
                if rms_guard_baseline is not None
                else float("nan")
            ),
            "rms_guard_candidate": (
                float(rms_guard_candidate)
                if rms_guard_candidate is not None
                else float("nan")
            ),
            "bottom_z_mean_error": (
                float(bottom_state.bottom_z_mean_error)
                if bottom_state is not None
                else data.bottom_z_mean_error
            ),
            "bottom_z_max_error": (
                float(bottom_state.bottom_z_max_error)
                if bottom_state is not None
                else data.bottom_z_max_error
            ),
            "bottom_z_E_z": (
                float(bottom_state.E_z)
                if bottom_state is not None
                else 0.0
            ),
        }
    )
    return LayerResult(
        layer_id=layer_id,
        x_sim=data.x_sim,
        v_sim=data.v_sim,
        error_metrics=metrics,
        field_command_next=FieldCommand(
            voltage=np.array([command.E_z], dtype=float),
            electrode_ids=["E_z"],
        ),
        max_deformation=data.max_error,
        rms_error=data.rms_error,
        success=bool(data.max_error < 2.0),
    )


# ---------------------------------------------------------------------------
# 主仿真入口
# ---------------------------------------------------------------------------

def run_demo(
    layers: int = 3,
    output: str | Path = "outputs/demo",
    field_control_mode: str | None = None,
) -> list[LayerResult]:
    """运行完整的水凝胶 DLP VBD 仿真演示。

    这是框架的主要入口函数，执行以下步骤：

    1. 创建输出目录结构（states / vtk / reports / gcode）
    2. 加载配置，生成多层共形四面体网格
    3. 对每一层：
       a. 激活该层（继承上一层的变形几何）
       b. 更新层间 CZM 损伤状态
       c. 执行 VBD 求解（含或不含平台提升）
       d. 计算底部节点平均垂度（sag）
       e. PID 控制器更新电场强度
       f. 保存 NPZ 状态快照和 VTU 可视化
    4. 输出 CSV 指标报告、JSON 回放文件、补偿 G-code

    Parameters
    ----------
    layers : int
        仿真层数，默认 3 层。
    output : str | Path
        输出根目录路径，所有结果文件将保存在其子目录中。
    field_control_mode : str or None
        电场控制模式：``scalar_pid``、``bottom_z`` 或
        ``bottom_z_guarded``。None 时使用配置默认值。

    Returns
    -------
    list[LayerResult]
        每层的求解结果列表，包含变形顶点、速度、误差指标、电场指令等。
    """
    # ── 1. 创建输出目录结构 ──
    output_dir = Path(output)
    states_dir = output_dir / "states"    # NPZ 状态快照
    vtk_dir = output_dir / "vtk"          # VTU 可视化文件
    reports_dir = output_dir / "reports"  # CSV/JSON 报告
    gcode_dir = output_dir / "gcode"      # 补偿 G-code
    for path in (states_dir, vtk_dir, reports_dir, gcode_dir):
        path.mkdir(parents=True, exist_ok=True)

    # ── 2. 初始化各组件 ──
    config = SimulationConfig(layer_thickness=5e-5)
    requested_mode = _validate_field_control_mode(
        field_control_mode
        if field_control_mode is not None
        else getattr(config, "field_control_mode", FIELD_CONTROL_SCALAR_PID)
    )
    rms_guard_tolerance = float(
        getattr(config, "rms_guard_tolerance", DEFAULT_RMS_GUARD_TOLERANCE)
    )
    # 2a. 生成多层共形四面体网格
    mesh, _ = ConformalMeshPipeline.create_demo(
        layers=layers,
        layer_thickness=config.layer_thickness,
        config=config,
    )
    # 保存目标（理想）形状作为误差参照
    target_vertices = mesh.ideal_vertices.copy()
    # 2b. 电场控制器
    scalar_controller = PIDFieldController(config)
    bottom_controller = BottomZFieldController(config)

    # ── 3. 逐层仿真循环 ──
    results: list[LayerResult] = []
    commands_by_layer: dict[int, PIDFieldState | BottomZFieldState] = {}

    for layer_id in range(layers):
        if requested_mode == FIELD_CONTROL_SCALAR_PID:
            data = _simulate_layer_once(
                mesh, config, layer_id, target_vertices, scalar_controller.E_z
            )
            pid_state = scalar_controller.update(err_avg=data.err_avg)
            bottom_state = bottom_controller.update(
                bottom_nodes=data.bottom,
                target_vertices=target_vertices,
                simulated_vertices=data.x_sim,
            )
            command = pid_state
            result = _build_layer_result(
                layer_id,
                data,
                command,
                requested_mode=requested_mode,
                effective_mode=FIELD_CONTROL_SCALAR_PID,
                bottom_state=bottom_state,
            )

        elif requested_mode == FIELD_CONTROL_BOTTOM_Z:
            data = _simulate_layer_once(
                mesh, config, layer_id, target_vertices, bottom_controller.E_z
            )
            scalar_controller.update(err_avg=data.err_avg)
            bottom_state = bottom_controller.update(
                bottom_nodes=data.bottom,
                target_vertices=target_vertices,
                simulated_vertices=data.x_sim,
            )
            command = bottom_state
            result = _build_layer_result(
                layer_id,
                data,
                command,
                requested_mode=requested_mode,
                effective_mode=FIELD_CONTROL_BOTTOM_Z,
                bottom_state=bottom_state,
            )

        else:
            pre_layer_mesh = copy.deepcopy(mesh)
            baseline_mesh = copy.deepcopy(pre_layer_mesh)
            candidate_mesh = copy.deepcopy(pre_layer_mesh)
            baseline_data = _simulate_layer_once(
                baseline_mesh,
                config,
                layer_id,
                target_vertices,
                scalar_controller.E_z,
            )
            candidate_data = _simulate_layer_once(
                candidate_mesh,
                config,
                layer_id,
                target_vertices,
                bottom_controller.E_z,
            )
            baseline_guard = LayerResult(
                layer_id=layer_id,
                x_sim=baseline_data.x_sim,
                v_sim=baseline_data.v_sim,
                error_metrics={},
                field_command_next=FieldCommand(np.array([scalar_controller.E_z])),
                max_deformation=baseline_data.max_error,
                rms_error=baseline_data.rms_error,
                success=True,
            )
            candidate_guard = LayerResult(
                layer_id=layer_id,
                x_sim=candidate_data.x_sim,
                v_sim=candidate_data.v_sim,
                error_metrics={},
                field_command_next=FieldCommand(np.array([bottom_controller.E_z])),
                max_deformation=candidate_data.max_error,
                rms_error=candidate_data.rms_error,
                success=True,
            )
            selected_guard, guard_passed = _select_rms_guarded_result(
                baseline_guard,
                candidate_guard,
                tolerance=rms_guard_tolerance,
            )
            if selected_guard is candidate_guard:
                mesh = candidate_mesh
                data = candidate_data
                effective_mode = FIELD_CONTROL_BOTTOM_Z
            else:
                mesh = baseline_mesh
                data = baseline_data
                effective_mode = FIELD_CONTROL_SCALAR_PID

            pid_state = scalar_controller.update(err_avg=data.err_avg)
            bottom_state = bottom_controller.update(
                bottom_nodes=data.bottom,
                target_vertices=target_vertices,
                simulated_vertices=data.x_sim,
            )
            command = bottom_state if guard_passed else pid_state
            result = _build_layer_result(
                layer_id,
                data,
                command,
                requested_mode=requested_mode,
                effective_mode=effective_mode,
                rms_guard_passed=guard_passed,
                rms_guard_baseline=baseline_data.rms_error,
                rms_guard_candidate=candidate_data.rms_error,
                bottom_state=bottom_state,
            )

        results.append(result)

        # ── 3g. 保存当前层状态 ──
        commands_by_layer[layer_id] = command
        save_layer_state(states_dir / f"layer_{layer_id:04d}.npz", result)
        write_vtu(
            vtk_dir / f"layer_{layer_id:04d}.vtu",
            mesh,
            point_data={"active": mesh.active_vertex_mask.astype(float)},
        )

    # ── 4. 输出汇总报告 ──
    # CSV 误差指标
    write_metrics_csv(reports_dir / "error_metrics.csv", results)

    # JSON 回放文件（含所有层的 PID 状态）
    command_payload = {
        "layers": [
            _command_json(layer_id, command)
            for layer_id, command in commands_by_layer.items()
        ]
    }
    (output_dir / "simulation_field_commands.json").write_text(
        json.dumps(command_payload, indent=2), encoding="utf-8"
    )

    # 补偿 G-code（含 M150 E... 电场指令）
    source_gcode = "".join(
        f";LAYER: {layer_id}\nG1 Z{layer_id * config.layer_thickness:.6f}\n"
        for layer_id in range(layers)
    )
    compensated = insert_pid_field_commands(source_gcode, commands_by_layer)
    (gcode_dir / "compensated_print.gcode").write_text(compensated, encoding="utf-8")

    return results


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main() -> None:
    """命令行入口：解析参数并运行仿真演示。"""
    parser = argparse.ArgumentParser(
        description="运行水凝胶 VBD 仿真演示循环。"
    )
    parser.add_argument("--layers", type=int, default=3, help="仿真层数")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/demo"),
        help="输出目录路径",
    )
    parser.add_argument(
        "--field-control-mode",
        choices=sorted(FIELD_CONTROL_MODES),
        default=None,
        help="电场控制模式",
    )
    args = parser.parse_args()
    run_demo(
        layers=args.layers,
        output=args.output,
        field_control_mode=args.field_control_mode,
    )


if __name__ == "__main__":
    main()
