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
import json
from pathlib import Path

import numpy as np

from typing import Any

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.control.field_controller import PIDFieldController, PIDFieldState
from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
from hydrogel_vbd.geometry.layer_activator import LayerActivator
from hydrogel_vbd.geometry.stl_slicer import load_stl, slice_stl
from hydrogel_vbd.io.gcode_exporter import insert_pid_field_commands
from hydrogel_vbd.io.npz_state import save_layer_state
from hydrogel_vbd.io.report_writer import write_metrics_csv
from hydrogel_vbd.io.vtk_writer import write_vtu
from hydrogel_vbd.physics.czm import update_czm_states
from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver
from hydrogel_vbd.core.state import FieldCommand, LayerResult, MeshState


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


def _command_json(layer_id: int, command: PIDFieldState) -> dict:
    """将 PID 控制器状态序列化为 JSON 友好的字典。

    用于生成 `simulation_field_commands.json` 回放文件，
    供后续离线分析或 G-code 补偿使用。

    Parameters
    ----------
    layer_id : int
        当前层编号（从 0 开始）。
    command : PIDFieldState
        PID 控制器的当前状态快照。

    Returns
    -------
    dict
        含层号、电场强度、误差、PID 积分项等信息的字典。
    """
    return {
        "layer_id": layer_id,
        "E_z": float(command.E_z),
        "err_avg": float(command.err_avg),
        "PID_integral": float(command.PID_integral),
        "prev_error": float(command.prev_error),
        "delta_E": float(command.delta_E),
    }


# ---------------------------------------------------------------------------
# 主仿真入口
# ---------------------------------------------------------------------------

def run_demo(
    layers: int = 3,
    output: str | Path = "outputs/demo",
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
    # 2a. 生成多层共形四面体网格
    mesh, _ = ConformalMeshPipeline.create_demo(
        layers=layers,
        layer_thickness=config.layer_thickness,
        config=config,
    )
    # 保存目标（理想）形状作为误差参照
    target_vertices = mesh.ideal_vertices.copy()
    # 2b. Chebyshev 半隐式 VBD 求解器
    solver = PythonReferenceVBDSolver(config)
    # 2c. 逐层激活器（保形继承 + 防穿透）
    activator = LayerActivator()
    # 2d. PID 电场控制器
    controller = PIDFieldController(config)

    # ── 3. 逐层仿真循环 ──
    results: list[LayerResult] = []
    commands_by_layer: dict[int, PIDFieldState] = {}

    for layer_id in range(layers):
        # ── 3a. 激活当前层 ──
        # 激活新层节点：标记为 active，初始化速度，
        # 并对离型膜（FEP）平面附近的节点进行防穿透处理
        activator.activate_with_inheritance(mesh, layer_id, z_fep=config.z_fep)

        # 获取当前层的底部表面节点（用于 CZM 和误差评估）
        bottom = mesh.bottom_nodes(layer_id)

        # ── 3b. 更新 CZM 内聚力损伤状态 ──
        # 对底部界面节点评估剥离应力，更新 FIXED→DAMAGING→FREE 状态机
        update_czm_states(
            mesh,
            bottom,
            internal_pull_z=np.full(len(bottom), config.T_max * 1.05),
            area=config.node_area,
            t_max=config.T_max,
            k_czm=config.K_czm,
            delta_f=config.delta_f,
            z_fep=config.z_fep,
            dt=config.dt,
        )

        # ── 3c. 执行 VBD 求解 ──
        # 根据是否启用平台提升选择不同的求解模式
        if config.v_lift > 0 and np.any(mesh.is_top_fixed):
            # 含平台提升的求解（顶部固定节点随平台上升）
            lifting_top = np.flatnonzero(mesh.is_top_fixed)
            solve_result = solver.solve_with_lift(
                mesh,
                layer_id=layer_id,
                e_z=controller.E_z,
                lifting_top=lifting_top,
            )
        else:
            # 标准求解（无提升，仅电场作用）
            solve_result = solver.solve_until_stable(
                mesh, layer_id=layer_id, e_z=controller.E_z
            )

        # ── 3d. 计算形状误差 ──
        # 提取求解后的顶点和速度
        x_sim, v_sim = solve_result.x, solve_result.v

        # 底部节点平均垂度（sag）：理想 Z - 实际 Z
        # 正值表示下垂（需要增大电场），负值表示过度提升
        if len(bottom):
            err_avg = float(np.mean(target_vertices[bottom, 2] - x_sim[bottom, 2]))
        else:
            err_avg = 0.0

        # ── 3e. PID 控制更新 ──
        # 根据当前误差计算下一层的电场强度 E_z
        pid_state = controller.update(err_avg=err_avg)

        # 计算额外的误差指标
        max_error = float(np.max(np.linalg.norm(target_vertices - x_sim, axis=1)))
        rms_error = float(
            np.sqrt(np.mean(np.sum((target_vertices - x_sim) ** 2, axis=1)))
        )

        # ── 3f. 组装结果 ──
        metrics = {
            "err_avg": err_avg,
            "E_z": pid_state.E_z,
            "PID_integral": pid_state.PID_integral,
            "kinetic_energy": solve_result.kinetic_energy,
            "stable_steps": float(solve_result.stable_steps),
            "max_dx": solve_result.max_dx,
            "all_free": float(solve_result.all_free),
            "max_error": max_error,
        }

        result = LayerResult(
            layer_id=layer_id,
            x_sim=x_sim.copy(),
            v_sim=v_sim.copy(),
            error_metrics=metrics,
            field_command_next=FieldCommand(
                voltage=np.array([pid_state.E_z]),
                electrode_ids=["E_z"],
            ),
            max_deformation=max_error,
            rms_error=rms_error,
            success=bool(max_error < 2.0),  # 最大变形 < 2m 视为成功
        )
        results.append(result)

        # ── 3g. 保存当前层状态 ──
        commands_by_layer[layer_id] = pid_state
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
# STL 流水线入口
# ---------------------------------------------------------------------------

def run_from_stl(
    stl_path: str | Path,
    layer_height: float = 5e-5,
    quality: float = 1.0,
    output: str | Path = "outputs/stl_sim",
    config: SimulationConfig | None = None,
) -> list[LayerResult]:
    """从 STL 文件运行完整打印仿真流水线。

    步骤：
    1. 切片 STL（用于预览 / 报告）
    2. 通过 TetGen 构建共形分层四面体网格
    3. 逐层执行：激活 → 求解 → 评估 → 补偿

    Parameters
    ----------
    stl_path : str | Path
        输入 STL 文件路径。
    layer_height : float
        打印层厚（与 STL 同单位），默认 5e-5 (0.05 mm)。
    quality : float
        TetGen 网格细化因子 (0.1 … 5.0，默认 1.0)。
    output : str | Path
        输出根目录路径。
    config : SimulationConfig or None
        仿真配置，None 则使用默认值。

    Returns
    -------
    list[LayerResult]
        每层的求解结果列表。
    """
    output_dir = Path(output)
    for sub in ("states", "vtk", "reports", "gcode", "slices"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    config = config or SimulationConfig(layer_thickness=layer_height)

    # ── 1. STL 切片 ──
    mesh_orig = load_stl(stl_path)
    z_min = float(mesh_orig.bounds[0][2])
    z_max = float(mesh_orig.bounds[1][2])
    num_layers = max(1, int((z_max - z_min) / layer_height))
    print(f"Model Z range: [{z_min:.6f}, {z_max:.6f}]  →  {num_layers} layers")

    slices = slice_stl(stl_path, layer_height, z_min, z_max)
    print(f"  Generated {len(slices)} slice contours")

    # ── 2. 共形分层四面体网格 ──
    print("Building conformal tet mesh …")
    mesh, _ = ConformalMeshPipeline.from_stl(
        stl_path, layer_height=layer_height, config=config, quality=quality,
    )
    target_vertices = mesh.ideal_vertices.copy()
    print(f"  Vertices: {len(mesh.vertices)}, Tets: {len(mesh.tets)}")

    # ── 3. 逐层仿真循环 ──
    solver = PythonReferenceVBDSolver(config)
    activator = LayerActivator()
    controller = PIDFieldController(config)

    results: list[LayerResult] = []
    commands_by_layer: dict[int, Any] = {}

    for layer_id in range(num_layers):
        activator.activate_with_inheritance(mesh, layer_id, z_fep=config.z_fep)

        bottom = mesh.bottom_nodes(layer_id)
        update_czm_states(
            mesh,
            bottom,
            internal_pull_z=np.full(len(bottom), config.T_max * 1.05),
            area=config.node_area,
            t_max=config.T_max,
            k_czm=config.K_czm,
            delta_f=config.delta_f,
            z_fep=config.z_fep,
            dt=config.dt,
        )

        solve_result = solver.solve_until_stable(mesh, layer_id=layer_id, e_z=controller.E_z)
        x_sim, v_sim = solve_result.x, solve_result.v

        err_avg = (
            float(np.mean(target_vertices[bottom, 2] - x_sim[bottom, 2]))
            if len(bottom)
            else 0.0
        )
        pid_state = controller.update(err_avg=err_avg)
        max_error = float(np.max(np.linalg.norm(target_vertices - x_sim, axis=1)))
        rms_error = float(
            np.sqrt(np.mean(np.sum((target_vertices - x_sim) ** 2, axis=1)))
        )
        metrics = {
            "err_avg": err_avg,
            "E_z": pid_state.E_z,
            "PID_integral": pid_state.PID_integral,
            "kinetic_energy": solve_result.kinetic_energy,
            "stable_steps": float(solve_result.stable_steps),
            "max_dx": solve_result.max_dx,
            "all_free": float(solve_result.all_free),
            "max_error": max_error,
        }

        result = LayerResult(
            layer_id=layer_id,
            x_sim=x_sim.copy(),
            v_sim=v_sim.copy(),
            error_metrics=metrics,
            field_command_next=FieldCommand(
                voltage=np.array([pid_state.E_z]), electrode_ids=["E_z"]
            ),
            max_deformation=max_error,
            rms_error=rms_error,
            success=bool(max_error < 2.0),
        )
        results.append(result)
        commands_by_layer[layer_id] = pid_state

        save_layer_state(output_dir / "states" / f"layer_{layer_id:04d}.npz", result)
        write_vtu(
            output_dir / "vtk" / f"layer_{layer_id:04d}.vtu",
            mesh,
            point_data={"active": mesh.active_vertex_mask.astype(float)},
        )

        print(
            f"  Layer {layer_id:3d}: err_avg={err_avg:.6e}  "
            f"E_z={pid_state.E_z:.6f}  "
            f"steps={solve_result.stable_steps}"
        )

    # ── 4. 输出报告 ──
    write_metrics_csv(output_dir / "reports" / "error_metrics.csv", results)
    command_payload = {
        "layers": [
            {
                "layer_id": lid,
                "E_z": float(c.E_z),
                "err_avg": float(c.err_avg),
                "PID_integral": float(c.PID_integral),
                "prev_error": float(c.prev_error),
                "delta_E": float(c.delta_E),
            }
            for lid, c in commands_by_layer.items()
        ]
    }
    (output_dir / "simulation_field_commands.json").write_text(
        json.dumps(command_payload, indent=2), encoding="utf-8"
    )

    source_gcode = "".join(
        f";LAYER: {lid}\nG1 Z{lid * layer_height:.6f}\n" for lid in range(num_layers)
    )
    compensated = insert_pid_field_commands(source_gcode, commands_by_layer)
    (output_dir / "gcode" / "compensated_print.gcode").write_text(
        compensated, encoding="utf-8"
    )

    print(f"\nDone — results in {output_dir.resolve()}")
    return results


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main() -> None:
    """命令行入口：解析参数并运行仿真演示。"""
    parser = argparse.ArgumentParser(
        description="运行水凝胶 VBD 仿真演示循环。"
    )
    parser.add_argument("--layers", type=int, default=3, help="合成柱体仿真层数")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"), help="输出目录路径")
    parser.add_argument("--stl", type=str, default=None, help="STL 文件路径（使用 STL 流水线）")
    parser.add_argument("--layer-height", type=float, default=5e-5, help="打印层厚（与 STL 同单位）")
    parser.add_argument("--quality", type=float, default=1.0, help="TetGen 网格质量因子 (0.1-5.0)")
    args = parser.parse_args()

    if args.stl:
        run_from_stl(
            stl_path=args.stl,
            layer_height=args.layer_height,
            quality=args.quality,
            output=args.output,
        )
    else:
        run_demo(layers=args.layers, output=args.output)


if __name__ == "__main__":
    main()
