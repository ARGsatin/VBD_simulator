from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hydrogel_vbd.config import SimulationConfig
from hydrogel_vbd.control.field_controller import PIDFieldController, PIDFieldState
from hydrogel_vbd.forces.czm import update_czm_states
from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
from hydrogel_vbd.geometry.layer_activator import LayerActivator
from hydrogel_vbd.geometry.stl_slicer import slice_stl, load_stl
from hydrogel_vbd.io.gcode_exporter import insert_pid_field_commands
from hydrogel_vbd.io.npz_state import save_layer_state
from hydrogel_vbd.io.report_writer import write_metrics_csv
from hydrogel_vbd.io.vtk_writer import write_vtu
from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver
from hydrogel_vbd.state import FieldCommand, LayerResult, MeshState


def create_demo_mesh(layers: int) -> MeshState:
    mesh, _ = ConformalMeshPipeline.create_demo(layers=layers, layer_thickness=0.05)
    return mesh


def run_from_stl(
    stl_path: str | Path,
    layer_height: float = 0.05,
    quality: float = 1.0,
    output: str | Path = "outputs/stl_sim",
    config: SimulationConfig | None = None,
) -> list[LayerResult]:
    """Run the full print-simulation loop from an STL file.

    Steps
    -----
    1. Slice the STL (for reporting / visualisation).
    2. Build a conformal layered tet mesh via TetGen.
    3. Run the per-layer activate → solve → evaluate → compensate loop.
    """
    output_dir = Path(output)
    for sub in ("states", "vtk", "reports", "gcode", "slices"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    config = config or SimulationConfig(layer_thickness=layer_height)

    # ---------- 1. STL slicing (optional preview) ----------
    mesh_orig = load_stl(stl_path)
    z_min = float(mesh_orig.bounds[0][2])
    z_max = float(mesh_orig.bounds[1][2])
    num_layers = max(1, int((z_max - z_min) / layer_height))
    print(f"Model Z range: [{z_min:.3f}, {z_max:.3f}]  →  {num_layers} layers")

    # Write slice images to disk (reuse the existing slicer)
    slices = slice_stl(stl_path, layer_height, z_min, z_max)
    print(f"  Generated {len(slices)} slice contours")

    # ---------- 2. Conformal layered tet mesh ----------
    print("Building conformal tet mesh …")
    mesh, _ = ConformalMeshPipeline.from_stl(
        stl_path, layer_height=layer_height, config=config, quality=quality,
    )
    target_vertices = mesh.ideal_vertices.copy()
    print(f"  Vertices: {len(mesh.vertices)}, Tets: {len(mesh.tets)}")

    # ---------- 3. Simulation loop ----------
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
            f"  Layer {layer_id:3d}: err_avg={err_avg:.3e}  "
            f"E_z={pid_state.E_z:.3f}  "
            f"steps={solve_result.stable_steps}"
        )

    # ---------- 4. Reports ----------
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

    # dummy G-code for demonstration
    source_gcode = "".join(
        f";LAYER: {lid}\nG1 Z{lid * layer_height:.3f}\n" for lid in range(num_layers)
    )
    compensated = insert_pid_field_commands(source_gcode, commands_by_layer)
    (output_dir / "gcode" / "compensated_print.gcode").write_text(
        compensated, encoding="utf-8"
    )

    print(f"\nDone — results in {output_dir.resolve()}")
    return results


def _command_json(layer_id: int, command: PIDFieldState) -> dict:
    return {
        "layer_id": layer_id,
        "E_z": float(command.E_z),
        "err_avg": float(command.err_avg),
        "PID_integral": float(command.PID_integral),
        "prev_error": float(command.prev_error),
        "delta_E": float(command.delta_E),
    }


def run_demo(layers: int = 3, output: str | Path = "outputs/demo") -> list[LayerResult]:
    output_dir = Path(output)
    states_dir = output_dir / "states"
    vtk_dir = output_dir / "vtk"
    reports_dir = output_dir / "reports"
    gcode_dir = output_dir / "gcode"
    for path in (states_dir, vtk_dir, reports_dir, gcode_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(layer_thickness=0.05)
    mesh, _ = ConformalMeshPipeline.create_demo(layers=layers, layer_thickness=config.layer_thickness, config=config)
    target_vertices = mesh.ideal_vertices.copy()
    solver = PythonReferenceVBDSolver(config)
    activator = LayerActivator()
    controller = PIDFieldController(config)

    results: list[LayerResult] = []
    commands_by_layer: dict[int, PIDFieldState] = {}
    for layer_id in range(layers):
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
        err_avg = float(np.mean(target_vertices[bottom, 2] - x_sim[bottom, 2])) if len(bottom) else 0.0
        pid_state = controller.update(err_avg=err_avg)
        max_error = float(np.max(np.linalg.norm(target_vertices - x_sim, axis=1)))
        rms_error = float(np.sqrt(np.mean(np.sum((target_vertices - x_sim) ** 2, axis=1))))
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
            field_command_next=FieldCommand(voltage=np.array([pid_state.E_z]), electrode_ids=["E_z"]),
            max_deformation=max_error,
            rms_error=rms_error,
            success=bool(max_error < 2.0),
        )
        results.append(result)
        commands_by_layer[layer_id] = pid_state
        save_layer_state(states_dir / f"layer_{layer_id:04d}.npz", result)
        write_vtu(vtk_dir / f"layer_{layer_id:04d}.vtu", mesh, point_data={"active": mesh.active_vertex_mask.astype(float)})

    write_metrics_csv(reports_dir / "error_metrics.csv", results)
    command_payload = {"layers": [_command_json(layer_id, command) for layer_id, command in commands_by_layer.items()]}
    (output_dir / "simulation_field_commands.json").write_text(json.dumps(command_payload, indent=2), encoding="utf-8")

    source_gcode = "".join(f";LAYER: {layer_id}\nG1 Z{layer_id * 0.05:.3f}\n" for layer_id in range(layers))
    compensated = insert_pid_field_commands(source_gcode, commands_by_layer)
    (gcode_dir / "compensated_print.gcode").write_text(compensated, encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hydrogel VBD MVP demo loop.")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--stl", type=str, default=None,
                        help="STL file path (use STL pipeline instead of demo mesh)")
    parser.add_argument("--layer-height", type=float, default=0.05,
                        help="Layer thickness (used with --stl, default 0.05)")
    parser.add_argument("--quality", type=float, default=1.0,
                        help="TetGen mesh quality (used with --stl, default 1.0)")
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
