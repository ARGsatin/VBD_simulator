from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hydrogel_vbd.control.field_controller import FieldController
from hydrogel_vbd.evaluation.shape_error import compare_shapes
from hydrogel_vbd.forces.aggregate import aggregate_forces
from hydrogel_vbd.forces.electric import ElectricForceModel
from hydrogel_vbd.forces.fluid_drag import fluid_drag_force
from hydrogel_vbd.forces.gravity import gravity_force
from hydrogel_vbd.forces.peel import peel_force
from hydrogel_vbd.forces.surface_tension import surface_tension_force
from hydrogel_vbd.geometry.layer_activator import LayerActivator
from hydrogel_vbd.io.gcode_exporter import insert_field_commands
from hydrogel_vbd.io.npz_state import save_layer_state
from hydrogel_vbd.io.report_writer import write_metrics_csv
from hydrogel_vbd.io.vtk_writer import write_vtu
from hydrogel_vbd.solver.constraints import fixed_z_constraints
from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver
from hydrogel_vbd.state import FieldCommand, LayerResult, MeshState


def create_demo_mesh(layers: int) -> MeshState:
    vertices: list[list[float]] = []
    layer_ids: list[int] = []
    for layer in range(layers):
        z = float(layer)
        vertices.extend(
            [
                [0.0, 0.0, z],
                [1.0, 0.0, z],
                [0.0, 1.0, z],
                [1.0, 1.0, z + 0.15 * np.sin(layer)],
            ]
        )
        layer_ids.extend([layer, layer, layer, layer])
    tets: list[list[int]] = []
    tet_layers: list[int] = []
    for layer in range(layers):
        base = layer * 4
        tets.append([base, base + 1, base + 2, base + 3])
        tet_layers.append(layer)
    mesh = MeshState(
        vertices=np.asarray(vertices, dtype=float),
        tets=np.asarray(tets, dtype=int),
        layer_id_per_vertex=np.asarray(layer_ids, dtype=int),
        layer_id_per_tet=np.asarray(tet_layers, dtype=int),
    )
    return mesh


def _command_json(layer_id: int, command: FieldCommand) -> dict:
    return {
        "layer_id": layer_id,
        "electrode_ids": command.electrode_ids,
        "voltage": [float(value) for value in command.voltage],
        "duration": float(command.duration),
    }


def run_demo(layers: int = 3, output: str | Path = "outputs/demo") -> list[LayerResult]:
    output_dir = Path(output)
    states_dir = output_dir / "states"
    vtk_dir = output_dir / "vtk"
    reports_dir = output_dir / "reports"
    gcode_dir = output_dir / "gcode"
    for path in (states_dir, vtk_dir, reports_dir, gcode_dir):
        path.mkdir(parents=True, exist_ok=True)

    mesh = create_demo_mesh(layers)
    target_vertices = mesh.vertices.copy()
    solver = PythonReferenceVBDSolver(damping=0.03)
    activator = LayerActivator()
    field_command = FieldCommand(voltage=np.zeros(2), duration=0.8, electrode_ids=["left", "right"])
    electric_model = ElectricForceModel(alpha=0.12, direction=(0.0, 0.0, 1.0))
    mapping = np.zeros((mesh.vertices.shape[0] * 3, 2), dtype=float)
    mapping[2::3, 0] = 0.12
    mapping[2::3, 1] = 0.12
    controller = FieldController(
        force_mapping=mapping,
        kp=0.8,
        kd=0.1,
        regularization=0.01,
        voltage_limits=(-5.0, 5.0),
        electrode_ids=["left", "right"],
    )

    results: list[LayerResult] = []
    commands_by_layer: dict[int, FieldCommand] = {}
    for layer_id in range(layers):
        activator.activate(mesh, layer_id)
        constraints = fixed_z_constraints(mesh, z_value=0.0)
        gravity = gravity_force(mesh, density=0.015, g=(0.0, 0.0, -9.81))
        peel = peel_force(mesh, pressure=0.05, normal=(0.0, 0.0, 1.0), vertex_area=1.0)
        electric = electric_model.compute(mesh, field_command)
        fluid = fluid_drag_force(mesh, coefficient=0.01)
        surface = surface_tension_force(mesh, coefficient=0.0)
        force_state = aggregate_forces(gravity, peel, fluid, surface, electric)
        x_sim, v_sim = solver.step(mesh, force_state.total, constraints, dt=0.05, substeps=2, iterations=4)
        metrics = compare_shapes(x_sim, target_vertices)
        nodal_error = (target_vertices - x_sim).reshape(-1)
        next_command = controller.compute(nodal_error, previous_command=field_command)
        next_command.duration = 0.8

        result = LayerResult(
            layer_id=layer_id,
            x_sim=x_sim.copy(),
            v_sim=v_sim.copy(),
            error_metrics=metrics,
            field_command_next=next_command,
            max_deformation=metrics["max_error"],
            rms_error=metrics["rms_error"],
            success=bool(metrics["max_error"] < 2.0),
        )
        results.append(result)
        commands_by_layer[layer_id] = next_command
        save_layer_state(states_dir / f"layer_{layer_id:04d}.npz", result)
        write_vtu(vtk_dir / f"layer_{layer_id:04d}.vtu", mesh, point_data={"active": mesh.active_vertex_mask.astype(float)})
        field_command = next_command

    write_metrics_csv(reports_dir / "error_metrics.csv", results)
    command_payload = {"layers": [_command_json(result.layer_id, result.field_command_next) for result in results]}
    (output_dir / "simulation_field_commands.json").write_text(json.dumps(command_payload, indent=2), encoding="utf-8")

    source_gcode = "".join(f";LAYER: {layer_id}\nG1 Z{layer_id * 0.05:.3f}\n" for layer_id in range(layers))
    compensated = insert_field_commands(source_gcode, commands_by_layer)
    (gcode_dir / "compensated_print.gcode").write_text(compensated, encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hydrogel VBD MVP demo loop.")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    run_demo(layers=args.layers, output=args.output)


if __name__ == "__main__":
    main()
