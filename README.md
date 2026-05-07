# Hydrogel VBD Simulation

This repository contains a Python-first reference framework for a VBD-based electric-field-assisted hydrogel DLP printing simulation.

The current architecture follows `修改点.docx` and `伪代码.txt`:

1. Slice an input STL into 2D cross-section contours (optional preview).
2. Build one global conformal tetrahedral mesh with shared layer-interface nodes — from a synthetic column or from an STL via TetGen.
3. Activate layers with FEP collision handling, inherited deformed geometry, and anti-penetration interpolation.
4. Assemble node-local force and Hessian terms for inertia, hyperelastic placeholder stiffness, damping, CZM softening, fluid suction, and electric lift.
5. Solve with a VBD-style local 3x3 Newton loop over graph-colored vertex batches.
6. Track CZM interface states: `FIXED -> DAMAGING -> FREE`.
7. Evaluate bottom-node average sag and update the PID-controlled electric field `E_z`.
8. Save NPZ state, VTU visualization, CSV reports, JSON replay data, and `M150 E...` G-code commands.

The Python solver is intentionally a reference implementation. The interfaces are shaped so a future C++17/Eigen/OpenMP/pybind11 VBD core or real PLC/TetGen geometry pipeline can replace the Python demo components.

## Configuration

The unified physical and control parameters live in `configs/config.yaml`. It includes the parameters from the pseudocode, including `g`, `rho`, `mu`, `kappa`, `k_d`, `c_shrink`, CZM constants, fluid cutoffs, VBD convergence controls, and PID/electric safety limits.

## Geometry pipeline

Three modules under `hydrogel_vbd.geometry` handle STL-to-mesh preprocessing:

| Module | Responsibility |
|--------|---------------|
| `stl_slicer` | Read an STL, extract 2D cross-section polygons at each layer height via `trimesh.section()`. |
| `tet_mesher` | Generate a linear tetrahedral mesh from an STL via TetGen (`tetgen` Python wrapper). |
| `conformal_pipeline` | `ConformalMeshPipeline.from_stl()` — tet mesh → layer ID assignment → `MeshState` with interface metadata. The static `create_demo()` method produces a synthetic rectangular column for testing. |

The STL entry point is `ConformalMeshPipeline.from_stl(stl_path, layer_height, config, quality)`.
It returns a fully initialised `MeshState` compatible with `LayerActivator` and the VBD solver.

## Simulation loop

The entry points live in `hydrogel_vbd.main_loop`:

- **`run_demo(layers, output)`** — build a synthetic column mesh and simulate `layers` print layers.
- **`run_from_stl(stl_path, layer_height, quality, output, config)`** — slice the STL, build a conformal tet mesh, then run the per-layer activate → solve → compensate loop.

Both write NPZ layer states, VTU visualisations, CSV error metrics, PID field-command JSON, and G-code to the output directory.

## Quickstart

### Demo (synthetic column)

```powershell
python -m pip install -e .
python -m hydrogel_vbd.main_loop --layers 3 --output outputs/demo
```

### From an STL file

```powershell
python -m hydrogel_vbd.main_loop --stl "path/to/model.stl" --layer-height 0.05 --output outputs/stl_sim
```

Optional flags:

- `--quality 1.0` — TetGen mesh refinement factor (0.1 … 5.0; larger = finer).
- `--layer-height 0.05` — print layer thickness (same unit as the STL).

### Run all tests

```powershell
python -m unittest discover -s tests -v
```

Test STL files are provided in `tests/data/` (`demo7.STL`, `长方体.STL`) for integration tests.

## Output structure

```
outputs/<name>/
├── states/             # NPZ files (one per layer)
├── vtk/                # VTU files for ParaView visualisation
├── reports/            # error_metrics.csv
├── slices/             # (STL path only) slice contour images
├── gcode/              # compensated_print.gcode
└── simulation_field_commands.json
```

The CSV report includes:

```text
layer_id, err_avg, E_z, PID_integral, kinetic_energy, stable_steps, max_dx, all_free, max_error
```
