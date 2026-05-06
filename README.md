# Hydrogel VBD Simulation

This repository contains a Python-first reference framework for a VBD-based electric-field-assisted hydrogel DLP printing simulation.

The current architecture follows `修改点.docx` and `伪代码.txt`:

1. Build one global conformal tetrahedral mesh with shared layer-interface nodes.
2. Activate layers with FEP collision handling, inherited deformed geometry, and anti-penetration interpolation.
3. Assemble node-local force and Hessian terms for inertia, hyperelastic placeholder stiffness, damping, CZM softening, fluid suction, and electric lift.
4. Solve with a VBD-style local 3x3 Newton loop over graph-colored vertex batches.
5. Track CZM interface states: `FIXED -> DAMAGING -> FREE`.
6. Evaluate bottom-node average sag and update the PID-controlled electric field `E_z`.
7. Save NPZ state, VTU visualization, CSV reports, JSON replay data, and `M150 E...` G-code commands.

The Python solver is intentionally a reference implementation. The interfaces are shaped so a future C++17/Eigen/OpenMP/pybind11 VBD core or real PLC/TetGen geometry pipeline can replace the Python demo components.

## Configuration

The unified physical and control parameters live in `configs/config.yaml`. It includes the parameters from the pseudocode, including `g`, `rho`, `mu`, `kappa`, `k_d`, `c_shrink`, CZM constants, fluid cutoffs, VBD convergence controls, and PID/electric safety limits.

## Quickstart

```powershell
python -m unittest discover -s tests -v
python -c "import sys; sys.path.insert(0, 'src'); from hydrogel_vbd.main_loop import run_demo; run_demo(layers=3, output='outputs/demo')"
```

After installing the package in editable mode, the demo can also be run as a module:

```powershell
python -m pip install -e .
python -m hydrogel_vbd.main_loop --layers 3 --output outputs/demo
```

The demo report at `outputs/demo/reports/error_metrics.csv` includes:

```text
layer_id, err_avg, E_z, kinetic_energy, stable_steps, max_dx, all_free
```
