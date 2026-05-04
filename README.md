# Hydrogel VBD Simulation

This repository contains a Python-first MVP framework for a VBD-based electric-field-assisted hydrogel DLP printing simulation.

The first version focuses on a runnable closed loop:

1. Activate a layered tetrahedral mesh.
2. Update hydrogel material parameters.
3. Compute simplified gravity, peel, and electric forces.
4. Step a replaceable VBD solver interface.
5. Compare simulated and target shapes.
6. Compute the next electric-field command.
7. Save state, reports, visualization files, and G-code annotations.

The Python solver is intentionally a reference implementation. The public solver interface is designed so a future C++17/Eigen/OpenMP/pybind11 VBD core can replace it.

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
