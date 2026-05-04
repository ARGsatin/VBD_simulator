# Hydrogel VBD Simulation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Python MVP framework for a VBD-based electric-field-assisted hydrogel DLP printing simulation loop.

**Architecture:** The first version uses Python for orchestration, data models, simplified force models, shape evaluation, field compensation, persistence, and reports. The VBD solver is exposed behind a small replaceable interface so a future C++/pybind11 core can replace the Python reference implementation without changing the rest of the pipeline.

**Tech Stack:** Python 3.10+, NumPy, unittest, JSON/CSV/NPZ file outputs.

---

## File Structure

- `pyproject.toml`: package metadata, pytest configuration, editable install support.
- `README.md`: quickstart and framework overview.
- `configs/*.json`: material, printer, electrode, and solver templates.
- `src/hydrogel_vbd/state.py`: shared dataclasses and validation helpers.
- `src/hydrogel_vbd/geometry/layer_activator.py`: active vertex/tet mask updates.
- `src/hydrogel_vbd/material/hydrogel_model.py`: curing-degree-dependent hydrogel material state.
- `src/hydrogel_vbd/forces/*.py`: gravity, peel, electric, fluid drag, surface tension, and total force aggregation.
- `src/hydrogel_vbd/solver/*.py`: solver protocol, constraints, graph coloring, elastic placeholder, and Python reference VBD stepper.
- `src/hydrogel_vbd/evaluation/*.py`: nodal and aggregate shape error metrics.
- `src/hydrogel_vbd/control/*.py`: PD desired-force controller and regularized voltage inversion.
- `src/hydrogel_vbd/io/*.py`: NPZ state IO, VTK legacy export, CSV report writing, and G-code field insertion.
- `src/hydrogel_vbd/main_loop.py`: demo mesh generation and layer-by-layer closed loop.
- `tests/*.py`: behavior tests for the framework MVP.

## Tasks

### Task 1: Project Skeleton and Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/hydrogel_vbd/__init__.py`
- Create: `configs/material_hydrogel.json`
- Create: `configs/printer_dlp.json`
- Create: `configs/electrode_config.json`
- Create: `configs/solver_vbd.json`
- Create: `README.md`

- [ ] **Step 1: Write failing import/config test**

Create `tests/test_package_and_configs.py` with tests that import `hydrogel_vbd` and validate the four JSON config templates contain required keys.

- [ ] **Step 2: Run failing test**

Run: `python -m unittest tests.test_package_and_configs -v`
Expected: failure because `hydrogel_vbd` and config files do not exist.

- [ ] **Step 3: Create package and config templates**

Add the package metadata, empty package initializer, and JSON files with concrete hydrogel/printer/electrode/solver defaults.

- [ ] **Step 4: Run passing test**

Run: `python -m unittest tests.test_package_and_configs -v`
Expected: pass.

### Task 2: Shared State and Layer Activation

**Files:**
- Create: `src/hydrogel_vbd/state.py`
- Create: `src/hydrogel_vbd/geometry/layer_activator.py`
- Create: `tests/test_state_and_activation.py`

- [ ] **Step 1: Write failing tests**

Test `MeshState.activate_layer()` sets vertex and tet active masks, validates array shapes, and initializes zero velocities when none are supplied.

- [ ] **Step 2: Run failing tests**

Run: `python -m unittest tests.test_state_and_activation -v`
Expected: failure because state classes do not exist.

- [ ] **Step 3: Implement dataclasses and activation**

Add `MeshState`, `MaterialState`, `ForceState`, `FieldCommand`, `LayerResult`, and `LayerActivator`.

- [ ] **Step 4: Run passing tests**

Run: `python -m unittest tests.test_state_and_activation -v`
Expected: pass.

### Task 3: Material, Forces, Solver, Evaluator, and Controller

**Files:**
- Create: `src/hydrogel_vbd/material/hydrogel_model.py`
- Create: `src/hydrogel_vbd/forces/gravity.py`
- Create: `src/hydrogel_vbd/forces/peel.py`
- Create: `src/hydrogel_vbd/forces/electric.py`
- Create: `src/hydrogel_vbd/forces/fluid_drag.py`
- Create: `src/hydrogel_vbd/forces/surface_tension.py`
- Create: `src/hydrogel_vbd/forces/aggregate.py`
- Create: `src/hydrogel_vbd/solver/vbd_solver.py`
- Create: `src/hydrogel_vbd/solver/constraints.py`
- Create: `src/hydrogel_vbd/solver/graph_coloring.py`
- Create: `src/hydrogel_vbd/solver/elastic_energy.py`
- Create: `src/hydrogel_vbd/evaluation/metrics.py`
- Create: `src/hydrogel_vbd/evaluation/shape_error.py`
- Create: `src/hydrogel_vbd/control/voltage_optimizer.py`
- Create: `src/hydrogel_vbd/control/field_controller.py`
- Create: `tests/test_models_solver_control.py`

- [ ] **Step 1: Write failing behavior tests**

Test Lamé conversion, gravity force shape/magnitude, electric force from `B U`, solver movement under upward force, RMS/max shape metrics, and voltage least-squares inversion.

- [ ] **Step 2: Run failing tests**

Run: `python -m unittest tests.test_models_solver_control -v`
Expected: failure because modules do not exist.

- [ ] **Step 3: Implement minimal behavior**

Implement vectorized NumPy reference behavior for each module. Keep the solver intentionally simple: semi-implicit external-force integration with damping and fixed-vertex masking.

- [ ] **Step 4: Run passing tests**

Run: `python -m unittest tests.test_models_solver_control -v`
Expected: pass.

### Task 4: IO, Demo Loop, and Documentation

**Files:**
- Create: `src/hydrogel_vbd/io/npz_state.py`
- Create: `src/hydrogel_vbd/io/vtk_writer.py`
- Create: `src/hydrogel_vbd/io/report_writer.py`
- Create: `src/hydrogel_vbd/io/gcode_exporter.py`
- Create: `src/hydrogel_vbd/main_loop.py`
- Create: `tests/test_io_and_main_loop.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing IO and demo tests**

Test NPZ roundtrip, CSV report content, G-code insertion, and demo loop output files.

- [ ] **Step 2: Run failing tests**

Run: `python -m unittest tests.test_io_and_main_loop -v`
Expected: failure because IO and demo loop modules do not exist.

- [ ] **Step 3: Implement IO and demo**

Implement NPZ persistence, legacy ASCII VTU-style output, CSV metrics, G-code field insertion, and a deterministic beam/arch-like demo mesh loop.

- [ ] **Step 4: Run passing tests**

Run: `python -m unittest tests.test_io_and_main_loop -v`
Expected: pass.

### Task 5: Final Verification

**Files:**
- Modify as needed: files created in Tasks 1-4.

- [ ] **Step 1: Run full tests**

Run: `python -m unittest discover -v`
Expected: all tests pass.

- [ ] **Step 2: Run demo module**

Run: `python -m hydrogel_vbd.main_loop --layers 3 --output outputs/demo`
Expected: creates `outputs/demo/reports/error_metrics.csv`, layer `.npz` files, VTK files, and `simulation_field_commands.json`.

- [ ] **Step 3: Inspect generated outputs**

Confirm the demo output directories contain states, vtk, reports, and gcode artifacts.

## Self-Review

- Spec coverage: covers MVP framework, configuration, layer activation, material state, gravity/peel/electric forces, shape evaluation, field control, state outputs, CSV report, VTK-style visualization, and G-code extension. Full STL slicing/Gmsh and C++ VBD core are intentionally deferred by the approved Python-first MVP.
- Placeholder scan: no `TBD` or open implementation placeholders are required for the MVP.
- Type consistency: tests and implementation use `MeshState`, `MaterialState`, `ForceState`, `FieldCommand`, and `LayerResult` consistently.
