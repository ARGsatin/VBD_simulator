import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SolverDiagnosticsTests(unittest.TestCase):
    def test_logged_lift_plan_explains_9991_steps(self) -> None:
        from hydrogel_vbd.solver.diagnostics import compute_lift_plan

        plan = compute_lift_plan(
            layer_thickness=0.0019981,
            v_lift=0.001,
            dt=0.001,
            lift_multiplier=5.0,
            avg_call_ms=24.536,
        )

        self.assertAlmostEqual(plan.lift_max, 0.0099905)
        self.assertAlmostEqual(plan.lift_step, 1.0e-6)
        self.assertEqual(plan.expected_steps, 9991)
        self.assertGreater(plan.estimated_wall_s, 240.0)

    def test_gui_default_lift_plan_uses_configurable_multiplier(self) -> None:
        from hydrogel_vbd.solver.diagnostics import compute_lift_plan

        plan = compute_lift_plan(
            layer_thickness=1.0e-4,
            v_lift=0.001,
            dt=0.001,
            lift_multiplier=1.5,
        )

        self.assertAlmostEqual(plan.lift_max, 1.5e-4)
        self.assertAlmostEqual(plan.lift_step, 1.0e-6)
        self.assertEqual(plan.expected_steps, 150)

    def test_default_performance_benchmark_matrix_covers_main_slow_modes(self) -> None:
        from hydrogel_vbd.solver.diagnostics import (
            default_performance_benchmark_matrix,
        )

        scenarios = default_performance_benchmark_matrix()

        self.assertEqual(len(scenarios), 8)
        self.assertTrue(all(s.model_path.name == "demo7.STEP" for s in scenarios))
        self.assertTrue(all(s.layers == 32 for s in scenarios))
        self.assertIn(
            (True, 5.0, True),
            {
                (
                    s.enable_czm,
                    s.lift_multiplier,
                    s.field_debug_enabled,
                )
                for s in scenarios
            },
        )

    def test_solver_step_diagnostics_records_counts_and_timing(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.geometry.layer_activator import LayerActivator
        from hydrogel_vbd.solver.diagnostics import (
            SolverStepDiagnostics,
            write_solver_diagnostics_csv,
        )

        config = SimulationConfig(layer_thickness=1.0e-4)
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=2, layer_thickness=config.layer_thickness, config=config
        )
        LayerActivator().activate_with_inheritance(mesh, 0, z_fep=config.z_fep)
        result = SimpleNamespace(iterations=20, stable_steps=0, max_dx=0.002)

        diag = SolverStepDiagnostics.from_mesh(
            mesh,
            layer_id=0,
            step=7,
            lift_max=5.0e-4,
            lift_step=1.0e-6,
            expected_steps=500,
            result=result,
            call_ms=12.5,
        )

        self.assertEqual(diag.layer_id, 0)
        self.assertEqual(diag.step, 7)
        self.assertEqual(diag.iterations, 20)
        self.assertTrue(diag.clipped)
        self.assertGreater(diag.active_vertices, 0)
        self.assertGreater(diag.active_tets, 0)
        self.assertGreater(diag.top_count, 0)
        self.assertIn("fixed", diag.czm_counts)

        out = ROOT / "outputs" / "test_solver_diagnostics.csv"
        if out.exists():
            out.unlink()
        write_solver_diagnostics_csv(out, [diag])
        text = out.read_text(encoding="utf-8")
        header = text.splitlines()[0].split(",")
        self.assertEqual(header, SolverStepDiagnostics.csv_fields())
        self.assertIn("12.5", text)

    def test_solver_step_diagnostics_counts_current_bottom_czm_only(self) -> None:
        from hydrogel_vbd.core.state import MeshState
        from hydrogel_vbd.physics.czm import CZMState
        from hydrogel_vbd.solver.diagnostics import SolverStepDiagnostics

        mesh = MeshState(
            vertices=np.zeros((2, 3), dtype=float),
            tets=np.zeros((0, 4), dtype=int),
            layer_id_per_vertex=np.array([0, 1], dtype=int),
            layer_id_per_tet=np.zeros(0, dtype=int),
            first_active_layer=np.array([0, 1], dtype=int),
            is_top_surface_of_layer=np.array([1, 2], dtype=int),
        )
        mesh.active_vertex_mask[:] = True
        mesh.czm_state[:] = [CZMState.FIXED, CZMState.FREE]

        diag = SolverStepDiagnostics.from_mesh(
            mesh,
            layer_id=1,
            step=0,
            lift_max=0.0,
            lift_step=0.0,
            expected_steps=0,
            result=SimpleNamespace(iterations=0, stable_steps=0, max_dx=0.0),
            call_ms=0.0,
        )

        self.assertEqual(diag.bottom_count, 1)
        self.assertEqual(diag.czm_counts, "fixed:0,damaging:0,free:1")

    def test_prepare_solver_diagnostics_csv_replaces_stale_header(self) -> None:
        from hydrogel_vbd.solver.diagnostics import (
            SolverStepDiagnostics,
            prepare_solver_diagnostics_csv,
        )

        out = ROOT / "outputs" / "test_solver_diagnostics_stale.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("layer_id,step,call_ms\n1,1,2.0\n", encoding="utf-8")

        prepare_solver_diagnostics_csv(out)

        lines = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, [",".join(SolverStepDiagnostics.csv_fields())])

    def test_layer_performance_diagnostics_csv_has_stable_header(self) -> None:
        from hydrogel_vbd.solver.diagnostics import (
            LayerPerformanceDiagnostics,
            prepare_layer_performance_diagnostics_csv,
            write_layer_performance_diagnostics_csv,
        )

        out = ROOT / "outputs" / "test_layer_performance_diagnostics.csv"
        out.unlink(missing_ok=True)
        row = LayerPerformanceDiagnostics(
            layer_id=2,
            mode="field_debug",
            elapsed_s=1.25,
            lift_steps=45,
            return_steps=45,
            no_field_ms=300.0,
            with_field_ms=310.0,
            cpp_solve_ms=500.0,
            czm_sync_ms=80.0,
            render_ms=12.0,
            solver_steps=90,
            solver_iterations=180,
            rms_error=1.0e-3,
            max_error=2.0e-3,
            guard_reason="pass",
        )

        prepare_layer_performance_diagnostics_csv(out)
        write_layer_performance_diagnostics_csv(out, [row])

        lines = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0].split(","), LayerPerformanceDiagnostics.csv_fields())
        self.assertIn("field_debug", lines[1])
        self.assertIn("310.0", lines[1])

    def test_diagnostic_runaway_guard_triggers_after_repeated_clipped_steps(self) -> None:
        from hydrogel_vbd.solver.diagnostics import (
            SolverRunawayGuard,
            SolverStepDiagnostics,
        )

        guard = SolverRunawayGuard(limit=3, max_iters=20)
        bad = SolverStepDiagnostics(
            layer_id=0,
            step=1,
            lift_max=1.0,
            lift_step=1.0e-6,
            expected_steps=100,
            iterations=20,
            stable_steps=0,
            max_dx=0.002,
            clipped=True,
            active_vertices=4,
            active_tets=1,
            top_count=1,
            bottom_count=1,
            czm_counts="fixed:1,damaging:0,free:3",
            z_fep=0.0,
            current_bottom_z_min=0.0,
            current_bottom_z_max=0.0,
            previous_bottom_z_min=float("nan"),
            previous_bottom_z_max=float("nan"),
            global_bottom_z_min=0.0,
            global_bottom_z_max=0.0,
            max_move_node=-1,
            max_move_kind="",
            max_move_czm_state="",
            max_move_dx=float("nan"),
            max_move_dy=float("nan"),
            max_move_dz=float("nan"),
            max_move_norm=float("nan"),
            max_move_z=float("nan"),
            czm_pull_max=float("nan"),
            czm_pull_stress_max=float("nan"),
            czm_gap_max=float("nan"),
            czm_onset_candidates=0,
            czm_failure_candidates=0,
            call_ms=1.0,
        )
        good = SolverStepDiagnostics(
            layer_id=0,
            step=4,
            lift_max=1.0,
            lift_step=1.0e-6,
            expected_steps=100,
            iterations=3,
            stable_steps=1,
            max_dx=1.0e-8,
            clipped=False,
            active_vertices=4,
            active_tets=1,
            top_count=1,
            bottom_count=1,
            czm_counts="fixed:1,damaging:0,free:3",
            z_fep=0.0,
            current_bottom_z_min=0.0,
            current_bottom_z_max=0.0,
            previous_bottom_z_min=float("nan"),
            previous_bottom_z_max=float("nan"),
            global_bottom_z_min=0.0,
            global_bottom_z_max=0.0,
            max_move_node=-1,
            max_move_kind="",
            max_move_czm_state="",
            max_move_dx=float("nan"),
            max_move_dy=float("nan"),
            max_move_dz=float("nan"),
            max_move_norm=float("nan"),
            max_move_z=float("nan"),
            czm_pull_max=float("nan"),
            czm_pull_stress_max=float("nan"),
            czm_gap_max=float("nan"),
            czm_onset_candidates=0,
            czm_failure_candidates=0,
            call_ms=1.0,
        )

        self.assertFalse(guard.observe(bad))
        self.assertFalse(guard.observe(bad))
        self.assertTrue(guard.observe(bad))
        self.assertEqual(guard.consecutive_bad_steps, 3)
        self.assertFalse(guard.observe(good))
        self.assertEqual(guard.consecutive_bad_steps, 0)

    def test_diagnostics_env_flag_is_off_by_default(self) -> None:
        from hydrogel_vbd.solver.diagnostics import diagnostics_enabled

        self.assertFalse(diagnostics_enabled({}))
        self.assertTrue(diagnostics_enabled({"HYDROGEL_VBD_SOLVER_DIAG": "1"}))

    def test_physics_ablation_records_force_hessian_and_solver_metrics(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.geometry.layer_activator import LayerActivator
        from hydrogel_vbd.solver.diagnostics import (
            collect_physics_ablation_diagnostics,
        )

        config = SimulationConfig(layer_thickness=1.0e-4, max_iters=2)
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=1, layer_thickness=config.layer_thickness, config=config
        )
        LayerActivator().activate_with_inheritance(mesh, 0, z_fep=config.z_fep)
        lifting_top = np.flatnonzero(mesh.is_top_fixed & mesh.active_vertex_mask)

        def fake_step(mesh_arg, config_arg, e_z_arg, layer_id_arg, lifting_top_arg):
            return SimpleNamespace(iterations=2, stable_steps=0, max_dx=0.002)

        rows = collect_physics_ablation_diagnostics(
            mesh,
            config,
            layer_id=0,
            e_z=123.0,
            lifting_top=lifting_top,
            solve_step=fake_step,
        )

        self.assertEqual(
            [row.case for row in rows],
            ["elastic_only", "plus_shrink", "plus_czm", "plus_fluid", "plus_electric"],
        )
        for row in rows:
            self.assertTrue(np.isfinite(row.force_norm))
            self.assertTrue(np.isfinite(row.hessian_min_eig))
            self.assertTrue(np.isfinite(row.hessian_max_eig))
            self.assertEqual(row.iterations, 2)
            self.assertAlmostEqual(row.max_dx, 0.002)

    def test_cpp_adapter_preparation_profile_detects_array_copies(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.solver.diagnostics import profile_cpp_adapter_preparation

        config = SimulationConfig(layer_thickness=1.0e-4)
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=1, layer_thickness=config.layer_thickness, config=config
        )
        mesh.tets = mesh.tets.astype(np.int64, copy=True)
        mesh.colors = mesh.colors.astype(np.int64, copy=True)

        profile = profile_cpp_adapter_preparation(mesh)

        self.assertTrue(profile.tets_copied)
        self.assertTrue(profile.colors_copied)
        self.assertFalse(profile.bottom_surface_copied)
        self.assertGreaterEqual(profile.elapsed_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
