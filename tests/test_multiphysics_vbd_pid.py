import csv
import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class MultiphysicsVbdPidTests(unittest.TestCase):
    def test_czm_state_machine_softens_then_releases_and_free_time_advances(self):
        from hydrogel_vbd.physics.czm import CZMState, update_czm_states
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline

        mesh, _ = ConformalMeshPipeline.create_demo(layers=1, layer_thickness=0.05)
        mesh.activate_layer(0)
        bottom = mesh.bottom_nodes(0)

        update_czm_states(mesh, bottom, internal_pull_z=np.full(len(bottom), 6000.0), area=1.0, t_max=5000.0, k_czm=1e8, delta_f=1e-4, z_fep=0.0, dt=0.01)
        self.assertTrue(np.all(mesh.czm_state[bottom] == CZMState.DAMAGING))

        mesh.vertices[bottom, 2] = 2e-4
        previous_damage = mesh.damage[bottom].copy()
        update_czm_states(mesh, bottom, internal_pull_z=np.zeros(len(bottom)), area=1.0, t_max=5000.0, k_czm=1e8, delta_f=1e-4, z_fep=0.0, dt=0.01)
        self.assertTrue(np.all(mesh.damage[bottom] >= previous_damage))
        self.assertTrue(np.all(mesh.czm_state[bottom] == CZMState.FREE))

        update_czm_states(mesh, bottom, internal_pull_z=np.zeros(len(bottom)), area=1.0, t_max=5000.0, k_czm=1e8, delta_f=1e-4, z_fep=0.0, dt=0.01)
        self.assertTrue(np.all(mesh.time_free[bottom] > 0.0))

    def test_local_physics_terms_have_force_and_hessian_and_fluid_is_cut_off(self):
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.physics.local_terms import build_local_physics_terms
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.physics.czm import CZMState

        config = SimulationConfig()
        mesh, _ = ConformalMeshPipeline.create_demo(layers=1, layer_thickness=0.05)
        mesh.activate_layer(0)
        bottom = mesh.bottom_nodes(0)
        mesh.czm_state[bottom] = CZMState.FREE

        # ── 构建纯流体拖曳场景：F = I 无弹性形变 ──
        mesh.vertices = mesh.ideal_vertices * config.c_shrink
        # 底部 z 向上微抬 d_min*10，prev 留在原始位置产生向上速度
        mesh.vertices[bottom, 2] += config.d_min * 10.0
        mesh.prev_vertices[bottom, 2] = mesh.ideal_vertices[bottom, 2] * config.c_shrink

        # (a) 流体激活跃：time_free 小 → 满足 t_fluid_max 条件
        mesh.time_free[bottom] = 0.0
        terms = build_local_physics_terms(mesh, config, e_z=0.0, x_prev=mesh.prev_vertices)

        self.assertEqual(terms.force.shape, mesh.vertices.shape)
        self.assertEqual(terms.hessian.shape, (mesh.vertices.shape[0], 3, 3))
        self.assertTrue(np.all(terms.force[bottom, 2] < 0.0))
        self.assertTrue(np.all(terms.hessian[bottom, 2, 2] > 0.0))

        # (b) 流体截止：time_free ≥ t_fluid_max → 跳过流体贡献
        mesh.time_free[bottom] = config.t_fluid_max + 1.0
        far_terms = build_local_physics_terms(mesh, config, e_z=0.0, x_prev=mesh.prev_vertices)

        # 流体 Hessian 被移除，总 Hessian 应下降（弹性部分不变）
        hessian_drop = terms.hessian[bottom, 2, 2] - far_terms.hessian[bottom, 2, 2]
        self.assertTrue(np.all(hessian_drop > 0.0),
                        f"fluid Hessian not removed (drop={hessian_drop})")

    def test_vbd_solver_skips_fixed_nodes_tracks_convergence_and_blocks_damaging_extrapolation(self):
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.physics.czm import CZMState
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver

        config = SimulationConfig(max_iters=4, epsilon=1e-12, N_stable=2)
        mesh, _ = ConformalMeshPipeline.create_demo(layers=1, layer_thickness=0.05)
        mesh.activate_layer(0)
        top = mesh.top_nodes(0)
        bottom = mesh.bottom_nodes(0)
        mesh.is_top_fixed[top] = True
        mesh.czm_state[bottom] = CZMState.DAMAGING

        original_top = mesh.vertices[top].copy()
        result = PythonReferenceVBDSolver(config).solve_until_stable(mesh, layer_id=0, e_z=20.0)

        np.testing.assert_allclose(mesh.vertices[top], original_top)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertLessEqual(result.iterations, config.max_iters)
        self.assertGreaterEqual(result.max_dx, 0.0)
        self.assertEqual(result.chebyshev_skipped_damaging, int(len(bottom)))

    def test_pid_controller_and_outputs_use_average_sag_and_m150(self):
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.control.field_controller import PIDFieldController
        from hydrogel_vbd.io.gcode_exporter import insert_pid_field_commands

        config = SimulationConfig(err_target=0.5, K_p=10.0, K_i=2.0, K_d=1.0, E_max=20.0, dt=0.1)
        controller = PIDFieldController(config)

        first = controller.update(err_avg=0.2)
        self.assertEqual(first.E_z, 0.0)
        second = controller.update(err_avg=1.0)
        self.assertGreater(second.E_z, 0.0)
        third = controller.update(err_avg=10.0)
        self.assertLessEqual(third.E_z, 20.0)

        gcode = insert_pid_field_commands(";LAYER: 0\nG1 Z0\n", {0: second})
        self.assertIn("M150 E", gcode)

    def test_demo_loop_exports_pid_report_json_and_gcode(self):
        from hydrogel_vbd.core.main_loop import run_demo

        output_dir = ROOT / "outputs" / "architecture_demo_test"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        try:
            results = run_demo(layers=3, output=output_dir)
            self.assertEqual(len(results), 3)

            with (output_dir / "reports" / "error_metrics.csv").open(newline="", encoding="utf-8") as handle:
                fieldnames = csv.DictReader(handle).fieldnames
            for name in ["layer_id", "err_avg", "E_z", "kinetic_energy", "stable_steps", "max_dx", "all_free"]:
                self.assertIn(name, fieldnames)

            payload = json.loads((output_dir / "simulation_field_commands.json").read_text(encoding="utf-8"))
            self.assertIn("PID_integral", payload["layers"][0])
            self.assertIn("err_avg", payload["layers"][0])
            self.assertIn("M150 E", (output_dir / "gcode" / "compensated_print.gcode").read_text(encoding="utf-8"))
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir)


if __name__ == "__main__":
    unittest.main()
