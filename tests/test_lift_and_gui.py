"""测试平台运动学求解器 & STL 网格加载 & GUI 参数配置。"""

import sys
import unittest
from pathlib import Path
import os
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class _FakeSignal:
    def connect(self, slot) -> None:  # noqa: ANN001
        pass


class LiftSolverTests(unittest.TestCase):
    """平台提升-静平衡求解器单元测试。"""

    def setUp(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline

        self.config = SimulationConfig(
            max_iters=10,
            epsilon=1e-12,
            N_stable=3,
            v_lift=0.005,
            layer_thickness=0.05,
        )
        self.mesh, _ = ConformalMeshPipeline.create_demo(
            layers=1, layer_thickness=0.05, config=self.config
        )
        self.mesh.activate_layer(0)
        top = self.mesh.top_nodes(0)
        self.mesh.is_top_fixed[top] = True

    def test_solve_with_lift_runs_and_produces_result(self) -> None:
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver

        solver = PythonReferenceVBDSolver(self.config)
        lifting_top = np.flatnonzero(self.mesh.is_top_fixed)
        result = solver.solve_with_lift(
            self.mesh, layer_id=0, e_z=0.0, lifting_top=lifting_top
        )
        self.assertGreaterEqual(result.iterations, 1)
        self.assertLessEqual(result.iterations, self.config.max_iters)
        self.assertGreaterEqual(result.max_dx, 0.0)
        self.assertEqual(result.x.shape, self.mesh.vertices.shape)

    def test_lift_raises_top_nodes(self) -> None:
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver

        original_top_z = self.mesh.vertices[self.mesh.is_top_fixed, 2].copy()
        solver = PythonReferenceVBDSolver(self.config)
        lifting_top = np.flatnonzero(self.mesh.is_top_fixed)
        result = solver.solve_with_lift(
            self.mesh, layer_id=0, e_z=0.0, lifting_top=lifting_top
        )
        new_top_z = result.x[self.mesh.is_top_fixed, 2]
        self.assertTrue(np.all(new_top_z > original_top_z))


class PythonLiftSolverStabilityTests(unittest.TestCase):
    """Python 提升求解器数值稳定性回归测试。"""

    @staticmethod
    def _single_vertex_mesh(*, bottom_interface: int = 1):
        from hydrogel_vbd.core.state import MeshState

        mesh = MeshState(
            vertices=np.zeros((1, 3), dtype=float),
            tets=np.zeros((0, 4), dtype=int),
            layer_id_per_vertex=np.array([0], dtype=int),
            layer_id_per_tet=np.zeros(0, dtype=int),
            first_active_layer=np.array([0], dtype=int),
            is_bottom_surface=np.array([bottom_interface == 0], dtype=bool),
            is_top_surface_of_layer=np.array([bottom_interface], dtype=int),
        )
        mesh.active_vertex_mask[:] = True
        mesh.active_tet_mask = np.zeros(0, dtype=bool)
        mesh.colors = np.zeros(1, dtype=int)
        mesh.node_mass = np.ones(1, dtype=float)
        mesh.czm_state[:] = 2  # FREE
        return mesh

    def test_disable_czm_does_not_fix_or_update_bottom_nodes(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.physics.czm import CZMState
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver

        config = SimulationConfig(
            enable_czm=False,
            dt=1.0e-3,
            max_iters=1,
            N_stable=1,
            v_lift=1.0e-3,
        )
        mesh = self._single_vertex_mesh()
        mesh.czm_state[:] = int(CZMState.FIXED)

        result = PythonReferenceVBDSolver(config).solve_with_lift(
            mesh, layer_id=0, e_z=0.0, lifting_top=np.array([], dtype=int)
        )

        self.assertEqual(mesh.czm_state[0], int(CZMState.FIXED))
        self.assertTrue(result.all_free)
        self.assertEqual(result.chebyshev_skipped_damaging, 0)

    def test_print_axis_maps_model_y_to_simulation_z(self) -> None:
        from hydrogel_vbd.geometry.stl_mesher import transform_points_to_print_z

        points = np.array([[1.0, 2.0, 3.0]], dtype=float)
        mapped = transform_points_to_print_z(points, "y")

        np.testing.assert_allclose(mapped, np.array([[1.0, -3.0, 2.0]]))

    def test_solve_with_lift_psd_projects_indefinite_hessian(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.physics.local_terms import LocalPhysicsTerms
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver

        config = SimulationConfig(
            dt=1.0,
            k_d=0.0,
            max_iters=1,
            N_stable=1,
            epsilon=1.0e-12,
            v_lift=0.0,
        )
        mesh = self._single_vertex_mesh()
        bad_hessian = -1.000000001 * np.eye(3)[None, :, :]
        terms = LocalPhysicsTerms(
            force=np.array([[1.0, 0.0, 0.0]], dtype=float),
            hessian=bad_hessian,
        )

        with patch(
            "hydrogel_vbd.solver.vbd_solver.build_local_physics_terms",
            return_value=terms,
        ):
            result = PythonReferenceVBDSolver(config).solve_with_lift(
                mesh, layer_id=0, e_z=0.0, lifting_top=np.array([], dtype=int)
            )

        self.assertEqual(result.iterations, 1)
        self.assertTrue(np.all(np.isfinite(result.x)))

    def test_indefinite_hessian_projection_clips_negative_modes(self) -> None:
        from hydrogel_vbd.solver.vbd_solver import _make_psd

        hessian = np.diag([-2.0, 0.5, 3.0])
        projected = _make_psd(hessian)

        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(projected))), 0.0)
        np.testing.assert_allclose(projected, np.diag([0.0, 0.5, 3.0]))

    def test_solve_with_lift_updates_czm_with_actual_pull(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.physics.czm import CZMState
        from hydrogel_vbd.physics.local_terms import LocalPhysicsTerms
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver

        config = SimulationConfig(
            dt=1.0e-3,
            max_iters=1,
            N_stable=1,
            v_lift=0.0,
            T_max=5000.0,
        )
        mesh = self._single_vertex_mesh(bottom_interface=1)
        mesh.is_bottom_surface[:] = True
        mesh.czm_state[:] = int(CZMState.FIXED)
        actual_pull = 123.0
        terms = LocalPhysicsTerms(
            force=np.array([[0.0, 0.0, actual_pull]], dtype=float),
            hessian=np.zeros((1, 3, 3), dtype=float),
        )

        with (
            patch(
                "hydrogel_vbd.solver.vbd_solver.build_local_physics_terms",
                return_value=terms,
            ),
            patch("hydrogel_vbd.physics.czm.update_czm_states") as update_mock,
        ):
            PythonReferenceVBDSolver(config).solve_with_lift(
                mesh, layer_id=0, e_z=0.0, lifting_top=np.array([], dtype=int)
            )

        self.assertEqual(update_mock.call_count, 1)
        pull_arg = update_mock.call_args.kwargs["internal_pull_z"]
        np.testing.assert_allclose(pull_arg, [actual_pull])

    def test_layer_one_solver_does_not_clamp_previous_global_bottom_to_current_fep(self) -> None:
        from dataclasses import replace

        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.geometry.layer_activator import LayerActivator
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver

        config = SimulationConfig(
            layer_thickness=0.0019981,
            dt=1.0e-3,
            v_lift=1.0e-3,
            max_iters=1,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=2, layer_thickness=config.layer_thickness, config=config
        )
        activator = LayerActivator()
        activator.activate_with_inheritance(mesh, 0, z_fep=0.0)
        previous_bottom = mesh.bottom_nodes(0)

        layer_config = replace(config, z_fep=0.0)
        activator.activate_with_inheritance(
            mesh, 1, z_fep=layer_config.z_fep
        )
        lifting_top = np.flatnonzero(mesh.is_top_fixed & mesh.active_vertex_mask)

        result = PythonReferenceVBDSolver(layer_config).solve_with_lift(
            mesh, layer_id=1, e_z=0.0, lifting_top=lifting_top
        )

        self.assertLess(result.max_dx, 1.0e-5)
        np.testing.assert_allclose(
            mesh.vertices[previous_bottom, 2],
            config.layer_thickness,
            atol=config.layer_thickness * 0.01,
        )
        self.assertGreaterEqual(float(np.min(mesh.vertices[previous_bottom, 2])), 0.0)

    def test_layer_one_interior_nodes_are_clamped_to_their_layer_floor(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.core.state import MeshState
        from hydrogel_vbd.physics.local_terms import LocalPhysicsTerms
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver

        config = SimulationConfig(
            layer_thickness=0.0019981,
            dt=1.0e-3,
            v_lift=0.0,
            max_iters=1,
            N_stable=1,
            d_fluid_max=0.0,
        )
        layer_config = config
        layer_config.z_fep = 0.0
        mesh = MeshState(
            vertices=np.array([[0.0, 0.0, layer_config.z_fep - 1.0e-3]], dtype=float),
            tets=np.zeros((0, 4), dtype=int),
            layer_id_per_vertex=np.array([1], dtype=int),
            layer_id_per_tet=np.zeros(0, dtype=int),
            first_active_layer=np.array([1], dtype=int),
            is_bottom_surface=np.array([False], dtype=bool),
            is_top_surface_of_layer=np.array([-1], dtype=int),
        )
        mesh.active_vertex_mask[:] = True
        mesh.colors = np.zeros(1, dtype=int)
        mesh.node_mass = np.ones(1, dtype=float)
        terms = LocalPhysicsTerms(
            force=np.zeros_like(mesh.vertices),
            hessian=np.zeros((mesh.vertices.shape[0], 3, 3), dtype=float),
        )

        with patch(
            "hydrogel_vbd.solver.vbd_solver.build_local_physics_terms",
            return_value=terms,
        ):
            PythonReferenceVBDSolver(layer_config).solve_with_lift(
                mesh, layer_id=1, e_z=0.0, lifting_top=np.array([], dtype=int)
            )

        self.assertGreaterEqual(
            float(mesh.vertices[0, 2]),
            layer_config.z_fep,
        )


class WorkerLiftControlTests(unittest.TestCase):
    """Worker 层内提升控制回归测试。"""

    def test_current_lifting_top_uses_active_layer_top_nodes(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.geometry.layer_activator import LayerActivator
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker

        config = SimulationConfig(layer_thickness=1.0e-4)
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=3, layer_thickness=config.layer_thickness, config=config
        )

        # Simulate the previous worker preprocess cache of the global highest face.
        z_max = float(np.max(mesh.vertices[:, 2]))
        mesh.top_ids = np.flatnonzero(np.isclose(mesh.vertices[:, 2], z_max))

        LayerActivator().activate_with_inheritance(
            mesh, current_layer=0, z_fep=config.z_fep
        )

        lifting_top = SimulationWorker._current_lifting_top(mesh)
        np.testing.assert_array_equal(lifting_top, mesh.top_nodes(0))
        self.assertTrue(np.all(mesh.active_vertex_mask[lifting_top]))

    def test_expected_lift_steps_uses_si_units(self) -> None:
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker

        lift_max = 5.0 * 1.0e-4
        lift_step = 1.0e-3 * 1.0e-3
        self.assertEqual(
            SimulationWorker._expected_lift_steps(lift_max, lift_step),
            500,
        )
        with self.assertRaises(ValueError):
            SimulationWorker._expected_lift_steps(lift_max, 0.0)

    def test_platform_return_distance_removes_temporary_lift(self) -> None:
        from hydrogel_vbd.solver.cpp_subprocess import _platform_return_distance

        self.assertAlmostEqual(
            _platform_return_distance(actual_lift=2.0e-6, next_gap=1.0e-6),
            2.0e-6,
        )
        self.assertEqual(
            _platform_return_distance(actual_lift=0.8e-6, next_gap=1.0e-6),
            0.8e-6,
        )

    def test_layer_contact_z_stays_at_fixed_fep(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker

        config = SimulationConfig(z_fep=0.01, layer_thickness=0.0019981)

        self.assertAlmostEqual(SimulationWorker._layer_contact_z(config, 0), 0.01)
        self.assertAlmostEqual(
            SimulationWorker._layer_contact_z(config, 2),
            0.01,
        )

    def test_cpp_subprocess_config_includes_lift_height(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.cpp_subprocess import _DoneMsg

        captured: dict[str, object] = {}

        class FakeCppSubprocessSolver:
            def __init__(self, mesh_dict, config_dict, n_layers, output_dir, **kwargs):  # noqa: ANN001
                captured["config_dict"] = config_dict

            def start(self) -> None:
                pass

            def iter_messages(self, timeout=0.2):  # noqa: ANN001
                yield _DoneMsg(results=[])

            def terminate(self) -> None:
                pass

        mesh = PythonLiftSolverStabilityTests._single_vertex_mesh()
        worker = SimulationWorker(
            mesh=mesh,
            config=SimulationConfig(lift_height=5.0e-3),
            n_layers=0,
            output_dir="outputs/gui",
            use_cpp=False,
        )
        worker._trace = lambda msg: None

        with patch(
            "hydrogel_vbd.solver.cpp_subprocess.CppSubprocessSolver",
            FakeCppSubprocessSolver,
        ):
            worker._run_cpp_subprocess()

        config_dict = captured["config_dict"]
        self.assertEqual(config_dict["lift_height"], 5.0e-3)

    def test_push_frame_includes_active_tet_mask(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker

        mesh = PythonLiftSolverStabilityTests._single_vertex_mesh()
        worker = SimulationWorker(
            mesh=mesh,
            config=SimulationConfig(),
            n_layers=0,
            output_dir="outputs/gui",
            use_cpp=False,
        )
        frames: list[dict] = []
        worker.frame_ready.connect(frames.append)
        active_tet_mask = np.array([True, False], dtype=bool)

        worker._push_frame(
            vertices=np.zeros((4, 3), dtype=float),
            tets=np.zeros((2, 4), dtype=int),
            active_mask=np.ones(4, dtype=bool),
            active_tet_mask=active_tet_mask,
            title="frame",
        )

        np.testing.assert_array_equal(frames[-1]["active_tet_mask"], active_tet_mask)

    def test_cpp_subprocess_config_includes_enable_czm(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.cpp_subprocess import _DoneMsg

        captured: dict[str, object] = {}

        class FakeCppSubprocessSolver:
            def __init__(self, mesh_dict, config_dict, n_layers, output_dir, **kwargs):  # noqa: ANN001
                captured["config_dict"] = config_dict

            def start(self) -> None:
                pass

            def iter_messages(self, timeout=0.2):  # noqa: ANN001
                yield _DoneMsg(results=[])

            def terminate(self) -> None:
                pass

        mesh = PythonLiftSolverStabilityTests._single_vertex_mesh()
        worker = SimulationWorker(
            mesh=mesh,
            config=SimulationConfig(enable_czm=False),
            n_layers=0,
            output_dir="outputs/gui",
            use_cpp=False,
        )
        worker._trace = lambda msg: None

        with patch(
            "hydrogel_vbd.solver.cpp_subprocess.CppSubprocessSolver",
            FakeCppSubprocessSolver,
        ):
            worker._run_cpp_subprocess()

        config_dict = captured["config_dict"]
        self.assertIs(config_dict["enable_czm"], False)

    def test_yaml_config_loads_lift_height(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig

        config = SimulationConfig.from_yaml(ROOT / "configs" / "config.yaml")

        self.assertAlmostEqual(config.lift_height, 5.0e-3)

    def test_cpp_done_payload_converts_to_layer_result(self) -> None:
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.core.state import LayerResult

        result = SimulationWorker._cpp_payload_to_layer_result({
            "layer_id": 8,
            "total_steps": 2998,
            "final_max_dx": 1.25e-6,
            "total_iterations": 59960,
            "max_iter_hits": 17,
            "clipped_steps": 11,
            "elapsed_s": 4.2,
            "avg_call_ms": 1.4,
            "success": True,
        })

        self.assertIsInstance(result, LayerResult)
        self.assertEqual(result.layer_id, 8)
        self.assertAlmostEqual(result.max_deformation, 1.25e-6)
        self.assertEqual(result.error_metrics["total_steps"], 2998)
        self.assertEqual(result.error_metrics["solver_total_steps"], 2998)
        self.assertAlmostEqual(result.error_metrics["solver_final_max_dx"], 1.25e-6)
        self.assertEqual(result.error_metrics["solver_max_iter_hits"], 17)
        self.assertEqual(result.error_metrics["solver_clipped_steps"], 11)
        self.assertEqual(result.error_metrics["shape_error_available"], 0.0)
        self.assertTrue(result.success)

    def test_worker_rejects_detach_before_solver_convergence(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(max_iters=20, N_stable=2, epsilon=1.0e-9)
        result = VBDSolveResult(
            x=np.zeros((0, 3)),
            v=np.zeros((0, 3)),
            iterations=config.max_iters,
            max_dx=SimulationWorker.DX_CLIP_DIAGNOSTIC,
            kinetic_energy=0.0,
            stable_steps=0,
            all_free=True,
            chebyshev_skipped_damaging=0,
        )

        with self.assertRaisesRegex(RuntimeError, "solver did not converge"):
            SimulationWorker._raise_if_detached_before_convergence(
                layer_id=3,
                layer_steps=2,
                result=result,
                config=config,
            )

    def test_worker_czm_update_uses_actual_local_pull(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.physics.local_terms import LocalPhysicsTerms
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker

        config = SimulationConfig(T_max=5000.0)
        mesh = PythonLiftSolverStabilityTests._single_vertex_mesh(
            bottom_interface=1
        )
        actual_pull = 321.0
        terms = LocalPhysicsTerms(
            force=np.array([[0.0, 0.0, actual_pull]], dtype=float),
            hessian=np.zeros((1, 3, 3), dtype=float),
        )

        with (
            patch(
                "hydrogel_vbd.physics.local_terms.build_local_physics_terms",
                return_value=terms,
            ),
            patch("hydrogel_vbd.physics.czm.update_czm_states") as update_mock,
        ):
            SimulationWorker._update_czm_from_current_terms(
                mesh, config, layer_id=0, e_z=0.0, x_prev=mesh.vertices.copy()
            )

        self.assertEqual(update_mock.call_count, 1)
        pull_arg = update_mock.call_args.kwargs["internal_pull_z"]
        np.testing.assert_allclose(pull_arg, [actual_pull])

    def test_solver_only_fixes_czm_nodes_on_current_bottom(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.physics.czm import CZMState
        from hydrogel_vbd.physics.local_terms import LocalPhysicsTerms
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver

        mesh = PythonLiftSolverStabilityTests._single_vertex_mesh(bottom_interface=1)
        mesh.czm_state[:] = int(CZMState.FIXED)
        force = np.array([[1.0, 0.0, 0.0]], dtype=float)
        terms = LocalPhysicsTerms(
            force=force,
            hessian=np.zeros((1, 3, 3), dtype=float),
        )
        config = SimulationConfig(
            dt=1.0,
            k_d=0.0,
            max_iters=1,
            N_stable=1,
            epsilon=1.0e-12,
            c_init=0.0,
        )

        with patch(
            "hydrogel_vbd.solver.vbd_solver.build_local_physics_terms",
            return_value=terms,
        ):
            PythonReferenceVBDSolver(config).solve_with_lift(
                mesh, layer_id=1, e_z=0.0, lifting_top=np.array([], dtype=int)
            )

        self.assertGreater(mesh.vertices[0, 0], 0.0)

    def test_local_terms_ignore_free_czm_away_from_current_bottom(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.physics.czm import CZMState
        from hydrogel_vbd.physics.local_terms import build_local_physics_terms

        mesh = PythonLiftSolverStabilityTests._single_vertex_mesh(bottom_interface=1)
        mesh.czm_state[:] = int(CZMState.FREE)
        mesh.vertices[0, 2] = 2.0e-6
        mesh.prev_vertices[0, 2] = 1.0e-6
        config = SimulationConfig(
            g=(0.0, 0.0, 0.0),
            q_ion=0.0,
            dt=1.0e-3,
            d_min=1.0e-9,
            d_fluid_max=1.0e-3,
            t_fluid_max=1.0,
        )

        terms = build_local_physics_terms(
            mesh, config, e_z=0.0, x_prev=mesh.prev_vertices, layer_id=1
        )

        np.testing.assert_allclose(terms.force[0], np.zeros(3))
        np.testing.assert_allclose(terms.hessian[0], np.zeros((3, 3)))

    def test_worker_stop_request_terminates_active_cpp_subprocess(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker

        mesh = PythonLiftSolverStabilityTests._single_vertex_mesh()
        worker = SimulationWorker(
            mesh=mesh,
            config=SimulationConfig(),
            n_layers=1,
            output_dir="outputs/gui",
            use_cpp=False,
        )

        class FakeCppSolver:
            def __init__(self) -> None:
                self.terminated = False

            def terminate(self) -> None:
                self.terminated = True

        cpp_solver = FakeCppSolver()
        worker._cpp_solver = cpp_solver

        worker.request_stop()

        self.assertTrue(worker._stop_flag)
        self.assertTrue(cpp_solver.terminated)

    def test_python_worker_diagnostic_guard_writes_csv_and_stops(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.diagnostics import SolverStepDiagnostics
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            dt=1.0e-3,
            v_lift=1.0e-3,
            layer_thickness=20.0e-6,
            lift_height=100.0e-6,
            max_iters=2,
            N_stable=1,
            epsilon=1.0e-9,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=1, layer_thickness=config.layer_thickness, config=config
        )
        out_dir = Path("outputs/test_python_worker_diag")
        csv_path = out_dir / "reports" / "solver_diagnostics.csv"
        csv_path.unlink(missing_ok=True)

        class FakeSolver:
            def __init__(self, cfg) -> None:  # noqa: ANN001
                self.config = cfg

            def solve_with_lift(self, mesh_arg, **kwargs):  # noqa: ANN001, ARG002
                return VBDSolveResult(
                    x=mesh_arg.vertices,
                    v=mesh_arg.velocities,
                    iterations=self.config.max_iters,
                    max_dx=0.002,
                    kinetic_energy=0.0,
                    stable_steps=0,
                    all_free=False,
                    chebyshev_skipped_damaging=0,
                )

        saved = {
            "HYDROGEL_VBD_SOLVER_DIAG": os.environ.get("HYDROGEL_VBD_SOLVER_DIAG"),
            "HYDROGEL_VBD_SOLVER_DIAG_STRIDE": os.environ.get(
                "HYDROGEL_VBD_SOLVER_DIAG_STRIDE"
            ),
        }
        worker = SimulationWorker(
            mesh=mesh,
            config=config,
            n_layers=1,
            output_dir=out_dir,
            use_cpp=False,
            solver_diagnostics_enabled=True,
            solver_diagnostics_stride=1,
        )
        worker._trace = lambda msg: None
        try:
            os.environ.pop("HYDROGEL_VBD_SOLVER_DIAG", None)
            os.environ.pop("HYDROGEL_VBD_SOLVER_DIAG_STRIDE", None)
            with patch(
                "hydrogel_vbd.solver.vbd_solver.PythonReferenceVBDSolver",
                FakeSolver,
            ):
                results = worker._run_layers()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertTrue(worker._stop_flag)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].error_metrics["solver_total_steps"], 50.0)
        self.assertFalse(results[0].success)
        header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertEqual(header, SolverStepDiagnostics.csv_fields())

    def test_python_worker_field_debug_compares_no_field_and_with_field(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            g=(0.0, 0.0, 0.0),
            q_ion=1.0,
            K_p=1.0,
            K_i=0.0,
            K_d=0.0,
            err_target=0.0,
            field_regularization=0.0,
            v_lift=0.0,
            max_iters=1,
            N_stable=1,
        )
        mesh = PythonLiftSolverStabilityTests._single_vertex_mesh()
        mesh.vertices[0, 2] = 1.0
        mesh.ideal_vertices[0, 2] = 1.0

        class FakeSolver:
            def __init__(self, cfg) -> None:  # noqa: ANN001
                self.config = cfg

            def solve_until_stable(self, mesh_arg, layer_id, e_z, on_iteration=None):  # noqa: ANN001, ARG002
                mesh_arg.vertices[0, 2] = 0.5 if float(e_z) <= 0.0 else 0.9
                return VBDSolveResult(
                    x=mesh_arg.vertices.copy(),
                    v=np.zeros_like(mesh_arg.vertices),
                    iterations=1,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=True,
                    chebyshev_skipped_damaging=0,
                )

        worker = SimulationWorker(
            mesh=mesh,
            config=config,
            n_layers=1,
            output_dir="outputs/test_worker_field_debug",
            use_cpp=False,
            field_debug_enabled=True,
        )
        worker._trace = lambda msg: None
        logs: list[str] = []
        worker.log_message.connect(logs.append)

        with patch(
            "hydrogel_vbd.solver.vbd_solver.PythonReferenceVBDSolver",
            FakeSolver,
        ):
            results = worker._run_layers()

        self.assertEqual(len(results), 1)
        metrics = results[0].error_metrics
        self.assertEqual(metrics["field_debug_enabled"], 1.0)
        self.assertAlmostEqual(metrics["field_no_field_rms"], 0.5)
        self.assertAlmostEqual(metrics["field_with_field_rms"], 0.1)
        self.assertAlmostEqual(metrics["field_no_field_bottom_z_mean"], 0.5)
        self.assertAlmostEqual(metrics["field_with_field_bottom_z_mean"], 0.1)
        self.assertAlmostEqual(metrics["field_derived_E_z"], 0.5)
        self.assertEqual(metrics["field_effective_mode"], "with_field")
        self.assertTrue(any("[field-debug]" in msg for msg in logs))

    def test_field_debug_uses_direct_cpp_adapter_when_requested(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            g=(0.0, 0.0, 0.0),
            q_ion=1.0,
            K_p=1.0,
            K_i=0.0,
            K_d=0.0,
            err_target=0.0,
            field_regularization=0.0,
            v_lift=0.0,
            max_iters=1,
            N_stable=1,
        )
        mesh = PythonLiftSolverStabilityTests._single_vertex_mesh()
        mesh.vertices[0, 2] = 1.0
        mesh.ideal_vertices[0, 2] = 1.0
        cpp_calls: list[float] = []

        def fake_cpp_solve(mesh_arg, cfg_arg, e_z, layer_id):  # noqa: ANN001, ARG001
            cpp_calls.append(float(e_z))
            mesh_arg.vertices[0, 2] = 0.5 if float(e_z) <= 0.0 else 0.9
            return VBDSolveResult(
                x=mesh_arg.vertices.copy(),
                v=np.zeros_like(mesh_arg.vertices),
                iterations=1,
                max_dx=0.0,
                kinetic_energy=0.0,
                stable_steps=1,
                all_free=True,
                chebyshev_skipped_damaging=0,
            )

        with patch(
            "hydrogel_vbd.gui.simulation_worker.is_cpp_available",
            return_value=True,
        ):
            worker = SimulationWorker(
                mesh=mesh,
                config=config,
                n_layers=1,
                output_dir="outputs/test_worker_field_debug_cpp",
                use_cpp=True,
                field_debug_enabled=True,
            )
        worker._trace = lambda msg: None
        worker._run_cpp_subprocess = lambda: (_ for _ in ()).throw(
            AssertionError("field debug should not use the C++ subprocess")
        )
        logs: list[str] = []
        worker.log_message.connect(logs.append)

        with (
            patch(
                "hydrogel_vbd.solver.cpp_adapter.solve_until_stable",
                side_effect=fake_cpp_solve,
            ),
            patch(
                "hydrogel_vbd.solver.vbd_solver.PythonReferenceVBDSolver",
                side_effect=AssertionError(
                    "field debug should use the direct C++ adapter"
                ),
            ),
        ):
            results = worker._run_layers()

        self.assertEqual(len(results), 1)
        self.assertEqual(len(cpp_calls), 2)
        self.assertAlmostEqual(cpp_calls[0], 0.0)
        self.assertGreater(cpp_calls[1], 0.0)
        metrics = results[0].error_metrics
        self.assertEqual(metrics["field_debug_solver_backend"], "cpp_adapter")
        self.assertEqual(metrics["field_effective_mode"], "with_field")
        self.assertTrue(any("C++ adapter" in msg for msg in logs))

    def test_field_debug_cpp_adapter_failure_falls_back_to_python(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            g=(0.0, 0.0, 0.0),
            q_ion=1.0,
            K_p=1.0,
            K_i=0.0,
            K_d=0.0,
            err_target=0.0,
            field_regularization=0.0,
            v_lift=0.0,
            max_iters=1,
            N_stable=1,
        )
        mesh = PythonLiftSolverStabilityTests._single_vertex_mesh()
        mesh.vertices[0, 2] = 1.0
        mesh.ideal_vertices[0, 2] = 1.0
        python_calls: list[float] = []

        class FakeSolver:
            def __init__(self, cfg) -> None:  # noqa: ANN001
                self.config = cfg

            def solve_until_stable(self, mesh_arg, layer_id, e_z, on_iteration=None):  # noqa: ANN001, ARG002
                python_calls.append(float(e_z))
                mesh_arg.vertices[0, 2] = 0.5 if float(e_z) <= 0.0 else 0.9
                return VBDSolveResult(
                    x=mesh_arg.vertices.copy(),
                    v=np.zeros_like(mesh_arg.vertices),
                    iterations=1,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=True,
                    chebyshev_skipped_damaging=0,
                )

        with patch(
            "hydrogel_vbd.gui.simulation_worker.is_cpp_available",
            return_value=True,
        ):
            worker = SimulationWorker(
                mesh=mesh,
                config=config,
                n_layers=1,
                output_dir="outputs/test_worker_field_debug_cpp_fallback",
                use_cpp=True,
                field_debug_enabled=True,
            )
        worker._trace = lambda msg: None
        logs: list[str] = []
        worker.log_message.connect(logs.append)

        with (
            patch(
                "hydrogel_vbd.solver.cpp_adapter.solve_until_stable",
                side_effect=RuntimeError("adapter failed"),
            ),
            patch(
                "hydrogel_vbd.solver.vbd_solver.PythonReferenceVBDSolver",
                FakeSolver,
            ),
        ):
            results = worker._run_layers()

        self.assertEqual(len(results), 1)
        self.assertEqual(len(python_calls), 2)
        metrics = results[0].error_metrics
        self.assertEqual(metrics["field_debug_solver_backend"], "python")
        self.assertEqual(metrics["field_debug_cpp_fallbacks"], 1.0)
        self.assertEqual(metrics["field_effective_mode"], "with_field")
        self.assertTrue(any("回退 Python" in msg for msg in logs))


    def test_field_debug_rejects_candidate_when_max_error_worsens(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.core.state import MeshState
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            g=(0.0, 0.0, 0.0),
            q_ion=1.0,
            K_p=1.0,
            K_i=0.0,
            K_d=0.0,
            err_target=0.0,
            field_regularization=0.0,
            rms_guard_tolerance=0.01,
            v_lift=0.0,
            max_iters=1,
            N_stable=1,
        )
        n_vertices = 100
        mesh = MeshState(
            vertices=np.ones((n_vertices, 3), dtype=float),
            tets=np.zeros((0, 4), dtype=np.int32),
            layer_id_per_vertex=np.zeros(n_vertices, dtype=np.int32),
            layer_id_per_tet=np.zeros(0, dtype=np.int32),
            first_active_layer=np.zeros(n_vertices, dtype=np.int32),
            is_top_surface_of_layer=np.full(n_vertices, -1, dtype=np.int32),
        )
        mesh.is_top_surface_of_layer[0] = 1
        mesh.ideal_vertices[:] = mesh.vertices
        mesh.active_vertex_mask[:] = True
        mesh.active_tet_mask = np.zeros(0, dtype=bool)
        mesh.colors = np.zeros(n_vertices, dtype=np.int32)
        mesh.node_mass = np.ones(n_vertices, dtype=float)

        class FakeSolver:
            def __init__(self, cfg) -> None:  # noqa: ANN001
                self.config = cfg

            def solve_until_stable(self, mesh_arg, layer_id, e_z, on_iteration=None):  # noqa: ANN001, ARG002
                if float(e_z) <= 0.0:
                    mesh_arg.vertices[:, 2] = 0.9
                else:
                    mesh_arg.vertices[:, 2] = 1.0
                    mesh_arg.vertices[1, 2] = 0.895
                return VBDSolveResult(
                    x=mesh_arg.vertices.copy(),
                    v=np.zeros_like(mesh_arg.vertices),
                    iterations=1,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=True,
                    chebyshev_skipped_damaging=0,
                )

        worker = SimulationWorker(
            mesh=mesh,
            config=config,
            n_layers=1,
            output_dir="outputs/test_worker_field_debug_max_guard",
            use_cpp=False,
            field_debug_enabled=True,
        )
        worker._trace = lambda msg: None

        with patch(
            "hydrogel_vbd.solver.vbd_solver.PythonReferenceVBDSolver",
            FakeSolver,
        ):
            results = worker._run_layers()

        metrics = results[0].error_metrics
        self.assertEqual(metrics["field_effective_mode"], "no_field")
        self.assertEqual(metrics["field_guard_passed"], 0.0)
        self.assertEqual(metrics["field_guard_reason"], "max_error_worse")
        self.assertAlmostEqual(metrics["E_z"], 0.0)
        self.assertGreater(
            metrics["field_with_field_max_error"],
            metrics["field_no_field_max_error"],
        )

    def test_field_debug_keeps_no_field_when_candidate_has_no_improvement(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            g=(0.0, 0.0, 0.0),
            q_ion=1.0,
            K_p=1.0,
            K_i=0.0,
            K_d=0.0,
            err_target=0.0,
            field_regularization=0.0,
            v_lift=0.0,
            max_iters=1,
            N_stable=1,
        )
        mesh = PythonLiftSolverStabilityTests._single_vertex_mesh()
        mesh.vertices[0, 2] = 1.0
        mesh.ideal_vertices[0, 2] = 1.0

        class FakeSolver:
            def __init__(self, cfg) -> None:  # noqa: ANN001
                self.config = cfg

            def solve_until_stable(self, mesh_arg, layer_id, e_z, on_iteration=None):  # noqa: ANN001, ARG002
                mesh_arg.vertices[0, 2] = 0.5
                return VBDSolveResult(
                    x=mesh_arg.vertices.copy(),
                    v=np.zeros_like(mesh_arg.vertices),
                    iterations=1,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=True,
                    chebyshev_skipped_damaging=0,
                )

        worker = SimulationWorker(
            mesh=mesh,
            config=config,
            n_layers=1,
            output_dir="outputs/test_worker_field_debug_no_improvement",
            use_cpp=False,
            field_debug_enabled=True,
        )
        worker._trace = lambda msg: None

        with patch(
            "hydrogel_vbd.solver.vbd_solver.PythonReferenceVBDSolver",
            FakeSolver,
        ):
            results = worker._run_layers()

        metrics = results[0].error_metrics
        self.assertEqual(metrics["field_effective_mode"], "no_field")
        self.assertEqual(metrics["field_guard_reason"], "no_improvement")
        self.assertAlmostEqual(metrics["field_derived_E_z"], 0.5)
        self.assertAlmostEqual(metrics["E_z"], 0.0)

    def test_field_debug_emits_only_selected_frame_and_records_perf_metrics(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            g=(0.0, 0.0, 0.0),
            q_ion=1.0,
            K_p=1.0,
            K_i=0.0,
            K_d=0.0,
            err_target=0.0,
            field_regularization=0.0,
            v_lift=0.0,
            max_iters=1,
            N_stable=1,
        )
        mesh = PythonLiftSolverStabilityTests._single_vertex_mesh()
        mesh.vertices[0, 2] = 1.0
        mesh.ideal_vertices[0, 2] = 1.0

        class FakeSolver:
            def __init__(self, cfg) -> None:  # noqa: ANN001
                self.config = cfg

            def solve_until_stable(self, mesh_arg, layer_id, e_z, on_iteration=None):  # noqa: ANN001, ARG002
                mesh_arg.vertices[0, 2] = 0.5 if float(e_z) <= 0.0 else 0.9
                return VBDSolveResult(
                    x=mesh_arg.vertices.copy(),
                    v=np.zeros_like(mesh_arg.vertices),
                    iterations=1,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=True,
                    chebyshev_skipped_damaging=0,
                )

        out_dir = Path("outputs/test_worker_field_debug_frames")
        perf_csv = out_dir / "reports" / "performance_diagnostics.csv"
        perf_csv.unlink(missing_ok=True)
        worker = SimulationWorker(
            mesh=mesh,
            config=config,
            n_layers=1,
            output_dir=out_dir,
            use_cpp=False,
            solver_diagnostics_enabled=True,
            field_debug_enabled=True,
        )
        worker._trace = lambda msg: None
        frames: list[dict] = []
        worker.frame_ready.connect(frames.append)

        with patch(
            "hydrogel_vbd.solver.vbd_solver.PythonReferenceVBDSolver",
            FakeSolver,
        ):
            results = worker._run_layers()

        titles = [frame["title"] for frame in frames]
        self.assertFalse(any("no-field" in title for title in titles))
        self.assertTrue(any("selected" in title for title in titles))

        metrics = results[0].error_metrics
        self.assertIn("perf_no_field_ms", metrics)
        self.assertIn("perf_with_field_ms", metrics)
        self.assertIn("perf_render_ms", metrics)
        self.assertGreaterEqual(metrics["perf_no_field_ms"], 0.0)
        self.assertGreaterEqual(metrics["perf_with_field_ms"], 0.0)
        header = perf_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertIn("no_field_ms", header)
        self.assertIn("with_field_ms", header)
        self.assertIn("czm_sync_ms", header)

    def test_field_debug_v2_applies_field_only_at_detach_and_peak_windows(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            g=(0.0, 0.0, 0.0),
            q_ion=1.0,
            K_p=1.0,
            K_i=0.0,
            K_d=0.0,
            err_target=0.0,
            field_regularization=0.0,
            dt=1.0,
            v_lift=1.0,
            layer_thickness=1.0,
            lift_height=5.0,
            max_iters=1,
            N_stable=1,
            epsilon=1.0,
        )
        from hydrogel_vbd.core.state import MeshState

        mesh = MeshState(
            vertices=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
            tets=np.zeros((0, 4), dtype=np.int32),
            layer_id_per_vertex=np.zeros(2, dtype=np.int32),
            layer_id_per_tet=np.zeros(0, dtype=np.int32),
            first_active_layer=np.zeros(2, dtype=np.int32),
            is_top_surface_of_layer=np.array([1, 0], dtype=np.int32),
        )
        mesh.ideal_vertices[:] = mesh.vertices
        mesh.active_vertex_mask[:] = True
        mesh.active_tet_mask = np.zeros(0, dtype=bool)
        mesh.colors = np.zeros(2, dtype=np.int32)
        mesh.node_mass = np.ones(2, dtype=float)
        call_ez: list[float] = []

        class FakeSolver:
            def __init__(self, cfg) -> None:  # noqa: ANN001
                self.config = cfg

            def solve_with_lift(self, mesh_arg, layer_id, e_z, lifting_top, on_iteration=None):  # noqa: ANN001, ARG002
                call_ez.append(float(e_z))
                mesh_arg.vertices[0, 2] = 0.9 if float(e_z) > 0.0 else 0.5
                step_in_branch = ((len(call_ez) - 1) % 5) + 1
                return VBDSolveResult(
                    x=mesh_arg.vertices.copy(),
                    v=np.zeros_like(mesh_arg.vertices),
                    iterations=1,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=(step_in_branch == 3),
                    chebyshev_skipped_damaging=0,
                )

        worker = SimulationWorker(
            mesh=mesh,
            config=config,
            n_layers=1,
            output_dir="outputs/test_worker_field_debug_v2_windows",
            use_cpp=False,
            field_debug_enabled=True,
        )
        worker._trace = lambda msg: None

        with patch(
            "hydrogel_vbd.solver.vbd_solver.PythonReferenceVBDSolver",
            FakeSolver,
        ):
            results = worker._run_layers()

        self.assertEqual(call_ez[:5], [0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(call_ez[5:10], [0.0, 0.0, 0.5, 0.5, 0.5])
        metrics = results[0].error_metrics
        self.assertEqual(metrics["field_timing_mode"], "event_windows_v2")
        self.assertEqual(metrics["field_window_detach_step"], 3.0)
        self.assertEqual(metrics["field_window_peak_start_step"], 5.0)
        self.assertEqual(metrics["field_window_applied_steps"], 3.0)

    def test_field_debug_v2_uses_separate_detach_and_peak_field_values(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.core.state import MeshState
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            g=(0.0, 0.0, 0.0),
            q_ion=1.0,
            K_p=1.0,
            K_i=0.0,
            K_d=0.0,
            err_target=0.0,
            field_regularization=0.0,
            dt=1.0,
            v_lift=1.0,
            layer_thickness=1.0,
            lift_height=5.0,
            max_iters=1,
            N_stable=1,
            epsilon=1.0,
        )
        mesh = MeshState(
            vertices=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
            tets=np.zeros((0, 4), dtype=np.int32),
            layer_id_per_vertex=np.zeros(2, dtype=np.int32),
            layer_id_per_tet=np.zeros(0, dtype=np.int32),
            first_active_layer=np.zeros(2, dtype=np.int32),
            is_top_surface_of_layer=np.array([1, 0], dtype=np.int32),
        )
        mesh.ideal_vertices[:] = mesh.vertices
        mesh.active_vertex_mask[:] = True
        mesh.active_tet_mask = np.zeros(0, dtype=bool)
        mesh.colors = np.zeros(2, dtype=np.int32)
        mesh.node_mass = np.ones(2, dtype=float)
        call_ez: list[float] = []

        class FakeSolver:
            def __init__(self, cfg) -> None:  # noqa: ANN001
                self.config = cfg

            def solve_with_lift(self, mesh_arg, layer_id, e_z, lifting_top, on_iteration=None):  # noqa: ANN001, ARG002
                call_ez.append(float(e_z))
                step_in_branch = ((len(call_ez) - 1) % 5) + 1
                if len(call_ez) <= 5:
                    mesh_arg.vertices[0, 2] = 0.3 if step_in_branch == 3 else 0.8
                else:
                    mesh_arg.vertices[0, 2] = 0.9
                return VBDSolveResult(
                    x=mesh_arg.vertices.copy(),
                    v=np.zeros_like(mesh_arg.vertices),
                    iterations=1,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=(step_in_branch == 3),
                    chebyshev_skipped_damaging=0,
                )

        worker = SimulationWorker(
            mesh=mesh,
            config=config,
            n_layers=1,
            output_dir="outputs/test_worker_field_debug_v2_split_fields",
            use_cpp=False,
            field_debug_enabled=True,
        )
        worker._trace = lambda msg: None

        with patch(
            "hydrogel_vbd.solver.vbd_solver.PythonReferenceVBDSolver",
            FakeSolver,
        ):
            results = worker._run_layers()

        self.assertEqual(call_ez[:5], [0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(call_ez[5:10], [0.0, 0.0, 0.7, 0.7, 0.2])
        metrics = results[0].error_metrics
        self.assertAlmostEqual(metrics["field_detach_E_z"], 0.7)
        self.assertAlmostEqual(metrics["field_peak_E_z"], 0.2)

    def test_field_debug_commits_detach_state_not_peak_guard_state(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.core.state import MeshState
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            g=(0.0, 0.0, 0.0),
            q_ion=1.0,
            K_p=1.0,
            K_i=0.0,
            K_d=0.0,
            err_target=0.0,
            field_regularization=0.0,
            dt=1.0,
            v_lift=1.0,
            layer_thickness=1.0,
            lift_height=5.0,
            max_iters=1,
            N_stable=1,
            epsilon=1.0,
        )
        mesh = MeshState(
            vertices=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
            tets=np.zeros((0, 4), dtype=np.int32),
            layer_id_per_vertex=np.zeros(2, dtype=np.int32),
            layer_id_per_tet=np.zeros(0, dtype=np.int32),
            first_active_layer=np.zeros(2, dtype=np.int32),
            is_top_surface_of_layer=np.array([1, 0], dtype=np.int32),
        )
        mesh.ideal_vertices[:] = mesh.vertices
        mesh.active_vertex_mask[:] = True
        mesh.active_tet_mask = np.zeros(0, dtype=bool)
        mesh.colors = np.zeros(2, dtype=np.int32)
        mesh.node_mass = np.ones(2, dtype=float)
        call_count = 0

        class FakeSolver:
            def __init__(self, cfg) -> None:  # noqa: ANN001
                self.config = cfg

            def solve_with_lift(self, mesh_arg, layer_id, e_z, lifting_top, on_iteration=None):  # noqa: ANN001, ARG002
                nonlocal call_count
                call_count += 1
                step_in_branch = ((call_count - 1) % 5) + 1
                mesh_arg.vertices[:, 2] = float(step_in_branch)
                return VBDSolveResult(
                    x=mesh_arg.vertices.copy(),
                    v=np.zeros_like(mesh_arg.vertices),
                    iterations=1,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=(step_in_branch == 2),
                    chebyshev_skipped_damaging=0,
                )

        worker = SimulationWorker(
            mesh=mesh,
            config=config,
            n_layers=1,
            output_dir="outputs/test_worker_field_debug_commit_detach",
            use_cpp=False,
            field_debug_enabled=True,
        )
        worker._trace = lambda msg: None
        frames: list[dict] = []
        worker.frame_ready.connect(frames.append)

        with patch(
            "hydrogel_vbd.solver.vbd_solver.PythonReferenceVBDSolver",
            FakeSolver,
        ):
            results = worker._run_layers()

        self.assertAlmostEqual(results[0].x_sim[0, 2], 2.0)
        metrics = results[0].error_metrics
        self.assertEqual(metrics["field_effective_mode"], "no_field")
        self.assertEqual(metrics["field_window_detach_step"], 2.0)
        self.assertEqual(metrics["field_commit_step"], 2.0)
        self.assertEqual(metrics["field_guard_step"], 5.0)
        selected_frames = [
            frame for frame in frames if "selected no_field" in frame["title"]
        ]
        self.assertEqual(len(selected_frames), 1)
        self.assertAlmostEqual(selected_frames[0]["vertices"][0, 2], 2.0)

    def test_field_debug_czm_disabled_commits_all_free_state_not_peak(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.core.state import MeshState
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            enable_czm=False,
            g=(0.0, 0.0, 0.0),
            q_ion=1.0,
            K_p=1.0,
            K_i=0.0,
            K_d=0.0,
            err_target=0.0,
            field_regularization=0.0,
            dt=1.0,
            v_lift=1.0,
            layer_thickness=1.0,
            lift_height=5.0,
            max_iters=1,
            N_stable=1,
            epsilon=1.0,
        )
        mesh = MeshState(
            vertices=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
            tets=np.zeros((0, 4), dtype=np.int32),
            layer_id_per_vertex=np.zeros(2, dtype=np.int32),
            layer_id_per_tet=np.zeros(0, dtype=np.int32),
            first_active_layer=np.zeros(2, dtype=np.int32),
            is_top_surface_of_layer=np.array([1, 0], dtype=np.int32),
        )
        mesh.ideal_vertices[:] = mesh.vertices
        mesh.active_vertex_mask[:] = True
        mesh.active_tet_mask = np.zeros(0, dtype=bool)
        mesh.colors = np.zeros(2, dtype=np.int32)
        mesh.node_mass = np.ones(2, dtype=float)
        call_count = 0

        class FakeSolver:
            def __init__(self, cfg) -> None:  # noqa: ANN001
                self.config = cfg

            def solve_with_lift(self, mesh_arg, layer_id, e_z, lifting_top, on_iteration=None):  # noqa: ANN001, ARG002
                nonlocal call_count
                call_count += 1
                step_in_branch = ((call_count - 1) % 5) + 1
                mesh_arg.vertices[:, 2] = float(step_in_branch)
                return VBDSolveResult(
                    x=mesh_arg.vertices.copy(),
                    v=np.zeros_like(mesh_arg.vertices),
                    iterations=1,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=(step_in_branch == 2),
                    chebyshev_skipped_damaging=0,
                )

        worker = SimulationWorker(
            mesh=mesh,
            config=config,
            n_layers=1,
            output_dir="outputs/test_worker_field_debug_czm_disabled_commit",
            use_cpp=False,
            field_debug_enabled=True,
        )
        worker._trace = lambda msg: None

        with patch(
            "hydrogel_vbd.solver.vbd_solver.PythonReferenceVBDSolver",
            FakeSolver,
        ):
            results = worker._run_layers()

        metrics = results[0].error_metrics
        self.assertAlmostEqual(results[0].x_sim[0, 2], 2.0)
        self.assertEqual(metrics["field_window_detach_step"], 2.0)
        self.assertEqual(metrics["field_commit_step"], 2.0)
        self.assertEqual(metrics["field_guard_step"], 5.0)

    def test_field_debug_cpp_czm_sync_commits_python_all_free_state(self) -> None:
        from types import SimpleNamespace

        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.core.state import MeshState
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.physics.czm import CZMState
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            enable_czm=True,
            g=(0.0, 0.0, 0.0),
            q_ion=1.0,
            K_p=1.0,
            K_i=0.0,
            K_d=0.0,
            err_target=0.0,
            field_regularization=0.0,
            dt=1.0,
            v_lift=1.0,
            layer_thickness=1.0,
            lift_height=5.0,
            max_iters=1,
            N_stable=1,
            epsilon=1.0,
        )
        mesh = MeshState(
            vertices=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
            tets=np.zeros((0, 4), dtype=np.int32),
            layer_id_per_vertex=np.zeros(2, dtype=np.int32),
            layer_id_per_tet=np.zeros(0, dtype=np.int32),
            first_active_layer=np.zeros(2, dtype=np.int32),
            is_top_surface_of_layer=np.array([1, 0], dtype=np.int32),
            is_top_fixed=np.array([True, False], dtype=bool),
        )
        mesh.ideal_vertices[:] = mesh.vertices
        mesh.active_vertex_mask[:] = True
        mesh.active_tet_mask = np.zeros(0, dtype=bool)
        mesh.colors = np.zeros(2, dtype=np.int32)
        mesh.node_mass = np.ones(2, dtype=float)
        cpp_calls = 0
        czm_updates = 0

        def fake_cpp_solve(mesh_arg, cfg_arg, e_z, layer_id, lifting_top):  # noqa: ANN001, ARG001
            nonlocal cpp_calls
            cpp_calls += 1
            step_in_branch = ((cpp_calls - 1) % 5) + 1
            mesh_arg.vertices[:, 2] = float(step_in_branch)
            return VBDSolveResult(
                x=mesh_arg.vertices.copy(),
                v=np.zeros_like(mesh_arg.vertices),
                iterations=1,
                max_dx=0.0,
                kinetic_energy=0.0,
                stable_steps=1,
                all_free=False,
                chebyshev_skipped_damaging=0,
            )

        def fake_update_czm_states(mesh_arg, bottom, **kwargs):  # noqa: ANN001, ARG001
            nonlocal czm_updates
            czm_updates += 1
            if ((czm_updates - 1) % 5) + 1 >= 2:
                mesh_arg.czm_state[bottom] = int(CZMState.FREE)

        def fake_branch_runner(mesh_arg, cfg_arg, layer_id, e_z, lifting_top, **kwargs):  # noqa: ANN001, ARG001
            commit_vertices = mesh_arg.vertices.copy()
            guard_vertices = mesh_arg.vertices.copy()
            commit_vertices[:, 2] = 2.0
            guard_vertices[:, 2] = 5.0
            return SimpleNamespace(
                commit_vertices=commit_vertices,
                commit_velocities=np.zeros_like(commit_vertices),
                commit_czm_state=np.full_like(mesh_arg.czm_state, int(CZMState.FREE)),
                commit_damage=mesh_arg.damage.copy(),
                commit_time_free=mesh_arg.time_free.copy(),
                guard_vertices=guard_vertices,
                guard_velocities=np.zeros_like(guard_vertices),
                guard_czm_state=np.full_like(mesh_arg.czm_state, int(CZMState.FREE)),
                guard_damage=mesh_arg.damage.copy(),
                guard_time_free=mesh_arg.time_free.copy(),
                commit_result=VBDSolveResult(
                    x=commit_vertices,
                    v=np.zeros_like(commit_vertices),
                    iterations=1,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=True,
                    chebyshev_skipped_damaging=0,
                ),
                guard_result=VBDSolveResult(
                    x=guard_vertices,
                    v=np.zeros_like(guard_vertices),
                    iterations=1,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=True,
                    chebyshev_skipped_damaging=0,
                ),
                commit_steps=2,
                executed_steps=5,
                total_iterations=5,
                max_iter_hits=0,
                clipped_steps=0,
                lift_max=5.0,
                info={
                    "timing_mode": "event_windows_v2",
                    "expected_steps": 5.0,
                    "detach_step": 2.0,
                    "commit_step": 2.0,
                    "guard_step": 5.0,
                    "return_steps": 0.0,
                    "platform_return_distance": 0.0,
                    "peak_start_step": 5.0,
                    "applied_steps": 0.0,
                    "detach_E_z": float(e_z),
                    "peak_E_z": float(e_z),
                    "cpp_solve_ms": 1.0,
                    "python_solve_ms": 0.0,
                    "czm_sync_ms": 0.0,
                    "return_ms": 0.0,
                    "snapshot_ms": 0.0,
                    "branch_runner": 1.0,
                },
            )

        with patch(
            "hydrogel_vbd.gui.simulation_worker.is_cpp_available",
            return_value=True,
        ):
            worker = SimulationWorker(
                mesh=mesh,
                config=config,
                n_layers=1,
                output_dir="outputs/test_worker_field_debug_cpp_czm_sync",
                use_cpp=True,
                field_debug_enabled=True,
            )
        worker._trace = lambda msg: None

        with (
            patch(
                "hydrogel_vbd.solver.cpp_adapter.solve_field_debug_branch",
                side_effect=fake_branch_runner,
            ),
            patch(
                "hydrogel_vbd.physics.local_terms.build_local_physics_terms",
                return_value=SimpleNamespace(force=np.zeros((2, 3), dtype=float)),
            ),
            patch(
                "hydrogel_vbd.physics.czm.update_czm_states",
                side_effect=fake_update_czm_states,
            ),
        ):
            results = worker._run_layers()

        metrics = results[0].error_metrics
        self.assertAlmostEqual(results[0].x_sim[0, 2], 2.0)
        self.assertEqual(metrics["field_debug_solver_backend"], "cpp_adapter")
        self.assertEqual(metrics["field_commit_step"], 2.0)
        self.assertEqual(metrics["field_guard_step"], 5.0)

    def test_field_debug_cpp_prefers_single_branch_runner_call(self) -> None:
        from types import SimpleNamespace

        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.core.state import MeshState
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            enable_czm=True,
            g=(0.0, 0.0, 0.0),
            q_ion=1.0,
            K_p=1.0,
            K_i=0.0,
            K_d=0.0,
            err_target=0.0,
            field_regularization=0.0,
            dt=1.0,
            v_lift=1.0,
            layer_thickness=1.0,
            lift_height=5.0,
            max_iters=1,
            N_stable=1,
            epsilon=1.0,
        )
        mesh = MeshState(
            vertices=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
            tets=np.zeros((0, 4), dtype=np.int32),
            layer_id_per_vertex=np.zeros(2, dtype=np.int32),
            layer_id_per_tet=np.zeros(0, dtype=np.int32),
            first_active_layer=np.zeros(2, dtype=np.int32),
            is_top_surface_of_layer=np.array([1, 0], dtype=np.int32),
            is_top_fixed=np.array([True, False], dtype=bool),
        )
        mesh.ideal_vertices[:] = mesh.vertices
        mesh.active_vertex_mask[:] = True
        mesh.active_tet_mask = np.zeros(0, dtype=bool)
        mesh.colors = np.zeros(2, dtype=np.int32)
        mesh.node_mass = np.ones(2, dtype=float)
        branch_calls: list[float] = []

        def fake_branch_runner(
            mesh_arg, cfg_arg, layer_id, e_z, lifting_top, **kwargs
        ):  # noqa: ANN001, ARG001
            branch_calls.append(float(e_z))
            commit_vertices = mesh_arg.vertices.copy()
            guard_vertices = mesh_arg.vertices.copy()
            if float(e_z) <= 0.0:
                commit_vertices[:, 2] = -0.5
                guard_vertices[:, 2] = -0.5
            else:
                commit_vertices[:, 2] = -0.2
                guard_vertices[:, 2] = -0.2
            return SimpleNamespace(
                commit_vertices=commit_vertices,
                commit_velocities=np.zeros_like(commit_vertices),
                commit_czm_state=mesh_arg.czm_state.copy(),
                commit_damage=mesh_arg.damage.copy(),
                commit_time_free=mesh_arg.time_free.copy(),
                guard_vertices=guard_vertices,
                guard_velocities=np.zeros_like(guard_vertices),
                guard_czm_state=mesh_arg.czm_state.copy(),
                guard_damage=mesh_arg.damage.copy(),
                guard_time_free=mesh_arg.time_free.copy(),
                commit_result=VBDSolveResult(
                    x=commit_vertices,
                    v=np.zeros_like(commit_vertices),
                    iterations=1,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=True,
                    chebyshev_skipped_damaging=0,
                ),
                guard_result=VBDSolveResult(
                    x=guard_vertices,
                    v=np.zeros_like(guard_vertices),
                    iterations=1,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=True,
                    chebyshev_skipped_damaging=0,
                ),
                commit_steps=2,
                executed_steps=5,
                total_iterations=5,
                max_iter_hits=0,
                clipped_steps=0,
                lift_max=5.0,
                info={
                    "timing_mode": "event_windows_v2",
                    "expected_steps": 5.0,
                    "detach_step": 2.0,
                    "commit_step": 2.0,
                    "guard_step": 5.0,
                    "return_steps": 0.0,
                    "platform_return_distance": 0.0,
                    "peak_start_step": 5.0,
                    "applied_steps": 3.0 if float(e_z) > 0.0 else 0.0,
                    "detach_E_z": float(e_z),
                    "peak_E_z": float(kwargs.get("peak_e_z") or e_z),
                    "cpp_solve_ms": 1.0,
                    "python_solve_ms": 0.0,
                    "czm_sync_ms": 0.0,
                    "return_ms": 0.0,
                    "snapshot_ms": 0.0,
                    "branch_runner": 1.0,
                },
            )

        with patch(
            "hydrogel_vbd.gui.simulation_worker.is_cpp_available",
            return_value=True,
        ):
            worker = SimulationWorker(
                mesh=mesh,
                config=config,
                n_layers=1,
                output_dir="outputs/test_worker_field_debug_branch_runner",
                use_cpp=True,
                field_debug_enabled=True,
            )
        worker._trace = lambda msg: None

        with (
            patch(
                "hydrogel_vbd.solver.cpp_adapter.solve_field_debug_branch",
                side_effect=fake_branch_runner,
            ),
            patch(
                "hydrogel_vbd.solver.cpp_adapter.solve_lift_and_relax",
                side_effect=AssertionError(
                    "branch runner should replace per-step C++ calls"
                ),
            ),
        ):
            results = worker._run_layers()

        self.assertEqual(branch_calls, [0.0, 0.5])
        metrics = results[0].error_metrics
        self.assertEqual(metrics["field_debug_solver_backend"], "cpp_adapter")
        self.assertEqual(metrics["field_branch_runner_enabled"], 1.0)
        self.assertEqual(metrics["field_commit_step"], 2.0)
        self.assertEqual(metrics["field_guard_step"], 5.0)

    def test_field_debug_cpp_returns_platform_before_committing_peak_no_detach(self) -> None:
        from types import SimpleNamespace

        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.core.state import MeshState
        from hydrogel_vbd.gui.simulation_worker import SimulationWorker
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            enable_czm=True,
            g=(0.0, 0.0, 0.0),
            q_ion=1.0,
            K_p=1.0,
            K_i=0.0,
            K_d=0.0,
            err_target=0.0,
            field_regularization=0.0,
            dt=1.0,
            v_lift=1.0,
            layer_thickness=1.0,
            lift_height=5.0,
            max_iters=1,
            N_stable=1,
            epsilon=1.0,
        )
        mesh = MeshState(
            vertices=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
            tets=np.zeros((0, 4), dtype=np.int32),
            layer_id_per_vertex=np.zeros(2, dtype=np.int32),
            layer_id_per_tet=np.zeros(0, dtype=np.int32),
            first_active_layer=np.zeros(2, dtype=np.int32),
            is_top_surface_of_layer=np.array([1, 0], dtype=np.int32),
            is_top_fixed=np.array([True, False], dtype=bool),
        )
        mesh.ideal_vertices[:] = mesh.vertices
        mesh.active_vertex_mask[:] = True
        mesh.active_tet_mask = np.zeros(0, dtype=bool)
        mesh.colors = np.zeros(2, dtype=np.int32)
        mesh.node_mass = np.ones(2, dtype=float)
        branch_calls: list[float] = []

        def fake_branch_runner(mesh_arg, cfg_arg, layer_id, e_z, lifting_top, **kwargs):  # noqa: ANN001, ARG001
            branch_calls.append(float(e_z))
            commit_vertices = mesh_arg.vertices.copy()
            guard_vertices = mesh_arg.vertices.copy()
            return SimpleNamespace(
                commit_vertices=commit_vertices,
                commit_velocities=np.zeros_like(commit_vertices),
                commit_czm_state=mesh_arg.czm_state.copy(),
                commit_damage=mesh_arg.damage.copy(),
                commit_time_free=mesh_arg.time_free.copy(),
                guard_vertices=guard_vertices,
                guard_velocities=np.zeros_like(guard_vertices),
                guard_czm_state=mesh_arg.czm_state.copy(),
                guard_damage=mesh_arg.damage.copy(),
                guard_time_free=mesh_arg.time_free.copy(),
                commit_result=VBDSolveResult(
                    x=commit_vertices,
                    v=np.zeros_like(commit_vertices),
                    iterations=5,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=False,
                    chebyshev_skipped_damaging=0,
                ),
                guard_result=VBDSolveResult(
                    x=guard_vertices,
                    v=np.zeros_like(guard_vertices),
                    iterations=5,
                    max_dx=0.0,
                    kinetic_energy=0.0,
                    stable_steps=1,
                    all_free=False,
                    chebyshev_skipped_damaging=0,
                ),
                commit_steps=5,
                executed_steps=5,
                total_iterations=10,
                max_iter_hits=0,
                clipped_steps=0,
                lift_max=5.0,
                info={
                    "timing_mode": "event_windows_v2",
                    "expected_steps": 5.0,
                    "detach_step": 0.0,
                    "commit_step": 5.0,
                    "guard_step": 5.0,
                    "return_steps": 5.0,
                    "platform_return_distance": 5.0,
                    "peak_start_step": 5.0,
                    "applied_steps": 0.0,
                    "detach_E_z": float(e_z),
                    "peak_E_z": float(e_z),
                    "cpp_solve_ms": 1.0,
                    "python_solve_ms": 0.0,
                    "czm_sync_ms": 0.0,
                    "return_ms": 0.0,
                    "snapshot_ms": 0.0,
                    "branch_runner": 1.0,
                },
            )

        with patch(
            "hydrogel_vbd.gui.simulation_worker.is_cpp_available",
            return_value=True,
        ):
            worker = SimulationWorker(
                mesh=mesh,
                config=config,
                n_layers=2,
                output_dir="outputs/test_worker_field_debug_cpp_return_commit",
                use_cpp=True,
                field_debug_enabled=True,
            )
        worker._trace = lambda msg: None

        with patch(
            "hydrogel_vbd.solver.cpp_adapter.solve_field_debug_branch",
            side_effect=fake_branch_runner,
        ):
            results = worker._run_layers()

        self.assertAlmostEqual(results[0].x_sim[0, 2], 0.0)
        metrics = results[0].error_metrics
        self.assertEqual(metrics["field_commit_step"], 5.0)
        self.assertEqual(metrics["field_guard_step"], 5.0)
        self.assertEqual(metrics["field_platform_return_steps"], 5.0)
        self.assertEqual(branch_calls[:1], [0.0])


class LayerActivatorTopFallbackTests(unittest.TestCase):
    """层顶面分类缺失时的激活回归测试。"""

    def test_final_layer_uses_geometry_top_when_interface_id_is_missing(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.geometry.layer_activator import LayerActivator

        config = SimulationConfig(layer_thickness=1.0e-4)
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=2, layer_thickness=config.layer_thickness, config=config
        )
        original_final_top = mesh.top_nodes(1)
        self.assertGreater(len(original_final_top), 0)

        # Simulate STL/OCC classification that lacks interface_id == n_layers.
        mesh.is_top_surface_of_layer[original_final_top] = -1

        LayerActivator().activate_with_inheritance(
            mesh, current_layer=1, z_fep=config.z_fep
        )

        lifting_top = np.flatnonzero(mesh.is_top_fixed & mesh.active_vertex_mask)
        np.testing.assert_array_equal(lifting_top, original_final_top)


class VbdSolverSolveUntilStableTests(unittest.TestCase):
    """确保 solve_until_stable 向后兼容。"""

    def setUp(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline

        self.config = SimulationConfig(max_iters=4, epsilon=1e-12, N_stable=2)
        self.mesh, _ = ConformalMeshPipeline.create_demo(
            layers=1, layer_thickness=0.05, config=self.config
        )
        self.mesh.activate_layer(0)

    def test_solve_until_stable_still_works(self) -> None:
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver

        solver = PythonReferenceVBDSolver(self.config)
        result = solver.solve_until_stable(self.mesh, layer_id=0, e_z=20.0)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertEqual(result.x.shape, self.mesh.vertices.shape)


class StlMesherTests(unittest.TestCase):
    """STL 加载、分层、四面体划分测试。"""

    def test_create_demo_or_stl_demo_fallback_runs(self) -> None:
        from hydrogel_vbd.geometry.stl_mesher import create_demo_or_stl

        mesh, n_layers = create_demo_or_stl(
            stl_path=None,
            layers=3,
            layer_thickness=0.05,
        )
        self.assertGreater(mesh.vertices.shape[0], 0)
        self.assertGreater(mesh.tets.shape[0], 0)
        self.assertEqual(mesh.tets.shape[1], 4)

    def test_create_demo_or_stl_produces_layers(self) -> None:
        from hydrogel_vbd.geometry.stl_mesher import create_demo_or_stl

        mesh, n_layers = create_demo_or_stl(
            stl_path=None,
            layers=5,
            layer_thickness=0.05,
        )
        self.assertGreaterEqual(n_layers, 3)
        self.assertLessEqual(n_layers, 5)

    def test_stl_mesher_class_exists_and_accepts_path(self) -> None:
        from hydrogel_vbd.geometry.stl_mesher import STLMesher

        mesher = STLMesher(
            stl_path="nonexistent.stl",
            layer_thickness=0.05,
            resolution=0.02,
        )
        self.assertEqual(mesher.stl_path, "nonexistent.stl")
        self.assertAlmostEqual(mesher.resolution, 0.02)

    def test_fine_print_layer_resolution_is_not_clamped_to_one_mm(self) -> None:
        from hydrogel_vbd.geometry.stl_mesher import (
            DelaunayTetMesher,
            OCCFragmentMesher,
        )

        for mesher_type in (OCCFragmentMesher, DelaunayTetMesher):
            mesher = mesher_type(
                stl_path="nonexistent.step",
                layer_thickness=5.0e-5,
                resolution=5.0e-5,
            )

            self.assertAlmostEqual(mesher.resolution, 5.0e-5)

    def test_occ_mesh_min_size_tracks_fine_layer_thickness(self) -> None:
        from hydrogel_vbd.geometry.stl_mesher import (
            _occ_gmsh_mesh_retry_attempts_mm,
            _occ_gmsh_mesh_size_bounds_mm,
        )

        max_size_mm, min_size_mm = _occ_gmsh_mesh_size_bounds_mm(
            resolution_m=1.0e-3,
            quality_factor=1.0,
            layer_thickness_m=5.0e-5,
        )

        self.assertAlmostEqual(max_size_mm, 1.0)
        self.assertAlmostEqual(min_size_mm, 0.0125)

        attempts = _occ_gmsh_mesh_retry_attempts_mm(
            max_size_mm, min_size_mm, layer_thickness_m=5.0e-5
        )

        self.assertGreaterEqual(len(attempts), 2)
        self.assertEqual(attempts[0][2:], (4, 5))
        self.assertEqual(attempts[1][2:], (1, 5))
        self.assertAlmostEqual(attempts[0][0], 1.0)
        self.assertAlmostEqual(attempts[0][1], 0.0125)


try:
    from PySide6.QtWidgets import QApplication  # noqa: F401

    _HAS_PYSIDE6 = True
except ImportError:
    _HAS_PYSIDE6 = False

@unittest.skipUnless(_HAS_PYSIDE6, "PySide6 未安装，跳过 GUI 测试")
class GuiParamConfigTests(unittest.TestCase):
    """GUI 参数面板配置测试（需要 PySide6）。"""

    def test_config_roundtrip_from_gui_param_panel(self) -> None:
        from hydrogel_vbd.gui.main_window import ParameterPanel
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        panel = ParameterPanel()
        config = panel.get_config()
        self.assertIsNotNone(config.mu)
        self.assertIsNotNone(config.kappa)
        self.assertIsNotNone(config.v_lift)
        self.assertIsNotNone(config.K_p)
        self.assertAlmostEqual(config.mu, 5000.0)
        self.assertAlmostEqual(config.kappa, 250000.0)
        self.assertAlmostEqual(config.T_max, 3000.0)
        self.assertAlmostEqual(config.delta_f, 0.002)
        self.assertAlmostEqual(config.k_d, 0.05)
        self.assertAlmostEqual(config.c_shrink, 1.0)
        self.assertEqual(config.max_iters, 50)
        self.assertEqual(config.N_stable, 3)
        self.assertAlmostEqual(config.layer_thickness, 0.7993e-3)
        self.assertAlmostEqual(config.v_lift, 0.01)
        self.assertNotIn("lift_multiplier", panel._spin_map)
        self.assertIn("lift_height", panel._spin_map)
        self.assertAlmostEqual(config.lift_height, 5.0e-3)
        self.assertIn("node_area", panel._spin_map)
        self.assertAlmostEqual(config.node_area, 1.0e-6)
        app.quit()

    def test_fluid_czm_parameters_are_exposed_in_gui_param_panel(self) -> None:
        from hydrogel_vbd.gui.main_window import ParameterPanel
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        panel = ParameterPanel()
        values = {
            "eta": 0.6,
            "C_0": 25.0,
            "fluid_radius": 0.0015,
            "d_fluid_max": 0.0004,
            "t_fluid_max": 0.15,
            "d_min": 2.0e-6,
        }
        for key, value in values.items():
            self.assertIn(key, panel._spin_map)
            panel._spin_map[key].setValue(value)

        config = panel.get_config()
        for key, value in values.items():
            self.assertAlmostEqual(getattr(config, key), value)
        app.quit()

    def test_main_window_init(self) -> None:
        from hydrogel_vbd.gui.main_window import MainWindow
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        win = MainWindow()
        self.assertEqual(win.windowTitle(), "Hydrogel VBD Simulator")
        self.assertIsNotNone(win._chk_solver_diag)
        self.assertFalse(win._chk_solver_diag.isChecked())
        self.assertIsNotNone(win._chk_field_debug)
        self.assertFalse(win._chk_field_debug.isChecked())
        self.assertGreaterEqual(win._left_layout.stretch(
            win._left_layout.indexOf(win._param_scroll_area)
        ), 1)
        self.assertTrue(win.isVisible() is False)
        app.quit()

    def test_main_window_resolution_control_accepts_fifty_microns(self) -> None:
        from hydrogel_vbd.gui.main_window import MainWindow
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        win = MainWindow()
        win._spin_resolution.setValue(0.05)

        self.assertLessEqual(win._spin_resolution.minimum(), 0.05)
        self.assertGreaterEqual(win._spin_resolution.decimals(), 2)
        self.assertAlmostEqual(win._spin_resolution.value(), 0.05)
        app.quit()

    def test_mesh_viewer_filters_deformed_mesh_to_active_tets(self) -> None:
        from hydrogel_vbd.gui.mesh_viewer import MeshViewer

        tets = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            dtype=int,
        )
        active_tets = np.array([False, True, False], dtype=bool)

        np.testing.assert_array_equal(
            MeshViewer._visible_tets(tets, active_tets),
            tets[[1]],
        )
        self.assertEqual(
            len(MeshViewer._visible_tets(tets, np.zeros(3, dtype=bool))),
            0,
        )
        np.testing.assert_array_equal(MeshViewer._visible_tets(tets, None), tets)

    def test_mesh_viewer_equal_axis_bounds_preserve_physical_aspect(self) -> None:
        from hydrogel_vbd.gui.mesh_viewer import MeshViewer

        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.020, 0.001, 0.005],
            ],
            dtype=float,
        )

        xlim, ylim, zlim = MeshViewer._equal_axis_bounds(vertices)

        spans = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
        np.testing.assert_allclose(spans, np.repeat(spans[0], 3))
        self.assertLess(xlim[0], 0.0)
        self.assertGreater(xlim[1], 0.020)
        self.assertLess(ylim[0], 0.0)
        self.assertGreater(ylim[1], 0.001)
        self.assertLess(zlim[0], 0.0)
        self.assertGreater(zlim[1], 0.005)

    def test_mesh_viewer_converts_internal_meters_to_display_mm(self) -> None:
        from hydrogel_vbd.gui.mesh_viewer import MeshViewer

        vertices_m = np.array(
            [
                [0.001, -0.002, 0.003],
                [0.010, 0.0005, 0.012],
            ],
            dtype=float,
        )

        np.testing.assert_allclose(
            MeshViewer._display_points_mm(vertices_m),
            np.array(
                [
                    [1.0, -2.0, 3.0],
                    [10.0, 0.5, 12.0],
                ],
                dtype=float,
            ),
        )

    def test_mesh_viewer_surface_polygons_follow_boundary_face_indices(self) -> None:
        from hydrogel_vbd.gui.mesh_viewer import MeshViewer

        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=float,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ],
            dtype=int,
        )

        polygons = MeshViewer._surface_polygons(vertices, faces)

        self.assertEqual(polygons.shape, (2, 3, 3))
        np.testing.assert_array_equal(polygons[0], vertices[[0, 1, 2]])
        np.testing.assert_array_equal(polygons[1], vertices[[0, 2, 3]])
        self.assertNotIn(10.0, polygons)

    def test_mesh_bbox_summary_reports_mm_extents(self) -> None:
        from hydrogel_vbd.gui.main_window import _format_bbox_mm

        vertices = np.array(
            [
                [0.0, -0.002, 0.001],
                [0.020, 0.003, 0.011],
            ],
            dtype=float,
        )

        summary = _format_bbox_mm(vertices)

        self.assertIn("X[0.000, 20.000]", summary)
        self.assertIn("Y[-2.000, 3.000]", summary)
        self.assertIn("Z[1.000, 11.000]", summary)
        self.assertIn("size[20.000, 5.000, 10.000]", summary)

    def test_worker_frame_layer_id_parses_cpp_and_python_titles(self) -> None:
        from hydrogel_vbd.gui.main_window import _frame_layer_id_from_title

        cpp_title = "\u7b2c 3 \u5c42 \u2014 \u4e0b\u653e\u5b8c\u6210/\u5c42\u7ed3\u675f"
        python_title = "\u7b2c 3/10 \u5c42 \u2014 \u63d0\u5347 1.000e-3 m"

        self.assertEqual(_frame_layer_id_from_title(cpp_title), 2)
        self.assertEqual(_frame_layer_id_from_title(python_title), 2)
        self.assertEqual(_frame_layer_id_from_title("仿真完成"), -1)

    def test_step_meshing_boundary_recovery_failure_does_not_retry_standard_for_many_layers(self) -> None:
        from hydrogel_vbd.gui.main_window import _should_retry_standard_meshing

        exc = RuntimeError("Gmsh 3D mesh failed: Could not recover boundary mesh: error 2")

        self.assertFalse(
            _should_retry_standard_meshing(
                algo_type="layered",
                is_step=True,
                actual_layers=190,
                exc=exc,
            )
        )
        self.assertFalse(
            _should_retry_standard_meshing(
                algo_type="standard",
                is_step=True,
                actual_layers=190,
                exc=exc,
            )
        )
        self.assertFalse(
            _should_retry_standard_meshing(
                algo_type="layered",
                is_step=True,
                actual_layers=20,
                exc=exc,
            )
        )

    def test_solver_diagnostics_checkbox_controls_run_environment(self) -> None:
        from hydrogel_vbd.gui.main_window import MainWindow
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        saved = os.environ.get("HYDROGEL_VBD_SOLVER_DIAG")
        try:
            os.environ["HYDROGEL_VBD_SOLVER_DIAG"] = "external"
            win = MainWindow()
            win._chk_solver_diag.setChecked(True)

            win._apply_solver_diagnostics_env_for_run()
            self.assertEqual(os.environ["HYDROGEL_VBD_SOLVER_DIAG"], "1")

            win._restore_solver_diagnostics_env_after_run()
            self.assertEqual(os.environ["HYDROGEL_VBD_SOLVER_DIAG"], "external")
        finally:
            if saved is None:
                os.environ.pop("HYDROGEL_VBD_SOLVER_DIAG", None)
            else:
                os.environ["HYDROGEL_VBD_SOLVER_DIAG"] = saved
            app.quit()

    def test_solver_diagnostics_checkbox_is_passed_to_worker(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.core.state import MeshState
        from hydrogel_vbd.gui.main_window import MainWindow
        from PySide6 import QtCore
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        mesh = MeshState(
            vertices=np.zeros((1, 3), dtype=float),
            tets=np.zeros((0, 4), dtype=np.int32),
            layer_id_per_vertex=np.zeros(1, dtype=np.int32),
            layer_id_per_tet=np.zeros(0, dtype=np.int32),
        )
        win = MainWindow()
        win._generated_mesh = mesh
        win._actual_layers = 1
        win._chk_solver_diag.setChecked(True)

        created = {}

        class FakeThread:
            def __init__(self, parent=None) -> None:  # noqa: ANN001
                self.started = _FakeSignal()
                self.finished = _FakeSignal()

            def start(self) -> None:
                pass

            def quit(self) -> None:
                pass

            def deleteLater(self) -> None:
                pass

            def requestInterruption(self) -> None:
                pass

        class FakeWorker:
            def __init__(self, **kwargs) -> None:  # noqa: ANN003
                created.update(kwargs)
                self.frame_ready = _FakeSignal()
                self.progress_update = _FakeSignal()
                self.log_message = _FakeSignal()
                self.finished = _FakeSignal()
                self.cancelled = _FakeSignal()
                self.error = _FakeSignal()
                self.sub_progress = _FakeSignal()
                self.layer_finished = _FakeSignal()

            def moveToThread(self, thread) -> None:  # noqa: ANN001
                pass

            def run(self) -> None:
                pass

            def deleteLater(self) -> None:
                pass

        with patch(
            "hydrogel_vbd.gui.main_window.SimulationWorker",
            FakeWorker,
        ), patch.object(QtCore, "QThread", FakeThread), patch(
            "hydrogel_vbd.gui.main_window.is_cpp_available",
            return_value=False,
        ):
            win._on_run()

        self.assertTrue(created["solver_diagnostics_enabled"])
        app.quit()

    def test_field_debug_checkbox_is_passed_to_worker(self) -> None:
        from hydrogel_vbd.core.state import MeshState
        from hydrogel_vbd.gui.main_window import MainWindow
        from PySide6 import QtCore
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        mesh = MeshState(
            vertices=np.zeros((1, 3), dtype=float),
            tets=np.zeros((0, 4), dtype=np.int32),
            layer_id_per_vertex=np.zeros(1, dtype=np.int32),
            layer_id_per_tet=np.zeros(0, dtype=np.int32),
        )
        win = MainWindow()
        win._generated_mesh = mesh
        win._actual_layers = 1
        win._chk_field_debug.setChecked(True)

        created = {}

        class FakeThread:
            def __init__(self, parent=None) -> None:  # noqa: ANN001
                self.started = _FakeSignal()
                self.finished = _FakeSignal()

            def start(self) -> None:
                pass

            def quit(self) -> None:
                pass

            def deleteLater(self) -> None:
                pass

            def requestInterruption(self) -> None:
                pass

        class FakeWorker:
            def __init__(self, **kwargs) -> None:  # noqa: ANN003
                created.update(kwargs)
                self.frame_ready = _FakeSignal()
                self.progress_update = _FakeSignal()
                self.log_message = _FakeSignal()
                self.finished = _FakeSignal()
                self.cancelled = _FakeSignal()
                self.error = _FakeSignal()
                self.sub_progress = _FakeSignal()
                self.layer_finished = _FakeSignal()

            def moveToThread(self, thread) -> None:  # noqa: ANN001
                pass

            def run(self) -> None:
                pass

            def deleteLater(self) -> None:
                pass

        with patch(
            "hydrogel_vbd.gui.main_window.SimulationWorker",
            FakeWorker,
        ), patch.object(QtCore, "QThread", FakeThread), patch(
            "hydrogel_vbd.gui.main_window.is_cpp_available",
            return_value=False,
        ):
            win._on_run()

        self.assertTrue(created["field_debug_enabled"])
        app.quit()

    def test_field_debug_cpp_checkbox_uses_direct_adapter_not_subprocess(self) -> None:
        from hydrogel_vbd.core.state import MeshState
        from hydrogel_vbd.gui.main_window import MainWindow
        from PySide6 import QtCore
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        mesh = MeshState(
            vertices=np.zeros((1, 3), dtype=float),
            tets=np.zeros((0, 4), dtype=np.int32),
            layer_id_per_vertex=np.zeros(1, dtype=np.int32),
            layer_id_per_tet=np.zeros(0, dtype=np.int32),
        )
        win = MainWindow()
        win._generated_mesh = mesh
        win._actual_layers = 1
        win._chk_field_debug.setChecked(True)
        win._chk_use_cpp.setChecked(True)

        created = {}

        class FakeThread:
            def __init__(self, parent=None) -> None:  # noqa: ANN001
                self.started = _FakeSignal()
                self.finished = _FakeSignal()

            def start(self) -> None:
                pass

            def quit(self) -> None:
                pass

            def deleteLater(self) -> None:
                pass

            def requestInterruption(self) -> None:
                pass

        class FakeWorker:
            def __init__(self, **kwargs) -> None:  # noqa: ANN003
                created.update(kwargs)
                self.frame_ready = _FakeSignal()
                self.progress_update = _FakeSignal()
                self.log_message = _FakeSignal()
                self.finished = _FakeSignal()
                self.cancelled = _FakeSignal()
                self.error = _FakeSignal()
                self.sub_progress = _FakeSignal()
                self.layer_finished = _FakeSignal()

            def moveToThread(self, thread) -> None:  # noqa: ANN001
                pass

            def run(self) -> None:
                pass

            def deleteLater(self) -> None:
                pass

        with patch(
            "hydrogel_vbd.gui.main_window.SimulationWorker",
            FakeWorker,
        ), patch.object(QtCore, "QThread", FakeThread), patch(
            "hydrogel_vbd.gui.main_window.is_cpp_available",
            return_value=True,
        ):
            win._on_run()

        self.assertFalse(created["use_cpp"])
        self.assertTrue(created["field_debug_use_cpp"])
        app.quit()

    def test_disable_chebyshev_checkbox_sets_runtime_rho_cheb_zero(self) -> None:
        from hydrogel_vbd.core.state import MeshState
        from hydrogel_vbd.gui.main_window import MainWindow
        from PySide6 import QtCore
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        mesh = MeshState(
            vertices=np.zeros((1, 3), dtype=float),
            tets=np.zeros((0, 4), dtype=np.int32),
            layer_id_per_vertex=np.zeros(1, dtype=np.int32),
            layer_id_per_tet=np.zeros(0, dtype=np.int32),
        )
        win = MainWindow()
        win._generated_mesh = mesh
        win._actual_layers = 1
        win._chk_disable_chebyshev.setChecked(True)

        created = {}

        class FakeThread:
            def __init__(self, parent=None) -> None:  # noqa: ANN001
                self.started = _FakeSignal()
                self.finished = _FakeSignal()

            def start(self) -> None:
                pass

            def quit(self) -> None:
                pass

            def deleteLater(self) -> None:
                pass

            def requestInterruption(self) -> None:
                pass

        class FakeWorker:
            def __init__(self, **kwargs) -> None:  # noqa: ANN003
                created.update(kwargs)
                self.frame_ready = _FakeSignal()
                self.progress_update = _FakeSignal()
                self.log_message = _FakeSignal()
                self.finished = _FakeSignal()
                self.cancelled = _FakeSignal()
                self.error = _FakeSignal()
                self.sub_progress = _FakeSignal()
                self.layer_finished = _FakeSignal()

            def moveToThread(self, thread) -> None:  # noqa: ANN001
                pass

            def run(self) -> None:
                pass

            def deleteLater(self) -> None:
                pass

        with patch(
            "hydrogel_vbd.gui.main_window.SimulationWorker",
            FakeWorker,
        ), patch.object(QtCore, "QThread", FakeThread), patch(
            "hydrogel_vbd.gui.main_window.is_cpp_available",
            return_value=False,
        ):
            win._on_run()

        self.assertEqual(created["config"].rho_cheb, 0.0)
        app.quit()

    def test_main_window_exposes_simulation_stop_button(self) -> None:
        from hydrogel_vbd.gui.main_window import MainWindow
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        win = MainWindow()
        self.assertEqual(win._btn_stop_sim.text(), "中断仿真")
        self.assertTrue(win._btn_stop_sim.isHidden())
        self.assertFalse(win._btn_stop_sim.isEnabled())
        app.quit()

    def test_stop_simulation_requests_worker_stop(self) -> None:
        from hydrogel_vbd.gui.main_window import MainWindow
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        class FakeWorker:
            def __init__(self) -> None:
                self.stop_requested = False

            def request_stop(self) -> None:
                self.stop_requested = True

        win = MainWindow()
        worker = FakeWorker()
        win._worker = worker
        win._set_simulation_running(True)

        win._on_stop_simulation()

        self.assertTrue(worker.stop_requested)
        self.assertTrue(win._simulation_stop_requested)
        self.assertFalse(win._btn_stop_sim.isEnabled())
        self.assertEqual(win._btn_stop_sim.text(), "正在中断...")
        app.quit()

    def test_launch_gui_importable(self) -> None:
        from hydrogel_vbd.gui.main_window import launch_gui
        self.assertTrue(callable(launch_gui))

    def test_report_uses_solver_diagnostic_mode_without_shape_error(self) -> None:
        from hydrogel_vbd.core.state import FieldCommand, LayerResult
        from hydrogel_vbd.gui.main_window import ElectricFieldPlotWindow

        results = [
            LayerResult(
                layer_id=0,
                x_sim=np.zeros((0, 3)),
                v_sim=np.zeros((0, 3)),
                error_metrics={
                    "E_z": 0.0,
                    "max_error": 2.0e-3,
                    "solver_final_max_dx": 2.0e-3,
                    "solver_max_iter_hit_pct": 100.0,
                    "solver_avg_call_ms": 23.4,
                    "shape_error_available": 0.0,
                },
                field_command_next=FieldCommand(np.array([])),
                max_deformation=2.0e-3,
                rms_error=0.0,
                success=True,
            )
        ]
        plot_data = ElectricFieldPlotWindow._build_plot_data(results)

        self.assertEqual(plot_data.mode, "solver")
        self.assertEqual(plot_data.primary_label, "solver max_dx (m)")
        self.assertEqual(plot_data.aux_label, "max_iter 命中率 (%)")
        np.testing.assert_allclose(plot_data.primary, [2.0e-3])
        np.testing.assert_allclose(plot_data.aux_pct, [100.0])

    def test_report_uses_shape_error_mode_when_metrics_exist(self) -> None:
        from hydrogel_vbd.core.state import FieldCommand, LayerResult
        from hydrogel_vbd.gui.main_window import ElectricFieldPlotWindow

        results = [
            LayerResult(
                layer_id=0,
                x_sim=np.zeros((0, 3)),
                v_sim=np.zeros((0, 3)),
                error_metrics={
                    "E_z": 1.0,
                    "shape_max_error": 3.0e-4,
                    "shape_rms_error": 2.0e-4,
                    "shape_error_available": 1.0,
                },
                field_command_next=FieldCommand(np.array([])),
                max_deformation=9.0e-3,
                rms_error=9.0e-3,
                success=True,
            )
        ]
        plot_data = ElectricFieldPlotWindow._build_plot_data(results)

        self.assertEqual(plot_data.mode, "shape")
        self.assertEqual(plot_data.primary_label, "max_error (m)")
        self.assertEqual(plot_data.secondary_label, "rms_error (m)")
        np.testing.assert_allclose(plot_data.primary, [3.0e-4])
        np.testing.assert_allclose(plot_data.secondary, [2.0e-4])

    def test_report_field_debug_aux_shows_rms_improvement_and_guard_status(self) -> None:
        from hydrogel_vbd.core.state import FieldCommand, LayerResult
        from hydrogel_vbd.gui.main_window import ElectricFieldPlotWindow

        results = [
            LayerResult(
                layer_id=0,
                x_sim=np.zeros((0, 3)),
                v_sim=np.zeros((0, 3)),
                error_metrics={
                    "E_z": 120.0,
                    "max_error": 4.0e-3,
                    "rms_error": 3.0e-3,
                    "field_no_field_rms": 4.0e-3,
                    "field_with_field_rms": 3.0e-3,
                    "field_guard_passed": 1.0,
                    "shape_error_available": 1.0,
                },
                field_command_next=FieldCommand(np.array([])),
                max_deformation=4.0e-3,
                rms_error=3.0e-3,
                success=True,
            ),
            LayerResult(
                layer_id=1,
                x_sim=np.zeros((0, 3)),
                v_sim=np.zeros((0, 3)),
                error_metrics={
                    "E_z": 0.0,
                    "max_error": 5.0e-3,
                    "rms_error": 4.0e-3,
                    "field_no_field_rms": 4.0e-3,
                    "field_with_field_rms": 5.0e-3,
                    "field_guard_passed": 0.0,
                    "shape_error_available": 1.0,
                },
                field_command_next=FieldCommand(np.array([])),
                max_deformation=5.0e-3,
                rms_error=4.0e-3,
                success=True,
            ),
        ]
        plot_data = ElectricFieldPlotWindow._build_plot_data(results)

        self.assertEqual(plot_data.aux_label, "RMS 改善率 (%)")
        np.testing.assert_allclose(plot_data.aux_pct, [25.0, -25.0])
        np.testing.assert_allclose(plot_data.guard_passed, [1.0, 0.0])


class CppAdapterStateWritebackTests(unittest.TestCase):
    """C++ 适配层可写状态数组回写回归测试。"""

    def test_cpp_config_scales_dx_clip_to_layer_thickness(self) -> None:
        from types import SimpleNamespace

        from hydrogel_vbd.core.config import SimulationConfig
        import hydrogel_vbd.solver.cpp_adapter as cpp_adapter

        class FakeSolverConfig:
            pass

        fake_cpp = SimpleNamespace(SolverConfig=FakeSolverConfig)
        config = SimulationConfig(
            layer_thickness=5.0e-5,
            dt=1.0e-3,
            v_lift=1.0e-2,
        )

        with patch.object(cpp_adapter, "hydrogel_vbd_cpp", fake_cpp):
            cpp_cfg = cpp_adapter._build_cpp_config(config)

        self.assertAlmostEqual(cpp_cfg.dx_clip, 2.5e-5)

    def test_clipped_max_iter_uses_scaled_dx_clip(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        import hydrogel_vbd.solver.cpp_adapter as cpp_adapter

        config = SimulationConfig(
            layer_thickness=5.0e-5,
            dt=1.0e-3,
            v_lift=1.0e-2,
            max_iters=50,
        )

        self.assertTrue(cpp_adapter._hit_clipped_max_iters(
            {"iterations": 50, "max_dx": 2.5e-5}, config
        ))
        self.assertFalse(cpp_adapter._hit_clipped_max_iters(
            {"iterations": 50, "max_dx": 1.0e-5}, config
        ))

    def test_cpp_adapter_passes_current_layer_bottom_mask(self) -> None:
        from types import SimpleNamespace

        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.core.state import MeshState
        import hydrogel_vbd.solver.cpp_adapter as cpp_adapter

        mesh = MeshState(
            vertices=np.zeros((2, 3), dtype=float),
            tets=np.zeros((0, 4), dtype=int),
            layer_id_per_vertex=np.array([0, 1], dtype=int),
            layer_id_per_tet=np.zeros(0, dtype=int),
            first_active_layer=np.array([0, 1], dtype=int),
            is_bottom_surface=np.array([True, False], dtype=bool),
            is_top_surface_of_layer=np.array([1, 2], dtype=int),
        )
        mesh.active_vertex_mask[:] = True
        mesh.active_tet_mask = np.zeros(0, dtype=bool)
        mesh.colors = np.zeros(2, dtype=int)
        mesh.node_mass = np.ones(2, dtype=float)

        captured: dict[str, np.ndarray] = {}

        class FakeSolverConfig:
            pass

        def fake_solve_lift_and_relax(*args):
            captured["global_bottom"] = np.asarray(args[8], dtype=bool).copy()
            captured["current_bottom"] = np.asarray(args[9], dtype=bool).copy()
            return {
                "max_dx": 0.0,
                "kinetic_energy": 0.0,
                "iterations": 0,
                "stable_steps": 0,
                "all_free": False,
                "chebyshev_skipped_damaging": 0,
            }

        fake_cpp = SimpleNamespace(
            SolverConfig=FakeSolverConfig,
            solve_lift_and_relax=fake_solve_lift_and_relax,
        )

        with patch.object(cpp_adapter, "_CPP_AVAILABLE", True), patch.object(
            cpp_adapter, "hydrogel_vbd_cpp", fake_cpp
        ):
            cpp_adapter.solve_lift_and_relax(
                mesh,
                SimulationConfig(),
                e_z=0.0,
                layer_id=1,
                lifting_top=np.array([], dtype=int),
            )

        np.testing.assert_array_equal(captured["global_bottom"], [True, False])
        np.testing.assert_array_equal(captured["current_bottom"], [False, True])

    def test_solve_lift_and_relax_writes_int64_czm_state_back(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.physics.czm import CZMState
        from hydrogel_vbd.solver.cpp_adapter import (
            is_cpp_available,
            solve_lift_and_relax,
        )

        if not is_cpp_available():
            self.skipTest("C++ solver module is not available")

        config = SimulationConfig(
            dt=1.0e-3,
            max_iters=1,
            N_stable=1,
            layer_thickness=1.0e-4,
            delta_f=1.0e-4,
            v_lift=1.0e-3,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=1, layer_thickness=config.layer_thickness, config=config
        )
        mesh.activate_layer(0)
        top = mesh.top_nodes(0)
        mesh.is_top_fixed[top] = True

        bottom = mesh.bottom_nodes(0)
        mesh.vertices[bottom, 2] = 6.0 * config.delta_f
        mesh.czm_state = mesh.czm_state.astype(np.int64, copy=True)

        solve_lift_and_relax(
            mesh,
            config,
            e_z=0.0,
            layer_id=0,
            lifting_top=top,
        )

        self.assertTrue(
            np.all(mesh.czm_state[bottom] == int(CZMState.DAMAGING))
        )

    def test_cpp_lift_does_not_clamp_previous_global_bottom_to_current_fep(self) -> None:
        from dataclasses import replace

        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.geometry.layer_activator import LayerActivator
        from hydrogel_vbd.solver.cpp_adapter import (
            is_cpp_available,
            solve_lift_and_relax,
        )

        if not is_cpp_available():
            self.skipTest("C++ solver module is not available")

        config = SimulationConfig(
            layer_thickness=0.0019981,
            dt=1.0e-3,
            v_lift=1.0e-3,
            max_iters=1,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=2, layer_thickness=config.layer_thickness, config=config
        )
        activator = LayerActivator()
        activator.activate_with_inheritance(mesh, 0, z_fep=0.0)
        previous_bottom = mesh.bottom_nodes(0)

        layer_config = replace(config, z_fep=0.0)
        activator.activate_with_inheritance(
            mesh, 1, z_fep=layer_config.z_fep
        )
        lifting_top = np.flatnonzero(
            mesh.is_top_fixed & mesh.active_vertex_mask
        ).astype(np.int32)

        result = solve_lift_and_relax(mesh, layer_config, 0.0, 1, lifting_top)

        self.assertLess(result.max_dx, 1.0e-5)
        np.testing.assert_allclose(
            mesh.vertices[previous_bottom, 2],
            config.layer_thickness,
            atol=config.layer_thickness * 0.01,
        )

    def test_cpp_lift_restores_internal_motion_when_clipped_at_max_iters(self) -> None:
        from types import SimpleNamespace

        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.core.state import MeshState
        import hydrogel_vbd.solver.cpp_adapter as cpp_adapter

        mesh = MeshState(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
            tets=np.zeros((0, 4), dtype=int),
            layer_id_per_vertex=np.zeros(2, dtype=int),
            layer_id_per_tet=np.zeros(0, dtype=int),
            first_active_layer=np.zeros(2, dtype=int),
            is_top_surface_of_layer=np.array([1, 0], dtype=int),
        )
        mesh.active_vertex_mask[:] = True
        mesh.active_tet_mask = np.zeros(0, dtype=bool)
        mesh.colors = np.zeros(2, dtype=np.int32)
        mesh.node_mass = np.ones(2, dtype=float)
        mesh.is_top_fixed[:] = [False, True]
        mesh.czm_state[:] = 2
        original = mesh.vertices.copy()
        lifting_top = np.array([1], dtype=np.int32)

        class FakeSolverConfig:
            pass

        def fake_solve_lift_and_relax(*args):
            vertices = args[0]
            vertices[0] += np.array([0.0, 0.1, 0.0])
            vertices[1, 2] += 1.0e-6
            return {
                "max_dx": 0.002,
                "kinetic_energy": 0.0,
                "iterations": 50,
                "stable_steps": 0,
                "all_free": True,
                "chebyshev_skipped_damaging": 0,
            }

        fake_cpp = SimpleNamespace(
            SolverConfig=FakeSolverConfig,
            solve_lift_and_relax=fake_solve_lift_and_relax,
        )
        config = SimulationConfig(
            max_iters=50,
            v_lift=1.0e-3,
            dt=1.0e-3,
            enable_czm=False,
        )

        with patch.object(cpp_adapter, "_CPP_AVAILABLE", True), patch.object(
            cpp_adapter, "hydrogel_vbd_cpp", fake_cpp
        ):
            result = cpp_adapter.solve_lift_and_relax(
                mesh,
                config,
                e_z=0.0,
                layer_id=0,
                lifting_top=lifting_top,
            )

        np.testing.assert_allclose(mesh.vertices[0], original[0])
        np.testing.assert_allclose(
            mesh.vertices[1],
            original[1] + np.array([0.0, 0.0, config.v_lift * config.dt]),
        )
        np.testing.assert_allclose(mesh.velocities[1], 0.0)
        self.assertEqual(result.iterations, config.max_iters)
        self.assertEqual(result.max_dx, 0.002)


class CppSubprocessRuntimeTests(unittest.TestCase):
    """C++ 子进程运行时配置测试。"""

    def test_configure_cpp_runtime_enables_requested_subprocess_threads(self) -> None:
        from hydrogel_vbd.solver.cpp_subprocess import (
            _configure_cpp_runtime_for_subprocess,
        )

        saved = {
            key: os.environ.get(key)
            for key in (
                "HYDROGEL_VBD_SUBPROCESS_THREADS",
                "HYDROGEL_VBD_OMP",
                "OMP_NUM_THREADS",
            )
        }
        try:
            os.environ["HYDROGEL_VBD_SUBPROCESS_THREADS"] = "4"
            os.environ.pop("HYDROGEL_VBD_OMP", None)
            os.environ.pop("OMP_NUM_THREADS", None)

            info = _configure_cpp_runtime_for_subprocess()

            self.assertEqual(info["threads"], 4)
            self.assertEqual(os.environ["HYDROGEL_VBD_OMP"], "1")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "4")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_cpp_lift_step_updates_current_layer_czm_boundary(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.solver.cpp_subprocess import _run_simulation
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            dt=1.0e-3,
            v_lift=1.0e-3,
            layer_thickness=1.0e-6,
            lift_height=1.5e-6,
            max_iters=2,
            N_stable=1,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=2, layer_thickness=config.layer_thickness, config=config
        )
        expected_layer1_bottom = mesh.bottom_nodes(1)
        mesh_dict = {
            attr: getattr(mesh, attr)
            for attr in (
                "vertices", "velocities", "prev_vertices", "ideal_vertices",
                "node_mass", "active_vertex_mask", "is_top_fixed",
                "is_bottom_surface", "czm_state", "damage", "time_free",
                "tets", "active_tet_mask", "dm_inv", "tet_volumes", "colors",
                "layer_id_per_vertex", "layer_id_per_tet", "first_active_layer",
                "is_top_surface_of_layer",
            )
            if getattr(mesh, attr, None) is not None
        }
        config_dict = {
            "dt": config.dt,
            "v_lift": config.v_lift,
            "layer_thickness": config.layer_thickness,
            "lift_height": config.lift_height,
            "max_iters": config.max_iters,
            "N_stable": config.N_stable,
        }

        class FakeConn:
            def __init__(self) -> None:
                self.messages = []

            def send(self, msg) -> None:  # noqa: ANN001
                self.messages.append(msg)

            def poll(self) -> bool:
                return False

        def fake_solve(mesh_arg, cfg_arg, e_z, layer_id, lifting_top):  # noqa: ANN001, ARG001
            return VBDSolveResult(
                x=mesh_arg.vertices,
                v=mesh_arg.velocities,
                iterations=1,
                max_dx=0.0,
                kinetic_energy=0.0,
                stable_steps=1,
                all_free=True,
                chebyshev_skipped_damaging=0,
            )

        conn = FakeConn()
        with (
            patch(
                "hydrogel_vbd.solver.cpp_adapter.solve_lift_and_relax",
                side_effect=fake_solve,
            ),
            patch("hydrogel_vbd.physics.czm.update_czm_states") as update_mock,
        ):
            _run_simulation(conn, mesh_dict, config_dict, n_layers=2, output_dir="outputs/gui")

        updated_node_sets = [
            np.asarray(call.kwargs["bottom_nodes"], dtype=int)
            if "bottom_nodes" in call.kwargs
            else np.asarray(call.args[1], dtype=int)
            for call in update_mock.call_args_list
        ]
        self.assertTrue(
            any(np.array_equal(nodes, expected_layer1_bottom) for nodes in updated_node_sets)
        )

    def test_cpp_subprocess_emits_activation_frame_and_trace_before_lift(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.solver.cpp_subprocess import _FrameMsg, _run_simulation
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            enable_czm=False,
            dt=1.0e-3,
            v_lift=1.0e-3,
            layer_thickness=1.0e-6,
            lift_height=1.0e-6,
            max_iters=1,
            N_stable=1,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=2, layer_thickness=config.layer_thickness, config=config
        )
        mesh_dict = {
            attr: getattr(mesh, attr)
            for attr in (
                "vertices", "velocities", "prev_vertices", "ideal_vertices",
                "node_mass", "active_vertex_mask", "is_top_fixed",
                "is_bottom_surface", "czm_state", "damage", "time_free",
                "tets", "active_tet_mask", "dm_inv", "tet_volumes", "colors",
                "layer_id_per_vertex", "layer_id_per_tet", "first_active_layer",
                "is_top_surface_of_layer",
            )
            if getattr(mesh, attr, None) is not None
        }
        config_dict = {
            "enable_czm": config.enable_czm,
            "dt": config.dt,
            "v_lift": config.v_lift,
            "layer_thickness": config.layer_thickness,
            "lift_height": config.lift_height,
            "max_iters": config.max_iters,
            "N_stable": config.N_stable,
        }
        out_dir = Path("outputs/test_cpp_activation_snapshot")
        trace_path = out_dir / "worker_trace.log"
        trace_path.unlink(missing_ok=True)

        class FakeConn:
            def __init__(self) -> None:
                self.messages = []

            def send(self, msg) -> None:  # noqa: ANN001
                self.messages.append(msg)

            def poll(self) -> bool:
                return False

        def fake_solve(mesh_arg, cfg_arg, e_z, layer_id, lifting_top):  # noqa: ANN001, ARG001
            return VBDSolveResult(
                x=mesh_arg.vertices,
                v=mesh_arg.velocities,
                iterations=1,
                max_dx=0.0,
                kinetic_energy=0.0,
                stable_steps=1,
                all_free=False,
                chebyshev_skipped_damaging=0,
            )

        conn = FakeConn()
        with patch(
            "hydrogel_vbd.solver.cpp_adapter.solve_lift_and_relax",
            side_effect=fake_solve,
        ):
            _run_simulation(conn, mesh_dict, config_dict, n_layers=2, output_dir=out_dir)

        frame_titles = [
            msg.title for msg in conn.messages if isinstance(msg, _FrameMsg)
        ]
        self.assertIn("第 1 层 — 激活后/上提前", frame_titles)
        self.assertIn("第 2 层 — 激活后/上提前", frame_titles)

        trace_text = trace_path.read_text(encoding="utf-8")
        self.assertIn("layer_1_activation_state", trace_text)
        self.assertIn("previous_bottom_z=", trace_text)
        self.assertIn("current_bottom_z=", trace_text)
        self.assertIn("target_gap=", trace_text)

    def test_cpp_subprocess_emits_layer_end_frame_before_next_activation(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.solver.cpp_subprocess import _FrameMsg, _run_simulation
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            enable_czm=False,
            dt=1.0e-3,
            v_lift=1.0e-3,
            layer_thickness=1.0e-6,
            lift_height=1.5e-6,
            max_iters=1,
            N_stable=1,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=2, layer_thickness=config.layer_thickness, config=config
        )
        mesh_dict = {
            attr: getattr(mesh, attr)
            for attr in (
                "vertices", "velocities", "prev_vertices", "ideal_vertices",
                "node_mass", "active_vertex_mask", "is_top_fixed",
                "is_bottom_surface", "czm_state", "damage", "time_free",
                "tets", "active_tet_mask", "dm_inv", "tet_volumes", "colors",
                "layer_id_per_vertex", "layer_id_per_tet", "first_active_layer",
                "is_top_surface_of_layer",
            )
            if getattr(mesh, attr, None) is not None
        }
        config_dict = {
            "enable_czm": config.enable_czm,
            "dt": config.dt,
            "v_lift": config.v_lift,
            "layer_thickness": config.layer_thickness,
            "lift_height": config.lift_height,
            "max_iters": config.max_iters,
            "N_stable": config.N_stable,
        }

        class FakeConn:
            def __init__(self) -> None:
                self.messages = []

            def send(self, msg) -> None:  # noqa: ANN001
                self.messages.append(msg)

            def poll(self) -> bool:
                return False

        def fake_solve(mesh_arg, cfg_arg, e_z, layer_id, lifting_top):  # noqa: ANN001, ARG001
            step = float(cfg_arg.v_lift) * float(cfg_arg.dt)
            mesh_arg.vertices[np.asarray(lifting_top, dtype=int), 2] += step
            return VBDSolveResult(
                x=mesh_arg.vertices,
                v=mesh_arg.velocities,
                iterations=1,
                max_dx=0.0,
                kinetic_energy=0.0,
                stable_steps=1,
                all_free=False,
                chebyshev_skipped_damaging=0,
            )

        conn = FakeConn()
        with patch(
            "hydrogel_vbd.solver.cpp_adapter.solve_lift_and_relax",
            side_effect=fake_solve,
        ):
            _run_simulation(
                conn, mesh_dict, config_dict,
                n_layers=2, output_dir="outputs/test_cpp_layer_end_frame",
            )

        titles = [msg.title for msg in conn.messages if isinstance(msg, _FrameMsg)]
        layer1_end = "\u7b2c 1 \u5c42 \u2014 \u4e0b\u653e\u5b8c\u6210/\u5c42\u7ed3\u675f"
        layer2_activation = "\u7b2c 2 \u5c42 \u2014 \u6fc0\u6d3b\u540e/\u4e0a\u63d0\u524d"
        layer2_final = "\u7b2c 2 \u5c42 \u2014 \u4e0a\u63d0\u5b8c\u6210/\u6700\u7ec8\u72b6\u6001"
        self.assertIn(layer1_end, titles)
        self.assertIn(layer2_final, titles)
        self.assertLess(titles.index(layer1_end), titles.index(layer2_activation))

    def test_cpp_subprocess_returns_platform_before_next_layer_activation(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.solver.cpp_subprocess import _run_simulation
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            enable_czm=False,
            dt=1.0e-3,
            v_lift=1.0e-3,
            layer_thickness=1.0e-6,
            lift_height=1.5e-6,
            max_iters=1,
            N_stable=1,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=2, layer_thickness=config.layer_thickness, config=config
        )
        mesh_dict = {
            attr: getattr(mesh, attr)
            for attr in (
                "vertices", "velocities", "prev_vertices", "ideal_vertices",
                "node_mass", "active_vertex_mask", "is_top_fixed",
                "is_bottom_surface", "czm_state", "damage", "time_free",
                "tets", "active_tet_mask", "dm_inv", "tet_volumes", "colors",
                "layer_id_per_vertex", "layer_id_per_tet", "first_active_layer",
                "is_top_surface_of_layer",
            )
            if getattr(mesh, attr, None) is not None
        }
        config_dict = {
            "enable_czm": config.enable_czm,
            "dt": config.dt,
            "v_lift": config.v_lift,
            "layer_thickness": config.layer_thickness,
            "lift_height": config.lift_height,
            "max_iters": config.max_iters,
            "N_stable": config.N_stable,
        }
        observed_calls: list[tuple[int, float]] = []

        class FakeConn:
            def send(self, msg) -> None:  # noqa: ANN001
                pass

            def poll(self) -> bool:
                return False

        def fake_solve(mesh_arg, cfg_arg, e_z, layer_id, lifting_top):  # noqa: ANN001, ARG001
            observed_calls.append((int(layer_id), float(cfg_arg.v_lift)))
            step = float(cfg_arg.v_lift) * float(cfg_arg.dt)
            mesh_arg.vertices[np.asarray(lifting_top, dtype=int), 2] += step
            return VBDSolveResult(
                x=mesh_arg.vertices,
                v=mesh_arg.velocities,
                iterations=1,
                max_dx=0.0,
                kinetic_energy=0.0,
                stable_steps=1,
                all_free=False,
                chebyshev_skipped_damaging=0,
            )

        with patch(
            "hydrogel_vbd.solver.cpp_adapter.solve_lift_and_relax",
            side_effect=fake_solve,
        ):
            _run_simulation(
                FakeConn(), mesh_dict, config_dict,
                n_layers=2, output_dir="outputs/test_cpp_platform_return",
            )

        layer0_calls = [v_lift for layer, v_lift in observed_calls if layer == 0]
        self.assertIn(config.v_lift, layer0_calls)
        return_calls = [v_lift for v_lift in layer0_calls if v_lift < 0.0]
        self.assertEqual(len(return_calls), 2)
        for v_lift in return_calls:
            self.assertAlmostEqual(v_lift, -config.v_lift)

    def test_cpp_platform_return_disables_czm_for_downstroke(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.solver.cpp_subprocess import _run_simulation
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            enable_czm=True,
            dt=1.0e-3,
            v_lift=1.0e-3,
            layer_thickness=1.0e-6,
            lift_height=1.5e-6,
            max_iters=1,
            N_stable=1,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=2, layer_thickness=config.layer_thickness, config=config
        )
        mesh_dict = {
            attr: getattr(mesh, attr)
            for attr in (
                "vertices", "velocities", "prev_vertices", "ideal_vertices",
                "node_mass", "active_vertex_mask", "is_top_fixed",
                "is_bottom_surface", "czm_state", "damage", "time_free",
                "tets", "active_tet_mask", "dm_inv", "tet_volumes", "colors",
                "layer_id_per_vertex", "layer_id_per_tet", "first_active_layer",
                "is_top_surface_of_layer",
            )
            if getattr(mesh, attr, None) is not None
        }
        config_dict = {
            "enable_czm": config.enable_czm,
            "dt": config.dt,
            "v_lift": config.v_lift,
            "layer_thickness": config.layer_thickness,
            "lift_height": config.lift_height,
            "max_iters": config.max_iters,
            "N_stable": config.N_stable,
        }
        observed_calls: list[tuple[int, float, bool]] = []

        class FakeConn:
            def send(self, msg) -> None:  # noqa: ANN001
                pass

            def poll(self) -> bool:
                return False

        def fake_solve(mesh_arg, cfg_arg, e_z, layer_id, lifting_top):  # noqa: ANN001, ARG001
            observed_calls.append(
                (int(layer_id), float(cfg_arg.v_lift), bool(cfg_arg.enable_czm))
            )
            step = float(cfg_arg.v_lift) * float(cfg_arg.dt)
            mesh_arg.vertices[np.asarray(lifting_top, dtype=int), 2] += step
            return VBDSolveResult(
                x=mesh_arg.vertices,
                v=mesh_arg.velocities,
                iterations=1,
                max_dx=0.0,
                kinetic_energy=0.0,
                stable_steps=1,
                all_free=False,
                chebyshev_skipped_damaging=0,
            )

        with patch(
            "hydrogel_vbd.solver.cpp_adapter.solve_lift_and_relax",
            side_effect=fake_solve,
        ):
            _run_simulation(
                FakeConn(), mesh_dict, config_dict,
                n_layers=2, output_dir="outputs/test_cpp_platform_return_czm",
            )

        lift_calls = [enable for layer, v_lift, enable in observed_calls if layer == 0 and v_lift > 0.0]
        return_calls = [enable for layer, v_lift, enable in observed_calls if layer == 0 and v_lift < 0.0]
        self.assertTrue(lift_calls)
        self.assertTrue(all(lift_calls))
        self.assertTrue(return_calls)
        self.assertTrue(all(enable is False for enable in return_calls))

    def test_cpp_subprocess_traces_hit_rate_quality_and_scaled_lift_epsilon(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.solver.cpp_subprocess import _DoneMsg, _run_simulation
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            enable_czm=False,
            dt=1.0e-3,
            v_lift=1.0,
            layer_thickness=1.0e-3,
            lift_height=1.0e-3,
            max_iters=3,
            N_stable=1,
            epsilon=1.0e-9,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=1, layer_thickness=config.layer_thickness, config=config
        )
        mesh_dict = {
            attr: getattr(mesh, attr)
            for attr in (
                "vertices", "velocities", "prev_vertices", "ideal_vertices",
                "node_mass", "active_vertex_mask", "is_top_fixed",
                "is_bottom_surface", "czm_state", "damage", "time_free",
                "tets", "active_tet_mask", "dm_inv", "tet_volumes", "colors",
                "layer_id_per_vertex", "layer_id_per_tet", "first_active_layer",
                "is_top_surface_of_layer",
            )
            if getattr(mesh, attr, None) is not None
        }
        config_dict = {
            "enable_czm": config.enable_czm,
            "dt": config.dt,
            "v_lift": config.v_lift,
            "layer_thickness": config.layer_thickness,
            "lift_height": config.lift_height,
            "max_iters": config.max_iters,
            "N_stable": config.N_stable,
            "epsilon": config.epsilon,
        }
        out_dir = Path("outputs/test_cpp_lift_epsilon_trace")
        trace_path = out_dir / "worker_trace.log"
        trace_path.unlink(missing_ok=True)
        observed_epsilons: list[float] = []

        class FakeConn:
            def __init__(self) -> None:
                self.messages = []

            def send(self, msg) -> None:  # noqa: ANN001
                self.messages.append(msg)

            def poll(self) -> bool:
                return False

        def fake_solve(mesh_arg, cfg_arg, e_z, layer_id, lifting_top):  # noqa: ANN001, ARG001
            observed_epsilons.append(float(cfg_arg.epsilon))
            return VBDSolveResult(
                x=mesh_arg.vertices,
                v=mesh_arg.velocities,
                iterations=cfg_arg.max_iters,
                max_dx=2.0e-6,
                kinetic_energy=0.0,
                stable_steps=0,
                all_free=False,
                chebyshev_skipped_damaging=0,
            )

        conn = FakeConn()
        with patch(
            "hydrogel_vbd.solver.cpp_adapter.solve_lift_and_relax",
            side_effect=fake_solve,
        ):
            _run_simulation(conn, mesh_dict, config_dict, n_layers=1, output_dir=out_dir)

        self.assertEqual(len(observed_epsilons), 1)
        self.assertAlmostEqual(observed_epsilons[0], 1.5e-6)
        done_messages = [msg for msg in conn.messages if isinstance(msg, _DoneMsg)]
        self.assertEqual(done_messages[0].results[0]["max_iter_hit_rate"], 100.0)

        trace_text = trace_path.read_text(encoding="utf-8")
        self.assertIn("active_tets=", trace_text)
        self.assertIn("tet_quality=", trace_text)
        self.assertIn("solver_epsilon=1.500000e-06", trace_text)
        self.assertIn("max_iter_hit_rate=100.00%", trace_text)

    def test_cpp_lift_rejects_detach_before_solver_convergence(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.physics.czm import CZMState
        from hydrogel_vbd.solver.cpp_subprocess import DX_CLIP_DIAGNOSTIC, _run_simulation
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            dt=1.0e-3,
            v_lift=1.0e-3,
            layer_thickness=1.0e-6,
            lift_height=1.5e-6,
            max_iters=2,
            N_stable=1,
            epsilon=1.0e-9,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=1, layer_thickness=config.layer_thickness, config=config
        )
        bottom = mesh.bottom_nodes(0)
        mesh_dict = {
            attr: getattr(mesh, attr)
            for attr in (
                "vertices", "velocities", "prev_vertices", "ideal_vertices",
                "node_mass", "active_vertex_mask", "is_top_fixed",
                "is_bottom_surface", "czm_state", "damage", "time_free",
                "tets", "active_tet_mask", "dm_inv", "tet_volumes", "colors",
                "layer_id_per_vertex", "layer_id_per_tet", "first_active_layer",
                "is_top_surface_of_layer",
            )
            if getattr(mesh, attr, None) is not None
        }
        config_dict = {
            "dt": config.dt,
            "v_lift": config.v_lift,
            "layer_thickness": config.layer_thickness,
            "lift_height": config.lift_height,
            "max_iters": config.max_iters,
            "N_stable": config.N_stable,
            "epsilon": config.epsilon,
        }

        class FakeConn:
            def send(self, msg) -> None:  # noqa: ANN001
                pass

            def poll(self) -> bool:
                return False

        def fake_solve(mesh_arg, cfg_arg, e_z, layer_id, lifting_top):  # noqa: ANN001, ARG001
            return VBDSolveResult(
                x=mesh_arg.vertices,
                v=mesh_arg.velocities,
                iterations=cfg_arg.max_iters,
                max_dx=DX_CLIP_DIAGNOSTIC,
                kinetic_energy=0.0,
                stable_steps=0,
                all_free=True,
                chebyshev_skipped_damaging=0,
            )

        def fake_czm_update(mesh_arg, bottom_nodes, **kwargs):  # noqa: ANN001, ARG001
            mesh_arg.czm_state[np.asarray(bottom_nodes, dtype=int)] = int(CZMState.FREE)

        with (
            patch(
                "hydrogel_vbd.solver.cpp_adapter.solve_lift_and_relax",
                side_effect=fake_solve,
            ),
            patch(
                "hydrogel_vbd.physics.czm.update_czm_states",
                side_effect=fake_czm_update,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "solver did not converge"):
                _run_simulation(
                    FakeConn(), mesh_dict, config_dict,
                    n_layers=1, output_dir="outputs/gui",
                )

    def test_cpp_lift_uses_actual_pull_and_restores_cpp_czm_mutation(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.physics.czm import CZMState
        from hydrogel_vbd.physics.local_terms import LocalPhysicsTerms
        from hydrogel_vbd.solver.cpp_subprocess import _run_simulation
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            dt=1.0e-3,
            v_lift=1.0e-3,
            layer_thickness=2.0e-7,
            lift_height=3.0e-7,
            max_iters=2,
            N_stable=1,
            epsilon=1.0e-9,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=1, layer_thickness=config.layer_thickness, config=config
        )
        bottom = mesh.bottom_nodes(0)
        mesh_dict = {
            attr: getattr(mesh, attr)
            for attr in (
                "vertices", "velocities", "prev_vertices", "ideal_vertices",
                "node_mass", "active_vertex_mask", "is_top_fixed",
                "is_bottom_surface", "czm_state", "damage", "time_free",
                "tets", "active_tet_mask", "dm_inv", "tet_volumes", "colors",
                "layer_id_per_vertex", "layer_id_per_tet", "first_active_layer",
                "is_top_surface_of_layer",
            )
            if getattr(mesh, attr, None) is not None
        }
        config_dict = {
            "dt": config.dt,
            "v_lift": config.v_lift,
            "layer_thickness": config.layer_thickness,
            "lift_height": config.lift_height,
            "max_iters": config.max_iters,
            "N_stable": config.N_stable,
            "epsilon": config.epsilon,
        }
        actual_pull = 321.0
        terms = LocalPhysicsTerms(
            force=np.full_like(mesh.vertices, [0.0, 0.0, actual_pull], dtype=float),
            hessian=np.zeros((mesh.vertices.shape[0], 3, 3), dtype=float),
        )

        class FakeConn:
            def __init__(self) -> None:
                self.messages = []

            def send(self, msg) -> None:  # noqa: ANN001
                self.messages.append(msg)

            def poll(self) -> bool:
                return False

        def fake_solve(mesh_arg, cfg_arg, e_z, layer_id, lifting_top):  # noqa: ANN001, ARG001
            mesh_arg.czm_state[bottom] = int(CZMState.FREE)
            return VBDSolveResult(
                x=mesh_arg.vertices,
                v=mesh_arg.velocities,
                iterations=1,
                max_dx=0.0,
                kinetic_energy=0.0,
                stable_steps=1,
                all_free=False,
                chebyshev_skipped_damaging=0,
            )

        def fake_update(mesh_arg, bottom_nodes, **kwargs):  # noqa: ANN001
            nodes = np.asarray(bottom_nodes, dtype=int)
            self.assertTrue(np.all(mesh_arg.czm_state[nodes] == int(CZMState.FIXED)))

        with (
            patch(
                "hydrogel_vbd.solver.cpp_adapter.solve_lift_and_relax",
                side_effect=fake_solve,
            ),
            patch(
                "hydrogel_vbd.physics.local_terms.build_local_physics_terms",
                return_value=terms,
            ),
            patch(
                "hydrogel_vbd.physics.czm.update_czm_states",
                side_effect=fake_update,
            ) as update_mock,
        ):
            _run_simulation(
                FakeConn(), mesh_dict, config_dict,
                n_layers=1, output_dir="outputs/gui",
            )

        pull_arg = update_mock.call_args.kwargs["internal_pull_z"]
        np.testing.assert_allclose(pull_arg, np.full(len(bottom), actual_pull))

    def test_cpp_diagnostic_runaway_guard_returns_done_without_error(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.physics.czm import CZMState
        from hydrogel_vbd.solver.cpp_subprocess import _DoneMsg, _ErrorMsg, _run_simulation
        from hydrogel_vbd.solver.diagnostics import SolverStepDiagnostics
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            dt=1.0e-3,
            v_lift=1.0e-3,
            layer_thickness=20.0e-6,
            lift_height=100.0e-6,
            max_iters=2,
            N_stable=1,
            epsilon=1.0e-9,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=1, layer_thickness=config.layer_thickness, config=config
        )
        bottom = mesh.bottom_nodes(0)
        mesh_dict = {
            attr: getattr(mesh, attr)
            for attr in (
                "vertices", "velocities", "prev_vertices", "ideal_vertices",
                "node_mass", "active_vertex_mask", "is_top_fixed",
                "is_bottom_surface", "czm_state", "damage", "time_free",
                "tets", "active_tet_mask", "dm_inv", "tet_volumes", "colors",
                "layer_id_per_vertex", "layer_id_per_tet", "first_active_layer",
                "is_top_surface_of_layer",
            )
            if getattr(mesh, attr, None) is not None
        }
        config_dict = {
            "dt": config.dt,
            "v_lift": config.v_lift,
            "layer_thickness": config.layer_thickness,
            "lift_height": config.lift_height,
            "max_iters": config.max_iters,
            "N_stable": config.N_stable,
            "epsilon": config.epsilon,
        }
        out_dir = Path("outputs/test_cpp_diagnostic_guard")
        csv_path = out_dir / "reports" / "solver_diagnostics.csv"
        perf_csv_path = out_dir / "reports" / "performance_diagnostics.csv"
        csv_path.unlink(missing_ok=True)
        perf_csv_path.unlink(missing_ok=True)

        class FakeConn:
            def __init__(self) -> None:
                self.messages = []

            def send(self, msg) -> None:  # noqa: ANN001
                self.messages.append(msg)

            def poll(self) -> bool:
                return False

        def fake_solve(mesh_arg, cfg_arg, e_z, layer_id, lifting_top):  # noqa: ANN001, ARG001
            return VBDSolveResult(
                x=mesh_arg.vertices,
                v=mesh_arg.velocities,
                iterations=cfg_arg.max_iters,
                max_dx=0.002,
                kinetic_energy=0.0,
                stable_steps=0,
                all_free=False,
                chebyshev_skipped_damaging=0,
            )

        def fake_update(mesh_arg, bottom_nodes, **kwargs):  # noqa: ANN001, ARG001
            mesh_arg.czm_state[np.asarray(bottom_nodes, dtype=int)] = int(CZMState.FIXED)

        saved = {
            "HYDROGEL_VBD_SOLVER_DIAG": os.environ.get("HYDROGEL_VBD_SOLVER_DIAG"),
            "HYDROGEL_VBD_SOLVER_DIAG_STRIDE": os.environ.get(
                "HYDROGEL_VBD_SOLVER_DIAG_STRIDE"
            ),
        }
        conn = FakeConn()
        try:
            os.environ.pop("HYDROGEL_VBD_SOLVER_DIAG", None)
            os.environ.pop("HYDROGEL_VBD_SOLVER_DIAG_STRIDE", None)
            with patch(
                "hydrogel_vbd.solver.cpp_adapter.solve_lift_and_relax",
                side_effect=fake_solve,
            ), patch(
                "hydrogel_vbd.physics.czm.update_czm_states",
                side_effect=fake_update,
            ):
                _run_simulation(
                    conn,
                    mesh_dict,
                    config_dict,
                    n_layers=1,
                    output_dir=out_dir,
                    diag_enabled_override=True,
                    diag_stride_override=1,
                )
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertFalse(any(isinstance(msg, _ErrorMsg) for msg in conn.messages))
        done_messages = [msg for msg in conn.messages if isinstance(msg, _DoneMsg)]
        self.assertEqual(len(done_messages), 1)
        self.assertEqual(done_messages[0].results[0]["total_steps"], 50)
        self.assertFalse(done_messages[0].results[0]["success"])
        header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertEqual(header, SolverStepDiagnostics.csv_fields())
        perf_header = perf_csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertIn("cpp_solve_ms", perf_header)
        self.assertIn("czm_sync_ms", perf_header)

    def test_cpp_diagnostics_off_does_not_write_csv(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.solver.cpp_subprocess import _run_simulation
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            dt=1.0e-3,
            v_lift=1.0e-3,
            layer_thickness=2.0e-7,
            lift_height=3.0e-7,
            max_iters=2,
            N_stable=1,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=1, layer_thickness=config.layer_thickness, config=config
        )
        mesh_dict = {
            attr: getattr(mesh, attr)
            for attr in (
                "vertices", "velocities", "prev_vertices", "ideal_vertices",
                "node_mass", "active_vertex_mask", "is_top_fixed",
                "is_bottom_surface", "czm_state", "damage", "time_free",
                "tets", "active_tet_mask", "dm_inv", "tet_volumes", "colors",
                "layer_id_per_vertex", "layer_id_per_tet", "first_active_layer",
                "is_top_surface_of_layer",
            )
            if getattr(mesh, attr, None) is not None
        }
        config_dict = {
            "dt": config.dt,
            "v_lift": config.v_lift,
            "layer_thickness": config.layer_thickness,
            "lift_height": config.lift_height,
            "max_iters": config.max_iters,
            "N_stable": config.N_stable,
        }
        out_dir = Path("outputs/test_cpp_diagnostics_off")
        csv_path = out_dir / "reports" / "solver_diagnostics.csv"
        csv_path.unlink(missing_ok=True)

        class FakeConn:
            def send(self, msg) -> None:  # noqa: ANN001
                pass

            def poll(self) -> bool:
                return False

        def fake_solve(mesh_arg, cfg_arg, e_z, layer_id, lifting_top):  # noqa: ANN001, ARG001
            return VBDSolveResult(
                x=mesh_arg.vertices,
                v=mesh_arg.velocities,
                iterations=1,
                max_dx=0.0,
                kinetic_energy=0.0,
                stable_steps=1,
                all_free=True,
                chebyshev_skipped_damaging=0,
            )

        saved = os.environ.pop("HYDROGEL_VBD_SOLVER_DIAG", None)
        try:
            with patch(
                "hydrogel_vbd.solver.cpp_adapter.solve_lift_and_relax",
                side_effect=fake_solve,
            ):
                _run_simulation(FakeConn(), mesh_dict, config_dict, n_layers=1, output_dir=out_dir)
        finally:
            if saved is not None:
                os.environ["HYDROGEL_VBD_SOLVER_DIAG"] = saved

        self.assertFalse(csv_path.exists())

    def test_cpp_subprocess_disable_czm_skips_czm_update_and_detach_break(self) -> None:
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.solver.cpp_subprocess import _DoneMsg, _run_simulation
        from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

        config = SimulationConfig(
            enable_czm=False,
            dt=1.0e-3,
            v_lift=1.0e-3,
            layer_thickness=1.0e-6,
            lift_height=1.5e-6,
            max_iters=2,
            N_stable=1,
        )
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=1, layer_thickness=config.layer_thickness, config=config
        )
        mesh_dict = {
            attr: getattr(mesh, attr)
            for attr in (
                "vertices", "velocities", "prev_vertices", "ideal_vertices",
                "node_mass", "active_vertex_mask", "is_top_fixed",
                "is_bottom_surface", "czm_state", "damage", "time_free",
                "tets", "active_tet_mask", "dm_inv", "tet_volumes", "colors",
                "layer_id_per_vertex", "layer_id_per_tet", "first_active_layer",
                "is_top_surface_of_layer",
            )
            if getattr(mesh, attr, None) is not None
        }
        config_dict = {
            "enable_czm": config.enable_czm,
            "dt": config.dt,
            "v_lift": config.v_lift,
            "layer_thickness": config.layer_thickness,
            "lift_height": config.lift_height,
            "max_iters": config.max_iters,
            "N_stable": config.N_stable,
        }

        class FakeConn:
            def __init__(self) -> None:
                self.messages = []

            def send(self, msg) -> None:  # noqa: ANN001
                self.messages.append(msg)

            def poll(self) -> bool:
                return False

        def fake_solve(mesh_arg, cfg_arg, e_z, layer_id, lifting_top):  # noqa: ANN001, ARG001
            return VBDSolveResult(
                x=mesh_arg.vertices,
                v=mesh_arg.velocities,
                iterations=1,
                max_dx=0.0,
                kinetic_energy=0.0,
                stable_steps=1,
                all_free=True,
                chebyshev_skipped_damaging=0,
            )

        conn = FakeConn()
        with (
            patch(
                "hydrogel_vbd.solver.cpp_adapter.solve_lift_and_relax",
                side_effect=fake_solve,
            ),
            patch("hydrogel_vbd.physics.czm.update_czm_states") as update_mock,
        ):
            _run_simulation(conn, mesh_dict, config_dict, n_layers=1, output_dir="outputs/gui")

        update_mock.assert_not_called()
        done_messages = [msg for msg in conn.messages if isinstance(msg, _DoneMsg)]
        self.assertEqual(len(done_messages), 1)
        self.assertEqual(done_messages[0].results[0]["total_steps"], 2)


if __name__ == "__main__":
    unittest.main()
