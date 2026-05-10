"""测试平台运动学求解器 & STL 网格加载 & GUI 参数配置。"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class LiftSolverTests(unittest.TestCase):
    """平台提升-静平衡求解器单元测试。"""

    def setUp(self) -> None:
        from hydrogel_vbd.config import SimulationConfig
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


class VbdSolverSolveUntilStableTests(unittest.TestCase):
    """确保 solve_until_stable 向后兼容。"""

    def setUp(self) -> None:
        from hydrogel_vbd.config import SimulationConfig
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
        app.quit()

    def test_main_window_init(self) -> None:
        from hydrogel_vbd.gui.main_window import MainWindow
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        win = MainWindow()
        self.assertEqual(win.windowTitle(), "Hydrogel VBD Simulator")
        self.assertTrue(win.isVisible() is False)
        app.quit()

    def test_launch_gui_importable(self) -> None:
        from hydrogel_vbd.gui.main_window import launch_gui
        self.assertTrue(callable(launch_gui))


if __name__ == "__main__":
    unittest.main()
