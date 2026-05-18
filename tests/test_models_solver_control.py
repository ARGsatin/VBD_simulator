import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ModelsSolverControlTests(unittest.TestCase):
    def test_hydrogel_model_computes_lame_parameters_from_curing_degree(self):
        from hydrogel_vbd.physics.hydrogel_model import HydrogelMaterialModel

        model = HydrogelMaterialModel(
            density=1000.0,
            young_modulus_min=100.0,
            young_modulus_max=1100.0,
            poisson_ratio=0.25,
            damping=0.1,
            curing_exponent=1.0,
            peel_stress_crit=50.0,
            electric_response_alpha=0.2,
        )

        material = model.create_state(np.array([0.0, 0.5, 1.0]))

        np.testing.assert_allclose(material.young_modulus, np.array([100.0, 600.0, 1100.0]))
        np.testing.assert_allclose(material.mu, material.young_modulus / 2.5)
        np.testing.assert_allclose(material.lam, material.young_modulus * 0.25 / (1.25 * 0.5))

    def test_force_models_compute_expected_vectors(self):
        from hydrogel_vbd.physics.electric import ElectricForceModel
        from hydrogel_vbd.physics.gravity import gravity_force
        from hydrogel_vbd.physics.peel import peel_force
        from hydrogel_vbd.core.state import FieldCommand, MeshState

        mesh = MeshState(
            vertices=np.zeros((3, 3)),
            tets=np.array([[0, 1, 2, 2]]),
            layer_id_per_vertex=np.array([0, 0, 1]),
            layer_id_per_tet=np.array([1]),
        )
        mesh.activate_layer(0)

        gravity = gravity_force(mesh, density=2.0, g=(0.0, 0.0, -9.8))
        np.testing.assert_allclose(gravity[:2], np.array([[0.0, 0.0, -19.6], [0.0, 0.0, -19.6]]))
        np.testing.assert_allclose(gravity[2], np.zeros(3))

        peel = peel_force(mesh, pressure=3.0, normal=(0.0, 0.0, 1.0), vertex_area=2.0)
        np.testing.assert_allclose(peel[:2], np.array([[0.0, 0.0, 6.0], [0.0, 0.0, 6.0]]))

        command = FieldCommand(voltage=np.array([2.0, -1.0]), electrode_ids=["left", "right"])
        electric = ElectricForceModel(alpha=0.5, direction=(0.0, 0.0, 1.0)).compute(mesh, command)
        np.testing.assert_allclose(electric[:2], np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]))
        np.testing.assert_allclose(electric[2], np.zeros(3))

    def test_reference_solver_moves_unfixed_active_vertices_under_force(self):
        from hydrogel_vbd.solver.constraints import fixed_z_constraints
        from hydrogel_vbd.solver.vbd_solver import PythonReferenceVBDSolver
        from hydrogel_vbd.core.state import MeshState

        mesh = MeshState(
            vertices=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            tets=np.array([[0, 1, 1, 1]]),
            layer_id_per_vertex=np.array([0, 0]),
            layer_id_per_tet=np.array([0]),
        )
        mesh.activate_layer(0)
        constraints = fixed_z_constraints(mesh, z_value=0.0)
        forces = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]])

        x_next, v_next = PythonReferenceVBDSolver(damping=0.0).step(
            mesh=mesh,
            forces=forces,
            constraints=constraints,
            dt=0.1,
            substeps=1,
            iterations=1,
        )

        np.testing.assert_allclose(x_next[0], np.array([0.0, 0.0, 0.0]))
        self.assertGreater(x_next[1, 2], 1.0)
        self.assertGreater(v_next[1, 2], 0.0)

    def test_shape_metrics_and_voltage_optimizer(self):
        from hydrogel_vbd.control.voltage_optimizer import solve_regularized_voltage
        from hydrogel_vbd.evaluation.shape_error import compare_shapes

        target = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        simulated = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        metrics = compare_shapes(simulated, target)

        self.assertAlmostEqual(metrics["max_error"], 1.0)
        self.assertAlmostEqual(metrics["rms_error"], np.sqrt(0.5))
        self.assertAlmostEqual(metrics["max_z_sag"], 1.0)

        mapping = np.eye(2)
        voltage = solve_regularized_voltage(mapping, np.array([1.0, -2.0]), regularization=0.0)
        np.testing.assert_allclose(voltage, np.array([1.0, -2.0]))

    def test_bottom_z_controller_maps_bottom_sag_to_positive_field(self):
        from hydrogel_vbd.control.field_controller import BottomZFieldController
        from hydrogel_vbd.core.config import SimulationConfig

        config = SimulationConfig(
            err_target=0.05,
            K_p=10.0,
            K_i=0.0,
            K_d=0.0,
            q_ion=2.0,
            E_max=10.0,
        )
        controller = BottomZFieldController(config, regularization=0.0)
        target = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        )
        simulated = np.array(
            [[0.0, 0.0, 0.8], [0.0, 0.0, 0.9], [0.0, 0.0, 1.0]]
        )

        state = controller.update(
            bottom_nodes=np.array([0, 1], dtype=int),
            target_vertices=target,
            simulated_vertices=simulated,
        )

        self.assertAlmostEqual(state.E_z, 0.5)
        self.assertAlmostEqual(state.bottom_z_mean_error, 0.15)
        self.assertAlmostEqual(state.bottom_z_max_error, 0.2)

    def test_bottom_z_controller_ignores_empty_or_overlifted_bottom(self):
        from hydrogel_vbd.control.field_controller import BottomZFieldController
        from hydrogel_vbd.core.config import SimulationConfig

        config = SimulationConfig(err_target=0.01, K_p=10.0, K_d=0.0, E_max=5.0)
        controller = BottomZFieldController(config, regularization=0.0)
        target = np.array([[0.0, 0.0, 1.0]])

        empty = controller.update(
            bottom_nodes=np.array([], dtype=int),
            target_vertices=target,
            simulated_vertices=target.copy(),
        )
        self.assertEqual(empty.E_z, 0.0)

        overlifted = controller.update(
            bottom_nodes=np.array([0], dtype=int),
            target_vertices=target,
            simulated_vertices=np.array([[0.0, 0.0, 1.2]]),
        )
        self.assertEqual(overlifted.E_z, 0.0)

    def test_bottom_z_controller_derivative_uses_dt_and_voltage_clip(self):
        from hydrogel_vbd.control.field_controller import BottomZFieldController
        from hydrogel_vbd.core.config import SimulationConfig

        config = SimulationConfig(
            err_target=0.0,
            K_p=0.0,
            K_i=0.0,
            K_d=2.0,
            dt=0.5,
            q_ion=1.0,
            E_max=0.25,
        )
        controller = BottomZFieldController(config, regularization=0.0)
        target = np.array([[0.0, 0.0, 1.0]])

        first = controller.update(
            bottom_nodes=np.array([0], dtype=int),
            target_vertices=target,
            simulated_vertices=np.array([[0.0, 0.0, 0.9]]),
        )
        self.assertEqual(first.E_z, 0.0)

        second = controller.update(
            bottom_nodes=np.array([0], dtype=int),
            target_vertices=target,
            simulated_vertices=np.array([[0.0, 0.0, 0.8]]),
        )
        self.assertAlmostEqual(second.unclipped_E_z, 0.4)
        self.assertAlmostEqual(second.E_z, 0.25)


if __name__ == "__main__":
    unittest.main()
