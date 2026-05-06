import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ConformalArchitectureTests(unittest.TestCase):
    def test_config_yaml_contains_required_physics_and_control_parameters(self):
        from hydrogel_vbd.config import SimulationConfig

        config = SimulationConfig.from_yaml(ROOT / "configs" / "config.yaml")

        self.assertEqual(config.g, (0.0, 0.0, -9.81))
        self.assertEqual(config.rho, 1050.0)
        self.assertGreater(config.kappa, config.mu)
        self.assertEqual(config.c_shrink, 0.98)
        self.assertEqual(config.N_stable, 10)
        self.assertEqual(config.E_max, 500.0)

    def test_conformal_demo_mesh_shares_layer_interface_nodes_and_colors_graph(self):
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline

        mesh, total_layers = ConformalMeshPipeline.create_demo(layers=3, layer_thickness=0.05)

        self.assertEqual(total_layers, 3)
        interface_ids = mesh.layer_interface_nodes(1)
        self.assertGreater(len(interface_ids), 0)
        self.assertTrue(np.all(mesh.first_active_layer[interface_ids] == 0))
        self.assertTrue(np.all(mesh.is_top_surface_of_layer[interface_ids] == 1))
        self.assertTrue(np.any(mesh.layer_id_per_tet == 1))

        for node_id, neighbors in enumerate(mesh.neighbors):
            for neighbor_id in neighbors:
                self.assertNotEqual(mesh.colors[node_id], mesh.colors[neighbor_id])

    def test_layer_activation_inherits_shape_clamps_fep_and_resets_new_velocities(self):
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.geometry.layer_activator import LayerActivator

        mesh, _ = ConformalMeshPipeline.create_demo(layers=2, layer_thickness=0.05)
        mesh.activate_layer(0)
        bottom = mesh.bottom_nodes(0)
        mesh.vertices[bottom, 2] = -0.01
        mesh.velocities[:] = 3.0

        LayerActivator().activate_with_inheritance(mesh, current_layer=1, z_fep=0.0)

        np.testing.assert_allclose(mesh.vertices[bottom, 2], 0.0)
        new_nodes = np.flatnonzero(mesh.first_active_layer == 1)
        self.assertGreater(len(new_nodes), 0)
        np.testing.assert_allclose(mesh.velocities[new_nodes], 0.0)
        self.assertTrue(np.all(mesh.vertices[new_nodes, 2] >= 0.0))
        self.assertTrue(np.all(mesh.vertices[new_nodes, 2] <= mesh.ideal_vertices[new_nodes, 2]))


if __name__ == "__main__":
    unittest.main()
