import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ConformalArchitectureTests(unittest.TestCase):
    def test_config_yaml_contains_required_physics_and_control_parameters(self):
        from hydrogel_vbd.core.config import SimulationConfig

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

    def test_layer_activation_leaves_one_layer_gap_above_fixed_fep(self):
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.geometry.layer_activator import LayerActivator

        layer_thickness = 0.05
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=2, layer_thickness=layer_thickness
        )
        mesh.activate_layer(0)
        active_before = mesh.active_vertex_mask.copy()
        previous_bottom = mesh.bottom_nodes(0)
        current_bottom = mesh.bottom_nodes(1)
        mesh.vertices[active_before, 2] += 0.15
        mesh.velocities[:] = 3.0

        LayerActivator().activate_with_inheritance(
            mesh, current_layer=1, z_fep=0.0
        )

        np.testing.assert_allclose(mesh.vertices[current_bottom, 2], 0.0)
        np.testing.assert_allclose(mesh.vertices[previous_bottom, 2], layer_thickness)
        np.testing.assert_allclose(mesh.velocities[active_before], 0.0)
        new_nodes = np.flatnonzero(mesh.first_active_layer == 1)
        self.assertGreater(len(new_nodes), 0)
        np.testing.assert_allclose(mesh.velocities[new_nodes], 0.0)
        self.assertTrue(np.all(mesh.vertices[new_nodes, 2] >= 0.0))
        self.assertTrue(np.all(mesh.vertices[new_nodes, 2] <= layer_thickness))

    def test_new_layer_tets_are_stress_free_at_activation(self):
        from hydrogel_vbd.core.config import SimulationConfig
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.geometry.layer_activator import LayerActivator
        from hydrogel_vbd.physics.elastic_energy import compute_tet_deformation_gradient

        layer_thickness = 0.05
        config = SimulationConfig(c_shrink=1.0, layer_thickness=layer_thickness)
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=2, layer_thickness=layer_thickness, config=config
        )
        activator = LayerActivator()

        activator.activate_with_inheritance(mesh, current_layer=0, z_fep=0.0)
        old_tets = np.flatnonzero(mesh.layer_id_per_tet == 0)
        old_dm_inv = mesh.dm_inv[old_tets].copy()
        old_volumes = mesh.tet_volumes[old_tets].copy()

        shared_interface = mesh.bottom_nodes(0)
        offsets = np.linspace(
            -0.2 * layer_thickness,
            0.2 * layer_thickness,
            len(shared_interface),
        )
        mesh.vertices[shared_interface, 2] += offsets
        expected_inherited_z = layer_thickness + offsets

        activator.activate_with_inheritance(mesh, current_layer=1, z_fep=0.0)

        np.testing.assert_allclose(
            mesh.vertices[shared_interface, 2],
            expected_inherited_z,
        )

        new_tets = np.flatnonzero(mesh.layer_id_per_tet == 1)
        self.assertGreater(len(new_tets), 0)
        for tet_id in new_tets:
            tet = mesh.tets[tet_id]
            F = compute_tet_deformation_gradient(
                mesh.vertices[tet],
                mesh.dm_inv[tet_id],
            )
            np.testing.assert_allclose(F, np.eye(3), atol=1e-10)

        np.testing.assert_allclose(mesh.dm_inv[old_tets], old_dm_inv)
        np.testing.assert_allclose(mesh.tet_volumes[old_tets], old_volumes)

    def test_top_down_activation_keeps_fixed_fep_and_platform_surface(self):
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.geometry.layer_activator import LayerActivator

        layer_thickness = 0.05
        mesh, _ = ConformalMeshPipeline.create_demo(
            layers=3, layer_thickness=layer_thickness
        )
        activator = LayerActivator()

        activator.activate_with_inheritance(mesh, current_layer=0, z_fep=0.0)
        first_bottom = mesh.bottom_nodes(0)
        platform = mesh.top_nodes(0)

        np.testing.assert_allclose(
            mesh.ideal_vertices[first_bottom, 2],
            2.0 * layer_thickness,
        )
        np.testing.assert_allclose(mesh.vertices[first_bottom, 2], 0.0)
        np.testing.assert_allclose(mesh.vertices[platform, 2], layer_thickness)
        np.testing.assert_array_equal(np.flatnonzero(mesh.is_top_fixed), platform)

        activator.activate_with_inheritance(mesh, current_layer=1, z_fep=0.0)
        second_bottom = mesh.bottom_nodes(1)

        np.testing.assert_allclose(
            mesh.ideal_vertices[second_bottom, 2],
            layer_thickness,
        )
        np.testing.assert_allclose(mesh.vertices[second_bottom, 2], 0.0)
        np.testing.assert_allclose(mesh.vertices[first_bottom, 2], layer_thickness)
        np.testing.assert_allclose(mesh.vertices[platform, 2], 2.0 * layer_thickness)
        np.testing.assert_array_equal(np.flatnonzero(mesh.is_top_fixed), platform)

    def test_activation_forces_platform_surface_active_on_first_layer(self):
        from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
        from hydrogel_vbd.geometry.layer_activator import LayerActivator

        mesh, _ = ConformalMeshPipeline.create_demo(layers=2, layer_thickness=0.05)
        platform = mesh.top_nodes(0)
        self.assertGreater(len(platform), 0)
        mesh.first_active_layer[platform] = 1

        LayerActivator().activate_with_inheritance(mesh, current_layer=0, z_fep=0.0)

        lifting_top = np.flatnonzero(mesh.is_top_fixed & mesh.active_vertex_mask)
        np.testing.assert_array_equal(lifting_top, platform)

    def test_occ_tet_layer_classifier_matches_top_down_print_order(self):
        from hydrogel_vbd.geometry.stl_mesher import _classify_occ_tets_to_layers

        tet_z = np.array([0.0005, 0.0015, 0.0025], dtype=float)

        layer_id = _classify_occ_tets_to_layers(
            tet_z,
            z_min_m=0.0,
            n_layers=3,
            layer_thickness_m=0.001,
        )

        np.testing.assert_array_equal(layer_id, np.array([2, 1, 0], dtype=int))

    def test_top_down_slice_grid_keeps_remainder_off_first_printed_layer(self):
        from hydrogel_vbd.geometry.stl_mesher import (
            _classify_occ_tets_to_layers,
            _classify_occ_vertices,
            _effective_top_down_layer_count,
            _top_aligned_slice_origin,
        )

        z_min = 0.0
        z_max = 9.5e-3
        layer_thickness = 0.4996e-3
        n_layers = _effective_top_down_layer_count(
            z_max - z_min,
            layer_thickness,
        )
        origin = _top_aligned_slice_origin(
            z_min,
            z_max,
            n_layers,
            layer_thickness,
        )

        self.assertEqual(n_layers, 19)
        self.assertGreater(origin, z_min)

        top_centroid = z_max - 0.5 * layer_thickness
        bottom_centroid = 0.5 * origin
        layer_id = _classify_occ_tets_to_layers(
            np.array([top_centroid, bottom_centroid]),
            origin,
            n_layers,
            layer_thickness,
        )
        np.testing.assert_array_equal(layer_id, np.array([0, n_layers - 1]))

        point_z = np.array(
            [z_min, origin + layer_thickness, z_max - layer_thickness, z_max]
        )
        first_active, surface_ids = _classify_occ_vertices(
            point_z,
            origin,
            n_layers,
            layer_thickness,
            tol=1e-10,
            model_z_min_m=z_min,
            model_z_max_m=z_max,
        )
        np.testing.assert_array_equal(
            surface_ids,
            np.array([n_layers, n_layers - 1, 1, 0]),
        )
        np.testing.assert_array_equal(
            first_active,
            np.array([n_layers - 1, n_layers - 2, 0, 0]),
        )


if __name__ == "__main__":
    unittest.main()
