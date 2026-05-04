import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class StateAndActivationTests(unittest.TestCase):
    def test_mesh_state_initializes_velocities_and_activation_masks(self):
        from hydrogel_vbd.state import MeshState

        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        mesh = MeshState(
            vertices=vertices,
            tets=tets,
            layer_id_per_vertex=np.array([0, 0, 0, 1, 2]),
            layer_id_per_tet=np.array([1, 2]),
        )

        np.testing.assert_array_equal(mesh.velocities, np.zeros_like(vertices))
        np.testing.assert_array_equal(mesh.active_vertex_mask, np.zeros(5, dtype=bool))
        np.testing.assert_array_equal(mesh.active_tet_mask, np.zeros(2, dtype=bool))

        mesh.activate_layer(1)

        np.testing.assert_array_equal(mesh.active_vertex_mask, np.array([True, True, True, True, False]))
        np.testing.assert_array_equal(mesh.active_tet_mask, np.array([True, False]))

    def test_mesh_state_rejects_invalid_shapes(self):
        from hydrogel_vbd.state import MeshState

        with self.assertRaisesRegex(ValueError, "vertices"):
            MeshState(
                vertices=np.array([0.0, 1.0, 2.0]),
                tets=np.array([[0, 1, 2, 3]]),
                layer_id_per_vertex=np.array([0]),
                layer_id_per_tet=np.array([0]),
            )

    def test_layer_activator_returns_updated_mesh(self):
        from hydrogel_vbd.geometry.layer_activator import LayerActivator
        from hydrogel_vbd.state import MeshState

        mesh = MeshState(
            vertices=np.zeros((4, 3)),
            tets=np.array([[0, 1, 2, 3]]),
            layer_id_per_vertex=np.array([0, 1, 1, 2]),
            layer_id_per_tet=np.array([2]),
        )

        result = LayerActivator().activate(mesh, current_layer=1)

        self.assertIs(result, mesh)
        np.testing.assert_array_equal(result.active_vertex_mask, np.array([True, True, True, False]))
        np.testing.assert_array_equal(result.active_tet_mask, np.array([False]))


if __name__ == "__main__":
    unittest.main()
