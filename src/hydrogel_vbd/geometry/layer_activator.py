from __future__ import annotations

from hydrogel_vbd.state import MeshState


class LayerActivator:
    """Updates mesh active masks for the current print layer."""

    def activate(self, mesh: MeshState, current_layer: int) -> MeshState:
        mesh.activate_layer(current_layer)
        return mesh
