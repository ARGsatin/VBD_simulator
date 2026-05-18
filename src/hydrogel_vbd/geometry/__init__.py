"""Geometry preprocessing and layer activation."""

from hydrogel_vbd.geometry.conformal_pipeline import ConformalMeshPipeline
from hydrogel_vbd.geometry.layer_activator import LayerActivator
from hydrogel_vbd.geometry.stl_slicer import load_stl, slice_stl
from hydrogel_vbd.geometry.tet_mesher import from_stl as tet_mesh_from_stl

__all__ = [
    "ConformalMeshPipeline",
    "LayerActivator",
    "load_stl",
    "slice_stl",
    "tet_mesh_from_stl",
]
