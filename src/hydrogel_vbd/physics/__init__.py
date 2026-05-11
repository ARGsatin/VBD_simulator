"""Physics models — forces, CZM, elastic energy, material models."""

from hydrogel_vbd.physics.czm import CZMState, update_czm_states
from hydrogel_vbd.physics.local_terms import LocalPhysicsTerms, build_local_physics_terms

__all__ = ["CZMState", "LocalPhysicsTerms", "build_local_physics_terms", "update_czm_states"]
