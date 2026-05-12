# -*- coding: utf-8 -*-
"""C++ 加速求解器适配层。

将 SimulationConfig 和 MeshState 转换为 C++ pybind11 模块所需的格式，
提供与 PythonReferenceVBDSolver 相同的接口。

若 C++ 模块不可用（未编译或平台不兼容），则优雅降级。
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.core.state import MeshState
from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

_CPP_AVAILABLE = False
_CPP_IMPORT_ERROR: str | None = None

try:
    import hydrogel_vbd_cpp  # type: ignore[import-untyped]

    _CPP_AVAILABLE = True
except ImportError as exc:
    _CPP_IMPORT_ERROR = str(exc)


def is_cpp_available() -> bool:
    """检查 C++ solver 模块是否已编译并可导入。"""
    return _CPP_AVAILABLE


def get_import_error() -> str | None:
    """获取 C++ 模块导入失败的错误信息（仅调试/日志用途）。"""
    return _CPP_IMPORT_ERROR


def refresh_availability() -> bool:
    """编译后重新检测 C++ 模块（支持热加载，无需重启进程）。

    若模块之前导入失败，编译后调用此函数可重新尝试导入。
    若模块已导入，则 reload 以使用最新编译的版本。

    Returns
    -------
    bool
        True 表示 C++ 模块当前可用。
    """
    global _CPP_AVAILABLE, _CPP_IMPORT_ERROR
    import importlib

    try:
        if "hydrogel_vbd_cpp" in sys.modules:
            importlib.reload(sys.modules["hydrogel_vbd_cpp"])
        else:
            import hydrogel_vbd_cpp  # type: ignore[import-untyped]
        _CPP_AVAILABLE = True
        _CPP_IMPORT_ERROR = None
    except ImportError as exc:
        _CPP_AVAILABLE = False
        _CPP_IMPORT_ERROR = str(exc)
    return _CPP_AVAILABLE


def _build_cpp_config(cfg: SimulationConfig) -> Any:
    """将 Python SimulationConfig 转换为 C++ SolverConfig。"""
    cpp_cfg = hydrogel_vbd_cpp.SolverConfig()
    cpp_cfg.dt = cfg.dt
    cpp_cfg.max_iters = cfg.max_iters
    cpp_cfg.N_stable = cfg.N_stable
    cpp_cfg.epsilon = cfg.epsilon
    cpp_cfg.k_d = cfg.k_d
    cpp_cfg.rho_cheb = cfg.rho_cheb
    cpp_cfg.mu = cfg.mu
    cpp_cfg.kappa = cfg.kappa
    cpp_cfg.c_shrink = cfg.c_shrink
    cpp_cfg.q_ion = cfg.q_ion
    cpp_cfg.g_x = cfg.g[0]
    cpp_cfg.g_y = cfg.g[1]
    cpp_cfg.g_z = cfg.g[2]
    cpp_cfg.T_max = cfg.T_max
    cpp_cfg.K_czm = cfg.K_czm
    cpp_cfg.delta_f = cfg.delta_f
    cpp_cfg.node_area = cfg.node_area
    cpp_cfg.z_fep = cfg.z_fep
    cpp_cfg.C_0 = cfg.C_0
    cpp_cfg.eta = cfg.eta
    cpp_cfg.fluid_radius = cfg.fluid_radius
    cpp_cfg.d_min = cfg.d_min
    cpp_cfg.d_fluid_max = cfg.d_fluid_max
    cpp_cfg.t_fluid_max = cfg.t_fluid_max
    cpp_cfg.v_lift = cfg.v_lift
    cpp_cfg.layer_thickness = cfg.layer_thickness
    cpp_cfg.c_init = cfg.c_init
    return cpp_cfg


def solve_until_stable(
    mesh: MeshState,
    config: SimulationConfig,
    e_z: float,
    layer_id: int,
) -> VBDSolveResult:
    """C++ 加速的静力平衡求解（与 PythonReferenceVBDSolver 接口一致）。

    Parameters
    ----------
    mesh : MeshState
        当前网格状态（原地修改）。
    config : SimulationConfig
        物理/求解器参数。
    e_z : float
        均匀电场强度 (V/m)。
    layer_id : int
        当前层编号。

    Returns
    -------
    VBDSolveResult
    """
    if not _CPP_AVAILABLE:
        raise ImportError(
            f"C++ solver 不可用。请编译 C++ 模块后重试。\n"
            f"导入错误: {_CPP_IMPORT_ERROR}"
        )

    cpp_cfg = _build_cpp_config(config)

    result_dict = hydrogel_vbd_cpp.solve_until_stable(
        mesh.vertices,
        mesh.velocities,
        mesh.ideal_vertices,
        mesh.masses,
        mesh.active_vertex_mask,
        mesh.is_top_fixed,
        mesh.is_bottom_surface.astype(bool),
        mesh.czm_state,
        mesh.damage,
        mesh.time_free,
        mesh.tets,
        mesh.active_tet_mask,
        mesh.dm_inv,
        mesh.tet_volumes,
        mesh.colors,
        cpp_cfg,
        e_z,
        layer_id,
    )

    return VBDSolveResult(
        x=mesh.vertices,
        v=mesh.velocities,
        iterations=result_dict["iterations"],
        max_dx=result_dict["max_dx"],
        kinetic_energy=result_dict["kinetic_energy"],
        stable_steps=result_dict["stable_steps"],
        all_free=result_dict["all_free"],
        chebyshev_skipped_damaging=result_dict["chebyshev_skipped_damaging"],
    )


def solve_with_lift(
    mesh: MeshState,
    config: SimulationConfig,
    e_z: float,
    layer_id: int,
    lifting_top: np.ndarray,
) -> VBDSolveResult:
    """C++ 加速的平台提升剥离 + 静平衡求解。

    Parameters
    ----------
    mesh : MeshState
        当前网格状态（原地修改）。
    config : SimulationConfig
        物理/求解器参数。
    e_z : float
        均匀电场强度 (V/m)。
    layer_id : int
        当前层编号。
    lifting_top : np.ndarray, shape (K,)
        平台提升阶段需要抬升的顶层节点索引。

    Returns
    -------
    VBDSolveResult
    """
    if not _CPP_AVAILABLE:
        raise ImportError(
            f"C++ solver 不可用。请编译 C++ 模块后重试。\n"
            f"导入错误: {_CPP_IMPORT_ERROR}"
        )

    cpp_cfg = _build_cpp_config(config)
    lifting_top_list = lifting_top.astype(int).tolist()

    result_dict = hydrogel_vbd_cpp.solve_with_lift(
        mesh.vertices,
        mesh.velocities,
        mesh.ideal_vertices,
        mesh.masses,
        mesh.active_vertex_mask,
        mesh.is_top_fixed,
        mesh.is_bottom_surface.astype(bool),
        mesh.czm_state,
        mesh.damage,
        mesh.time_free,
        mesh.tets,
        mesh.active_tet_mask,
        mesh.dm_inv,
        mesh.tet_volumes,
        mesh.colors,
        cpp_cfg,
        e_z,
        layer_id,
        lifting_top_list,
    )

    return VBDSolveResult(
        x=mesh.vertices,
        v=mesh.velocities,
        iterations=result_dict["iterations"],
        max_dx=result_dict["max_dx"],
        kinetic_energy=result_dict["kinetic_energy"],
        stable_steps=result_dict["stable_steps"],
        all_free=result_dict["all_free"],
        chebyshev_skipped_damaging=result_dict["chebyshev_skipped_damaging"],
    )
